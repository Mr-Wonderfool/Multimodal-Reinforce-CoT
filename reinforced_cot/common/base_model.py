import os
import json
import torch
import threading
import torch.nn as nn
from datetime import timedelta
from peft import get_peft_model, LoraConfig, PeftModel
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoModelForCausalLM, AutoConfig

from .utils import Logger, Recorder
from reinforced_cot.utils.utils import set_seed
from reinforced_cot.utils.preprocess import DatasetPreprocessor


class BaseVLM:

    # timeout < runtime error < compile error < partial pass < full pass
    REWARD_CONFIG = {
        "未提供代码": -0.5,
        "未找到 Solution 类": -0.3,
        "Solution 类中未找到方法": -0.5,
        "执行主体出错": -1.0,
        "执行超时": -1.5,
        # internal error for dataset
        "未提供测试": 0.0,
        "无测试用例": 0.0,
    }

    def __init__(self, config):
        self.train_config = config["pipeline"]["train"]
        self.test_config = config["pipeline"]["test"]
        self.optimizer_config = self.train_config["optimizer"]
        self.config = config

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))],
        )
        set_seed(self.train_config["seed"] + self.accelerator.process_index)

        # placeholder for model, tokenizer
        self.model = None
        self.tokenizer = None

        # placeholder for logger and recorder
        self.recorder = None
        self.logger = None

        # ! logger and recorder should only be created and called in the main process
        if self.accelerator.is_main_process:
            self.logger = Logger(log_level="DEBUG", resume=False, log_dir=config["log_dir"], tag=str(self))
            self.logger.INFO(str(json.dumps(config, indent=4)))
            self.recorder = Recorder(self.logger.tb_dir)

        # evaluation metric
        self.ret_dict = {"reward": 0.0, "info": ""}

    def inference(
        self,
        problem_description: str,
        max_new_tokens: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Generates a Chain-of-Thought and an answer code for a single Leetcode problem.

        Parameters:
            problem_description (str): The full text of the Leetcode problem.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            tuple[str, str]: A tuple containing the predicted Chain-of-Thought
                and the predicted answer code.
        """
        self.model.eval()
        self.model.to(device)

        inputs = self.tokenizer(problem_description, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_output = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        generated_text_only = full_output[len(problem_description) :]

        if DatasetPreprocessor.answer_trigger in generated_text_only:
            parts = generated_text_only.split(DatasetPreprocessor.answer_trigger, 1)
            pred_cot = parts[0].strip()
            pred_answer_code = parts[1].strip()
            return pred_cot, pred_answer_code
        else:
            print("Warning: No answer code found in the generated text.")
            return generated_text_only.strip(), ""

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save(self, save_path: str):
        """
        Save function that is only intended to be called by the main process
        """
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def __str__(self):
        return "BaseCodeModel"

    def _run_with_timeout(self, func, timeout: float):
        result = {}

        def wrapper():
            try:
                func()
                result["status"] = "success"
            except AssertionError:
                result["status"] = "assert_error"
            except Exception:
                result["status"] = "runtime_error"

        thread = threading.Thread(target=wrapper)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return "timeout"
        return result.get("status", "runtime_error")

    def _prepare_eval_return(self, ret_info: str):
        return {"reward": self.REWARD_CONFIG.get(ret_info, 0.0), "info": ret_info, "status": ret_info}

    def evaluate_code_generation(self, code: str, prompt: str, test_func: str) -> float:
        """
        Run all the test cases and return eval metrics from compiler
        Returns:
            ret_dict: {[reward]: float, [info]: str}
        """

        if code is None:
            return self._prepare_eval_return("未提供代码")

        if test_func is None:
            return self._prepare_eval_return("未提供测试")

        # 提取测试代码中所有 assert 语句
        assert_lines = [line.strip() for line in test_func.strip().split("\n") if line.strip().startswith("assert ")]
        total = len(assert_lines)
        if total == 0:
            return self._prepare_eval_return("无测试用例")

        correct_results = 0
        wrong_results = 0
        runtime_errors = 0

        full_code = f"{prompt}\n\n{code}"
        global_env = {}

        def exec_main_code():
            exec(full_code, global_env)

        status = self._run_with_timeout(exec_main_code, timeout=15.0)
        if status == "timeout":
            return self._prepare_eval_return("执行超时")
        elif status == "runtime_error":
            return self._prepare_eval_return("执行主体出错")

        Solution = global_env.get("Solution", None)
        if Solution is None:
            return self._prepare_eval_return("未找到 Solution 类")

        method_names = [
            name for name in dir(Solution) if not name.startswith("__") and callable(getattr(Solution, name))
        ]
        if not method_names:
            return self._prepare_eval_return("Solution 类中未找到方法")

        method_name = method_names[0]

        def candidate(**kwargs):
            return getattr(Solution(), method_name)(**kwargs)

        global_env["candidate"] = candidate

        for line in assert_lines:

            def exec_assert():
                exec(line, global_env)

            status = self._run_with_timeout(exec_assert, timeout=5.0)

            if status == "success":
                correct_results += 1
            elif status == "assert_error":
                wrong_results += 1
            elif status == "timeout":
                runtime_errors += 1
            else:
                runtime_errors += 1

        score = correct_results / total if total > 0 else 0.0
        ret_info = f"执行成功: {total}个断言, {correct_results}个通过, {wrong_results}个失败, {runtime_errors}个错误."
        return {
            "reward": round(score, 4),
            "info": ret_info,
            "status": "测试通过",
        }  # 返回 0.0000 ~ 1.0000 的得分


class PolicyAndValueModel(nn.Module):
    def __init__(self, model_path: str, config=None, peft_config: LoraConfig = None):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(model_path)

        self.policy_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)

        hidden_size = self.policy_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False, dtype=torch.bfloat16)

        self.config = self.policy_model.config

        if peft_config is not None:
            if not isinstance(self.policy_model, PeftModel):
                self.policy_model = get_peft_model(self.policy_model, peft_config)
                self.policy_model.print_trainable_parameters()
            else:
                self.logger.INFO("Model is already a PeftModel. Skipping LoRA application.")

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor = None, **kwargs):
        outputs = self.policy_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states or outputs.base_model_past_key_values.hidden_states

        last_hidden_state = hidden_states[-1]
        values = self.value_head(last_hidden_state).squeeze(-1)

        return logits, values

    def generate(self, *args, **kwargs):
        return self.policy_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str):
        print(f"Saving model components to {save_directory}")
        self.policy_model.save_pretrained(save_directory)

        value_head_path = os.path.join(save_directory, "value_head.pth")
        torch.save(self.value_head.state_dict(), value_head_path)

        print(f"LoRA adapters and value head saved successfully.")


# class RewardModel(PreTrainedModel):
#     config_class = AutoConfig

#     def __init__(self, config: PretrainedConfig, model_path: str):
#         super().__init__(config)
#         self.reward_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

#         hidden_size = self.reward_model.config.hidden_size
#         self.reward_head = nn.Linear(hidden_size, 1, bias=False)

#     def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor = None, **kwargs):
#         outputs = self.reward_model(
#             input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs
#         )
#         last_hidden_state = outputs.hidden_states[-1]

#         # 使用最后一个有效token的隐藏状态来计算分数
#         sequence_lengths = torch.ne(input_ids, self.reward_model.config.pad_token_id).sum(-1) - 1
#         last_token_hidden_state = last_hidden_state[
#             torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), sequence_lengths
#         ]

#         scores = self.reward_head(last_token_hidden_state).squeeze(-1)
#         return scores


class RewardModel:
    def __init__(self, constant_reward: float = 1.0):

        self.constant_reward = constant_reward
        print(f"✅ ConstantRewardModel initialized. It will always return {self.constant_reward}.")

    def __call__(self, prompt: str, response: str, **kwargs) -> float:

        # This model ignores the prompt and response to return a fixed score.
        return self.constant_reward
