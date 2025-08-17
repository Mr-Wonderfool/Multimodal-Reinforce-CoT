import os
import re
import json
import torch
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs

from .utils import Logger, Recorder
from reinforced_cot.utils.utils import set_seed
from reinforced_cot.utils.preprocess import DatasetPreprocessor


class BaseVLM:

    def __init__(self, config):
        self.train_config = config["pipeline"]["train"]
        self.test_config = config["pipeline"]["test"]
        self.optimizer_config = self.train_config["optimizer"]
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))],
        )
        set_seed(self.train_config["seed"] + self.accelerator.process_index)

        # placeholder for model
        self.model = None
        self.processor = None

        # placeholder for logger and recorder
        self.recorder = None
        self.logger = None

        # ! logger and recorder should only be created and called in the main process
        if self.accelerator.is_main_process:
            self.logger = Logger(log_level="DEBUG", resume=False, log_dir=config["log_dir"], tag=str(self))
            self.logger.INFO(str(json.dumps(config, indent=4)))
            self.recorder = Recorder(self.logger.tb_dir)

    def inference(
        self,
        image,  # PIL Image 或 tensor
        instruction,  # str
        max_new_tokens: int = 128,
    ):
        """
        对单个样本做推理，生成思维链和答案
        """
        self.model.eval()
        self.model.to(self.device)

        # 构造对话模板
        system_prompt = DatasetPreprocessor.INSTRUCTION
        user_prompt = DatasetPreprocessor.QUESTION_TRIGGER + instruction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.config.get("max_input_length", 1024),
            truncation=True,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            output_ids = unwrapped_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # isolate only the newly generated tokens, excluding the input prompt
        input_token_len = model_inputs["input_ids"].shape[1]
        generated_token_ids = output_ids[0, input_token_len:]
        # decode only the generated part
        generated_text = self.processor.decode(
            generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # parse answer contents within tags
        think_match = re.search(r"<think>\n?(.*?)\n?</think>", generated_text, re.DOTALL)
        answer_match = re.search(r"<answer>\n?(.*?)\n?</answer>", generated_text, re.DOTALL)
        pred_cot = ""
        pred_answer = ""

        if think_match:
            pred_cot = think_match.group(1).strip()

        if answer_match:
            pred_answer = answer_match.group(1).strip()

        # provide a sensible fallback if the model completely fails to generate the tags.
        if not think_match and not answer_match:
            # If no tags are found, we can treat the entire un-tagged output as the CoT.
            pred_cot = generated_text.strip()

        return {"cot": pred_cot, "answer": pred_answer}

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
        return "BaseMultiModalVQAModel"
