import os
import re
import json
import tqdm
import torch
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs

from .utils import Logger, Recorder
from reinforced_cot.utils.utils import set_seed
from reinforced_cot.utils import DatasetPreprocessor, AnswerProcessor


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

        # placeholder for dataloaders
        self.train_dataloader = None
        self.test_dataloader = None

        # declare common parameters
        self._declare_common_parameters()

        # placeholder for logger and recorder
        self.recorder = None
        self.logger = None

        # ! logger and recorder should only be created and called in the main process
        if self.accelerator.is_main_process:
            self.logger = Logger(log_level="DEBUG", resume=False, log_dir=config["log_dir"], tag=str(self))
            self.logger.INFO(str(json.dumps(config, indent=4)))
            self.recorder = Recorder(self.logger.tb_dir)

    def _declare_common_parameters(self):
        self.max_prompt_length = self.config.get("max_prompt_length", 2024)
        self.max_completion_length = self.config.get("max_completion_length", 1024)

    def inference(
        self,
        image,
        instruction,
        max_new_tokens: int = 128,
    ):
        """
        Inference on single sample and output CoT and answer.
        """
        self.model.eval()
        self.model.to(self.device)

        # 构造对话模板
        system_prompt = DatasetPreprocessor.INSTRUCTION
        user_prompt = DatasetPreprocessor.QUESTION_TRIGGER + instruction
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
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

    def evaluate(self, tag: str = None):
        self.model.eval()

        total_correct = 0
        total_consist = 0
        total_count = 0

        local_log_samples = []

        for batch in tqdm(self.test_dataloader, disable=not self.accelerator.is_main_process, desc="Eval Loop"):
            images = batch["images"]
            sys_prompts = batch["system_prompt"]
            user_prompts = batch["user_prompt"]
            answers = batch["answers"]
            image_ids = batch.get("image_id", [None] * len(images))

            messages_batch = []
            for img, instr, sp in zip(images, user_prompts, sys_prompts):
                messages_batch.append(
                    [
                        {"role": "system", "content": sp},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": instr},
                            ],
                        },
                    ]
                )

            text_batch = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
            model_inputs = self.processor(
                text=text_batch,
                images=images,
                padding="max_length",
                max_length=self.max_prompt_length,
                truncation=True,
                return_tensors="pt",
            )
            model_inputs = {k: v.to(self.accelerator.device) for k, v in model_inputs.items()}

            bad_words_ids = self.processor.tokenizer(["<tool_call>"], add_special_tokens=False).input_ids

            with torch.no_grad():
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                output_ids = unwrapped_model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_completion_length,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    bad_words_ids=bad_words_ids,
                )

            gen_only = output_ids[:, model_inputs["input_ids"].shape[1] :]
            output_texts = self.processor.batch_decode(
                gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            for img_id, gt_ans, pred_text, instr in zip(image_ids, answers, output_texts, user_prompts):
                pred_think, pred_ans = AnswerProcessor.extract_answer_and_cot(pred_text)
                is_correct = int(AnswerProcessor.is_match(pred_ans, gt_ans))
                is_consist = int(AnswerProcessor.is_match(pred_think, gt_ans)) if is_correct == 1 else 0

                total_correct += is_correct
                total_consist += is_consist
                total_count += 1

                local_log_samples.append(
                    {
                        "image_id": img_id,
                        "instruction": instr,
                        "gt_answer": gt_ans,
                        "prediction_raw": pred_text,
                        "pred_think": pred_think,
                        "pred_answer": pred_ans,
                        "is_correct": is_correct,
                        "is_consist": is_consist,
                    }
                )

        results_tensor = torch.tensor(
            [total_correct, total_consist, total_count], dtype=torch.float32, device=self.accelerator.device
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(results_tensor, op=torch.distributed.ReduceOp.SUM)
        global_correct, global_consist, global_total = (
            int(results_tensor[0].item()),
            int(results_tensor[1].item()),
            int(results_tensor[2].item()),
        )

        accuracy = global_correct / global_total if global_total > 0 else 0.0
        consistency = global_consist / global_correct if global_correct > 0 else 0.0
        if self.accelerator.is_main_process:
            json_path = "eval_samples.json" if not tag else f"eval_samples_{tag}.json"
            if local_log_samples:
                res_path = os.path.join(self.logger.result_dir, json_path)
                with open(res_path, "w", encoding="utf-8") as f:
                    json.dump(local_log_samples, f, indent=2, ensure_ascii=False)
                self.logger.INFO(f"Saved {len(local_log_samples)} evaluation samples to {res_path}")
                self.logger.INFO(f"Evaluation accuracy: {accuracy:.4f}")
                self.logger.INFO(f"Consistent samples percentage: {consistency:.4f}")
        return accuracy

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
