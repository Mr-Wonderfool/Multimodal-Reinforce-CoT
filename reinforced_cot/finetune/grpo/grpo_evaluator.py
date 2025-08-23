import os
import json
import torch
from tqdm import tqdm
from peft import PeftModel
from reinforced_cot.common import BaseVLM
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from reinforced_cot.utils import DatasetPreprocessor, AnswerProcessor


class GRPOEvaluator(BaseVLM):
    def __init__(self, config):
        super().__init__(config)
        # load pre-trained weights
        self._setup_models()
        # prepare dataloaders
        self._prepare_datasets()
        # store constants
        self.max_input_length = self.config["max_prompt_length"]
        self.max_response_length = self.config["max_completion_length"]

    def _setup_models(self):
        quantization_config = None
        if self.config["quantization"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.processor = AutoProcessor.from_pretrained(self.config["tokenizer_path"])
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # base model without lora adapters
        base_model = AutoModelForVision2Seq.from_pretrained(
            self.config["model_path"],
            quantization_config=quantization_config,
        )

        grpo_adapters_path = self.config["grpo_model_path"]
        if self.accelerator.is_main_process:
            self.logger.INFO(f"Loading LoRA adapter from {grpo_adapters_path}...")
        self.model = PeftModel.from_pretrained(base_model, grpo_adapters_path)

    def _prepare_datasets(self):
        # hack batch size for train and val
        self.config["pipeline"]["train"]["batch_size"] = 1
        self.config["pipeline"]["val"]["batch_size"] = 1

        (
            (self.train_dataset, self.train_dataloader),
            (self.val_dataset, self.val_dataloader),
            (self.test_dataset, self.test_dataloader),
        ) = DatasetPreprocessor.prepare_datasets_and_data_loaders(
            args=self.config, accelerator=self.accelerator, processor=self.processor
        )

    def evaluate(self, tag: str = None):
        self.model.eval()

        total_correct = 0
        total_consist = 0
        total_count = 0

        local_log_samples = []

        for batch in tqdm(self.test_dataloader, disable=not self.accelerator.is_main_process, desc="Eval Loop"):
            images = batch["images"]  # list[PIL.Image] 或 tensor
            sys_prompts = batch["system_prompt"]
            user_prompts = batch["user_prompt"]  # list[str]
            answers = batch["answers"]  # list[str]
            image_ids = batch.get("image_id", [None] * len(images))  # 假设 batch 可能包含 "image_id"

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

            # 用processor得到输入
            text_batch = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
            model_inputs = self.processor(
                text=text_batch,
                images=images,
                padding="max_length",
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            model_inputs = {k: v.to(torch.device("cuda:0")) for k, v in model_inputs.items()}

            bad_words_ids = self.processor.tokenizer(["<tool_call>"], add_special_tokens=False).input_ids

            with torch.no_grad():
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                output_ids = unwrapped_model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_response_length,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    bad_words_ids=bad_words_ids,
                )

            # 只取生成段
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
                        "prediction_raw": pred_text,  # 原始整段生成
                        "pred_think": pred_think,  # 抽取的思维链
                        "pred_answer": pred_ans,  # 抽取的答案
                        "is_correct": is_correct,
                        "is_consist": is_consist,
                    }
                )

        # 分布式结果聚合
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

    def __str__(self):
        return "GRPOEvaluator"
