import os
import json
import torch
from tqdm import tqdm
import torch.distributed
from torch.optim import AdamW
from collections import defaultdict
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, get_linear_schedule_with_warmup

from reinforced_cot.common import BaseVLM
from reinforced_cot.utils.preprocess import DatasetPreprocessor


class SupervisedFineTuning(BaseVLM):
    def __init__(self, sft_config: dict):
        super().__init__(sft_config)
        # apply quantization if specified
        quantization_config = None
        if sft_config["quantization"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # 加载多模态processor和模型
        self.processor = AutoProcessor.from_pretrained(sft_config["tokenizer_path"])
        self.model = AutoModelForVision2Seq.from_pretrained(
            sft_config["model_path"], quantization_config=quantization_config
        )
        self.tokenizer = self.processor.tokenizer  
        # apply LoRA configuration if specified
        if "lora" in sft_config:
            if self.accelerator.is_main_process:
                self.logger.INFO("Applying LoRA configuration...")

            lora_config = LoraConfig(**sft_config["lora"])
            self.model = get_peft_model(self.model, lora_config)

            if self.accelerator.is_main_process:
                self.logger.INFO("LoRA applied successfully.")
                self.model.print_trainable_parameters()

        (self.train_dataset, self.train_dataloader), (self.val_dataset, self.val_dataloader), (self.test_dataset, self.test_dataloader) = (
            DatasetPreprocessor.prepare_datasets_and_data_loaders(
                args=sft_config, accelerator=self.accelerator, processor=self.processor
            )
        )

        self.n_epochs = self.train_config["n_epochs"]
        self.max_input_length = sft_config["max_input_length"]
        self.max_response_length = sft_config["max_response_length"]
        
        num_training_steps = (
            len(self.train_dataloader) // self.accelerator.num_processes * self.n_epochs
        ) // sft_config["gradient_accumulation_steps"]
        # automatically adjust warm up steps
        warmup_step = int(0.1 * num_training_steps)
        self.clip_grad_norm = self.train_config["clip_grad_norm"]

        optimizer_grouped_parameters = (
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                    ],
                    "weight_decay": self.optimizer_config["weight_decay"],
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if "lora" not in sft_config
            else self.model.parameters()
        )

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.optimizer_config["learning_rate"], eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=num_training_steps,
        )
        # further modify dataloader with deepseed
        self.model, self.optimizer, self.train_dataloader, self.test_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.test_dataloader
        )

        self.global_step = 0
        self.evaluating_epoch_freq = self.test_config["evaluating_epoch_freq"]
        self.logging_epoch_freq = self.train_config["logging_epoch_freq"]
        self.saving_epoch_freq = self.train_config["saving_epoch_freq"]

        # log updated configuration
        if self.accelerator.is_main_process:
            self.logger.INFO(
                str(
                    json.dumps(
                        {
                            "pad_token_id": self.tokenizer.pad_token_id,
                            "eos_token_id": self.tokenizer.eos_token_id,
                            "vocab_size": len(self.tokenizer),
                        },
                        indent=4,
                    )
                )
            )

    def _train_one_epoch(self):
        self.model.train()
        epoch_result_dict = defaultdict(list)
        with tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            disable=not self.accelerator.is_main_process,
            desc="Train Loop",
        ) as t:
            for idx, batch in t:
                with self.accelerator.accumulate(self.model):
                    # pop logging info
                    batch = {k: v for k, v in batch.items() if k != "forward_kwargs"}  # 保险起见
                    output = self.model(**batch)
                    loss = output.loss
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                    self.optimizer.step()
                    if self.accelerator.sync_gradients:
                        self.scheduler.step()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    epoch_result_dict["loss"].append(loss.item())

                    if self.accelerator.is_main_process:
                        self.recorder.record("train/batch_loss", loss, self.global_step)

        return epoch_result_dict

    def train(self):
        best_pass_rate = -float("inf")
        curr_pass_rate = 0

        with tqdm(range(1, self.n_epochs + 1), total=self.n_epochs, disable=False) as t:
            for epoch in t:
                epoch_result_dict = self._train_one_epoch()
                curr_loss = sum(epoch_result_dict["loss"]) / len(epoch_result_dict["loss"])
                if epoch % self.evaluating_epoch_freq == 0:
                    curr_pass_rate = self.evaluate(tag=str(epoch))
                    if self.accelerator.is_main_process:
                        # record pass rates
                        self.recorder.record("test/pass_rate", curr_pass_rate, step=self.global_step)
                    # model saving
                    if epoch % self.saving_epoch_freq == 0:
                        # Make sure all processes have the same pass_rate before checking
                        self.accelerator.wait_for_everyone()
                        if curr_pass_rate > best_pass_rate:
                            best_pass_rate = curr_pass_rate
                            if self.accelerator.is_main_process:
                                save_path = os.path.join(
                                    self.logger.ckpt_dir,
                                    f"epoch_{epoch}_loss_{curr_loss:.2f}_pass_{curr_pass_rate:.2f}",
                                )
                                self.logger.INFO(
                                    f"New best model found at epoch {epoch} with loss {curr_loss:.4f} and pass rate: {curr_pass_rate}. Saving to {save_path} ..."
                                )
                                self.save(save_path)
                                self.logger.INFO(f"Finish saving model to {save_path}")

                # logging
                if self.accelerator.is_main_process:
                    epoch_result_dict["epoch"] = epoch
                    epoch_result_dict["loss"] = curr_loss
                    self.logger.INFO(f"IN EPOCH {epoch}: loss {curr_loss:.4f}")
                    self.recorder.record("train/epoch_loss", curr_loss, step=self.global_step)

        # final dump of records
        if self.accelerator.is_main_process:
            self.recorder.close()

    def evaluate(self, tag: str = None):
        self.model.eval()
        # dataloader = self.val_dataloader if use_val and self.val_dataloader is not None else self.test_dataloader

        total_correct = 0
        total_count = 0

        local_log_samples = []

        for batch in tqdm(
            self.test_dataloader, disable=not self.accelerator.is_main_process, desc="Eval Loop"
        ):
            images = batch["images"]           # list[PIL.Image] 或 tensor
            instructions = batch["instructions"]  # list[str]
            answers = batch["answers"]             # list[str]
            messages_batch = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": instr}
                    ]
                }
                for img, instr in zip(images, instructions)
            ]


            # 用processor得到输入
            text_batch = [self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages_batch]
            model_inputs = self.processor(
                text=text_batch,
                images=images,
                padding="max_length",
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt"
            )
            model_inputs = {k: v.to(self.accelerator.device) for k, v in model_inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_response_length,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            # 解码输出
            output_texts = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # 逐条判分（字符串比较，可按实际需要自定义）
            for i, (gt_ans, pred_text, instr) in enumerate(zip(answers, output_texts, instructions)):
                gt_ans_clean = gt_ans.strip().lower()
                pred_clean = pred_text.strip().lower()

                # 简单包含式判分，也可做模糊匹配或后处理
                is_correct = int(gt_ans_clean in pred_clean)
                total_correct += is_correct
                total_count += 1

                # log样本
                local_log_samples.append({
                    "instruction": instr,
                    "gt_answer": gt_ans,
                    "prediction": pred_text,
                    "is_correct": is_correct
                })

        # 分布式结果聚合
        results_tensor = torch.tensor([total_correct, total_count], dtype=torch.float32, device=self.accelerator.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(results_tensor, op=torch.distributed.ReduceOp.SUM)
        global_correct, global_total = int(results_tensor[0].item()), int(results_tensor[1].item())

        # 日志保存，只在主进程
        if self.accelerator.is_main_process:
            accuracy = global_correct / global_total if global_total > 0 else 0.0
            json_path = "eval_samples.json" if not tag else f"eval_samples_{tag}.json"
            if local_log_samples:
                res_path = os.path.join(self.logger.result_dir, json_path)
                with open(res_path, "w", encoding="utf-8") as f:
                    json.dump(local_log_samples, f, indent=2, ensure_ascii=False)
                self.logger.INFO(f"Saved {len(local_log_samples)} evaluation samples to {res_path}")
            self.logger.INFO(f"Evaluation accuracy: {accuracy:.4f}")

            accuracy = global_correct / global_total if global_total > 0 else 0.0
        return accuracy
 

    def __str__(self):
        return "SFT"

    @classmethod
    def load(cls, model_path: str, base_model_path: str = None, device: str = "cuda"):
        """
         Loads a Qwen2.5-VL model and processor from the specified path.
         Supports both full models and LoRA adapters.
        """
        processor = AutoProcessor.from_pretrained(model_path)

        # 检查是否为 LoRA adapter
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)

        if is_lora:
            print(f"###### Detected LoRA adapter in {model_path}. Loading adapter. ######")
            if base_model_path is None:
                raise ValueError("A `base_model_path` must be provided to load a LoRA adapter.")

            model = AutoModelForVision2Seq.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, device_map={"": device}
            )

        
            model = PeftModel.from_pretrained(model, model_path)
            print("##### Successfully loaded LoRA adapter. #####")

        else:
            print(f"##### No LoRA adapter detected. Loading full model from {model_path}. #####")
            model = AutoModelForVision2Seq.from_pretrained(
                 model_path, torch_dtype=torch.bfloat16, device_map={"": device}
           )
            print("##### Successfully loaded full model. #####")

        model.eval()

        return model, processor