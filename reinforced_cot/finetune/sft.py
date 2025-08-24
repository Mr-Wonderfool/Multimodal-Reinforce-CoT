import os
import json
import torch
from tqdm import tqdm
from torch.optim import AdamW
from collections import defaultdict
from peft import get_peft_model, LoraConfig
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, get_linear_schedule_with_warmup

from reinforced_cot.common import BaseVLM
from reinforced_cot.utils import DatasetPreprocessor


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

        (
            (self.train_dataset, self.train_dataloader),
            (self.val_dataset, self.val_dataloader),
            (self.test_dataset, self.test_dataloader),
        ) = DatasetPreprocessor.prepare_datasets_and_data_loaders(
            args=sft_config, accelerator=self.accelerator, processor=self.processor
        )

        self.n_epochs = self.train_config["n_epochs"]
        self.max_prompt_length = sft_config["max_prompt_length"]
        self.max_completion_length = sft_config["max_completion_length"]

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

                should_eval = (epoch % self.evaluating_epoch_freq == 0) or (epoch == self.n_epochs)
                if should_eval:
                    curr_pass_rate = self.evaluate(tag=str(epoch))
                    if self.accelerator.is_main_process:
                        self.recorder.record("test/pass_rate", curr_pass_rate, step=self.global_step)

                    if epoch % self.saving_epoch_freq == 0:
                        self.accelerator.wait_for_everyone()
                        if curr_pass_rate > best_pass_rate:
                            best_pass_rate = curr_pass_rate
                            if self.accelerator.is_main_process:
                                save_path = os.path.join(
                                    self.logger.ckpt_dir,
                                    f"epoch_{epoch}_loss_{curr_loss:.2f}_pass_{curr_pass_rate:.2f}",
                                )
                                self.logger.INFO(
                                    f"New best model found at epoch {epoch} with loss {curr_loss:.4f} and pass rate: {curr_pass_rate:.2f}. Saving to {save_path} ..."
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

    def __str__(self):
        return "SFT"
