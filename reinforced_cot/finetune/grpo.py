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


class GroupRelativePolicyOptimization(BaseVLM):
    def __init__(self, grpo_config: dict):
        super().__init__(grpo_config)
        # apply quantization if specified
        quantization_config = None
        if grpo_config["quantization"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        self.processor = AutoProcessor.from_pretrained(grpo_config["tokenizer_path"])
        self.model = AutoModelForVision2Seq.from_pretrained(
            grpo_config["model_path"], quantization_config=quantization_config
        )
        # load reference model
        self.model_ref = AutoModelForVision2Seq.from_pretrained(
            grpo_config["sft_model_path"], quantization_config=quantization_config
        )

        if self.accelerator.is_main_process:
            self.logger.INFO("Initializing GRPO policy model with SFT model weights...")
        self.model.load_state_dict(self.model_ref.state_dict())

        self.tokenizer = self.processor.tokenizer
        # apply LoRA configuration if specified
        if "lora" in grpo_config:
            if self.accelerator.is_main_process:
                self.logger.INFO("Applying LoRA configuration...")

            lora_config = LoraConfig(**grpo_config["lora"])
            self.model = get_peft_model(self.model, lora_config)

            if self.accelerator.is_main_process:
                self.logger.INFO("LoRA applied successfully.")
                self.model.print_trainable_parameters()

        (self.train_dataset, self.train_dataloader), (self.val_dataset, self.val_dataloader), (self.test_dataset, self.test_dataloader) = (
            DatasetPreprocessor.prepare_datasets_and_data_loaders(
                args=grpo_config, accelerator=self.accelerator, processor=self.processor
            )
        )

        self.n_epochs = self.train_config["n_epochs"]
        self.max_input_length = grpo_config["max_input_length"]
        self.max_response_length = grpo_config["max_response_length"]

        num_training_steps = (
            len(self.train_dataloader) // self.accelerator.num_processes * self.n_epochs
        ) // grpo_config["gradient_accumulation_steps"]
        # decrease warm up steps for GRPO
        warmup_step = grpo_config.get("warmup_steps", 50)
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
            if "lora" not in grpo_config
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

    def __str__(self):
        return "GRPO"