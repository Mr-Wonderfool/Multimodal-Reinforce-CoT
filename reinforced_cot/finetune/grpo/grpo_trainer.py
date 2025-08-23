import os
import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from .reward import combined_reward
from reinforced_cot.common import BaseVLM
from .grpo_config import GRPOTrainerConfig
from reinforced_cot.utils.utils import load_jsonl
from reinforced_cot.utils import DatasetPreprocessor


class GRPOTrainerWrapper(BaseVLM):
    def __init__(self, grpo_config: dict):
        super().__init__(grpo_config)
        custom_config, grpo_config, lora_config = GRPOTrainerConfig.construct_config(grpo_config)
        self.grpo_config = GRPOConfig(**grpo_config)
        self.custom_config = custom_config
        if lora_config is not None:
            if self.accelerator.is_main_process:
                self.logger.INFO("Applying LoRA configuration...")
            self.lora_config = LoraConfig(**lora_config)

        self._setup_models()
        self._prepare_dataset()

        self.trainer = GRPOTrainer(
            model=self.model,
            args=self.grpo_config,
            reward_funcs=[combined_reward],
            train_dataset=self.train_dataset,
            processing_class=self.processor,
            peft_config=self.lora_config,
        )

    def _setup_models(self):
        quantization_config = None
        if self.custom_config["quantization"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.processor = AutoProcessor.from_pretrained(self.custom_config["tokenizer_path"])
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # load merged sft weights
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.custom_config["model_path"],
            quantization_config=quantization_config,
        )
        # sft_model = PeftModel.from_pretrained(base_model, self.custom_config["sft_model_path"])

        # if want SFT knowledge to be frozen (dont continue training on SFT weights)
        # self.model = sft_model.merge_and_unload()

        # currently we are continue training on SFT adapters
        # ! in this case, dont provide lora config to trainer again
        # self.model = sft_model
        # self.lora_config = None

    def _prepare_dataset(self):

        def create_prompt(sample):
            instruction = sample.get("instruction", sample.get("question", ""))
            system_prompt = DatasetPreprocessor.INSTRUCTION
            user_prompt_text = DatasetPreprocessor.QUESTION_TRIGGER + instruction

            user_message_content = [{"type": "image"}, {"type": "text", "content": user_prompt_text}]

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message_content}]

            prompt_string = self.processor.apply_chat_template(
                messages,
                tokenize=False,
            )

            image = Image.open(os.path.join(self.custom_config["image_dir"], f"{sample['imageId']}.jpg")).convert("RGB")

            return {
                "prompt": prompt_string,
                "image": image,
                "ground_truth_answer": sample["answer"],
            }

        train_data = load_jsonl(self.custom_config["train_file"], max_length=500)
        self.train_dataset = Dataset.from_list(train_data).map(create_prompt)

    def train(self):
        if self.accelerator.is_main_process:
            self.logger.INFO("---- Starting GRPO Training ----")
        self.trainer.train()
        # save the model after full training
        self.save()

    def save(self):
        if self.accelerator.is_main_process:
            # Create a specific path for the final model within the main output directory
            final_save_path = os.path.join(self.grpo_config.output_dir, "final_checkpoint")
            self.logger.INFO(f"--- Training finished. Saving final model adapter to {final_save_path} ---")
            self.trainer.save_model(final_save_path)
            self.logger.INFO(f"Model adapter saved successfully.")

    def __str__(self):
        return "GRPO"
