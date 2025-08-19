import os
import re
import torch
from tqdm import tqdm
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig, PeftModelForVision2Seq
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from reinforced_cot.utils.utils import load_jsonl
from reinforced_cot.utils.preprocess import DatasetPreprocessor

class GRPOTrainingPipeline:
    def __init__(self, config: dict):
        self.config = config

        self._setup_config()
        self._setup_models_and_processor()
        self._prepare_dataset()

        # Initialize the GRPOTrainer from TRL
        self.trainer = GRPOTrainer(
            model=self.model,
            ref_model=self.model_ref,
            config=self.grpo_config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset=self.train_dataset,
            reward_function=self._compute_rewards,
            peft_config=self.lora_config if hasattr(self, 'lora_config') else None,
        )

    def _setup_config(self):
        self.grpo_config = GRPOConfig(
            output_dir=self.config["log_dir"],
            epochs=self.config["pipeline"]["train"]["n_epochs"],
            learning_rate=self.config["pipeline"]["train"]["optimizer"]["learning_rate"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            per_device_train_batch_size=self.config["pipeline"]["train"]["batch_size"],
            ppo_epochs=self.config["pipeline"]["train"].get("gradient_steps", 4),
            # ? may consider disable KL penalty
            adap_kl_ctrl=False,
            init_kl_coef=0.1,
            max_new_tokens=self.config["max_response_length"],
            logging_dir=os.path.join(self.config["log_dir"], "logs"),
            report_to="tensorboard",
            remove_unused_columns=False,
        )

    def _setup_models_and_processor(self):
        quantization_config = None
        if self.config.get("quantization", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )

        self.processor = AutoProcessor.from_pretrained(self.config["tokenizer_path"])
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config["model_path"], quantization_config=quantization_config, device_map="auto"
        )
        self.model_ref = AutoModelForVision2Seq.from_pretrained(
            self.config["sft_model_path"], quantization_config=quantization_config, device_map="auto"
        )

        if self.accelerator.is_main_process:
            self.logger.INFO("Initializing policy model with SFT weights...")
            self.model.load_state_dict(self.model_ref.state_dict(), strict=False)
        
        # apply LoRA configuration if specified
        if "lora" in self.config:
            if self.accelerator.is_main_process:
                self.logger.INFO("Applying LoRA configuration...")

            lora_config = LoraConfig(**self.config["lora"])
            self.model = get_peft_model(self.model, lora_config)

            if self.accelerator.is_main_process:
                self.logger.INFO("LoRA applied successfully.")
                self.model.print_trainable_parameters()

    def _prepare_dataset(self):

        def create_prompt(sample):
            prompt = DatasetPreprocessor.INSTRUCTION + sample["instruction"]
            return {
                "prompt": prompt,
                "image_path": os.path.join(self.config["image_dir"], f"{sample['imageId']}.jpg"),
                "ground_truth_answer": sample["answer"]
            }

        train_data = load_jsonl(self.config["pipeline"]["train"]["train_file"])
        self.train_dataset = Dataset.from_list(train_data).map(create_prompt)

    def _compute_rewards(self, generated_texts: list[str], ground_truth_answers: list[str], **kwargs) -> list[float]:
        """Rule-based reward function passed to the trainer."""
        rewards = []
        for i in range(len(generated_texts)):
            text = generated_texts[i]
            gt_answer = ground_truth_answers[i]
            reward = 0.0

            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

            if not think_match or not answer_match:
                reward = -0.5
            else:
                pred_answer = answer_match.group(1).strip().lower()
                if pred_answer == gt_answer.lower():
                    reward = 1.0
                else:
                    reward = -0.5
            rewards.append(torch.tensor(reward, dtype=torch.float32))
        return rewards

    def train(self):
        """Executes the main training loop."""
        print("--- Starting GRPO Training Loop ---")
        generation_kwargs = {
            "max_new_tokens": self.grpo_config.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        for epoch in range(self.grpo_config.epochs):
            print(f"--- Epoch {epoch + 1}/{self.grpo_config.epochs} ---")
            for batch in tqdm(self.trainer.dataloader, desc=f"Epoch {epoch + 1}"):
                response_texts, _ = self.trainer.generate(
                    batch["prompt"],
                    image_path=batch["image_path"],
                    generation_kwargs=generation_kwargs,
                    return_prompt_and_completion=False,
                )
                
                rewards = self.trainer.compute_reward(response_texts, **batch)
                
                stats = self.trainer.step(response_texts, rewards, **batch)
                
                self.trainer.log_stats(stats, batch, rewards)

    def save(self):
        """Saves the final trained model."""
        print("--- Training finished. Saving model. ---")
        final_path = os.path.join(self.grpo_config.output_dir, "final_checkpoint")
        self.trainer.save_model(final_path)
        print(f"Model saved to {final_path}")

# Example of how to run the pipeline
# if __name__ == '__main__':
#     # Load your config from a file (e.g., YAML or JSON)
#     with open("path/to/your/grpo_config.json", "r") as f:
#         config = json.load(f)
#
#     pipeline = GRPOTrainingPipeline(config=config)
#     pipeline.train()
#     pipeline.save()