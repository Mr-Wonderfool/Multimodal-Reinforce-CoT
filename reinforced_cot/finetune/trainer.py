import os
import json
import torch
from torch.optim import AdamW
from typing import Dict, Any, List
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from accelerate.utils import pad_across_processes
from peft import LoraConfig
from typing import Union, List
from accelerate.utils import DistributedType

from reinforced_cot.finetune.updater import PPOUpdater
from reinforced_cot.common.base_model import PolicyAndValueModel, RewardModel
from reinforced_cot.utils.preprocess import DatasetPreprocessor
from reinforced_cot.common import BaseVLM


class ProximalPolicyOptimizationTrainer(BaseVLM):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], use_fast=True)
        self.tokenizer.pad_token_id = 1
        self.tokenizer.eos_token_id = 2

        peft_config = LoraConfig(**config["lora"]) if "lora" in config else None

        self.policy_model = PolicyAndValueModel(model_path=config["sft_model_path"], peft_config=peft_config)

        self.reference_model = AutoModelForCausalLM.from_pretrained(
            config["sft_model_path"], torch_dtype=torch.bfloat16
        )

        self.reward_model = RewardModel

        self.policy_model.policy_model.resize_token_embeddings(len(self.tokenizer))
        self.reference_model.resize_token_embeddings(len(self.tokenizer))

        self.policy_model = self.policy_model.to(torch.bfloat16)
        self.reference_model = self.reference_model.to(torch.bfloat16)

        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        (self.train_dataset, self.train_dataloader), (self.test_dataset, self.test_dataloader) = (
            DatasetPreprocessor.prepare_datasets_and_data_loaders(
                args=config, accelerator=self.accelerator, tokenizer=self.tokenizer
            )
        )

        distributed_strategy = self.accelerator.state.distributed_type

        if distributed_strategy == DistributedType.DEEPSPEED:
            self.optimizer = AdamW(self.policy_model.parameters(), lr=config["learning_rate"])
            num_training_steps = len(self.train_dataloader) * config["num_epochs"]
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
            )

            (
                self.policy_model,
                self.optimizer,
                self.train_dataloader,
                self.test_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.policy_model,
                self.optimizer,
                self.train_dataloader,
                self.test_dataloader,
                self.scheduler,
            )
            self.reference_model.to(self.accelerator.device)
            # self.reward_model.to(self.accelerator.device)
        else:
            self.policy_model, self.reference_model = self.accelerator.prepare(self.policy_model, self.reference_model)

            self.optimizer = AdamW(self.policy_model.parameters(), lr=config["learning_rate"])
            num_training_steps = len(self.train_dataloader) * config["num_epochs"]
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
            )

            (
                self.optimizer,
                self.train_dataloader,
                self.test_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(self.optimizer, self.train_dataloader, self.test_dataloader, self.scheduler)

        self.updater = PPOUpdater(
            model=self.policy_model,
            optimizer=self.optimizer,
            accelerator=self.accelerator,
            ppo_epochs=config["ppo_epochs"],
            clip_grad_norm=config["clip_grad_norm"],
        )
        self.global_step = 0
        self.evaluating_epoch_freq = config.get("evaluating_epoch_freq", 1)
        self.saving_epoch_freq = config.get("saving_epoch_freq", 1)

    @torch.no_grad()
    def _generate_rollouts(
        self,
        query_tensors: torch.Tensor,
        attention_mask: torch.Tensor,
        import_prompts: List[str],
        test_funcs: List[str],
    ) -> Dict[str, Any]:
        self.policy_model.eval()

        unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
        model_dtype = unwrapped_model.policy_model.dtype

        generation_output = unwrapped_model.generate(
            input_ids=query_tensors,
            attention_mask=attention_mask,
            max_new_tokens=self.config["response_length"],
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        full_sequences = generation_output
        query_length = query_tensors.shape[1]
        response_tensors = full_sequences[:, query_length:]

        # retrieve varying length code from the response
        completed_tensors = pad_across_processes(
            response_tensors, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=False
        )
        completed_texts = self.tokenizer.batch_decode(
            completed_tensors.cpu().numpy().tolist(), skip_special_tokens=True
        )
        answer_codes = [
            CodePreprocessor.extract_code_from_response(
                response.strip(), remove_comments=True, wrap_extracted_code=False
            )
            for response in completed_texts
        ]
        # combine answer code with test cases and pass to compiler
        # ! dont shuffle data from this point on
        compiler_reward = []
        for code, prompt, test_func in zip(answer_codes, import_prompts, test_funcs):
            eval_dict = self.evaluate_code_generation(code=code, prompt=prompt, test_func=test_func)
            compiler_reward.append(eval_dict["reward"])

        query_mask = attention_mask
        response_mask = torch.ones_like(response_tensors)
        full_attention_mask = torch.cat([query_mask, response_mask], dim=1).to(self.accelerator.device)

        full_logits, full_values = unwrapped_model(full_sequences, attention_mask=full_attention_mask)

        response_logits = full_logits[:, query_length - 1 : -1, :]
        response_values = full_values[:, query_length - 1 : -1]
        log_probs = torch.gather(F.log_softmax(response_logits, dim=-1), 2, response_tensors.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            reference_logits = self.reference_model(full_sequences, attention_mask=full_attention_mask).logits
            reference_response_logits = reference_logits[:, query_length - 1 : -1, :]

        kl_div = (
            F.kl_div(
                F.log_softmax(reference_response_logits, dim=-1),
                F.softmax(response_logits, dim=-1),
                log_target=True,
                reduction="none",
            )
            .sum(dim=-1)
            .to(model_dtype)
        )

        with torch.no_grad():
            rm_scores = torch.randn(full_sequences.size(0), device=self.accelerator.device, dtype=model_dtype)

        rewards = -self.config["kl_coef"] * kl_div
        rewards = rewards.to(model_dtype)
        # add compiler reward to the last entry
        compiler_reward_tensor = torch.tensor(compiler_reward, device=self.accelerator.device, dtype=model_dtype)
        rewards[:, -1] += compiler_reward_tensor

        self.policy_model.train()

        return {
            "query_tensors": query_tensors,
            "response_tensors": response_tensors,
            "log_probs": log_probs,
            "values": response_values,
            "rewards": rewards,
            "attention_mask": response_mask,
            "query_mask": query_mask,
        }

    def _train_one_epoch(self):
        self.policy_model.train()
        epoch_results = defaultdict(list)
        pbar = tqdm(self.train_dataloader, disable=not self.accelerator.is_main_process, desc="PPO Epoch Loop")
        for batch in pbar:
            prefix_data = batch["generate_prefix_kwargs"]
            query_tensors = prefix_data["input_ids"]
            query_attention_mask = prefix_data["attention_mask"]

            rollouts = self._generate_rollouts(
                query_tensors=query_tensors,
                attention_mask=query_attention_mask,
                import_prompts=prefix_data["prompts"],
                test_funcs=prefix_data["test_funcs"],
            )
            mean_batch_reward = rollouts["rewards"].mean().item()
            epoch_results["train/reward"].append(mean_batch_reward)
            if self.accelerator.is_main_process:
                self.recorder.record("train/batch_reward", mean_batch_reward, step=self.global_step)
            self.updater.update(rollouts)
            if self.accelerator.sync_gradients:
                self.scheduler.step()

            if self.accelerator.sync_gradients:
                self.global_step += 1
                avg_epoch_reward = sum(epoch_results["train/reward"]) / len(epoch_results["train/reward"])
        return {"reward": avg_epoch_reward}

    def evaluate(self):
        self.logger.INFO("--- Running Evaluation ---")
        self.policy_model.eval()

        all_prompts = []
        all_responses = []
        all_rewards = []

        pbar = tqdm(self.test_dataloader, disable=not self.accelerator.is_main_process, desc="Eval Loop")
        for batch in pbar:
            prefix_data = batch["generate_prefix_kwargs"]
            query_tensors = prefix_data["input_ids"]
            attention_mask = prefix_data["attention_mask"]

            with torch.no_grad():
                unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
                generation_output = unwrapped_model.generate(
                    input_ids=query_tensors,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config["response_length"],
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                padded_generation_output = self.accelerator.pad_across_processes(
                    generation_output, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=True
                )
                padded_queries = self.accelerator.pad_across_processes(
                    query_tensors, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=True
                )

                gathered_generations = self.accelerator.gather(padded_generation_output)
                gathered_queries = self.accelerator.gather(padded_queries)

                full_attention_mask = (gathered_generations != self.tokenizer.pad_token_id).long()

                # rm_scores = self.reward_model(gathered_generations, attention_mask=full_attention_mask)
                # 占位符
                rm_scores = torch.randn(gathered_generations.size(0), device=self.accelerator.device)
                all_rewards.extend(rm_scores.cpu().numpy())

            if self.accelerator.is_main_process:
                prompts = self.tokenizer.batch_decode(gathered_queries, skip_special_tokens=True)
                all_prompts.extend(prompts)

                full_texts = self.tokenizer.batch_decode(gathered_generations, skip_special_tokens=True)
                responses = [full[len(prompt) :] for prompt, full in zip(prompts, full_texts)]
                all_responses.extend(responses)

        if self.accelerator.is_main_process:
            mean_eval_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

            self.logger.INFO(f"Evaluation finished. Average Reward: {mean_eval_reward:.4f}")
            self.recorder.record("eval/reward", mean_eval_reward, step=self.global_step)

            eval_results_to_log = []
            for i in range(min(len(all_prompts), 3)):  # 最多记录3个样本
                eval_results_to_log.append(
                    {"prompt": all_prompts[i], "response": all_responses[i], "reward": float(all_rewards[i])}
                )

            res_path = os.path.join(self.logger.result_dir, f"eval_epoch_{self.current_epoch}.json")
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(eval_results_to_log, f, indent=2, ensure_ascii=False)

        self.accelerator.wait_for_everyone()

        return {"reward": mean_eval_reward if "mean_eval_reward" in locals() else 0}

    def train(self):
        self.logger.INFO("--- Start PPO Training with Real Models ---")
        best_eval_reward = -float("inf")
        for epoch in range(1, self.config["num_epochs"] + 1):
            self.current_epoch = epoch
            if self.accelerator.is_main_process:
                self.logger.INFO(f"\n--- Epoch: {epoch}/{self.config['num_epochs']} ---")
            train_results = self._train_one_epoch()
            avg_train_reward = train_results["reward"]
            if self.accelerator.is_main_process:
                self.logger.INFO(f"IN EPOCH {epoch}: Average Train Reward: {avg_train_reward:.4f}")
                self.recorder.record("train/epoch_reward", avg_train_reward, step=epoch)
            if epoch % self.evaluating_epoch_freq == 0:
                eval_results = self.evaluate()

                current_eval_reward = eval_results["reward"]

                if current_eval_reward > best_eval_reward:
                    best_eval_reward = current_eval_reward

                    self.accelerator.wait_for_everyone()

                    if self.accelerator.is_main_process:
                        save_path = os.path.join(
                            self.logger.ckpt_dir, f"epoch_{epoch}_reward_{current_eval_reward:.4f}"
                        )
                        os.makedirs(save_path, exist_ok=True)
                        self.logger.INFO(f"New best model found!")
                        unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
                        self.logger.INFO(f"Saving to {save_path}")
                        unwrapped_model.save_pretrained(save_path)
                        self.tokenizer.save_pretrained(save_path)
                        self.logger.INFO(f"Model saved to {save_path}")

        if self.accelerator.is_main_process:
            self.logger.INFO("\n--- PPO Training Finished ---")

    def infer(self, prompt_text: Union[str, List[str]], **generation_kwargs):
        self.policy_model.eval()

        is_single_input = isinstance(prompt_text, str)
        prompts = [prompt_text] if is_single_input else prompt_text

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.accelerator.device)
        attention_mask = inputs["attention_mask"].to(self.accelerator.device)

        default_generation_kwargs = {
            "max_new_tokens": self.config.get("response_length", 512),
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        default_generation_kwargs.update(generation_kwargs)

        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
            generation_output = unwrapped_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **default_generation_kwargs
            )

            full_attention_mask = (generation_output != self.tokenizer.pad_token_id).long()

            _, values = unwrapped_model(input_ids=generation_output, attention_mask=full_attention_mask)

            sequence_lengths = full_attention_mask.sum(dim=1) - 1

            last_token_values = torch.gather(values, 1, sequence_lengths.unsqueeze(-1)).squeeze(-1)

        decoded_responses = []

        full_decoded_texts = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)

        for i, full_text in enumerate(full_decoded_texts):
            prompt = prompts[i]
            response_only_text = full_text[len(prompt) :]
            decoded_responses.append(response_only_text)

        results = list(zip(decoded_responses, last_token_values.tolist()))

        self.logger.INFO("--- Inference Finished ---")

        return results[0] if is_single_input else results

    def __str__(self):
        return "PPO"
