import os
import json
import torch
import numpy as np
import torch.distributed
from tqdm import tqdm
from torch.optim import AdamW
from collections import defaultdict
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup

from reinforced_cot.common import BaseVLM
from reinforced_cot.utils.preprocess import DatasetPreprocessor, CodePreprocessor


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

        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(sft_config["tokenizer_path"], use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            sft_config["model_path"], quantization_config=quantization_config
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        # ? change id for special token
        self.tokenizer.pad_token_id = 1
        self.tokenizer.eos_token_id = 2

        # apply LoRA configuration if specified
        if "lora" in sft_config:
            if self.accelerator.is_main_process:
                self.logger.INFO("Applying LoRA configuration...")

            lora_config = LoraConfig(**sft_config["lora"])
            self.model = get_peft_model(self.model, lora_config)

            if self.accelerator.is_main_process:
                self.logger.INFO("LoRA applied successfully.")
                self.model.print_trainable_parameters()

        (self.train_dataset, self.train_dataloader), (self.test_dataset, self.test_dataloader) = (
            DatasetPreprocessor.prepare_datasets_and_data_loaders(
                args=sft_config, accelerator=self.accelerator, tokenizer=self.tokenizer
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
                    batch["forward_kwargs"].pop("question_ids")
                    output = self.model(**batch["forward_kwargs"])
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

        total_pass_rate_on_this_process = 0.0
        num_items_on_this_process = 0

        local_log_samples = []

        for batch in tqdm(
            self.test_dataloader, disable=not self.accelerator.is_main_process, desc="Parallel Eval Loop"
        ):
            generation_kwargs = batch["generate_prefix_kwargs"]
            input_ids = generation_kwargs.pop("input_ids")
            attention_mask = generation_kwargs.pop("attention_mask")

            output_sequences = self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_response_length,
                do_sample=False,
                top_p=None,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # extract only the answer (get rid of input prompt
            prompt_length = input_ids.shape[1]
            generated_tokens = output_sequences[:, prompt_length:]
            predictions = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # parallel scoring on cpu
            prompts = batch["generate_prefix_kwargs"]["prompts"]
            test_funcs = batch["generate_prefix_kwargs"]["test_funcs"]
            question_ids = batch["generate_prefix_kwargs"]["question_ids"]

            for i in range(len(predictions)):
                pred_text = predictions[i]

                eval_dict = self.evaluate_code_generation(
                    CodePreprocessor.extract_code_from_response(pred_text), prompts[i], test_funcs[i]
                )

                total_pass_rate_on_this_process += eval_dict.get("pass_percent", 0.0)

                # 5% change of logging
                if np.random.rand() < 0.05:
                    # Reconstruct the loggable response dictionary
                    pred_answer_code = CodePreprocessor.extract_code_from_response(
                        pred_text, remove_comments=True, wrap_extracted_code=False
                    )
                    wrapped_answer_code = f"```python\n{pred_answer_code}```"

                    log_item = {
                        "question_id": question_ids[i],
                        "pred_cot": pred_text.strip(),
                        "pred_answer_code": wrapped_answer_code.strip(),
                        "eval_result": eval_dict,
                    }
                    local_log_samples.append(log_item)

            num_items_on_this_process += len(predictions)

        # gather all log samples to the main process
        gathered_log_samples = [None] * self.accelerator.num_processes
        torch.distributed.all_gather_object(gathered_log_samples, local_log_samples)

        if self.accelerator.is_main_process:
            all_samples = [item for sublist in gathered_log_samples for item in sublist]
            json_path = "eval_samples.json" if not tag else f"eval_samples_{tag}.json"

            if all_samples:
                res_path = os.path.join(self.logger.result_dir, json_path)
                # Open the file once and dump the entire list
                with open(res_path, "w", encoding="utf-8") as f:
                    json.dump(all_samples, f, indent=2, ensure_ascii=False)
                self.logger.INFO(f"Saved {len(all_samples)} evaluation samples to {res_path}")

        # aggregate numeric results across processes
        local_results = torch.tensor(
            [total_pass_rate_on_this_process, num_items_on_this_process], device=self.accelerator.device
        )
        torch.distributed.all_reduce(local_results, op=torch.distributed.ReduceOp.SUM)

        global_total_pass_rate = local_results[0].item()
        global_total_items = local_results[1].item()

        mean_pass_rate = global_total_pass_rate / global_total_items if global_total_items > 0 else 0.0

        return mean_pass_rate

    def __str__(self):
        return "SFT"

    @classmethod
    def load(cls, model_path: str, base_model_path: str = None, device: str = "cuda"):
        """
        Loads a model and tokenizer from a specified path.
        This method automatically detects if the checkpoint is a full model or a
        LoRA adapter and loads it appropriately.

        :param model_path: Path to the saved model checkpoint directory.
        :param base_model_path: (Optional) Path to the original base model.
                            Required only if loading a LoRA adapter.
        :return: A tuple of (model, tokenizer) ready for inference.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check if the checkpoint is a LoRA adapter
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)

        if is_lora:
            print(f"###### Detected LoRA adapter in {model_path}. Loading adapter. ######")
            if base_model_path is None:
                raise ValueError("A `base_model_path` must be provided to load a LoRA adapter.")

            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, device_map={"": device}
            )

            model = PeftModel.from_pretrained(model, model_path)
            print("##### Successfully loaded LoRA adapter. #####")

        else:
            print(f"##### No LoRA adapter detected. Loading full model from {model_path}. #####")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map={"": device}
            )
            print("##### Successfully loaded full model. #####")

        model.eval()

        return model, tokenizer
