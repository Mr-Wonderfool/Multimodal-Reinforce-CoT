import torch
from peft import PeftModel
from reinforced_cot.common import BaseVLM
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from reinforced_cot.utils import DatasetPreprocessor


class Evaluator(BaseVLM):
    def __init__(self, config):
        super().__init__(config)
        # load pre-trained weights
        self._setup_models()
        # prepare dataloaders
        self._prepare_datasets()
        # prepare model and dataset on gpus
        self.model, self.test_dataloader = self.accelerator.prepare(self.model, self.test_dataloader)

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

        lora_adapters_path = self.config["lora_adapter_path"]
        if self.accelerator.is_main_process:
            self.logger.INFO(f"Loading LoRA adapter from {lora_adapters_path}...")
        self.model = PeftModel.from_pretrained(base_model, lora_adapters_path)

    def _prepare_datasets(self):
        # hack batch size for train and val
        self.config["pipeline"]["train"]["batch_size"] = 1
        self.config["pipeline"]["val"]["batch_size"] = 1

        (
            (self.train_dataset, self.train_dataloader),
            (self.val_dataset, self.val_dataloader),
            (self.test_dataset, self.test_dataloader),
        ) = DatasetPreprocessor.prepare_datasets_and_data_loaders(
            args=self.config,
            accelerator=self.accelerator,
            processor=self.processor,
            evaluate=True,
        )

    def __str__(self):
        return "GRPOEvaluator"
