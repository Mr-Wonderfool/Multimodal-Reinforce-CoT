import copy


class GRPOTrainerConfig:
    """
    Process custom yaml file into `GRPOConfig` while maintaining
    custom parameters.
    """

    @classmethod
    def construct_config(cls, config: dict):
        grpo_config = copy.deepcopy(config)
        custom_config = {}
        lora_config = None

        custom_keys = [
            "model_path",
            "tokenizer_path",
            "sft_model_path",
            "image_dir",
            "test_image_dir",
            "quantization",
            "log_dir",
        ]
        pipeline_config = grpo_config.pop("pipeline")
        if "lora" in grpo_config:
            lora_config = grpo_config.pop("lora")

        # construct custom config
        for key in custom_keys:
            custom_config[key] = grpo_config.pop(key)

        train_config = pipeline_config.pop("train")
        val_config = pipeline_config.pop("val")
        test_config = pipeline_config.pop("test")

        custom_config["train_file"] = train_config.pop("train_file")
        custom_config["optimizer"] = train_config.pop("optimizer")
        custom_config["val_file"] = val_config.pop("val_file")
        custom_config["test_file"] = test_config.pop("test_file")

        grpo_config.update(train_config)

        return custom_config, grpo_config, lora_config
