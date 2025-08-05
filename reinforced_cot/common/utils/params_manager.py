import os
import yaml


class ParamsManager:

    @classmethod
    def parse(cls, config_file_path: str, tag: str) -> dict:
        config_dir_path = os.path.dirname(config_file_path)

        with open(config_file_path, "r", encoding="utf-8") as config_file:
            config = yaml.load(config_file, yaml.FullLoader)

        config = config[tag]
        # change relative path to abs path
        config["log_dir"] = os.path.join(config_dir_path, config["log_dir"])
        config["image_dir"] = os.path.join(config_dir_path, config["image_dir"])
        config["model_path"] = os.path.join(config_dir_path, config["model_path"])
        if "sft_model_path" in config:
            config["sft_model_path"] = os.path.join(config_dir_path, config["sft_model_path"])
        config["tokenizer_path"] = os.path.join(config_dir_path, config["tokenizer_path"])

        config["pipeline"]["train"]["train_file"] = os.path.join(
            config_dir_path, config["pipeline"]["train"]["train_file"]
        )
        config["pipeline"]["val"]["val_file"] = os.path.join(config_dir_path, config["pipeline"]["val"]["val_file"])
        config["pipeline"]["test"]["test_file"] = os.path.join(config_dir_path, config["pipeline"]["test"]["test_file"])

        return config
