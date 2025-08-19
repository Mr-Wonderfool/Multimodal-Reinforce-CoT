import os
from typing import Optional
from dataclasses import is_dataclass

from .utils import ParamsManager


class BaseTrainerConfig:
    @classmethod
    def from_yaml(cls, config_file: str, config_section: Optional[str] = None):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"No yaml found at {config_file}")

        parsed_dict = ParamsManager.parse(config_file)
        if config_section:
            if config_section not in parsed_dict:
                raise KeyError(f"Section '{config_section}' not found in the config file.")
            config_dict = parsed_dict[config_section]
        else:
            config_dict = parsed_dict

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BaseTrainerConfig":
        class_fields = {f.name for f in cls.__dataclass_fields__.values()}
        # recursively instantiate nested dataclasses
        nested_instances = {}
        for name, field_type in cls.__annotations__.items():
            if is_dataclass(field_type) and isinstance(config_dict.get(name), dict):
                nested_instances[name] = field_type.from_dict(config_dict[name])

        # filter the input dict to only include fields defined in the dataclass
        # and update with any nested dataclass instances just created
        filtered_kwargs = {k: v for k, v in config_dict.items() if k in class_fields}
        filtered_kwargs.update(nested_instances)

        return cls(**filtered_kwargs)