import os
import yaml
from typing import Any, Dict


class ParamsManager:
    """
    A utility class for managing and processing nested YAML configuration files.
    """
    @classmethod
    def parse(cls, config_file: str) -> dict:
        """
        Parse a nested YAML configuration file and convert relative paths to absolute paths.

        Parameters:
            config_file (str): The path to the main configuration file.

        Returns:
            dict: The processed complete configuration dictionary.
        """
        base_path = os.path.abspath(os.path.dirname(config_file))
        config = cls.processConfig(cls.load(config_file), os.path.dirname(config_file), base_path)
        return config

    @classmethod
    def load(cls, config_file: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Parameters:
            config_file (str): The path to the YAML file.

        Returns:
            dict: The loaded configuration as a dictionary.
        """
        with open(config_file, "r") as f:
            try:
                return yaml.load(f, Loader=yaml.FullLoader) or {}
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file {config_file}: {exc}")
                return {}

    @classmethod
    def processConfig(cls, config: Dict[str, Any], current_dir: str, root_dir: str) -> Dict[str, Any]:
        """
        Recursively process a configuration dictionary.

        Parameters:
            config (dict): The configuration dictionary to process.
            current_dir (str): The current directory for resolving relative paths.
            root_dir (str): The root directory for resolving relative paths.

        Returns:
            dict: The processed configuration dictionary.
        """
        processed = {}
        for key, value in config.items():
            processed[key] = cls.processValue(value, current_dir, root_dir)
        return processed

    @classmethod
    def processValue(cls, value: Any, current_dir: str, root_dir: str) -> Any:
        """
        Process individual values in the configuration.

        Parameters:
            value (Any): The configuration value to process (could be dict, list, str, etc.).
            current_dir (str): The current directory for resolving relative paths.
            root_dir (str): The root directory for resolving relative paths.

        Returns:
            Any: The processed value.
        """
        if isinstance(value, dict):
            return cls.processConfig(value, current_dir, root_dir)
        elif isinstance(value, list):
            return [cls.processItem(item, current_dir, root_dir) for item in value]
        elif isinstance(value, str):
            return cls.processString(value, current_dir, root_dir)
        return value

    @classmethod
    def processItem(cls, item: Any, current_dir: str, root_dir: str) -> Any:
        """
        Process an item in a list.

        Parameters:
            item (Any): The list item to process.
            current_dir (str): The current directory for resolving relative paths.
            root_dir (str): The root directory for resolving relative paths.

        Returns:
            Any: The processed item.
        """
        if isinstance(item, dict):
            return cls.processConfig(item, current_dir, root_dir)
        elif isinstance(item, str):
            return cls.processString(item, current_dir, root_dir)
        return item

    @classmethod
    def processString(cls, s: str, current_dir: str, root_dir: str) -> str:
        """
        Process a string configuration value.

        Parameters:
            s (str): The string to process.
            current_dir (str): The current directory for resolving relative paths.
            root_dir (str): The root directory for resolving relative paths.

        Returns:
            str: The processed string.
        """
        if s.endswith((".yaml", ".yml")):
            abs_path = os.path.abspath(os.path.join(current_dir, s))
            if os.path.exists(abs_path):
                config = cls.load(abs_path)
                return cls.processConfig(config, os.path.dirname(abs_path), root_dir)
            return s

        # Handle regular paths (assuming non-YAML paths)
        if any([s.startswith(("./", "../", "/")) or "/" in s]):
            return os.path.abspath(os.path.join(current_dir, s))
        return s