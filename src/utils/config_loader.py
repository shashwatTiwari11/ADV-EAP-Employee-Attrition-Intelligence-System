import yaml
from pathlib import Path
from typing import Dict

from src.exceptions.data_exceptions import ConfigLoadException


class ConfigLoader:

    @staticmethod
    def load_config(config_path: str) -> Dict:
        try:
            path = Path(config_path)

            if not path.exists():
                raise ConfigLoadException(f"Config not found: {path}")

            with open(path, "r") as f:
                config = yaml.safe_load(f)

            # 🔥 THIS IS THE IMPORTANT FIX
            if config is None:
                raise ConfigLoadException(
                    f"Config file is empty or has invalid YAML: {path}"
                )

            return config

        except Exception as e:
            raise ConfigLoadException(str(e))
