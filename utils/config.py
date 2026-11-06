import yaml

def load_config(config_path: str) -> dict[str, any]:
    """Loads a YAML configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found at {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config: {exc}") from exc
