import os
import yaml
import tempfile
import mlflow
from typing import Optional

def setup_mlflow(config: dict[str, any]):
    """Sets up MLflow experiment from config."""
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlruns.db")
    experiment_name = mlflow_config.get("experiment_name", "default_experiment")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_params(config: dict[str, any]):
    """Logs flat hyperparameter config to MLflow."""
    flat_params = {}
    for section, params in config.items():
        if isinstance(params, dict):
            for k, v in params.items():
                flat_params[f"{section}.{k}"] = v
        else:
            flat_params[section] = params
            
    mlflow.log_params(flat_params)

def log_metrics(metrics: dict[str, float], step: Optional[int] = None):
    """Logs evaluation metrics to MLflow."""
    mlflow.log_metrics(metrics, step=step)

def log_artifact(local_path: str, artifact_path: Optional[str] = None):
    """Logs a local artifact (like a model checkpoint)."""
    mlflow.log_artifact(local_path, artifact_path)

def log_config_as_artifact(config: dict[str, any]):
    """Logs the configuration dictionary as an artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        mlflow.log_artifact(config_path)

def log_model_summary(model, **kwargs):
    """Logs model summary using torchinfo."""
    from torchinfo import summary
    stats = summary(model, verbose=0, **kwargs)
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = os.path.join(tmpdir, "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write(str(stats))
        mlflow.log_artifact(summary_path)
