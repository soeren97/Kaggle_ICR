"""Main training script for LightGBM."""
import os
import random
import warnings
from multiprocessing import Process
from typing import Dict

import lightgbm
import numpy as np
import wandb
from wandb.lightgbm import log_summary, wandb_callback

from source.constants import PROJECT_NAME, WANDB_DIR
from source.utils.data import augment_data, load_data
from source.utils.training import evaluate_loss

np.random.seed(42)
random.seed(42)

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DIR"] = WANDB_DIR

# Filter out specific warning
warnings.filterwarnings(
    "ignore",
    message="[LightGBM] [Warning] No further splits with positive gain, best gain: -inf",  # noqa : E501
)


def train():
    """Train lightgbm model."""
    X_train, y_train, X_valid, y_valid = load_data()

    X_train, y_train = augment_data(X_train, y_train)

    feature_names = X_train.columns.tolist()

    # Convert to lightgbm dataset
    train_dataset = lightgbm.Dataset(
        data=X_train, label=y_train, feature_name=feature_names, params={"verbose": -1}
    )
    valid_dataset = lightgbm.Dataset(
        data=X_valid,
        label=y_valid,
        feature_name=feature_names,
        params={"verbose": -1},
    )

    wandb.init()

    config_copy = dict(wandb.config)

    # Train the LightGBM model
    model = lightgbm.train(
        config_copy,
        train_dataset,
        valid_sets=[valid_dataset],
        feval=evaluate_loss,
        callbacks=[wandb_callback(), lightgbm.log_evaluation(period=0)],
    )

    # Log feature importance
    log_summary(model)


def setup_sweep_config() -> Dict:
    """Initialize sweep configuration for wandb.

    Returns:
        Dict: Dict of hyperparameter space.
    """
    depth = list(range(3, 15))
    depth.append(-1)

    # TODO: Move to json file
    hyperparameter_space = {
        "device": {"values": ["gpu"]},
        "boost": {"values": ["dart"]},
        "num_leaves": {"values": np.random.randint(10, 50, 30).tolist()},
        "learning_rate": {"max": 0.5, "min": 0.001},
        "max_depth": {"values": depth},
        "feature_fraction": {"values": np.random.uniform(0.7, 1, 30).tolist()},
        "verbose": {"values": [-1]},
    }

    # Convert NumPy arrays to Python lists
    for key, value in hyperparameter_space.items():
        if isinstance(value, np.ndarray):
            hyperparameter_space[key] = value.tolist()

    # TODO: Move to json file
    sweep_config = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": hyperparameter_space,
        "early_terminate": {"type": "hyperband", "min_iter": 10},
        "run_cap": 300,
    }

    return sweep_config


# Define the sweep agent function
def sweep_agent():
    """Agent to control sweeps."""
    sweep_config = setup_sweep_config()

    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    # TODO: Move to json file
    init_config = {
        "device": "gpu",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "max_depth": -1,
        "feature_fraction": 0.9,
    }

    wandb.init(project=PROJECT_NAME, config=init_config)

    # Start multiple child processes
    processes = []
    for _ in range(4):
        p = Process(target=sweep_agent)
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Finish the Wandb run
    wandb.finish()
