"""Main training script for LightGBM."""
import os
import random
import warnings
from functools import partial
from multiprocessing import Process
from typing import Callable, Dict

import numpy as np
import wandb

from source.constants import (
    PATH_TO_CONFIG_FOLDER,
    PROJECT_NAME,
    WANDB_CONFIG,
    WANDB_DIR,
)
from source.models.lightgbm import train_lightGBM_model  # noqa
from source.utils.data import load_config

np.random.seed(42)
random.seed(42)

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DIR"] = WANDB_DIR

# Filter out specific warning
warnings.filterwarnings(
    "ignore",
    message="[LightGBM] [Warning] No further splits with positive gain, best gain: -inf",  # noqa : E501
)


def setup_sweep_config(config: Dict) -> Dict:
    """Initialize sweep configuration for wandb.

    Returns:
        Dict: Dict of hyperparameter space.
    """
    depth = list(range(3, 15))
    depth.append(-1)

    hyperparameter_space = config["HYPERPARAMETER_SPACE"]

    # Convert NumPy arrays to Python lists
    # for key, value in hyperparameter_space.items():
    #     if isinstance(value, np.ndarray):
    #         hyperparameter_space[key] = value.tolist()

    sweep_config = config["SWEEP_CONFIG"]
    sweep_config["parameters"] = hyperparameter_space

    return sweep_config


# Define the sweep agent function
def sweep_agent(train_function: Callable, config: Dict):
    """Agent to control sweeps.

    Args:
        train_function (Callable): Decide what type of model is sweept.
        config (Dict): Config file for sweep.
    """
    sweep_config = setup_sweep_config(config)

    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train_function)


if __name__ == "__main__":
    wandb_config = load_config(WANDB_CONFIG)

    config = load_config(PATH_TO_CONFIG_FOLDER + wandb_config["METHOD"] + ".yaml")

    init_config = config["INIT_CONFIG"]

    training_function = eval(f"train_{wandb_config['METHOD']}_model")

    partial_function = partial(sweep_agent, training_function, config)

    wandb.init(project=PROJECT_NAME, config=init_config)

    # Start multiple child processes
    processes = []
    for _ in range(4):
        p = Process(target=partial_function)
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Finish the Wandb run
    wandb.finish()
