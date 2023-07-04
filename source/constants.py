"""Constants used throughout the repo."""
import os

# Repocitory
CWD = os.getcwd()

# Data
DATA_FOLDER = CWD + "/Data/"
TRAIN_DATASET_LOCATION = DATA_FOLDER + "training_set.csv"
VALIDATION_DATASET_LOCATION = DATA_FOLDER + "validation_set.csv"

# wandb
WANDB_DIR = DATA_FOLDER + "wandb/"
PROJECT_NAME = "Kaggle ICR"

# Plotting
FIGURE_LOCATION = CWD + "/Figures/"
