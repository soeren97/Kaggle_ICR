import pandas as pd
import numpy as np
from lightgbm import Dataset
from typing import Tuple

def split_data():
    """Split and save full dataset into training and validation set.
    """    
    data = pd.read_csv("Data/train.csv")

    data.drop(columns='Id', inplace=True)

    data.replace(['A', 'B'], [0, 1], inplace=True)

    train_set = data.sample(frac=.8, random_state=42)

    validation_set = data.drop(train_set.index)

    train_set.to_csv("Data/training_set.csv")
    validation_set.to_csv("Data/validation_set.csv")

def evaluate_loss(preds: Dataset, eval_data: Dataset) -> Tuple[str, float, bool]:
    """Evaluate balanced log loss.

    Args:
        preds (Dataset): Predictions.
        eval_data (Dataset): Targets.

    Returns:
        Tuple[str, float, bool]: Name of loss fn, loss value, minimize or maximize.
    """
    eval_name = 'balanced logarithmic loss'

    eval_data = eval_data.get_label()

    class_0 = eval_data[eval_data == 0]
    class_1 = eval_data[eval_data == 1]

    n_0 = len(class_0)
    n_1 = len(class_1)

    loss_0 = (-1/n_0) * sum(eval_data == 0 * np.log(preds[0]))
    loss_1 = (-1/n_1) * sum(eval_data == 1 * np.log(preds[1]))

    eval_result = (loss_0 - loss_1) / 2

    is_high_better = False

    return eval_name, eval_result, is_high_better

if __name__ == "__main__":
    pass