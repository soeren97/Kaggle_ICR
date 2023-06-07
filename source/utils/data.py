"""Functions used to process data."""
from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import Dataset


def split_data():
    """Split and save full dataset into training and validation set."""
    data = pd.read_csv("Data/train.csv")

    data.drop(columns="Id", inplace=True)

    data.replace(["A", "B"], [0, 1], inplace=True)

    train_set = data.sample(frac=0.8, random_state=42)

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
    eval_name = "balanced logarithmic loss"

    eval_data = eval_data.get_label()

    class_0 = eval_data[eval_data == 0]
    class_1 = eval_data[eval_data == 1]

    n_0 = len(class_0)
    n_1 = len(class_1)

    loss_0 = (-1 / n_0) * sum(eval_data == 0 * np.log(preds[0]))
    loss_1 = (-1 / n_1) * sum(eval_data == 1 * np.log(preds[1]))

    eval_result = (loss_0 - loss_1) / 2

    is_high_better = False

    return eval_name, eval_result, is_high_better


def augment_categorical_column(column: pd.Series, p: float) -> pd.Series:
    """Augment categorical column, maintaining the proportion of categories of classes.

    Args:
        column (pd.Series): Categorical column to be augmented.
        p (float): Probability of selecting a value of 1 during augmentation.

    Returns:
        pd.Series: Augmented categorical column.
    """
    augmented_column = column.copy()
    mask_1 = column == 1
    mask_0 = ~mask_1
    n_augmented_1 = int(p * np.sum(mask_1))
    n_augmented_0 = int(p * np.sum(mask_0))
    augmented_column.loc[mask_1] = np.random.choice(
        [0, 1], size=n_augmented_1, p=[1 - p, p]
    )
    augmented_column.loc[mask_0] = np.random.choice(
        [0, 1], size=n_augmented_0, p=[1 - p, p]
    )
    return augmented_column


def augment_numerical_column(column: pd.Series, std: float) -> pd.Series:
    """Augment a numerical column by adding random noise.

    Args:
        column (pd.Series): Numerical column to be augmented.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        pd.Series: Augmented numerical column.
    """
    noise = np.random.normal(0, std, size=len(column))
    augmented_column = column + noise
    return augmented_column


def augment_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Augment data by applying data augmentation to categorical and numerical columns.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Augmented features and target variable.
    """
    # Probability of selecting a value of 1 during categorical column augmentation
    # for each class.
    p_categorical = (256 / 407, 66 / 87)

    augmented_X = pd.concat(
        [
            augment_categorical_column(X["EJ"], p_categorical[y.iloc[i]])
            for i in range(len(X))
        ],
        axis=1,
    )
    augmented_X.columns = X.columns
    augmented_y = pd.concat([y] * len(X), ignore_index=True)
    return augmented_X, augmented_y


if __name__ == "__main__":
    pass
