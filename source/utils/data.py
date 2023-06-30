"""Functions used to process data."""
from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import Dataset


def split_data():
    """Split and save full dataset into training and validation set."""
    data = pd.read_csv("Data/train.csv")

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

    # Number of observations in each class
    n_0 = len(class_0)
    n_1 = len(class_1)

    # Weights of each class
    w_0 = 1 / n_0
    w_1 = 1 / n_1

    # Limit predicted probability
    preds = np.clip(preds, 1e-15, 1 - 1e-15)

    p_0 = 1 - preds

    p_1 = preds

    loss_0 = -w_0 * np.sum((1 - eval_data) * np.log(p_0))
    loss_1 = -w_1 * np.sum(eval_data * np.log(p_1))

    eval_result = (loss_0 + loss_1) / (w_0 + w_1)

    is_high_better = True

    return eval_name, eval_result, is_high_better


def augment_categorical_column(
    column: pd.Series, p: Tuple[float, float], targets: pd.Series
) -> pd.Series:
    """Augment categorical column, maintaining the proportion of categories of classes.

    Args:
        column (pd.Series): Categorical column to be augmented.
        p (float): Probability of selecting a value of 1 during augmentation.

    Returns:
        pd.Series: Augmented categorical column.
    """
    augmented_column = column.copy()

    index_0 = targets == 0
    index_1 = targets == 1

    augmented_column.iloc[index_0] = np.random.choice(
        [0, 1], size=sum(index_0), p=[1 - p[0], p[0]]
    )
    augmented_column.iloc[index_1] = np.random.choice(
        [0, 1], size=sum(index_1), p=[1 - p[1], p[1]]
    )
    return pd.Series(augmented_column)


def augment_numerical_columns(columns: pd.Series) -> pd.Series:
    """Augment a numerical column by adding random noise.

    Args:
        column (pd.Series): Numerical column to be augmented.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        pd.Series: Augmented numerical column.
    """
    std = columns.std() / 10
    noise = np.random.normal(-std / 2, std / 2, size=columns.shape)
    augmented_values = columns + noise
    augmented_values = np.maximum(augmented_values, 0)

    return augmented_values


def augment_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Augment data by applying augmentation to categorical and numerical columns.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Augmented features and target variable.
    """
    p_categorical = (256 / 407, 66 / 87)

    augmented_X = pd.DataFrame()

    # Augment categorical column
    augmented_X["EJ"] = augment_categorical_column(X["EJ"], p_categorical, y)

    # Augment numerical columns
    numerical_cols = X.columns.drop("EJ")
    augmented_X[numerical_cols] = augment_numerical_columns(X[numerical_cols])

    # Copy target variable
    augmented_y = y.copy()

    return augmented_X, augmented_y


if __name__ == "__main__":
    pass
