"""Functions used to process and load data."""
from os.path import isfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml  # type: ignore
from sklearn.preprocessing import StandardScaler

from source.constants import TRAIN_DATASET_LOCATION, VALIDATION_DATASET_LOCATION


def load_config(path: str) -> Dict:
    """Load in configuration file for model.

    Used to easily change variables such as learning rate,
    loss function and such.

    Args:
        path (str): path to config file.

    Returns:
        Dict: Model config.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def split_data():
    """Split and save full dataset into training and validation set."""
    data = pd.read_csv("Data/train.csv")

    data.replace(["A", "B"], [0, 1], inplace=True)

    train_set = data.sample(frac=0.8, random_state=42)

    validation_set = data.drop(train_set.index)

    train_set.to_csv("Data/training_set.csv")
    validation_set.to_csv("Data/validation_set.csv")


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the data from csv file.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            Train and validation feature and targets.
    """
    if not (isfile(TRAIN_DATASET_LOCATION) and isfile(VALIDATION_DATASET_LOCATION)):
        split_data()

    train_df = pd.read_csv(TRAIN_DATASET_LOCATION, index_col=0).drop(columns="Id")
    valid_df = pd.read_csv(VALIDATION_DATASET_LOCATION, index_col=0).drop(columns="Id")

    X_valid = valid_df.iloc[:, 1:-1]
    y_valid = valid_df.Class

    X_train = train_df.iloc[:, 1:-1]
    y_train = train_df.Class

    return X_train, y_train, X_valid, y_valid


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

    X_train = pd.concat([X, augmented_X]).reset_index(drop=True)
    y_train = pd.concat([y, augmented_y]).reset_index(drop=True)

    return X_train, y_train


def remove_nan(
    X: pd.DataFrame, y: pd.Series, method: str = "drop"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove nan values from dataset.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Targets.
        method (str, optional): Method of removal. Defaults to "drop".

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Cleaned dataset.
    """
    # index = np.where(X.isna())

    if method == "drop":
        X["Class"] = y
        X.dropna(inplace=True)

        y = X["Class"]
        del X["Class"]

    if method == "average":
        # TODO: impliment other methods.
        # averages = X.mean()
        # X.iloc[]
        pass
    return X, y


def scale_data(
    X_train: pd.DataFrame, X_valid: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale features.

    Args:
        X_train (pd.DataFrame): Training features.
        X_valid (pd.DataFrame): Validation Features

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled train and validation features.
    """
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train)
    scaled_valid = scaler.transform(X_valid)
    return scaled_train, scaled_valid


if __name__ == "__main__":
    pass
