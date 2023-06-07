"""Main training script for LightGBM."""
from os.path import isfile

import lightgbm
import pandas as pd

from source.utils.data import augment_data, evaluate_loss, split_data

if not (isfile("Data/training_set.csv") and isfile("Data/training_set.csv")):
    split_data()

train_df = pd.read_csv("Data/training_set.csv", index_col=0).drop(columns="Id")
valid_df = pd.read_csv("Data/validation_set.csv", index_col=0).drop(columns="Id")

feature_names = train_df.columns.drop("Class").to_list()

# Augment train_df
X_train, y_train = augment_data(train_df.iloc[:, :-1], train_df.iloc[:, -1])
X_train = X_train[train_df.columns.drop("Class")]

X_train = pd.concat([train_df.iloc[:, :-1], X_train]).reset_index(drop=True)
y_train = pd.concat([train_df.iloc[:, -1], y_train]).reset_index(drop=True)

# Convert to lightgbm dataset
train_dataset = lightgbm.Dataset(
    data=X_train, label=y_train, feature_name=feature_names
)
valid_dataset = lightgbm.Dataset(
    data=valid_df.iloc[:, 1:-1], label=valid_df.Class, feature_name=feature_names
)

lightgbm_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "num_leaves": 31,
    "learning_rate": 0.005,
    "max_depth": 3,
}

model = lightgbm.train(
    lightgbm_params,
    train_dataset,
    num_boost_round=100,
    valid_sets=[valid_dataset],
    early_stopping_rounds=10,
    categorical_feature=["EJ"],
    feval=evaluate_loss,
)
