from source.utils.data import split_data
import lightgbm
import pandas as pd
from os.path import isfile
from source.utils.data import evaluate_loss

if not (isfile("Data/training_set.csv") and isfile("Data/training_set.csv")):
    split_data()

train_df = pd.read_csv("Data/training_set.csv")
valid_df = pd.read_csv("Data/validation_set.csv")

train_dataset = lightgbm.Dataset(data = train_df.iloc[:, 1:-1], label=train_df.iloc[:,-1])
valid_dataset = lightgbm.Dataset(data = valid_df.iloc[:, 1:-1], label=valid_df.iloc[:,-1])

lightgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 31,
    'learning_rate': 0.005,
    'max_depth': 3,
}

model = lightgbm.train(lightgbm_params, train_dataset, num_boost_round=100, valid_sets=[valid_dataset], early_stopping_rounds=10, categorical_feature=['EJ'], feval=evaluate_loss)

