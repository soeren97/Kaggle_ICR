"""Training of the lightGBM model."""
import lightgbm
import wandb
from wandb.lightgbm import log_summary, wandb_callback

from source.utils.data import augment_data, load_data
from source.utils.training import evaluate_loss


def train_lightGBM_model():
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


if __name__ == "__main__":
    pass
