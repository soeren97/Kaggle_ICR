"""Training utilities."""
from typing import Tuple

import numpy as np
from lightgbm import Dataset


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


if __name__ == "__main__":
    pass
