import numpy as np


def calculate_mymetrics(
    pred_x,
    pred_y_mean,
    pred_y_std,
    target_x,
    target_y,
    context_x,
    context_y,
    closeness_thresh=0.1,
    std_multiplier=1,
):  # 2 std = 95% confidence interval
    """
    Calculate various metrics for the predictions.

    Args:
        pred_x, pred_y_mean, pred_y_std: Prediction data (x, mean, std)
        target_x, target_y: True values
        context_x, context_y: Context points used for prediction
        std_multiplier: Number of standard deviations for confidence interval
    """

    # Convert tensors to numpy arrays
    pred_y_mean = pred_y_mean.numpy()
    pred_y_std = pred_y_std.numpy()

    # 1. Fraction of truth points inside prediction uncertainty
    upper_bound = pred_y_mean + std_multiplier * pred_y_std
    lower_bound = pred_y_mean - std_multiplier * pred_y_std
    truth_inside = np.mean((target_y < upper_bound) & (target_y > lower_bound))

    sigma_mean = pred_y_std.mean()
    truth_inside_sigma = truth_inside / sigma_mean

    # 2. Fraction of context points inside prediction uncertainty
    # first find idx in pred_x which are closest to context_x
    context_idx = [np.argmin(np.abs(pred_x.ravel() - cx)) for cx in context_x.ravel()]
    upper_bound_c = (
        pred_y_mean[:, context_idx, :] + std_multiplier * pred_y_std[:, context_idx, :]
    )
    lower_bound_c = (
        pred_y_mean[:, context_idx, :] - std_multiplier * pred_y_std[:, context_idx, :]
    )

    sigma_c = pred_y_std[:, context_idx, :]

    context_inside = np.mean((context_y < upper_bound_c) & (context_y > lower_bound_c))
    context_inside_sigma = np.mean(
        ((context_y < upper_bound_c) & (context_y > lower_bound_c)) / sigma_c
    )

    # 3. Fraction of context points within 0.01 of prediction mean
    context_close = np.mean(
        np.abs(context_y - pred_y_mean[:, context_idx, :]) < closeness_thresh
    )

    return {
        "truth_inside_uncertainty": float(truth_inside),
        "truth_inside_uncertainty_div_sigma": float(truth_inside_sigma),
        "context_inside_uncertainty": float(context_inside),
        "context_inside_uncertainty_div_sigma": float(context_inside_sigma),
        "context_close_to_mean": float(context_close),
        "mean_confidence": float(1 / sigma_mean),
    }
