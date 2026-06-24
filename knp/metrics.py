import numpy as np
from scipy.stats import norm

# basic scaled 1d metrics
def msse_1d(y_true, y_pred):
    """
    Calculates Mean Scaled Squared Error (MSSE) for 1D arrays.
    Formula: MSE / Mean(Diff(Truth)^2)
    This assumes y_true and y_pred are already sorted by time.
    """
    # Numerator: MSE
    num = np.mean((y_true - y_pred) ** 2)

    # Denominator: Mean Squared difference of consecutive points
    den = np.mean(np.diff(y_true) ** 2)

    if np.isclose(den, 0):
        msse = 0.0 if np.isclose(num, 0) else np.inf
    else:
        msse = num / den
    return msse


def mase_1d(y_true, y_pred):
    """
    Calculates Mean Absolute Scaled Error (MASE) for 1D arrays.
    Formula: MAE / Mean(Abs(Diff(Truth)))
    This assumes y_true and y_pred are already sorted by time.
    """
    # Numerator: MAE
    num = np.mean(np.abs(y_true - y_pred))

    # Denominator: Mean Absolute difference of consecutive points
    den = np.mean(np.abs(np.diff(y_true)))

    if np.isclose(den, 0):
        mase = 0.0 if np.isclose(num, 0) else np.inf
    else:
        mase = num / den
    return mase


def msle_1d(y_true, y_pred):
    """
    Calculates Mean Scaled Logarithmic Error (MSLE) for 1D arrays.
    Formula: MSLE / Mean(Diff(Log(Truth))^2)
    This assumes y_true and y_pred are already sorted by time.
    """
    # Ensure no negative values for log
    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        raise ValueError("MSLE is not defined for non-positive values.")

    # Numerator: MSLE
    num = np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

    # Denominator: Mean Squared difference of log of consecutive points
    den = np.mean(np.diff(np.log1p(y_true)) ** 2)

    if np.isclose(den, 0):
        msle = 0.0 if np.isclose(num, 0) else np.inf
    else:
        msle = num / den
    return msle


def mape_1d(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE) for 1D arrays.
    Formula: Mean(Abs((Truth - Pred) / Truth)) * 100
    This assumes y_true and y_pred are already sorted by time.
    """
    # Avoid division by zero
    if np.any(y_true == 0):
        raise ValueError("MAPE is not defined when y_true contains zero values.")

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def mrmse_1d(y_true, y_pred):
    """
    Calculates Mean Relative Mean Squared Error (MRMSE) for 1D arrays.
    Formula: Mean((Truth - Pred)^2 / Truth^2)
    This assumes y_true and y_pred are already sorted by time.
    """
    # Avoid division by zero
    if np.any(y_true == 0):
        raise ValueError("MRMSE is not defined when y_true contains zero values.")

    mrmse = np.mean(((y_true - y_pred) ** 2) / (y_true**2))
    return mrmse


def chi2_1d(y_true, y_pred, err):
    return np.sum(((y_true - y_pred) / err) ** 2)


def nrmseo_1d(y_true, y_pred, err):
    return np.sqrt(np.mean(((y_true - y_pred) ** 2) / (2 * (err**2))))

def nrmse_po_1d(y_true, y_pred, y_err):
    # nrsme_p -> if y_err is predicted error
    # nrsme_o -> if y_err is observed error
    return np.sqrt(np.mean((y_true - y_pred)**2 / y_err**2))

def picp_mpiw_1d(y_true, y_pred, err, confidence=0.95):
    # y_err is predicted error
    p_left, p_right = norm.interval(confidence=confidence, loc=y_pred, scale=err)    
    picp = np.mean((y_true > p_left) * (y_true <= p_right)) * 100.0
    mpiw = np.mean(p_right - p_left)
    return picp, mpiw


def rmse_1d(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae_1d(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rse_1d(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.sqrt(num / den) if not np.isclose(den, 0) else np.inf


def rae_1d(y_true, y_pred):
    num = np.sum(np.abs(y_true - y_pred))
    den = np.sum(np.abs(y_true - np.mean(y_true)))
    return num / den if not np.isclose(den, 0) else np.inf


def nlpd_1d(y_true, y_pred, err):
    # As per equation: 0.5 * log(2pi) + (1/m)*sum[ log(err) + (y - mu)^2 / (2 * err^2) ]
    term1 = 0.5 * np.log(2 * np.pi)
    term2 = np.mean(np.log(err) + ((y_true - y_pred) ** 2) / (2 * (err**2)))
    return term1 + term2


def nrmsep_1d(y_true, y_pred, err):
    # As per equation: (1/m) * sum[ (y - mu)^2 / (2 * err^2) ]
    return np.mean(((y_true - y_pred) ** 2) / (2 * (err**2)))


def picp_1d(y_true, y_pred, err, alpha=0.05):
    # Uses normal distribution approx for limits
    # For a given coverage probability 1 - alpha, we use the percent point function (inverse CDF)
    z = norm.ppf(1 - alpha / 2)
    lower = y_pred - z * err
    upper = y_pred + z * err
    coverage = (y_true >= lower) & (y_true <= upper)
    return np.mean(coverage) * 100.0


def mpiw_1d(y_true, y_pred, err, alpha=0.05):
    """
    Calculates the Mean Prediction Interval Width (MPIW).
    Measures the 'sharpness' (tightness) of the uncertainty bounds.
    Note: y_true and y_pred are included to maintain a consistent
    callable signature with other uncertainty metrics.
    """
    # Use the exact same normal approximation as picp_1d
    z = norm.ppf(1 - alpha / 2)

    # Width of the interval is Upper Bound - Lower Bound
    # Upper = y_pred + z * err, Lower = y_pred - z * err
    # Width = 2 * z * err
    widths = 2 * z * err

    return np.mean(widths)

