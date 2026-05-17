import numpy as np
import pandas as pd
import warnings
from keras import ops
import sklearn
from collections import defaultdict
from scipy.stats import norm
import polars as pl
from tqdm.auto import tqdm
from .data import prepare_phys_arrays, get_phys_updown_errs, unscale_values, flux2mag
from .CONSTANTS import translate_filternos

lup_const = 2.5 / np.log(10)


def flux_to_luptitudes(flux, zp=27.5):
    return zp - (lup_const * np.arcsinh(flux / 2))


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
    return np.sqrt(np.mean((y_true - y_pred) ** 2 / y_err**2))


def picp_mpiw_1d(y_true, y_pred, err, confidence=0.95):
    # y_err is predicted error
    p_left, p_right = norm.interval(confidence=confidence, loc=y_pred, scale=err)
    picp = np.mean((y_true > p_left) * (y_true <= p_right)) * 100.0
    mpiw = np.mean(p_right - p_left)
    return picp, mpiw


def peak_time_fluxmag_error_1d(y_true, y_pred, mjd_common):
    # first, shorthands for useful quantities
    idx_true = np.argmax(y_true)
    idx_pred = np.argmax(y_pred)
    peak_true = mjd_common[idx_true]
    peak_pred = mjd_common[idx_pred]
    peak_flux_true = y_true[idx_true]
    peak_flux_pred = y_pred[idx_pred]
    peak_mags_true = flux_to_luptitudes(peak_flux_true)
    peak_mags_pred = flux_to_luptitudes(peak_flux_pred)

    peak_time_diff = peak_pred - peak_true  # signed error in days of peak estimate
    peak_time_absdiff = np.abs(
        peak_pred - peak_true
    )  # absolute error in days of peak estimate
    peak_mags_absdiff = np.abs(peak_mags_pred - peak_mags_true)  # absdiff in mag

    # frac_err = (
    #     (peak_flux_pred - peak_flux_true) / peak_flux_true
    #     if peak_flux_true != 0
    #     else np.nan
    # )
    # return peak_time_error, frac_err, peak_flux_true, peak_flux_pred

    return peak_time_diff, peak_time_absdiff, peak_mags_absdiff


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


def one_objd_metrics_df(df_pred, lc_target):
    assert len(lc_target) == 1 and len(df_pred) == 1
    assert lc_target["objid"][0] == df_pred["objid"][0]
    cur_objid = lc_target["objid"][0]

    pred_row = df_pred.row(0, named=True)
    true_row = lc_target.row(0, named=True)
    return one_obj_metrics(pred_row, true_row)


def one_obj_metrics(pred_row, true_row):
    """
    Calculates a suite of metrics for a single object by comparing predicted
    and true light curves across all photometric filters.

    Parameters
    ----------
    df_pred : pl.DataFrame, 1 row
        Predicted light curve with list-type columns: mjd, fltnum, flux,
        and optionally flux_err.
    lc_target : pl.DataFrame, 1 row
        True (target) light curve with list-type columns: mjd, fltnum, flux.

    Returns
    -------
    pl.DataFrame with one row containing scalar metric columns.
    """
    pred_mjd = np.round(np.asarray(pred_row["mjd"], dtype=float), 3)
    pred_flt = np.asarray(pred_row["fltnum"], dtype=int)
    pred_flux = np.asarray(pred_row["flux"], dtype=float)
    pred_err = (
        np.asarray(pred_row["flux_err"], dtype=float)
        if "flux_err" in pred_row and pred_row["flux_err"] is not None
        else None
    )

    true_mjd = np.round(np.asarray(true_row["mjd"], dtype=float), 3)
    true_flt = np.asarray(true_row["fltnum"], dtype=int)
    true_flux = np.asarray(true_row["flux"], dtype=float)

    # Metric Groups
    STANDARD_METRICS = {
        "msse": msse_1d,
        "mase": mase_1d,
        "mape": mape_1d,
        "mse": mean_squared_error,
        "rmse": rmse_1d,
        "mae": mae_1d,
        "rse": rse_1d,
        "rae": rae_1d,
    }
    UNCERTAINTY_METRICS = {
        "chi2": chi2_1d,
        "nrmseo": nrmseo_1d,
        "nrmsep": nrmsep_1d,
        "nlpd": nlpd_1d,
        "picp": picp_1d,
        "mpiw": mpiw_1d,
    }

    # 1. Global Peak Estimation
    try:
        assert (true_mjd == pred_mjd).all()

        global_peak_time_diff, global_peak_time_absdiff, global_peak_mags_absdiff = (
            peak_time_fluxmag_error_1d(
                y_true=true_flux, y_pred=pred_flux, mjd_common=true_mjd
            )
        )
    except Exception as e:
        global_peak_time_diff, global_peak_time_absdiff, global_peak_mags_absdiff = (
            np.nan,
            np.nan,
            np.nan,
        )

    metric_results = defaultdict(list)

    # 2. Per-band metrics; averaged over bands
    for fltnum in translate_filternos.keys():
        p_mask = pred_flt == fltnum
        t_mask = true_flt == fltnum

        if not np.any(p_mask) or not np.any(t_mask):
            continue

        pred_mjd_band = np.round(pred_mjd[p_mask], 3)
        true_mjd_band = np.round(true_mjd[t_mask], 3)
        pred_flux_band = pred_flux[p_mask]
        true_flux_band = true_flux[t_mask]
        pred_err_band = pred_err[p_mask] if pred_err is not None else None

        # sort
        p_sort = np.argsort(pred_mjd_band)
        t_sort = np.argsort(true_mjd_band)
        pred_mjd_band, pred_flux_band = pred_mjd_band[p_sort], pred_flux_band[p_sort]
        true_mjd_band, true_flux_band = true_mjd_band[t_sort], true_flux_band[t_sort]
        pred_err_band = pred_err_band[p_sort] if pred_err is not None else None

        if len(pred_mjd_band) != len(true_mjd_band) or not np.array_equal(
            pred_mjd_band, true_mjd_band
        ):
            warnings.warn(
                f"MJD mismatch for objid={pred_row['objid']}, fltnum={fltnum}; proceeding with available aligned points."
            )
            shared_mjd, p_idx, t_idx = np.intersect1d(
                pred_mjd_band, true_mjd_band, return_indices=True
            )
            if len(shared_mjd) < 2:
                continue
            pred_flux_band = pred_flux_band[p_idx]
            true_flux_band = true_flux_band[t_idx]
            if pred_err_band is not None:
                pred_err_band = pred_err_band[p_idx]
            true_mjd_band = shared_mjd

        if len(true_flux_band) < 2:
            continue

        def _safe_metric(func, *args, **kwargs):
            try:
                return float(func(*args, **kwargs))
            except Exception:
                return np.nan

        # 1. Standard Regression Metrics
        for m_name, m_func in STANDARD_METRICS.items():
            metric_results[m_name].append(
                _safe_metric(m_func, true_flux_band, pred_flux_band)
            )

        # 2. Peak Estimation Metrics
        try:
            band_peak_time_diff, band_peak_time_absdiff, band_peak_mags_absdiff = (
                peak_time_fluxmag_error_1d(
                    true_flux_band, pred_flux_band, true_mjd_band
                )
            )
        except Exception:
            band_peak_time_diff, band_peak_time_absdiff, band_peak_mags_absdiff = (
                np.nan,
                np.nan,
                np.nan,
            )

        metric_results["band_peak_time_diff"].append(band_peak_time_diff)
        metric_results["band_peak_time_absdiff"].append(band_peak_time_absdiff)
        metric_results["band_peak_mags_absdiff"].append(band_peak_mags_absdiff)

        # 3. Uncertainty-based Metrics
        if (
            pred_err_band is None
            or np.any(pred_err_band <= 0)
            or np.any(~np.isfinite(pred_err_band))
        ):
            for m_name in UNCERTAINTY_METRICS.keys():
                metric_results[m_name].append(np.nan)
            metric_results["chi2_reduced"].append(np.nan)
        else:
            for m_name, m_func in UNCERTAINTY_METRICS.items():
                metric_results[m_name].append(
                    _safe_metric(m_func, true_flux_band, pred_flux_band, pred_err_band)
                )

            # chi2 reduced special case
            chi2 = _safe_metric(chi2_1d, true_flux_band, pred_flux_band, pred_err_band)
            metric_results["chi2_reduced"].append(
                chi2 / len(true_flux_band) if len(true_flux_band) > 0 else np.nan
            )

    # Aggregate metrics across filters
    out_dict = {"objid": int(true_row["objid"])}

    expected_metrics = (
        list(STANDARD_METRICS.keys())
        + list(UNCERTAINTY_METRICS.keys())
        + [
            "chi2_reduced",
            "band_peak_time_diff",
            "band_peak_time_absdiff",
            "band_peak_mags_absdiff",
        ]
    )

    for m_name in expected_metrics:
        vals = metric_results.get(m_name, [])
        out_dict[f"{m_name}_mean"] = float(np.nanmean(vals)) if vals else np.nan
        # out_dict[f"{m_name}_median"] = float(np.nanmedian(vals)) if vals else np.nan
        # Append raw list of metrics per filter
        # out_dict[m_name] = vals

    # Inject global metrics
    out_dict["global_peak_time_diff"] = global_peak_time_diff
    out_dict["global_peak_time_absdiff"] = global_peak_time_absdiff
    out_dict["global_peak_mags_absdiff"] = global_peak_mags_absdiff

    return out_dict


def compute_physical_metrics(
    model,
    all_scenarios,
    X_test,
    y_test,
    time_scaler,
    flux_scaler,
    ADDFLUX_FOR_MAG_CONST=0,
):
    results = []
    num_objects = X_test.shape[0]

    for run_idx, choose_scenario in enumerate(all_scenarios):
        for strategy_name, (ctx_x_batch, ctx_y_batch) in choose_scenario.items():
            for obj_idx in range(num_objects):
                c_x = ops.expand_dims(ctx_x_batch[obj_idx], 0)
                c_y = ops.expand_dims(ctx_y_batch[obj_idx], 0)

                t_x = ops.expand_dims(X_test[obj_idx], 0)
                t_y = ops.expand_dims(y_test[obj_idx], 0)

                pred_mean_norm, pred_std_norm = model.test_step(c_x, c_y, t_x)

                pred_mean_norm = pred_mean_norm[0]
                pred_std_norm = pred_std_norm[0]
                target_x_norm = t_x[0]
                target_y_norm = t_y[0]

                target_x_phys, target_y_phys = prepare_phys_arrays(
                    target_x_norm, target_y_norm, time_scaler, flux_scaler
                )
                pred_x_phys, pred_y_phys = prepare_phys_arrays(
                    target_x_norm, pred_mean_norm, time_scaler, flux_scaler
                )

                y_norm_down_bound, y_norm_up_bound = get_phys_updown_errs(
                    pred_mean_norm, pred_std_norm
                )

                pred_phys_down_bound = unscale_values(y_norm_down_bound, flux_scaler)
                pred_phys_up_bound = unscale_values(y_norm_up_bound, flux_scaler)

                pred_yerr_phys = (
                    pred_phys_up_bound - pred_phys_down_bound
                ) / 2  ## APPROX

                # mags
                pred_mags = flux2mag(pred_y_phys + ADDFLUX_FOR_MAG_CONST)

                # note: up-down flip due to mag is inverse
                pred_mags_down_bound = flux2mag(
                    pred_phys_up_bound + ADDFLUX_FOR_MAG_CONST
                )
                pred_mags_up_bound = flux2mag(
                    pred_phys_down_bound + ADDFLUX_FOR_MAG_CONST
                )

                target_mags = flux2mag(target_y_phys + ADDFLUX_FOR_MAG_CONST)

                pred_mags_err = (
                    pred_mags_up_bound - pred_mags_down_bound
                ) / 2  ## APPROX

                flux_mse = mean_squared_error(target_y_phys, pred_y_phys)
                flux_msse = calculate_msse_weighted(
                    target_x_phys, target_y_phys, pred_y_phys
                )
                flux_chisq = chisq(
                    target_y_phys.ravel(), pred_y_phys.ravel(), pred_yerr_phys.ravel()
                )

                try:
                    mag_mse = mean_squared_error(target_mags, pred_mags)
                    mag_msse = calculate_msse_weighted(
                        target_x_phys, target_mags, pred_mags
                    )
                    mag_chisq = chisq(
                        target_mags.ravel(), pred_mags.ravel(), pred_mags_err.ravel()
                    )
                except ValueError:
                    mag_mse = np.nan
                    mag_msse = np.nan
                    mag_chisq = np.nan

                results.append(
                    {
                        "Run_ID": run_idx,
                        "Strategy": strategy_name,
                        "Object_ID": obj_idx,
                        "flux_mse": flux_mse,
                        "mag_mse": mag_mse,
                        "flux_msse": flux_msse,
                        "mag_msse": mag_msse,
                        "flux_chisq": flux_chisq,
                        "mag_chisq": mag_chisq,
                    }
                )

    return pd.DataFrame(results)


def calculate_msse_weighted(target_x, y_true, y_pred):
    """
    Calculates Mean Scaled Squared Error (MSSE) weighted by point count per channel.
    Formula: MSE / Mean(Diff(Truth)^2)
    """
    # Unpack tensor inputs
    times = target_x[:, 0]
    filters = target_x[:, 1].astype(int)

    # Flatten arrays
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]

    unique_channels, counts = np.unique(filters, return_counts=True)
    channel_msses = []
    channel_weights = []

    for ch in unique_channels:
        mask = filters == ch

        # 1. NaN Filtering (Crucial for Magnitude space)
        # We need to filter based on y_true and y_pred NaNs if any
        valid_indices = np.where(mask & ~np.isnan(y_true) & ~np.isnan(y_pred))[0]

        if len(valid_indices) < 2:
            continue

        curr_time = times[valid_indices]
        curr_true = y_true[valid_indices]
        curr_pred = y_pred[valid_indices]

        # 2. Time Sorting (Crucial because your data loader shuffles points)
        sort_idx = np.argsort(curr_time)
        sorted_true = curr_true[sort_idx]
        sorted_pred = curr_pred[sort_idx]

        # 3. Calculate Metric using helper
        val = msse_1d(sorted_true, sorted_pred)

        channel_msses.append(val)
        channel_weights.append(len(sorted_true))

    if not channel_msses:
        return np.nan

    return np.average(channel_msses, weights=channel_weights)


def chisq(y_true, y_pred, y_err_pred):
    residuals = y_true - y_pred
    normalized_residuals = residuals / y_err_pred
    chi_square = np.sum(normalized_residuals**2)

    return chi_square


def compute_metrics_for_row(args):
    """
    Computes global and per-band metrics for a single row (object).
    Designed to be used with multiprocessing.Pool.
    """
    pred_row, true_row = args

    pred_mjd = np.round(np.asarray(pred_row["mjd"], dtype=float), 3)
    pred_flt = np.asarray(pred_row["fltnum"], dtype=int)
    pred_flux = np.asarray(pred_row["flux"], dtype=float)
    pred_err = (
        np.asarray(pred_row["flux_err"], dtype=float)
        if "flux_err" in pred_row and pred_row["flux_err"] is not None
        else None
    )

    true_mjd = np.round(np.asarray(true_row["mjd"], dtype=float), 3)
    true_flt = np.asarray(true_row["fltnum"], dtype=int)
    true_flux = np.asarray(true_row["flux"], dtype=float)

    true_err = (
        np.asarray(true_row["flux_err"], dtype=float)
        if "flux_err" in true_row and true_row["flux_err"] is not None
        else None
    )

    # ensure its sorted by time
    assert (np.sort(true_mjd) == true_mjd).all()
    assert (np.sort(pred_mjd) == pred_mjd).all()

    ### global metrics
    assert np.array_equal(true_mjd, pred_mjd)

    global_peak_time_diff, global_peak_time_absdiff, global_peak_mags_absdiff = (
        peak_time_fluxmag_error_1d(
            y_true=true_flux, y_pred=pred_flux, mjd_common=true_mjd
        )
    )

    global_mse = sklearn.metrics.mean_squared_error(y_true=true_flux, y_pred=pred_flux)

    global_nrmse_p = nrmse_po_1d(
        y_true=true_flux, y_pred=pred_flux, y_err=pred_err
    )  # nrsme_p -> if y_err is predicted error
    global_nrmse_o = nrmse_po_1d(
        y_true=true_flux, y_pred=pred_flux, y_err=true_err
    )  # nrsme_o -> if y_err is observed error

    global_chi2 = chi2_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err
    )  # chi square uses pred error, assumes no error in true val
    global_chi2_reduced = global_chi2 / len(true_flux)

    global_nlpd = nlpd_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err
    )  # y_err is predicted error
    global_picp68, global_mpiw68 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.68268
    )  # 1 sigma; y_err is predicted error
    global_picp95, global_mpiw95 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.95450
    )  # 2 sigmas; y_err is predicted error

    #### EXPERIMENTAL START
    global_picp0, global_mpiw0 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.00
    )
    global_picp5, global_mpiw5 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.05
    )
    global_picp10, global_mpiw10 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.10
    )
    global_picp15, global_mpiw15 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.15
    )
    global_picp20, global_mpiw20 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.20
    )
    global_picp25, global_mpiw25 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.25
    )
    global_picp30, global_mpiw30 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.30
    )
    global_picp35, global_mpiw35 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.35
    )
    global_picp40, global_mpiw40 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.40
    )
    global_picp45, global_mpiw45 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.45
    )
    global_picp50, global_mpiw50 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.50
    )
    global_picp55, global_mpiw55 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.55
    )
    global_picp60, global_mpiw60 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.60
    )
    global_picp65, global_mpiw65 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.65
    )
    global_picp70, global_mpiw70 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.70
    )
    global_picp75, global_mpiw75 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.75
    )
    global_picp80, global_mpiw80 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.80
    )
    global_picp85, global_mpiw85 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.85
    )
    global_picp90, global_mpiw90 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=0.90
    )
    global_picp100, global_mpiw100 = picp_mpiw_1d(
        y_true=true_flux, y_pred=pred_flux, err=pred_err, confidence=1.00
    )
    #### EXPERIMENTAL END

    ### per-band metrics
    metric_results = defaultdict(list)

    for fltnum in translate_filternos.keys():
        p_mask = pred_flt == fltnum
        t_mask = true_flt == fltnum

        if not np.any(p_mask) or not np.any(t_mask):
            continue

        pred_mjd_band = np.round(pred_mjd[p_mask], 3)
        true_mjd_band = np.round(true_mjd[t_mask], 3)
        pred_flux_band = pred_flux[p_mask]
        true_flux_band = true_flux[t_mask]
        pred_err_band = pred_err[p_mask] if pred_err is not None else None
        true_err_band = true_err[t_mask] if true_err is not None else None

        # sort
        p_sort = np.argsort(pred_mjd_band)
        t_sort = np.argsort(true_mjd_band)
        pred_mjd_band, pred_flux_band = pred_mjd_band[p_sort], pred_flux_band[p_sort]
        true_mjd_band, true_flux_band = true_mjd_band[t_sort], true_flux_band[t_sort]
        pred_err_band = pred_err_band[p_sort] if pred_err is not None else None
        true_err_band = true_err_band[t_sort] if true_err is not None else None

        assert np.array_equal(
            pred_mjd_band, true_mjd_band
        )  # check no MJD mismatch for objid/fltnum

        _, peak_time_absdiff_band, peak_mags_absdiff_band = peak_time_fluxmag_error_1d(
            y_true=true_flux_band, y_pred=pred_flux_band, mjd_common=true_mjd_band
        )

        mse_band = sklearn.metrics.mean_squared_error(
            y_true=true_flux_band, y_pred=pred_flux_band
        )

        msse_band = msse_1d(y_true=true_flux_band, y_pred=pred_flux_band)
        mase_band = mase_1d(y_true=true_flux_band, y_pred=pred_flux_band)
        mape_band = mape_1d(y_true=true_flux_band, y_pred=pred_flux_band)

        nrmse_p_band = nrmse_po_1d(
            y_true=true_flux_band, y_pred=pred_flux_band, y_err=pred_err_band
        )
        nrmse_o_band = nrmse_po_1d(
            y_true=true_flux_band, y_pred=pred_flux_band, y_err=true_err_band
        )

        chi2_band = chi2_1d(
            y_true=true_flux_band, y_pred=pred_flux_band, err=pred_err_band
        )
        chi2_reduced_band = chi2_band / len(true_flux_band)

        nlpd_band = nlpd_1d(
            y_true=true_flux_band, y_pred=pred_flux_band, err=pred_err_band
        )
        picp68_band, mpiw68_band = picp_mpiw_1d(
            y_true=true_flux_band,
            y_pred=pred_flux_band,
            err=pred_err_band,
            confidence=0.68268,
        )
        picp95_band, mpiw95_band = picp_mpiw_1d(
            y_true=true_flux_band,
            y_pred=pred_flux_band,
            err=pred_err_band,
            confidence=0.95450,
        )

        metric_results["peak_time_absdiff_band"].append(peak_time_absdiff_band)
        metric_results["peak_mags_absdiff_band"].append(peak_mags_absdiff_band)
        metric_results["mse_band"].append(mse_band)
        metric_results["msse_band"].append(msse_band)
        metric_results["mase_band"].append(mase_band)
        metric_results["mape_band"].append(mape_band)
        metric_results["nrmse_p_band"].append(nrmse_p_band)
        metric_results["nrmse_o_band"].append(nrmse_o_band)
        metric_results["chi2_band"].append(chi2_band)
        metric_results["chi2_reduced_band"].append(chi2_reduced_band)
        metric_results["nlpd_band"].append(nlpd_band)
        metric_results["mpiw68_band"].append(mpiw68_band)
        metric_results["picp68_band"].append(picp68_band)
        metric_results["mpiw95_band"].append(mpiw95_band)
        metric_results["picp95_band"].append(picp95_band)

    out_dict = {"objid": int(true_row["objid"])}

    out_dict["global_peak_time_diff"] = global_peak_time_diff
    out_dict["global_peak_time_absdiff"] = global_peak_time_absdiff
    out_dict["global_peak_mags_absdiff"] = global_peak_mags_absdiff
    out_dict["global_picp68"] = global_picp68
    out_dict["global_mpiw68"] = global_mpiw68
    out_dict["global_picp95"] = global_picp95
    out_dict["global_mpiw95"] = global_mpiw95
    out_dict["global_mse"] = global_mse
    out_dict["global_nrmse_p"] = global_nrmse_p
    out_dict["global_nrmse_o"] = global_nrmse_o
    out_dict["global_chi2"] = global_chi2
    out_dict["global_chi2_reduced"] = global_chi2_reduced
    out_dict["global_nlpd"] = global_nlpd

    #### EXPERIMENTAL START
    out_dict["global_picp0"] = global_picp0
    out_dict["global_mpiw0"] = global_mpiw0
    out_dict["global_picp5"] = global_picp5
    out_dict["global_mpiw5"] = global_mpiw5
    out_dict["global_picp10"] = global_picp10
    out_dict["global_mpiw10"] = global_mpiw10
    out_dict["global_picp15"] = global_picp15
    out_dict["global_mpiw15"] = global_mpiw15
    out_dict["global_picp20"] = global_picp20
    out_dict["global_mpiw20"] = global_mpiw20
    out_dict["global_picp25"] = global_picp25
    out_dict["global_mpiw25"] = global_mpiw25
    out_dict["global_picp30"] = global_picp30
    out_dict["global_mpiw30"] = global_mpiw30
    out_dict["global_picp35"] = global_picp35
    out_dict["global_mpiw35"] = global_mpiw35
    out_dict["global_picp40"] = global_picp40
    out_dict["global_mpiw40"] = global_mpiw40
    out_dict["global_picp45"] = global_picp45
    out_dict["global_mpiw45"] = global_mpiw45
    out_dict["global_picp50"] = global_picp50
    out_dict["global_mpiw50"] = global_mpiw50
    out_dict["global_picp55"] = global_picp55
    out_dict["global_mpiw55"] = global_mpiw55
    out_dict["global_picp60"] = global_picp60
    out_dict["global_mpiw60"] = global_mpiw60
    out_dict["global_picp65"] = global_picp65
    out_dict["global_mpiw65"] = global_mpiw65
    out_dict["global_picp70"] = global_picp70
    out_dict["global_mpiw70"] = global_mpiw70
    out_dict["global_picp75"] = global_picp75
    out_dict["global_mpiw75"] = global_mpiw75
    out_dict["global_picp80"] = global_picp80
    out_dict["global_mpiw80"] = global_mpiw80
    out_dict["global_picp85"] = global_picp85
    out_dict["global_mpiw85"] = global_mpiw85
    out_dict["global_picp90"] = global_picp90
    out_dict["global_mpiw90"] = global_mpiw90
    out_dict["global_picp100"] = global_picp100
    out_dict["global_mpiw100"] = global_mpiw100
    #### EXPERIMENTAL END

    for m_name in metric_results.keys():
        vals = metric_results.get(m_name, [])
        for i, val in enumerate(vals):
            out_dict[m_name.replace("_band", f"_{translate_filternos[i]}")] = val
        with np.errstate(invalid="ignore"):  # Ignore warnings for all-nan slices
            out_dict[m_name.replace("_band", "_bandsmean")] = np.nanmean(vals)

    return out_dict


def process_model(run_num, model_name, pred_path, lc_target, pool=None):
    """
    Process all predictions for a single model run.
    """
    pred_df = pl.read_parquet(pred_path)

    if pred_df.height != lc_target.height:
        raise ValueError(
            f"Run {run_num}, model {model_name}: pred/target row mismatch ({pred_df.height} vs {lc_target.height})"
        )

    pred_df = pred_df.sort("objid")
    # lc_target is expected to be already sorted by objid

    pred_rows = list(pred_df.iter_rows(named=True))
    target_rows = list(lc_target.iter_rows(named=True))

    is_aligned = len(pred_rows) == len(target_rows) and all(
        (p["objid"] == t["objid"]) and (len(p["mjd"]) == len(t["mjd"]))
        for p, t in zip(pred_rows, target_rows)
    )

    assert is_aligned, f"Data alignment failed for run {run_num}, model {model_name}"

    iterator = list(zip(pred_rows, target_rows))
    tot_rows = len(iterator)

    metrics = []

    if pool is not None:
        # Multiprocessing for speedup
        for out_dict in tqdm(
            pool.imap(compute_metrics_for_row, iterator, chunksize=100),
            total=tot_rows,
            leave=False,
            desc=f"Run {run_num} | {model_name}",
        ):
            metrics.append(out_dict)
    else:
        # Fallback to serial processing
        for args in tqdm(
            iterator,
            total=tot_rows,
            leave=False,
            desc=f"Run {run_num} | {model_name}",
        ):
            metrics.append(compute_metrics_for_row(args))

    metrics_df = pl.DataFrame(metrics).with_columns(
        model_name=pl.lit(model_name),
        run_num=pl.lit(run_num),
    )

    metrics_df = (
        metrics_df.drop(["run_num", "model_name"])
        .insert_column(1, metrics_df.get_column("run_num"))
        .insert_column(2, metrics_df.get_column("model_name"))
    )

    return metrics_df
