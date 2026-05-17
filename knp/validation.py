import matplotlib.pyplot as plt
import polars as pl

pl.Config.set_tbl_rows(100)

from .metrics import compute_physical_metrics
from .plotting import plot_hist_msse, plot_lc_pred_scenarios


def val_step_physical(
    model,
    all_scenarios,
    time_scaler,
    flux_scaler,
    X_test,
    y_test,
    TEST_OBJ_CHOOSE,
    SCENARIO_CHOOSE,
    ADDFLUX_FOR_MAG_CONST,
    epochnum=None,
):
    print("*" * 20, f"Epoch {epochnum}", "*" * 20)
    results = compute_physical_metrics(
        model,
        all_scenarios,
        X_test,
        y_test,
        time_scaler,
        flux_scaler,
        ADDFLUX_FOR_MAG_CONST=ADDFLUX_FOR_MAG_CONST,
    )

    final_msse_mean = results["flux_msse"].mean()
    final_msse_std = results["flux_msse"].std()
    print(rf"Total averaged val set MSSE: {final_msse_mean:.3f} ± {final_msse_std:.3f}")

    # results

    evaluation_summary = results.groupby(["Object_ID", "Strategy"]).agg(
        # flux_mse_mean=("flux_mse", "mean"),
        # flux_mse_std=("flux_mse", "std"),
        flux_msse_mean=("flux_msse", "mean"),
        flux_msse_std=("flux_msse", "std"),
        # flux_chisq_mean=("flux_chisq", "mean"),
        # flux_chisq_std=("flux_chisq", "std"),
        #
        # mag_mse_mean=("mag_mse", "mean"),
        # mag_mse_std=("mag_mse", "std"),
        mag_msse_mean=("mag_msse", "mean"),
        mag_msse_std=("mag_msse", "std"),
        # mag_chisq_mean=("mag_chisq", "mean"),
        # mag_chisq_std=("mag_chisq", "std"),
        #
    )
    plot_hist_msse(evaluation_summary)
    print(pl.from_pandas(evaluation_summary.reset_index(drop=False).round(3)))
    plt.title(f"Epoch {epochnum}")
    plt.show()

    plot_lc_pred_scenarios(
        model,
        all_scenarios,
        X_test,
        y_test,
        time_scaler,
        flux_scaler,
        TEST_OBJ_CHOOSE=TEST_OBJ_CHOOSE,
        ADDFLUX_FOR_MAG_CONST=ADDFLUX_FOR_MAG_CONST,
        SCENARIO_CHOOSE=SCENARIO_CHOOSE,
        result_verify=results,
    )
    plt.suptitle(f"Epoch {epochnum}")
    plt.tight_layout()
    plt.show()

    return results
