import numpy as np
import matplotlib.pyplot as plt
from keras import ops
from sklearn.metrics import mean_squared_error

from .CONSTANTS import translate_filternos, mycolors
from .data import (
    prepare_phys_arrays,
    unscale_values,
    flux2mag,
    get_context_set_dense,
    get_phys_updown_errs,
)
from .metrics import calculate_msse_weighted, chisq


def plot_lc_dataframe(
    df,
    ax=None,
    mjd_col="mjd",
    flux_col="flux",  # add support for mag
    fluxerr_col="flux_err",  # and magerr
    flt_col="fltnum",
    filter_map=translate_filternos,
    color_map=mycolors,
    show_labels=True,
    **kwargs,
):
    if len(df) != 1:
        raise NotImplementedError(
            "Currently only supports plotting one object at a time (one row in the DataFrame)."
        )

    row = df.rows(named=True)[0]
    # Pop plotting kwargs with defaults (like old code)
    fmt = kwargs.pop("fmt", "-")
    alpha = kwargs.pop("alpha", 0.7)
    markersize = kwargs.pop("markersize", 5)
    fill_alpha = kwargs.pop("fill_alpha", 0.1)
    label_prefix = kwargs.pop("label_prefix", None)

    if ax is None:
        fig, ax = plt.subplots()

    # Iterate over each object in the DataFrame (usually just one for plotting)
    mjds = np.array(row[mjd_col])
    fluxes = np.array(row[flux_col])
    flts = np.array(row[flt_col])
    fluxerrs = np.array(row[fluxerr_col]) if fluxerr_col in row else None

    unique_filters = sorted(list(filter_map.keys()))

    for it_num, f_idx in enumerate(unique_filters):
        mask = flts == f_idx
        sorted_idx = np.argsort(mjds[mask])

        if show_labels:
            label = filter_map[f_idx] if filter_map else f"Filter {f_idx}"
            label = f"{label_prefix}{label}" if label_prefix is not None else label
        else:
            label = None

        if color_map:
            if isinstance(color_map, dict):
                color = color_map.get(f_idx, "black")
            elif isinstance(color_map, list):
                color = color_map[it_num % len(color_map)]

        t = mjds[mask][sorted_idx]
        y = fluxes[mask][sorted_idx]

        ax.errorbar(
            x=t,
            y=y,
            fmt=fmt,
            color=color,
            label=label,
            alpha=alpha,
            markersize=markersize,
            **kwargs,
        )

        if fluxerrs is not None:
            yerr = fluxerrs[mask][sorted_idx]
            ax.fill_between(
                x=t,
                y1=y - yerr,
                y2=y + yerr,
                color=color,
                alpha=fill_alpha,
            )

    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux")
    return ax


def plot_lightcurve_by_channel(
    x,
    y,
    yerr_range=None,
    ax=None,
    filter_map=translate_filternos,
    color_map=mycolors,
    **kwargs,
):
    x = np.array(x) if not hasattr(x, "numpy") else x.numpy()
    y = np.array(y) if not hasattr(y, "numpy") else y.numpy()

    fmt = kwargs.pop("fmt", "-")
    alpha = kwargs.pop("alpha", 0.7)
    markersize = kwargs.pop("markersize", 5)
    fill_alpha = kwargs.pop("fill_alpha", 0.1)

    dates = x[:, 0]
    filters = x[:, 1]
    fluxes = y[:, 0]

    fluxerrs_down, fluxerrs_up = None, None  # error bars
    if yerr_range is not None:
        yerr_down = (
            np.asarray(yerr_range[0])
            if not hasattr(yerr_range[0], "numpy")
            else yerr_range[0].numpy()
        )
        yerr_up = (
            np.asarray(yerr_range[1])
            if not hasattr(yerr_range[1], "numpy")
            else yerr_range[1].numpy()
        )
        fluxerrs_down = yerr_down[:, 0]
        fluxerrs_up = yerr_up[:, 0]

    if ax is None:
        fig, ax = plt.subplots()

    unique_filters = sorted(np.unique(filters))

    for it_num, f_idx in enumerate(unique_filters):
        mask = filters == f_idx
        sorted_idx = np.argsort(dates[mask])
        label = filter_map[f_idx] if filter_map else f"Filter {f_idx}"
        color = color_map[it_num % len(color_map)] if color_map else None

        ax.errorbar(
            x=dates[mask][sorted_idx],
            y=fluxes[mask][sorted_idx],
            fmt=fmt,
            color=color,
            label=label,
            alpha=alpha,
            markersize=markersize,
            **kwargs,
        )
        if yerr_range is not None:
            ax.fill_between(
                x=dates[mask][sorted_idx],
                y1=fluxerrs_down[mask][sorted_idx],
                y2=fluxerrs_up[mask][sorted_idx],
                color=color,
                alpha=fill_alpha,
            )

    return ax


def gplike_plot_functions(
    ax,
    pred_x,
    pred_y_mean,
    pred_y_std,
    target_x,
    target_y,
    context_x,
    context_y,
    mycolors,
    translate_filters,
    objnum=0,
):
    pred_x, pred_y_mean, pred_y_std, target_x, target_y, context_x, context_y = map(
        ops.convert_to_numpy,
        [pred_x, pred_y_mean, pred_y_std, target_x, target_y, context_x, context_y],
    )

    translate_filternos = {v: k for k, v in translate_filters.items()}

    y_channels = np.unique(target_x[objnum, :, 1])

    for y_channel in y_channels:
        channel_idx = int(y_channel)
        filter_name = translate_filternos.get(channel_idx, f"Filt_{channel_idx}")

        color = mycolors[channel_idx % len(mycolors)]

        # ground truth (dotted)
        target_mask = target_x[objnum, :, 1] == y_channel
        sort_indices = np.argsort(target_x[objnum, target_mask, 0])
        ax.plot(
            target_x[objnum, target_mask, 0][sort_indices],
            target_y[objnum, target_mask, 0][sort_indices],
            color=color,
            linestyle=":",
            linewidth=2,
            label=f"Truth ({filter_name})",
        )

        # mean preds (solid line) ---
        pred_mask = pred_x[objnum, :, 1] == y_channel
        sort_indices = np.argsort(pred_x[objnum, pred_mask, 0])
        ax.plot(
            pred_x[objnum, pred_mask, 0][sort_indices],
            pred_y_mean[objnum, pred_mask, 0][sort_indices],
            color=color,
            linewidth=2,
            label=f"Prediction ({filter_name})",
        )
        # uncertainty preds (shaded)
        ax.fill_between(
            pred_x[objnum, pred_mask, 0][sort_indices],
            pred_y_mean[objnum, pred_mask, 0]
            - pred_y_std[objnum, pred_mask, 0][sort_indices],
            pred_y_mean[objnum, pred_mask, 0]
            + pred_y_std[objnum, pred_mask, 0][sort_indices],
            alpha=0.02,
            color=color,
        )

        # context (large circles)
        context_mask = context_x[objnum, :, 1] == y_channel
        if np.any(context_mask):
            ax.plot(
                context_x[objnum, context_mask, 0],
                context_y[objnum, context_mask, 0],
                "o",  # Circle marker
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label=f"Context ({filter_name})",
            )

    ax.set_facecolor("white")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    return ax


def plot_lc_pred_scenarios(
    model,
    all_scenarios,
    X_test,
    y_test,
    time_scaler,
    flux_scaler,
    TEST_OBJ_CHOOSE=1,
    ADDFLUX_FOR_MAG_CONST=0,
    SCENARIO_CHOOSE=0,
    result_verify=None,
):

    choose_scenario = all_scenarios[SCENARIO_CHOOSE]
    fig, axs = plt.subplots(2, 2, figsize=(6.4 * 2, 4.8 * 2))

    for i, choose_strategy in enumerate(choose_scenario.keys()):
        ax = axs.ravel()[i]

        ctx_x_batch, ctx_y_batch = choose_scenario[choose_strategy]
        c_x = ops.expand_dims(ctx_x_batch[TEST_OBJ_CHOOSE], 0)
        c_y = ops.expand_dims(ctx_y_batch[TEST_OBJ_CHOOSE], 0)

        t_x = ops.expand_dims(X_test[TEST_OBJ_CHOOSE], 0)
        t_y = ops.expand_dims(y_test[TEST_OBJ_CHOOSE], 0)

        pred_mean_norm, pred_std_norm = model.test_step(c_x, c_y, t_x)

        pred_mean_norm = pred_mean_norm[0]
        pred_std_norm = pred_std_norm[0]
        target_x_norm = t_x[0]
        target_y_norm = t_y[0]
        context_x_norm = c_x[0]
        context_y_norm = c_y[0]

        target_x_phys, target_y_phys = prepare_phys_arrays(
            target_x_norm, target_y_norm, time_scaler, flux_scaler
        )
        pred_x_phys, pred_y_phys = prepare_phys_arrays(
            target_x_norm, pred_mean_norm, time_scaler, flux_scaler
        )

        context_x_phys, context_y_phys = prepare_phys_arrays(
            context_x_norm, context_y_norm, time_scaler, flux_scaler
        )

        y_norm_down_bound, y_norm_up_bound = get_phys_updown_errs(
            pred_mean_norm, pred_std_norm
        )

        pred_phys_down_bound = unscale_values(y_norm_down_bound, flux_scaler)
        pred_phys_up_bound = unscale_values(y_norm_up_bound, flux_scaler)

        pred_yerr_phys = (pred_phys_up_bound - pred_phys_down_bound) / 2  ## APPROX

        # mags
        pred_mags = flux2mag(pred_y_phys + ADDFLUX_FOR_MAG_CONST)

        # note: up-down flip due to mag is inverse
        pred_mags_down_bound = flux2mag(pred_phys_up_bound + ADDFLUX_FOR_MAG_CONST)
        pred_mags_up_bound = flux2mag(pred_phys_down_bound + ADDFLUX_FOR_MAG_CONST)

        target_mags = flux2mag(target_y_phys + ADDFLUX_FOR_MAG_CONST)

        pred_mags_err = (pred_mags_up_bound - pred_mags_down_bound) / 2  ## APPROX

        flux_mse = mean_squared_error(target_y_phys, pred_y_phys)  # noqa
        flux_msse = calculate_msse_weighted(target_x_phys, target_y_phys, pred_y_phys)
        flux_chisq = chisq(  # noqa
            target_y_phys.ravel(), pred_y_phys.ravel(), pred_yerr_phys.ravel()
        )

        try:
            mag_mse = mean_squared_error(target_mags, pred_mags)
            mag_msse = calculate_msse_weighted(target_x_phys, target_mags, pred_mags)
            mag_chisq = chisq(
                target_mags.ravel(), pred_mags.ravel(), pred_mags_err.ravel()
            )
        except ValueError:
            mag_mse = np.nan  # noqa
            mag_msse = np.nan  # noqa
            mag_chisq = np.nan  # noqa

        plot_lightcurve_by_channel(context_x_phys, context_y_phys, fmt="o", ax=ax)
        plot_lightcurve_by_channel(
            target_x_phys, target_y_phys, fmt="x", ax=ax, alpha=0.5
        )
        yl2 = ax.get_ylim()
        plot_lightcurve_by_channel(
            pred_x_phys,
            pred_y_phys,
            [pred_phys_down_bound, pred_phys_up_bound],
            fmt="-",
            ax=ax,
        )
        ax.set_xlabel(r"$\Delta$ MJD")
        ax.set_ylabel("Flux (Jy)")
        ax.set_ylim(yl2)
        ax.set_title(f"Scenario {choose_strategy} -- Flux MSSE: {flux_msse:.2f}")

        if result_verify is not None:
            assert np.isclose(
                result_verify[
                    (result_verify["Run_ID"] == SCENARIO_CHOOSE)
                    & (result_verify["Object_ID"] == TEST_OBJ_CHOOSE)
                ]["flux_msse"].to_numpy()[i],
                flux_msse,
            )


def plot_hist_msse(evaluation_summary):

    df = evaluation_summary.reset_index(drop=False).round(3)

    plot_means = df.pivot(
        index="Object_ID", columns="Strategy", values="flux_msse_mean"
    )
    plot_errors = df.pivot(
        index="Object_ID", columns="Strategy", values="flux_msse_std"
    )

    fig, ax = plt.subplots()

    plot_means.plot(
        kind="bar",
        yerr=plot_errors,
        ax=ax,
        capsize=4,
        rot=0,
        alpha=0.8,
        edgecolor="black",
        # log=True
    )

    ax.set_ylabel("Flux MSSE")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
    return ax


# TODO FIX time_scaler, flux_scaler usage and suptitlename
def viz_lc_selection(
    X_train_list,
    y_train_list,
    target_x_batch,
    target_y_batch,
    original_indices_batch,
    time_scaler,
    flux_scaler,
    i=0,
    suptitlename=None,
):
    print("Note: Un-normalizing data for plotting!")
    target_x_sample = target_x_batch[i]
    target_y_sample = target_y_batch[i]
    original_index = original_indices_batch[i].numpy()

    # Retrieve the original, full, variable-length light curve using the index
    original_x_sample = X_train_list[original_index]
    original_y_sample = y_train_list[original_index]

    # Sample a context set from normalised target set ---
    unique_filters = sorted(np.unique(original_x_sample[:, 1]).astype(int))
    num_unique_filters = len(unique_filters)

    num_context_sample = np.random.choice(
        np.linspace(
            *[num_unique_filters * 2, num_unique_filters * 10], num=10, dtype="int"
        )
    )
    context_x_sample, context_y_sample = get_context_set_dense(
        ops.expand_dims(target_x_sample, 0),
        ops.expand_dims(target_y_sample, 0),
        num_context_sample,
    )
    context_x_sample = ops.squeeze(context_x_sample, axis=0)
    context_y_sample = ops.squeeze(context_y_sample, axis=0)

    # get physical unit arrays
    orig_x_phys, orig_y_phys = prepare_phys_arrays(
        original_x_sample, original_y_sample, time_scaler, flux_scaler
    )
    targ_x_phys, targ_y_phys = prepare_phys_arrays(
        target_x_sample, target_y_sample, time_scaler, flux_scaler
    )
    cont_x_phys, cont_y_phys = prepare_phys_arrays(
        context_x_sample, context_y_sample, time_scaler, flux_scaler
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)

    # plot full lc
    plot_lightcurve_by_channel(
        orig_x_phys,
        orig_y_phys,
        ax=axes[0],
        alpha=0.7,
        fmt="o",
    )
    axes[0].set_title(
        f"1. Original full light curve ({original_x_sample[:,0].shape[0]} pts)"
    )

    # ploot target set
    plot_lightcurve_by_channel(
        targ_x_phys,
        targ_y_phys,
        ax=axes[1],
        alpha=0.7,
        fmt="o",
    )

    axes[1].set_title(f"2. Cropped target set ({target_x_sample[:,0].shape[0]} pts)")

    # plot context set
    plot_lightcurve_by_channel(
        targ_x_phys,
        targ_y_phys,
        ax=axes[2],
        alpha=0.1,
        fmt="o",
    )
    plot_lightcurve_by_channel(
        cont_x_phys,
        cont_y_phys,
        ax=axes[2],
        markersize=8,
        markeredgecolor="black",
        alpha=1.0,  # Highlighting context
        fmt="o",
    )
    axes[2].set_title(f"3. Context Set ({num_context_sample} pts)")

    # wrapping stuff up
    # plt.suptitle(
    #     f'Light Curve SNID{train_meta["SIM_TYPE_NAME"].index.to_numpy()[original_index]} ({train_meta["SIM_TYPE_NAME"].to_numpy()[original_index]})'
    # )
    # TODO FIX suptitlename
    plt.suptitle(suptitlename)

    # plot 1 shaded region: can plot now
    mintime_flux = target_y_sample[:, 0].numpy()[target_x_sample[:, 0].numpy().argmin()]
    maxtime_flux = target_y_sample[:, 0].numpy()[target_x_sample[:, 0].numpy().argmax()]
    slice_t_min = original_x_sample[:, 0][
        np.argwhere(original_y_sample[:, 0] == mintime_flux)[0][0]
    ]
    slice_t_min = unscale_values([slice_t_min], time_scaler)[0]
    slice_t_max = original_x_sample[:, 0][
        np.argwhere(original_y_sample[:, 0] == maxtime_flux)[0][0]
    ]
    slice_t_max = unscale_values([slice_t_max], time_scaler)[0]
    axes[0].axvspan(
        slice_t_min, slice_t_max, color="gray", alpha=0.2, label="Sampled Region"
    )

    plt.tight_layout()
    plt.show()
