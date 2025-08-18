import warnings

from keras import ops

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

import tensorflow_probability as tfp


def get_context_set(
    target_x,
    target_y,
    num_context,
    # either a single no. (total or each)
    # or a list ([each1, each2 ...eachn])
    # where n=#ofchannels aka np.unique(target_x[:,:,1])
    num_context_mode="all",  # or "each"
    seed=None,
    # ADD SUPPORT FOR RANGE LATER - then sample different num
    # context_frac_range=[0.01, 0.05],
):
    """
    Selects a subset of points from the target set to form the context set.
    Sid note: shape is (samples, points, channels) for x and y both

    Args:
        target_x (np.array or tf.Tensor):
            Shape (batch_size, num_points, x_channels).
            For multi-channel data in 'long' format, x_channels is typically
            2 for (time, channel_id).
        target_y (np.array or tf.Tensor):
            Shape (batch_size, num_points, y_channels).
        num_context (int or list/tuple):
            - If mode is "all": An integer specifying the total number of
            context points.
            - If mode is "each": An integer for sampling that many points from *each*
              channel, OR a list/tuple specifying the number of points for each
              channel individually (e.g., [10, 5] for 2 channels).
        num_context_mode (str):
            - "all": Randomly sample `num_context` points from the entire set,
              regardless of channel.
            - "each": Sample a specified number of points from each channel separately.
        seed (int, optional): Seed for the random number generator.

    """

    assert target_x.shape[:-1] == target_y.shape[:-1]
    # the inputs should have same no. of samples and points, channels may differ

    if isinstance(num_context, (list, tuple)) and num_context_mode != "each":
        raise ValueError(
            "For a list-like collection of num_context, the mode must be 'each'."
        )

    if tf.is_tensor(target_x) and tf.is_tensor(target_y) and num_context_mode == "all":
        context_x, context_y = _get_context_set_tf(
            target_x, target_y, num_context, seed
        )
    else:
        context_x, context_y = _get_context_set_np(
            target_x,
            target_y,
            num_context,
            num_context_mode=num_context_mode,
            seed=seed,
        )

    return context_x, context_y


def _get_context_set_tf(
    target_x,
    target_y,
    num_context,
    seed=None,
    # ADD SUPPORT FOR RANGE LATER - then sample different num
    # context_frac_range=[0.01, 0.05],
):
    # Validate early (static when possible)
    n_points_static = target_x.shape[1]
    if (n_points_static is not None) and (num_context > n_points_static):
        raise ValueError(
            f"Requesting {num_context} context points but",
            " only {n_points_static} available.",
        )

    n_points = tf.shape(target_x)[1]

    # Runtime guard too (works under tf.function)
    tf.debugging.assert_less_equal(
        tf.cast(num_context, tf.int32),
        n_points,
        message="num_context exceeds number of points",
    )

    indices = tf.range(n_points)
    if seed is not None:
        seed_vec = tf.convert_to_tensor([seed, 0], dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(
            indices, seed=seed_vec
        )[:num_context]
    else:
        shuffled_indices = tf.random.shuffle(indices)[:num_context]

    context_x = tf.gather(target_x, shuffled_indices, axis=1)
    context_y = tf.gather(target_y, shuffled_indices, axis=1)

    return context_x, context_y


def _get_context_set_np(
    target_x,
    target_y,
    num_context,
    # either a single no. (total or each)
    # or a list ([each1, each2 ...eachn])
    # where n=#ofchannels aka np.unique(target_x[:,:,1])
    num_context_mode="all",  # or "each"
    seed=None,
    # ADD SUPPORT FOR RANGE LATER - then sample different num
    # context_frac_range=[0.01, 0.05],
):
    n_points = target_x.shape[1]
    # Convert to array if tensor
    target_x_np, target_y_np = ops.convert_to_numpy(target_x), ops.convert_to_numpy(
        target_y
    )
    rng = np.random.default_rng(seed)

    final_indices = []

    if num_context_mode == "all":
        # Sample `num_context` indices from the total number of points.
        final_indices = rng.choice(n_points, num_context, replace=False)
    elif num_context_mode == "each":
        # --- NEW: Vectorized Grouping using argsort ---
        # This is much faster than looping and calling np.where.

        # 1. Get all channel IDs from the first sample (structure is same for all)
        all_channel_ids = target_x_np[0, :, 1]

        # 2. Get the indices that would sort the channel IDs
        sort_idx = np.argsort(all_channel_ids, kind="stable")

        # 3. Find the locations where the channel
        # ID changes. These are the boundaries of each group.
        #    We use np.diff on the sorted channel IDs to find these boundaries.
        sorted_channel_ids = all_channel_ids[sort_idx]
        # Adding a value at the end to ensure the last group is captured
        diff_array = np.diff(np.append(sorted_channel_ids, np.inf))
        split_points = np.where(diff_array)[0] + 1

        # 4. `split` the sorted original indices into groups based on the boundaries.
        #    Each element in this list is an array of original indices for one channel.
        grouped_original_indices = np.split(sort_idx, split_points[:-1])
        # --- End of Vectorized Grouping ---

        channels = np.unique(sorted_channel_ids)
        n_channels = len(channels)

        if isinstance(num_context, (list, tuple)):
            assert (
                len(num_context) == n_channels
            ), f"Length of num_context list ({len(num_context)}) "
            f"must match the number of channels ({n_channels})"

        all_channel_indices = []
        # This loop is now very fast as it just operates on pre-calculated groups
        for i, channel_indices in enumerate(grouped_original_indices):
            if isinstance(num_context, (list, tuple)):
                num_to_sample = num_context[i]
            else:
                num_to_sample = num_context

            if num_to_sample > len(channel_indices):
                raise ValueError(
                    f"Requesting {num_to_sample} context points "
                    f"for channel {channels[i]}, "
                    f"but only {len(channel_indices)} are available."
                )

            chosen_indices = rng.choice(channel_indices, num_to_sample, replace=False)
            all_channel_indices.append(chosen_indices)

        final_indices = np.concatenate(all_channel_indices)

    else:
        raise AttributeError("num_context_mode can only be 'all' or 'each'.")

    context_x = target_x_np[:, final_indices, :]
    context_y = target_y_np[:, final_indices, :]

    if tf.is_tensor(target_x):
        context_x = tf.constant(context_x)
        context_y = tf.constant(context_y)

    return context_x, context_y


def get_train_batch(X_train, y_train, batch_size=64):
    """
    Sidnote: shape is (samples, points, channels) for x and y both
    """
    assert X_train.shape[:-1] == y_train.shape[:-1]
    tot_samples = X_train.shape[0]
    # choose batch_size points from tot_batches e.g. 64 from 10000 randomly
    batch_indices = np.random.choice(tot_samples, batch_size, replace=False)

    if isinstance(X_train, np.ndarray):
        X_train_batch = X_train[batch_indices, :, :]
        y_train_batch = y_train[batch_indices, :, :]
    elif tf.is_tensor(X_train):
        X_train_batch = tf.gather(X_train, batch_indices, axis=0)
        y_train_batch = tf.gather(y_train, batch_indices, axis=0)
    else:
        print("X_train is neither a NumPy array nor a TensorFlow tensor")

    return X_train_batch, y_train_batch


def gplike_calculate_mymetrics(
    pred_x,
    pred_y_mean,
    pred_y_std,
    target_x,
    target_y,
    context_x,
    context_y,
    closeness_thresh=0.1,
    std_multiplier=1,
):
    """
    Calculates metrics directly on long-format data.
    This version is robust to uneven numbers of context points per channel.
    """
    # Convert all to numpy for calculation
    pred_x, pred_y_mean, pred_y_std, target_x, target_y, context_x, context_y = map(
        ops.convert_to_numpy,
        [pred_x, pred_y_mean, pred_y_std, target_x, target_y, context_x, context_y],
    )

    # 1. Target Coverage: Fraction of truth points within 1-sigma uncertainty
    upper_bound = pred_y_mean + pred_y_std
    lower_bound = pred_y_mean - pred_y_std
    target_in_bounds = (target_y >= lower_bound) & (target_y <= upper_bound)
    target_coverage = np.mean(target_in_bounds)

    # 2. Context Reconstruction Coverage
    # For each context point, find the closest prediction and check if it's in bounds.
    batch_size = context_x.shape[0]
    context_coverage_scores = []
    context_is_close_scores = []

    for i in range(batch_size):
        # Find predictions corresponding to context points
        # This is simpler as target_x is the same as the full x space of the function
        # We assume context_x points exist within target_x
        # Create a mapping from a tuple of coords to an index
        target_map = {tuple(row): idx for idx, row in enumerate(target_x[i])}
        context_indices_in_target = [target_map[tuple(row)] for row in context_x[i]]

        pred_mean_at_context = pred_y_mean[i, context_indices_in_target]
        true_val_at_context = target_y[i, context_indices_in_target]
        pred_std_at_context = pred_y_std[i, context_indices_in_target]

        is_context_close = (
            np.abs(true_val_at_context - pred_mean_at_context) <= closeness_thresh
        )
        context_is_close_scores.append(is_context_close)

        upper = pred_mean_at_context + pred_std_at_context
        lower = pred_mean_at_context - pred_std_at_context

        coverage = (context_y[i] >= lower) & (context_y[i] <= upper)
        context_coverage_scores.append(np.mean(coverage))

    return {
        "target_coverage": target_coverage,
        "fraction_context_close": np.mean(context_is_close_scores),
        "context_reconstruction_coverage": np.mean(context_coverage_scores),
        "mean_predictive_confidence": np.mean(pred_y_std),
    }


def gplike_plot_functions(
    ax,
    pred_x,
    pred_y_mean,
    pred_y_std,
    target_x,
    target_y,
    context_x,
    context_y,
    objnum=0,
):
    """
    Plots data in the new [time, channel_id] -> value format.
    It separates the data by channel_id for plotting.
    """
    pred_x, pred_y_mean, pred_y_std, target_x, target_y, context_x, context_y = map(
        ops.convert_to_numpy,
        [pred_x, pred_y_mean, pred_y_std, target_x, target_y, context_x, context_y],
    )
    # Define colors for the channels

    trueval_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#999999"]
    predval_colors = ["#56B4E9", "#E69F00", "#66c5ab", "#e0afca", "#f6ef8e", "#c2c2c2"]
    std_bgcolors = ["#9AC2E0", "#F3C4A5", "#b2e2d5", "#f0d7e5", "#faf7c6", "#e0e0e0"]

    # If 1 channel only
    if pred_x.shape[-1] == 1:
        y_channels = [0]  # Assign a default channel ID
    # else Multi-channel
    else:
        y_channels = np.unique(pred_x[:, :, 1])

    n_y_channels = len(y_channels)
    if n_y_channels > len(trueval_colors):
        warnings.warn(f"Plotting supports a maximum of {len(trueval_colors)} channels.")

    for i, y_channel in enumerate(y_channels):
        if i >= len(trueval_colors):
            break
        ych_num = i + 1

        # Single channel mask: True for all points
        if pred_x.shape[-1] == 1:
            channel_mask = np.ones(target_x.shape[1], dtype=bool)
            channel_mask_ctx = np.ones(context_x.shape[1], dtype=bool)
            channel_mask_pred = np.ones(pred_x.shape[1], dtype=bool)
        # For multi-channel mask: filter by channel ID
        else:
            channel_mask = target_x[objnum, :, 1] == y_channel
            channel_mask_ctx = context_x[objnum, :, 1] == y_channel
            channel_mask_pred = pred_x[objnum, :, 1] == y_channel

        # Plot truth first
        ax.plot(
            target_x[objnum, channel_mask, 0],
            target_y[objnum, channel_mask, 0],
            color=trueval_colors[i],
            linestyle=":",
            linewidth=2,
            label=f"truth$_{{y{ych_num}}}$",
        )

        # Plot context points second
        ax.plot(
            context_x[objnum, channel_mask_ctx, 0],
            context_y[objnum, channel_mask_ctx, 0],
            color=trueval_colors[i],
            linestyle="",
            marker="o",
            markersize=10,
            label=f"context$_{{y{ych_num}}}$",
        )

        # Plot predictions third
        ax.plot(
            pred_x[objnum, channel_mask_pred, 0],
            pred_y_mean[objnum, channel_mask_pred, 0],
            color=predval_colors[i],
            linewidth=2,
            label=f"pred$_{{y{ych_num}}}$",
        )

        ax.fill_between(
            pred_x[objnum, channel_mask_pred, 0],
            pred_y_mean[objnum, channel_mask_pred, 0]
            - pred_y_std[objnum, channel_mask_pred, 0],
            pred_y_mean[objnum, channel_mask_pred, 0]
            + pred_y_std[objnum, channel_mask_pred, 0],
            alpha=0.2,
            facecolor=std_bgcolors[i],
            interpolate=True,
        )
    # ax.set_yticks([-2, 0, 2])
    # ax.set_xticks([-2, 0, 2])
    # ax.set_ylim([-2, 2])
    ax.set_facecolor("white")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    return ax


def gplike_fixed_sets(
    X_val, y_val, pred_points, seed, fixed_val_num_context, num_context_mode
):
    # no default seed as very important for valstep choice

    context_x_val_fixed, context_y_val_fixed = get_context_set(
        X_val,
        y_val,
        num_context=fixed_val_num_context,
        seed=seed,
        num_context_mode=num_context_mode,
    )
    target_x_val_fixed, target_y_val_fixed = X_val, y_val
    # pred_x_val_fixed = np.linspace(-2, 2, pred_points).reshape(
    #     target_x_val_fixed.shape[0], pred_points, target_x_val_fixed.shape[2]
    # )
    pred_x_val_fixed = target_x_val_fixed

    return (
        context_x_val_fixed,
        context_y_val_fixed,
        target_x_val_fixed,
        target_y_val_fixed,
        pred_x_val_fixed,
    )


def _is_per_channel_range(num_context_range):
    """Checks if num_context_range is in the per-channel format."""
    return (
        isinstance(num_context_range, (list, tuple))
        and len(num_context_range) > 0
        and all(isinstance(i, (list, tuple)) for i in num_context_range)
    )


def _sample_num_context(num_context_range):
    """
    Samples the number of context points based on the format of num_context_range.

    Args:
        num_context_range (list or list of lists):
            - [min, max]: Samples a single integer from this range.
            - [[min1, max1], [min2, max2], ...]: Samples one integer from each
              sub-list's range and returns a list of integers.

    Returns:
        int or list: The number of context points to sample.
    """
    if _is_per_channel_range(num_context_range):
        # Per-channel format: sample one integer from each sub-list's range.
        return [np.random.randint(low=r[0], high=r[1] + 1) for r in num_context_range]
    else:
        # Simple format: sample a single integer.
        return np.random.randint(
            low=num_context_range[0], high=num_context_range[1] + 1
        )


def _get_fixed_num_context(num_context_range):
    """
    Gets the fixed number of context points (mean of the range)
    based on the format of num_context_range.

    Args:
        num_context_range (list or list of lists):
            - [min, max]: Returns the integer mean of the range.
            - [[min1, max1], [min2, max2], ...]: Returns a list containing the
              integer mean of each sub-list's range.

    Returns:
        int or list: The fixed number of context points for validation.
    """
    if _is_per_channel_range(num_context_range):
        # Per-channel format: return a list of the means.
        return [int(np.mean(r)) for r in num_context_range]
    else:
        # Simple format: return the single mean.
        return int(np.mean(num_context_range))


def gplike_new_sets(X_val, y_val, pred_points, num_context_range, num_context_mode):
    num_context = _sample_num_context(num_context_range)

    context_x_val, context_y_val = get_context_set(
        X_val,
        y_val,
        num_context=num_context,
        num_context_mode=num_context_mode,
    )
    target_x_val, target_y_val = X_val, y_val
    # pred_x_val = np.linspace(-2, 2, pred_points).reshape(
    #     target_x_val.shape[0], pred_points, target_x_val.shape[2]
    # )
    pred_x_val = target_x_val
    return context_x_val, context_y_val, target_x_val, target_y_val, pred_x_val


def gplike_val_step(
    model, X_val, y_val, pred_points, num_context_range, num_context_mode, epoch, seed
):
    # -------- Part 1: Fixed validation set --------

    fixed_val_num_context = _get_fixed_num_context(num_context_range)
    # Get fixed set
    (
        context_x_val_fixed,
        context_y_val_fixed,
        target_x_val_fixed,
        target_y_val_fixed,
        pred_x_val_fixed,
    ) = gplike_fixed_sets(
        X_val,
        y_val,
        pred_points,
        seed,
        fixed_val_num_context=fixed_val_num_context,
        num_context_mode=num_context_mode,
    )

    # Get model predictions and uncertainty estimates
    pred_y_mean_fixed, pred_y_std_fixed = model.test_step(
        context_x_val_fixed, context_y_val_fixed, pred_x_val_fixed
    )

    # Get model predictions and uncertainty estimates
    dist_fixed = tfp.distributions.MultivariateNormalDiag(
        pred_y_mean_fixed, pred_y_std_fixed
    )
    val_loss_fixed = -ops.mean(dist_fixed.log_prob(target_y_val_fixed))
    print(f"\nEpoch{epoch}, val_loss_fixed={val_loss_fixed} (lower is better)")

    # -------- Part 2: Calculate and print goodness
    # of fit metrics for fixed validation set  --------

    mymetrics = gplike_calculate_mymetrics(
        pred_x_val_fixed,
        pred_y_mean_fixed,
        pred_y_std_fixed,
        target_x_val_fixed,
        target_y_val_fixed,
        context_x_val_fixed,
        context_y_val_fixed,
    )

    for metric_name, value in mymetrics.items():
        print(f"{metric_name}: {value:.1%} (higher is better)")

    # -------- Part 3: New random validation set (in addition to fixed set) --------

    # Generate new set
    context_x_val, context_y_val, target_x_val, target_y_val, pred_x_val = (
        gplike_new_sets(
            X_val,
            y_val,
            pred_points,
            num_context_range,
            num_context_mode=num_context_mode,
        )
    )

    # Get model predictions and uncertainty estimates
    pred_y_mean, pred_y_std = model.test_step(context_x_val, context_y_val, pred_x_val)

    # # Calc and print val_loss for non fixed data
    # dist = tfp.distributions.MultivariateNormalDiag(pred_y_mean, pred_y_std)
    # val_loss = -ops.mean(dist.log_prob(target_y_val))
    # print(f"\nEpoch{epoch}, val_loss={val_loss} (lower is better)")

    # -------- Part 4: Plot val predictions for fixed and random set  --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    gplike_plot_functions(
        ax=ax1,
        pred_x=pred_x_val_fixed,
        pred_y_mean=pred_y_mean_fixed,
        pred_y_std=pred_y_std_fixed,
        target_x=target_x_val_fixed,
        target_y=target_y_val_fixed,
        context_x=context_x_val_fixed,
        context_y=context_y_val_fixed,
    )
    ax1.set_title("Fixed validation set")

    gplike_plot_functions(
        ax=ax2,
        pred_x=pred_x_val,
        pred_y_mean=pred_y_mean_fixed,
        pred_y_std=pred_y_std_fixed,
        target_x=target_x_val,
        target_y=target_y_val,
        context_x=context_x_val,
        context_y=context_y_val,
    )
    ax2.set_title("Random validation set")
    # ax2.legend(bbox_to_anchor=(1.27, 1), loc="upper right")

    # Create a single legend for the figure
    handles, labels = ax2.get_legend_handles_labels()
    # fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5))

    fig.legend(
        handles,
        labels,
        loc="lower center",  # Anchor point is the bottom-center of the legend box
        bbox_to_anchor=(0.5, 0.78),  # Position legend's bottom at y=0.88
        ncol=3,  # Arrange in 3 columns; you can tweak this number
    )

    fig.suptitle(f"Epoch {epoch}")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()
    plt.close()


def plot_2dimage(
    ax,
    image,
    title=None,
    cmap="gray",
    vmin=None,
    vmax=None,
    plot_context=False,
    context_x=None,
):
    # 1. Plot the image
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")

    # Add a colorbar, which is especially useful for the uncertainty plot
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. Plot context points if requested
    if plot_context:
        if context_x is None:
            raise ValueError(
                "`context_x` must be provided when `plot_context` is True."
            )

        height, width = image.shape
        # De-normalize context coordinates from [-1, 1] to pixel space [0, 27]
        context_x_pixels = (context_x[0, :, 0] + 1) * (width - 1) / 2
        context_y_pixels = (context_x[0, :, 1] + 1) * (height - 1) / 2

        ax.scatter(
            context_x_pixels,
            context_y_pixels,
            s=40,
            alpha=0.7,
            facecolors="none",
            edgecolors="red",
            linewidth=1.5,
            label="Context",
        )

    # 3. Set plot aesthetics
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def mnist_val_step(model, X_val, y_val, num_context_range, epoch, seed):
    if _is_per_channel_range(num_context_range):
        raise ValueError(
            "MNIST dataset is single-channel, but num_context_range "
            "was provided in the per-channel format. Please use [min, max]."
        )
    # -------- Part 1: Fixed validation set --------
    fixed_val_num_context = int(np.mean(num_context_range))
    context_x_val, context_y_val = get_context_set(
        X_val, y_val, num_context=fixed_val_num_context, seed=seed
    )
    target_x_val = X_val

    # Get model predictions and uncertainty estimates
    pred_y_mean, pred_y_std = map(
        ops.convert_to_numpy,
        model.test_step(context_x_val, context_y_val, target_x_val),
    )

    # -------- Part 2: Plot fixed validation set --------

    # --- Create Plots ---
    # We now create a 1x3 grid to include the standard deviation plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Epoch {epoch}", fontsize=16)

    # Reshape images from flat vectors to 2D arrays
    # Assuming batch size is 1 for validation plotting
    IMAGE_HEIGHT, IMAGE_WIDTH = 28, 28  # MNIST
    gt_img = y_val[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    pred_img = pred_y_mean[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    std_img = pred_y_std[0].reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    # Plot 1: Ground Truth with Context Points
    plot_2dimage(
        axes[0],
        gt_img,
        "Ground Truth with Context",
        vmin=-0.5,
        vmax=0.5,
        plot_context=True,
        context_x=context_x_val,
    )

    # Plot 2: Prediction (Mean)
    plot_2dimage(axes[1], pred_img, "Prediction (Mean)", vmin=-0.5, vmax=0.5)

    # Plot 3: Uncertainty (Standard Deviation)
    # Using a different colormap like 'viridis' or 'magma' is great for heatmaps
    plot_2dimage(axes[2], std_img, "Uncertainty (Std Dev)", cmap="magma")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close()
