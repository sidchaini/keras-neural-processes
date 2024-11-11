import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# IMPROVE BELOW
# - move model predict outside loop
# - check for other consistencies with the 1d method
def plot_predictionsnd(
    model,
    context_x,
    context_y,
    target_x,
    target_y,
    epoch_title,
    plot_batch=None,
    n_points=100,
):
    # Get dimensions
    xdim = context_x.shape[-1]
    ydim = context_y.shape[-1]

    # Validate input dimensions
    if xdim != ydim:
        raise ValueError(
            f"X and Y dimensions must be equal. Got xdim={xdim}, ydim={ydim}"
        )

    # Set up colors for different dimensions using seaborn color palette
    colors = sns.color_palette("bright")
    if xdim > len(colors):
        colors = plt.cm.rainbow(np.linspace(0, 1, xdim))

    min_vals = target_x.min(axis=1)  # shape: (batches, xdims)
    max_vals = target_x.max(axis=1)  # shape: (batches, xdims)

    x_test = np.linspace(min_vals, max_vals, n_points).T
    x_test = x_test.reshape(-1, n_points, xdim)

    y_pred_mean, y_pred_std = model.predict([context_x, context_y, x_test], verbose=0)

    if plot_batch is None:
        plot_batch = np.arange(len(target_x))

    for batch_num in plot_batch:
        for dim in range(xdim):
            # Target points
            plt.scatter(
                target_x[batch_num, :, dim],
                target_y[batch_num, :, dim],
                label=f"$t_{dim}$",
                alpha=0.5,
                color=colors[dim],
                marker="x",
                s=20,
            )
            # Context points
            plt.scatter(
                context_x[batch_num, :, dim],
                context_y[batch_num, :, dim],
                label=f"$c_{dim}$",
                alpha=0.5,
                color=colors[dim],
                marker="o",
                s=20,
            )
            # Mean pred
            plt.scatter(
                x_test[batch_num, :, dim],
                y_pred_mean[batch_num, :, dim],
                label=f"pred$_{dim}$",
                marker=".",
                s=20,
                alpha=0.5,
                lw=0,
                color=colors[dim],
            )
            # Errorbar
            plt.fill_between(
                x_test[batch_num, :, dim],
                y_pred_mean[batch_num, :, dim] - y_pred_std[batch_num, :, dim],
                y_pred_mean[batch_num, :, dim] + y_pred_std[batch_num, :, dim],
                alpha=0.1,
                color=colors[dim],
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(f"Epoch {epoch_title}, Batch {batch_num}")
        plt.show()
        plt.close()


def plot_predictions(
    model,
    context_x,
    context_y,
    target_x,
    target_y,
    epoch_title,
    plot_batch=None,
    n_points=100,
):

    # Validate input dimensions
    if not all(x.shape[-1] == 1 for x in [context_x, context_y, target_x, target_y]):
        invalid_shapes = {
            "context_x": context_x.shape,
            "context_y": context_y.shape,
            "target_x": target_x.shape,
            "target_y": target_y.shape,
        }
        raise ValueError(
            f"All inputs must have 1 dimension for their values. Got shapes: {invalid_shapes}. "
            "Expected shape for input arrays is (batches, points, 1)."
        )

    # Will have to change ravel below when not doing 1D
    min_vals = target_x.min(axis=1).ravel()
    max_vals = target_x.max(axis=1).ravel()

    x_test = np.linspace(min_vals, max_vals, n_points).T
    x_test = x_test.reshape(-1, n_points, 1)

    y_pred_mean, y_pred_std = model.predict([context_x, context_y, x_test], verbose=0)

    if plot_batch is None:
        plot_batch = np.arange(len(target_x))

    for batch_num in plot_batch:
        plt.scatter(
            target_x[batch_num, :, 0],
            target_y[batch_num, :, 0],
            label="target",
            alpha=0.5,
            color="red",
        )
        plt.scatter(
            context_x[batch_num, :, 0],
            context_y[batch_num, :, 0],
            label="context",
            alpha=0.5,
            color="blue",
        )
        plt.scatter(
            x_test[batch_num, :, 0],
            y_pred_mean[batch_num, :, 0],
            label="pred",
            alpha=0.5,
            color="black",
        )
        plt.fill_between(
            x_test[batch_num, :, 0],
            y_pred_mean[batch_num, :, 0] - y_pred_std[batch_num, :, 0],
            y_pred_mean[batch_num, :, 0] + y_pred_std[batch_num, :, 0],
            alpha=0.1,
            color="black",
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(f"Epoch {epoch_title}, Batch {batch_num}")
        plt.show()
        plt.close()
