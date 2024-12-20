import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_functions1d(
    target_x, target_y, context_x, context_y, pred_y, std, plot_batch=0
):
    """Deprecated: Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape batchsize x number_targets x 1 that contains the
          x values of the target points.
      target_y: An array of shape batchsize x number_targets x 1 that contains the
          y values of the target points.
      context_x: An array of shape batchsize x number_context x 1 that contains
          the x values of the context points.
      context_y: An array of shape batchsize x number_context x 1 that contains
          the y values of the context points.
      pred_y: An array of shape batchsize x number_targets x 1  that contains the
          predicted means of the y values at the target points in target_x.
      pred_y: An array of shape batchsize x number_targets x 1  that contains the
          predicted variance of the y values at the target points in target_x.
      plot_batch: An integer denoting which batch to plot in the given dataset.
    """
    # Plot everything
    plt.plot(target_x[plot_batch], pred_y[plot_batch], "b", linewidth=2, label="pred")
    plt.plot(
        target_x[plot_batch], target_y[plot_batch], "k:", linewidth=2, label="truth"
    )
    plt.plot(
        context_x[plot_batch],
        context_y[plot_batch],
        "ko",
        markersize=10,
        label="context",
    )
    plt.fill_between(
        target_x[plot_batch, :, 0],
        pred_y[plot_batch, :, 0] - std[plot_batch, :, 0],
        pred_y[plot_batch, :, 0] + std[plot_batch, :, 0],
        alpha=0.2,
        facecolor="#65c9f7",
        interpolate=True,
    )

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    ax = plt.gca()
    ax.set_facecolor("white")
    return ax


def plot_functions(
    target_x,
    target_y,
    context_x,
    context_y,
    pred_y,
    std,
    plot_batch=0,
):
    """Plots the predicted mean and variance and the context points for multi-dimensional data.

    Args:
      target_x: An array of shape batchsize x number_targets x xdims that contains the
          x values of the target points.
      target_y: An array of shape batchsize x number_targets x ydims that contains the
          y values of the target points.
      context_x: An array of shape batchsize x number_context x xdims that contains
          the x values of the context points.
      context_y: An array of shape batchsize x number_context x ydims that contains
          the y values of the context points.
      pred_y: An array of shape batchsize x number_targets x ydims that contains the
          predicted means of the y values at the target points in target_x.
      std: An array of shape batchsize x number_targets x ydims that contains the
          predicted standard deviation of the y values at the target points in target_x.
      plot_batch: Specific batch indices to plot. If None, all batches are plotted.
    """
    # Get dimensions
    xdims = context_x.shape[-1]
    ydims = context_y.shape[-1]
    assert (
        (target_x.shape[-1] == xdims)
        and (target_y.shape[-1] == ydims)
        and (pred_y.shape[-1] == ydims)
        and (std.shape[-1] == ydims)
    )

    assert xdims == ydims  # FOR NOW - later will add another loop

    # Set up colors for different dimensions using seaborn color palette
    colors = sns.color_palette("bright")
    if xdims > len(colors):
        colors = plt.cm.viridis(np.linspace(0, 1, xdims))

    if xdims == 1 and ydims == 1:
        # Single dimension case, mimic plot_functions
        plt.plot(
            target_x[plot_batch], pred_y[plot_batch], "b", linewidth=2, label="pred"
        )
        plt.plot(
            target_x[plot_batch], target_y[plot_batch], "k:", linewidth=2, label="truth"
        )
        plt.plot(
            context_x[plot_batch],
            context_y[plot_batch],
            "ko",
            markersize=10,
            label="context",
        )
        plt.fill_between(
            target_x[plot_batch, :, 0],
            pred_y[plot_batch, :, 0] - std[plot_batch, :, 0],
            pred_y[plot_batch, :, 0] + std[plot_batch, :, 0],
            alpha=0.2,
            facecolor="#65c9f7",
            interpolate=True,
        )
        plt.yticks([-2, 0, 2], fontsize=16)
        plt.xticks([-2, 0, 2], fontsize=16)
        plt.ylim([-2, 2])
        ax = plt.gca()
        ax.set_facecolor("white")
        return ax
    else:
        for xdim in range(xdims):
            ydim = xdim
            # Target points
            plt.scatter(
                target_x[plot_batch, :, xdim],
                target_y[plot_batch, :, ydim],
                label=f"$t_{xdim}$",
                alpha=0.5,
                color=colors[xdim],
                marker="x",
                s=20,
            )
            # Context points
            plt.scatter(
                context_x[plot_batch, :, xdim],
                context_y[plot_batch, :, ydim],
                label=f"$c_{xdim}$",
                alpha=0.5,
                color=colors[xdim],
                marker="o",
                s=20,
            )
            # Mean pred
            plt.scatter(
                target_x[plot_batch, :, xdim],
                pred_y[plot_batch, :, ydim],
                label=rf"$\mu_{xdim}$",
                marker=".",
                s=20,
                alpha=0.5,
                lw=0,
                color=colors[xdim],
            )
            # Errorbar
            plt.fill_between(
                target_x[plot_batch, :, xdim],
                pred_y[plot_batch, :, ydim] - std[plot_batch, :, ydim],
                pred_y[plot_batch, :, ydim] + std[plot_batch, :, ydim],
                alpha=0.1,
                color=colors[xdim],
            )
        # Make the plot pretty
        plt.yticks([-2, 0, 2], fontsize=16)
        plt.xticks([-2, 0, 2], fontsize=16)
        plt.ylim([-2, 2])
        plt.legend(
            # bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        ax = plt.gca()
        ax.set_facecolor("white")
        return ax
