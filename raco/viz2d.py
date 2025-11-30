"""
2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.

From glue-factory https://github.com/cvg/glue-factory/blob/main/gluefactory/visualization/viz2d.py
"""

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np
import torch


def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def cm_GnRd(x):
    """Custom colormap: green (0) -> yellow (0.5) -> red (1)."""
    x = np.clip(x, 0, 1)
    return cm_RdGn(1 - x)


def cm_BlRdGn(x_):
    """Custom colormap: blue (-1) -> red (0.0) -> green (1)."""
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0, 1.0]])

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 0.1, 1, 1.0]]) + (2 - xn) * np.array([[1.0, 0, 0, 1.0]])
    out = np.clip(np.where(x_[..., None] < 0, cn, c), 0, 1)
    return out


def cm_prune(x_):
    """Custom colormap to visualize pruning"""
    if isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
    max_i = max(x_)
    norm_x = np.where(x_ == max_i, -1, (x_ - 1) / 9)
    return cm_BlRdGn(norm_x)


def cm_grad2d(xy):
    """2D grad. colormap: yellow (0, 0) -> green (1, 0) -> red (0, 1) -> blue (1, 1)."""
    tl = np.array([1.0, 0, 0])  # red
    tr = np.array([0, 0.0, 1])  # blue
    ll = np.array([1.0, 1.0, 0])  # yellow
    lr = np.array([0, 1.0, 0])  # green

    xy = np.clip(xy, 0, 1)
    x = xy[..., :1]
    y = xy[..., -1:]
    rgb = (1 - x) * (1 - y) * ll + x * (1 - y) * lr + x * y * tr + (1 - x) * y * tl
    return rgb.clip(0, 1)


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: list of NumPy RGB (H, W, 3) or PyTorch RGB (3, H, W) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    # conversion to (H, W, 3) for torch.Tensor
    imgs = [
        (
            img.permute(1, 2, 0).cpu().numpy()
            if (isinstance(img, torch.Tensor) and img.dim() == 3)
            else img
        )
        for img in imgs
    ]

    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)
    return ax


def plot_keypoints(kpts, colors="lime", ps=4, axes=None, a=1.0):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        if isinstance(k, torch.Tensor):
            k = k.cpu().numpy()
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_covariance_ellipses(
    kpts, covariances, colors=None, sigma=3, lw=2, alpha=0.6, axes=None
):
    """Plot covariance ellipses for keypoints (super fast using EllipseCollection).
    Args:
        kpts: list of ndarrays of size (N, 2) or single ndarray of keypoint coordinates.
        covariances: list of ndarrays of size (N, 2, 2) or single ndarray of covariance matrices.
        colors: string, list of colors, or colormap. If None, uses tab10 colormap.
        sigma: number of standard deviations for the ellipse size (default: 3).
        lw: line width of the ellipse edges.
        alpha: transparency of the ellipses.
        axes: matplotlib axes to plot on. If None, uses current figure axes.
    """
    if axes is None:
        axes = plt.gcf().axes

    # Handle single axis case
    if not isinstance(axes, list):
        axes = [axes]
        kpts = [kpts]
        covariances = [covariances]

    # Ensure inputs are lists
    if not isinstance(kpts, list):
        kpts = [kpts]
    if not isinstance(covariances, list):
        covariances = [covariances]

    for ax, keypoints, covs in zip(axes, kpts, covariances):
        # Convert to numpy if needed
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(covs, torch.Tensor):
            covs = covs.cpu().numpy()

        if len(keypoints) == 0:
            continue

        # Prepare arrays for batch processing
        n_points = len(keypoints)
        widths = np.zeros(n_points)
        heights = np.zeros(n_points)
        angles = np.zeros(n_points)

        # Vectorized eigenvalue decomposition and ellipse parameter calculation
        for i, (mean, cov) in enumerate(zip(keypoints, covs)):
            # Eigenvalue decomposition
            vals, vecs = np.linalg.eigh(cov)

            # Sort eigenvalues and eigenvectors in descending order
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            # Calculate ellipse parameters
            # Ensure positive eigenvalues for numerical stability
            vals = np.maximum(vals, 1e-8)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width = sigma * 2 * np.sqrt(vals[0])
            height = sigma * 2 * np.sqrt(vals[1])

            widths[i] = width
            heights[i] = height
            angles[i] = angle

        # Handle colors
        if colors is None:
            # Use tab10 colormap cycling through colors
            cmap = plt.cm.tab10
            color_list = [cmap(i % 10) for i in range(n_points)]
        elif isinstance(colors, str):
            color_list = [colors] * n_points
        elif callable(colors):  # colormap
            color_list = [colors(i % 10) for i in range(n_points)]
        else:
            color_list = colors

        # Create ellipse collection for fast rendering
        ellipses = EllipseCollection(
            widths=widths,
            heights=heights,
            angles=angles,
            offsets=keypoints,
            units="xy",
            edgecolors=color_list,
            facecolors=color_list,
            linewidths=lw,
            alpha=alpha,
            transOffset=ax.transData,
        )

        ax.add_collection(ellipses)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    assert len(kpts0) == len(kpts1)
    if color is None:
        kpts_norm = (kpts0 - kpts0.min(axis=0, keepdims=True)) / np.ptp(
            kpts0, axis=0, keepdims=True
        )
        color = cm_grad2d(kpts_norm)  # gradient color
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
):
    ax = plt.gcf().axes[idx]
    t = ax.text(
        *pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
