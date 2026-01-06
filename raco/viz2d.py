"""
2D visualization primitives based on Matplotlib.
1) Plot images with `plot_images`.
2) Call `plot_keypoints`
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.

From glue-factory https://github.com/cvg/glue-factory/blob/main/gluefactory/visualization/viz2d.py
"""

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import EllipseCollection


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
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array(
        [[1.0, 0, 0, 1.0]]
    )

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 0.1, 1, 1.0]]) + (2 - xn) * np.array(
        [[1.0, 0, 0, 1.0]]
    )
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
    rgb = (
        (1 - x) * (1 - y) * ll
        + x * (1 - y) * lr
        + x * y * tr
        + (1 - x) * y * tl
    )
    return rgb.clip(0, 1)


def plot_images(
    imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True
):
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

    ratios = [i.shape[1] / i.shape[0] for i in imgs] if adaptive else [4 / 3] * n 
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
    """Plot covariance ellipses for keypoints.
    Args:
        kpts: list of ndarrays of size (N, 2) or single ndarray of keypoint coordinates.
        covariances: list of ndarrays of size (N, 2, 2) or 
                     single ndarray of covariance matrices.
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
        for i, cov in enumerate(covs):
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
        *pos,
        text,
        fontsize=fs,
        ha=ha,
        va=va,
        color=color,
        transform=ax.transAxes,
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
