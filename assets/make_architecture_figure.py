"""Draws assets/slc_helm_architecture.{svg,png}: the SLC-HELM architecture of the paper (Fig. 1).

    python assets/make_architecture_figure.py      # needs only matplotlib
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))


def mix(rgb, p):
    """xcolor's rgb!p: p% of the color, the rest white."""
    return tuple(p / 100 * c + (1 - p / 100) for c in rgb)


TEAL, RED, GOLD, BLUE = (0, 0.5, 0.5), (0.85, 0.1, 0.1), (0.8, 0.65, 0), (0.15, 0.25, 0.85)
GREEN, MAGENTA, ORANGE = (0.1, 0.6, 0.1), (0.75, 0, 0.75), (0.95, 0.5, 0)
VIOLET, GRAY, ARROW = (0.5, 0, 0.5), (0.45, 0.45, 0.45), (0.45, 0.45, 0.45)
STYLE = {  # fill, border (colors of the TikZ styles of the paper)
    "netvlad": (mix((0, 0.5, 0.5), 20), TEAL), "gp": (mix((1, 0, 0), 20), RED), "slc": (mix((1, 1, 0), 22), GOLD),
    "mgp": (mix((0, 0, 1), 10), BLUE), "rgr": (mix((0, 1, 0), 16), GREEN), "cl": (mix((1, 0, 1), 30), MAGENTA),
    "linear": (mix((1, 0.5, 0), 23), ORANGE), "norm": (mix((1, 0, 0), 20), RED), "conv": (mix((0.5, 0, 0.5), 30), VIOLET),
    "mlp": (mix((0.5, 0.5, 0.5), 20), GRAY),
}


def box(ax, x, y, w, h, text, style, size=11, weight="normal"):
    fill, edge = STYLE[style]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, zorder=4, linespacing=1.0)
    return x - w / 2, x + w / 2


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.5, mutation_scale=12, shrinkA=0, shrinkB=0))


def line(ax, xs, ys, **kw):
    ax.plot(xs, ys, color=kw.pop("color", ARROW), lw=kw.pop("lw", 1.5), zorder=2, solid_capstyle="butt", **kw)


def main():
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=(14, 6.3))
    ax.set_xlim(-0.2, 14.9)
    ax.set_ylim(0.0, 6.05)
    ax.set_aspect("equal")
    ax.axis("off")

    # Top row: the network
    y, w, h = 5.2, 0.95, 0.62
    ax.text(0.55, y, r"$\{x_p\}_{p \in \Omega_i}$", ha="center", va="center", fontsize=15)
    nv = box(ax, 2.55, y, 1.6, h, "NetVLAD", "netvlad", 12, "bold")
    arrow(ax, 1.2, y, nv[0], y)
    blocks = [("GP", "gp"), ("SLC", "slc"), ("MGP", "mgp"), ("SLC", "slc"), ("MGP", "mgp"), ("SLC", "slc"),
              ("RGR", "rgr"), ("CL", "cl")]
    xs = [4.55 + 1.25 * i for i in range(len(blocks))]
    edges = [box(ax, x, y, w, h, name, style, 13, "bold") for x, (name, style) in zip(xs, blocks)]
    arrow(ax, nv[1], y, edges[0][0], y)
    ax.text((nv[1] + edges[0][0]) / 2, y + 0.3, r"$h_i^0$", ha="center", va="bottom", fontsize=15)
    for (_, right), (left, _) in zip(edges, edges[1:]):
        arrow(ax, right, y, left, y)
    arrow(ax, edges[-1][1], y, edges[-1][1] + 0.55, y)
    ax.text(edges[-1][1] + 0.62, y, r"$\hat{y}$", ha="left", va="center", fontsize=15)

    # Bottom: the multigraph processing block, zoomed from the first MGP
    bx0, bx1, by0, by1 = 0.1, 13.75, 0.45, 3.95
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor=STYLE["mgp"][0], edgecolor=BLUE, lw=1.4,
                           zorder=1))
    mgp_left, mgp_right = edges[2]
    line(ax, [mgp_left, bx0], [y - h / 2, by1], color=BLUE, lw=0.9, ls=(0, (4, 3)))
    line(ax, [mgp_right, bx1], [y - h / 2, by1], color=BLUE, lw=0.9, ls=(0, (4, 3)))
    ax.text((bx0 + bx1) / 2, by0 - 0.12, "Multigraph Processing (MGP) block", ha="center", va="top", fontsize=13,
            weight="bold")

    yc, ys, yh = 2.2, 3.0, 1.4
    ax.text(0.65, yc, r"$h_i^{\ell}$", ha="center", va="center", fontsize=17)
    line(ax, [0.95, 1.3], [yc, yc])
    line(ax, [1.3, 1.3], [yh, ys])
    names = [("1-layer\nLinear", "linear"), ("Batch\nNorm.", "norm"), ("GATv2\nConv.", "conv"),
             ("Batch\nNorm.", "norm"), ("1-layer\nLinear", "linear"), ("Batch\nNorm.", "norm")]
    bxs = [2.15 + 1.25 * i for i in range(len(names))]
    for yb, relation, label_y, va in ((ys, r"spatial edges $E^{\mathbf{S}}$", ys + 0.5, "bottom"),
                                      (yh, r"hierarchical edges $E^{\mathbf{H}}$", yh - 0.5, "top")):
        sides = [box(ax, x, yb, 0.98, 0.66, name, style, 9.5) for x, (name, style) in zip(bxs, names)]
        arrow(ax, 1.3, yb, sides[0][0], yb)
        for (_, right), (left, _) in zip(sides, sides[1:]):
            arrow(ax, right, yb, left, yb)
        arrow(ax, sides[-1][1], yb, sides[-1][1] + 0.3, yb)
        ax.text(bxs[2], label_y, relation, ha="center", va=va, fontsize=11, color=VIOLET)
    ax.text(9.24, ys, r"$h_i^{\mathbf{S},\,\ell+1}$", ha="left", va="center", fontsize=15)
    ax.text(9.24, yh, r"$h_i^{\mathbf{H},\,\ell+1}$", ha="left", va="center", fontsize=15)
    cx = 11.1
    line(ax, [10.4, 10.65, 10.65], [ys, ys, yc])
    line(ax, [10.4, 10.65, 10.65], [yh, yh, yc])
    arrow(ax, 10.65, yc, cx - 0.22, yc)
    ax.add_patch(Circle((cx, yc), 0.22, facecolor="white", edgecolor="black", lw=1.2, zorder=3))
    ax.text(cx, yc, r"$\|$", ha="center", va="center", fontsize=13, zorder=4)
    mlp = box(ax, 12.1, yc, 0.85, 0.62, "MLP", "mlp", 11)
    arrow(ax, cx + 0.22, yc, mlp[0], yc)
    arrow(ax, mlp[1], yc, mlp[1] + 0.3, yc)
    ax.text(mlp[1] + 0.33, yc, r"$h_i^{\ell+1}$", ha="left", va="center", fontsize=15)

    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"slc_helm_architecture.{ext}"), dpi=200, bbox_inches="tight", pad_inches=0.12,
                    facecolor="white")


if __name__ == "__main__":
    main()
