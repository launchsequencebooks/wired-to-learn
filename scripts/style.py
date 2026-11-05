"""Shared figure style for the book (print-oriented, light surface)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repository root
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

CYAN, GOLD, MAG, VIOLET = "#0B84A5", "#C98A00", "#D6339F", "#5B4BC4"
NAVY, MUTED, GRID = "#1B2A4A", "#5A6478", "#E3E8EF"

# Grayscale-safe series style (P3-BOOK-005). The interior is printed in black and
# white, where CYAN, MAG and VIOLET have nearly the same luminance. So every series
# is told apart by THREE things at once: a colour of distinct luminance (dark / mid /
# light), a line style and a marker. Use SERIES in order, and for a line plot take the
# whole style with series_style(i); for bars use HATCHES[i]; for sign maps use
# DIVERGING with an explicit vmin/vmax and a colorbar.
SERIES = [NAVY, CYAN, GOLD, "#9FB3C8"]     # luminance about 15 %, 45 %, 60 %, 70 %
LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]
HATCHES = ["", "///", "...", "xxx"]
DIVERGING = "PuOr"                              # neutral mid-point survives grayscale


def series_style(i, markevery=None):
    """Colour, line style and marker for series i: ax.plot(x, y, **series_style(0))."""
    k = i % len(SERIES)
    style = {"color": SERIES[k], "linestyle": LINESTYLES[k], "marker": MARKERS[k],
             "markersize": 4, "markevery": markevery}
    return style
FIGDIR = str(ROOT / "figs") + "/"        # the LaTeX project reads figs/
Path(FIGDIR).mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Inter",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlecolor": NAVY,
    "axes.labelcolor": NAVY,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 2,
    # default colour cycle has distinct luminances; line styles and markers are set
    # explicitly per series (series_style) so that nothing changes by accident
    "axes.prop_cycle": cycler(color=SERIES),
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Pagella",
    "mathtext.it": "TeX Gyre Pagella:italic",
    "mathtext.bf": "TeX Gyre Pagella:bold",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def save(fig, name):
    fig.savefig(FIGDIR + name + ".pdf")
    plt.close(fig)
    print("saved", name)
