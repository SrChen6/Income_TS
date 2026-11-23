"""Shared matplotlib style: one validated categorical palette, recessive axes."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# Categorical slots, assigned in fixed order (never cycled).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
MUTED = "#a9a8a3"  # context / raw data drawn behind the main series

# Fixed colour per model so a model keeps its colour in every figure.
MODEL_COLORS = {
    "Naive": MUTED,
    "Seasonal naive": "#7d7c77",
    "Seasonal mean": "#c9c8c2",
    "ETS": AQUA,
    "SARIMA": BLUE,
    "DHR (SARIMA + Fourier)": VIOLET,
    "Gradient boosting": ORANGE,
    "Ensemble": GREEN,
}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": TEXT_2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "text.color": TEXT,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
