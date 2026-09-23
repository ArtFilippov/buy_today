"""Shared rendering defaults for local, reproducible report figures."""

from matplotlib.typing import RcKeyType

REPORT_STYLE: dict[RcKeyType, object] = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 120,
}
