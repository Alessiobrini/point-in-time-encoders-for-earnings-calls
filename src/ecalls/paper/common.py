"""Paths, widths and style shared by every exhibit script.

The widths were read off a compile of the IEEEtran conference class (`\\the\\columnwidth`,
`\\the\\textwidth`), so a figure built at COLUMN_PT and included at `\\columnwidth` is not rescaled.
"""
from ecalls.config import paths as P

PAPER_DIR = P.PROJECT_ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"
SEEDS = P.REPORTS_DIR / "seeds"
PORTFOLIO = P.REPORTS_DIR / "portfolio"

COLUMN_PT = 252.0      # IEEEtran conference, \columnwidth
TEXT_PT = 516.0        # IEEEtran conference, \textwidth

# IEEEtran sets body text at 10 pt and captions at 8 pt, so figure text at 8 pt reads as caption-sized.
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
      "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0,
      "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    return plt


def num(v: float, digits: int, signed: bool = False, bold: bool = False) -> str:
    """A number for a table cell, in math mode so a negative prints a minus sign, never a hyphen.
    A value that rounds to zero prints unsigned, so no cell reads -0.0000."""
    r = round(v, digits)
    if r == 0:
        r = 0.0
    s = f"{r:+.{digits}f}" if signed and r != 0 else f"{r:.{digits}f}"
    s = s.replace("-", "-")
    tex = f"${s}$"
    return r"{\boldmath" + tex + "}" if bold else tex
