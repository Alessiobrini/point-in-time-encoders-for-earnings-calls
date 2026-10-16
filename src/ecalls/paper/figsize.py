"""Figure dimensions that match the paper's column, so LaTeX includes every figure without rescaling.

Copied from the FINTECH 540 course utilities (utils/figsize.py), the helper every paper figure uses.
"""


def set_size(width: float, fraction: float = 1, subplots: tuple[int, int] = (1, 1)) -> tuple[float, float]:
    """Width and height in inches for a figure `fraction` of `width` points wide, at the golden ratio
    per subplot row."""
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5 ** 0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return fig_width_in, fig_height_in
