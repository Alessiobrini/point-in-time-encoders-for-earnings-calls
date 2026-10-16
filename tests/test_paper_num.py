"""Table cells print negatives with a minus sign and never as -0."""
from ecalls.paper import common as C


def test_negative_numbers_are_in_math_mode():
    assert C.num(-0.0023, 4, signed=True) == "$-0.0023$"
    assert C.num(0.0044, 4, signed=True) == "$+0.0044$"


def test_a_value_that_rounds_to_zero_prints_unsigned():
    assert C.num(-0.00001, 4, signed=True) == "$0.0000$"
    assert C.num(-0.00001, 4) == "$0.0000$"


def test_bold_wraps_math_in_boldmath():
    assert C.num(0.009, 4, signed=True, bold=True) == r"{\boldmath$+0.0090$}"
