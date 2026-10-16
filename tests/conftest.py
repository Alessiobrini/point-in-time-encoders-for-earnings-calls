"""Shared fixtures. Machine-dependent artifacts skip rather than fail."""
import random

import pytest

from ecalls.config import paths as P


@pytest.fixture(autouse=True)
def deterministic():
    """Autouse, because a test that passes only on a lucky draw is worse than no test."""
    random.seed(0)
    try:
        import numpy as np
        np.random.seed(0)
    except ImportError:
        pass


@pytest.fixture
def components_path():
    if not P.COMPONENTS.exists():
        pytest.skip(f"no consolidated corpus on this machine ({P.COMPONENTS})")
    return P.COMPONENTS


@pytest.fixture
def membership_path():
    if not P.SP500_MEMBERSHIP.exists():
        pytest.skip(f"no membership artifact on this machine ({P.SP500_MEMBERSHIP})")
    return P.SP500_MEMBERSHIP
