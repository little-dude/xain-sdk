"""Tests for Coordinator."""

import pytest

from xain_sdk.sdk import coordinator


@pytest.mark.xfail
def test_start() -> None:
    """Test start."""

    coordinator.start()
