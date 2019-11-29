import pytest

from xain_sdk.sdk import coordinator


@pytest.mark.xfail
def test_start():
    coordinator.start()
