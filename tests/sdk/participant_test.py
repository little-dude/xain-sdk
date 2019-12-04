"""Tests for Participant."""

from typing import Any

import pytest

from xain_sdk.sdk import participant
from xain_sdk.sdk.use_case import UseCase


@pytest.mark.xfail
def test_start() -> None:
    """Test start."""

    class MyUseCase(UseCase):
        """Custom UseCase."""

        def __init__(self, model: Any) -> None:
            super().__init__(model)
            self.model = model

        def set_weights(self, weights: Any) -> None:
            pass

        def get_weights(self) -> None:
            pass

        def train(self) -> None:
            pass

    my_use_case = MyUseCase(model={})

    participant.start(coordinator_url="http://localhost:8601", use_case=my_use_case)
