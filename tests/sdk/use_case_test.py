"""Tests for UseCase."""

from typing import Any

from xain_sdk.sdk.use_case import UseCase


def test_usecase() -> None:
    """Test UseCase."""

    class MyUseCase(UseCase):
        """Custom UseCase."""

        def __init__(self) -> None:
            super().__init__(self)

        def set_weights(self, weights: Any) -> None:
            pass

        def get_weights(self) -> None:
            pass

        def train(self) -> None:
            pass

    use_case = MyUseCase()

    print(use_case)
