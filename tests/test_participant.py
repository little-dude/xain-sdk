"""Tests for the participant API."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from tensorflow import Tensor as TFTensor
from tensorflow.keras import Input, Model  # pylint: disable=import-error
from tensorflow.keras.layers import Dense  # pylint: disable=import-error
import torch
from torch import Tensor as PTTensor
from torch.nn import Linear, Module, init

from xain_sdk.participant import Participant


@pytest.fixture
def test_participant() -> Participant:
    """Fixture to create a test participant."""

    class TestParticipant(Participant):
        """Test participant."""

        def train_round(
            self, weights: Optional[np.ndarray], epochs: int, epoch_base: int
        ) -> Tuple[np.ndarray, int, Dict[str, np.ndarray]]:
            """Dummy train round."""

            return weights, 0, {}

    return TestParticipant()


def test_get_set_tensorflow_weights(  # pylint: disable=redefined-outer-name
    test_participant: Participant,
) -> None:
    """Test the getting and setting of tensorflow weights."""

    # define model layers
    input_layer: TFTensor = Input(shape=(10,), dtype="float32")
    hidden_layer: TFTensor = Dense(
        units=6, use_bias=True, kernel_initializer="zeros", bias_initializer="zeros",
    )(inputs=input_layer)
    output_layer: TFTensor = Dense(
        units=2, use_bias=True, kernel_initializer="zeros", bias_initializer="zeros",
    )(inputs=hidden_layer)
    model: Model = Model(inputs=[input_layer], outputs=[output_layer])
    shapes: List[Tuple[int, ...]] = [weight.shape for weight in model.get_weights()]

    # get weights
    weights: np.ndarray = test_participant.get_tensorflow_weights(model=model)
    np.testing.assert_almost_equal(
        actual=weights,
        desired=np.zeros(shape=(np.sum([np.prod(shape) for shape in shapes]),)),
    )

    # set weights
    test_participant.set_tensorflow_weights(weights=weights, shapes=shapes, model=model)
    for weight, shape in zip(model.get_weights(), shapes):
        np.testing.assert_almost_equal(actual=weight, desired=np.zeros(shape=shape))


def test_get_set_pytorch_weights(  # pylint: disable=redefined-outer-name
    test_participant: Participant,
) -> None:
    """Test the getting and setting of pytorch weights."""

    # define model layers
    class TestModel(Module):
        """Test neural network."""

        def __init__(self) -> None:
            super(TestModel, self).__init__()
            self.linear1: Linear = Linear(in_features=10, out_features=6)
            self.linear2: Linear = Linear(in_features=6, out_features=2)
            init.constant_(self.linear1.weight, 0)
            init.constant_(self.linear1.bias, 0)
            init.constant_(self.linear2.weight, 0)
            init.constant_(self.linear2.bias, 0)

        def forward(self, input_layer: PTTensor) -> PTTensor:  # type: ignore  # pylint: disable=arguments-differ
            hidden_layer: PTTensor = self.linear1(input_layer)
            output_layer: PTTensor = self.linear2(hidden_layer)
            return output_layer

    model: TestModel = TestModel()
    model.forward(torch.zeros((10,)))  # pylint: disable=no-member
    shapes: List[Tuple[int, ...]] = [
        tuple(weight.shape) for weight in model.state_dict().values()
    ]

    # get weights
    weights: np.ndarray = test_participant.get_pytorch_weights(model=model)
    np.testing.assert_almost_equal(
        actual=weights,
        desired=np.zeros(shape=(np.sum([np.prod(shape) for shape in shapes]),)),
    )

    # set weights
    test_participant.set_pytorch_weights(weights=weights, shapes=shapes, model=model)
    for weight, shape in zip(model.state_dict().values(), shapes):
        np.testing.assert_almost_equal(actual=weight, desired=np.zeros(shape=shape))
