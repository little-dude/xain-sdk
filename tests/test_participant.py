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
def participant() -> Participant:
    """Fixture to create a test participant."""

    class TestParticipant(Participant):
        """Test participant."""

        def train_round(
            self, weights: Optional[np.ndarray], epochs: int, epoch_base: int
        ) -> Tuple[np.ndarray, int, Dict[str, np.ndarray]]:
            """Dummy train round."""

            return weights, 0, {}

    return TestParticipant()


@pytest.fixture
def tensorflow_model() -> Model:
    """Fixture to create a test tensorflow model."""

    # define model layers
    input_layer: TFTensor = Input(shape=(10,), dtype="float32")
    hidden_layer: TFTensor = Dense(
        units=6, use_bias=True, kernel_initializer="zeros", bias_initializer="zeros"
    )(inputs=input_layer)
    output_layer: TFTensor = Dense(
        units=2, use_bias=True, kernel_initializer="zeros", bias_initializer="zeros"
    )(inputs=hidden_layer)
    model: Model = Model(inputs=[input_layer], outputs=[output_layer])

    return model


@pytest.fixture
def pytorch_model() -> Module:
    """Fixture to create a test pytorch model."""

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

        def forward(  # type: ignore  # pylint: disable=arguments-differ
            self, input_layer: PTTensor
        ) -> PTTensor:
            hidden_layer: PTTensor = self.linear1(input_layer)
            output_layer: PTTensor = self.linear2(hidden_layer)
            return output_layer

    model: TestModel = TestModel()
    model.forward(torch.zeros((10,)))  # pylint: disable=no-member

    return model


def test_get_tensorflow_shapes(  # pylint: disable=redefined-outer-name
    participant: Participant, tensorflow_model: Model
) -> None:
    """Test the getting of tensorflow weight shapes."""

    shapes: List[Tuple[int, ...]] = participant.get_tensorflow_shapes(
        model=tensorflow_model
    )
    assert shapes == [(10, 6), (6,), (6, 2), (2,)]


def test_get_set_tensorflow_weights(  # pylint: disable=redefined-outer-name
    participant: Participant, tensorflow_model: Model
) -> None:
    """Test the getting and setting of tensorflow weights."""

    model: Model = tensorflow_model
    shapes: List[Tuple[int, ...]] = participant.get_tensorflow_shapes(model=model)

    # get weights
    weights: np.ndarray = participant.get_tensorflow_weights(model=model)
    np.testing.assert_almost_equal(
        actual=weights,
        desired=np.zeros(shape=(np.sum([np.prod(shape) for shape in shapes]),)),
    )

    # set weights
    participant.set_tensorflow_weights(weights=weights, shapes=shapes, model=model)
    for weight, shape in zip(model.get_weights(), shapes):
        np.testing.assert_almost_equal(actual=weight, desired=np.zeros(shape=shape))


def test_get_pytorch_shapes(  # pylint: disable=redefined-outer-name
    participant: Participant, pytorch_model: Module
) -> None:
    """Test the getting of pytorch weight shapes."""

    shapes: List[Tuple[int, ...]] = participant.get_pytorch_shapes(model=pytorch_model)
    assert shapes == [(6, 10), (6,), (2, 6), (2,)]


def test_get_set_pytorch_weights(  # pylint: disable=redefined-outer-name
    participant: Participant, pytorch_model: Module
) -> None:
    """Test the getting and setting of pytorch weights."""

    model: Module = pytorch_model
    shapes: List[Tuple[int, ...]] = participant.get_pytorch_shapes(model=model)

    # get weights
    weights: np.ndarray = participant.get_pytorch_weights(model=model)
    np.testing.assert_almost_equal(
        actual=weights,
        desired=np.zeros(shape=(np.sum([np.prod(shape) for shape in shapes]),)),
    )

    # set weights
    participant.set_pytorch_weights(weights=weights, shapes=shapes, model=model)
    for weight, shape in zip(model.state_dict().values(), shapes):
        np.testing.assert_almost_equal(actual=weight, desired=np.zeros(shape=shape))
