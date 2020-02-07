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
    """Fixture to create a test participant.

    Returns:
        ~xain_sdk.participant.Participant: A test participant.
    """

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
    """Fixture to create a test tensorflow model.

    Returns:
        ~tensorflow.keras.Model: A test tensorflow model.
    """

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
    """Fixture to create a test pytorch model.

    Returns:
        ~torch.nn.Module: A test pytorch model.
    """

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
    """Test the getting of tensorflow weight shapes.

    Args:
        participant (nxain_sdk.participant.Participant): A test participant.
        tensorflow_model (~tensorflow.keras.Model): A test tensorflow model.
    """

    shapes: List[Tuple[int, ...]] = participant.get_tensorflow_shapes(
        model=tensorflow_model
    )
    assert shapes == [(10, 6), (6,), (6, 2), (2,)]


def test_get_set_tensorflow_weights(  # pylint: disable=redefined-outer-name
    participant: Participant, tensorflow_model: Model
) -> None:
    """Test the getting and setting of tensorflow weights.

    Args:
        participant (nxain_sdk.participant.Participant): A test participant.
        tensorflow_model (~tensorflow.keras.Model): A test tensorflow model.
    """

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
    """Test the getting of pytorch weight shapes.

    Args:
        participant (nxain_sdk.participant.Participant): A test participant.
        pytorch_model (~torch.nn.Module): A test pytorch model.
    """

    shapes: List[Tuple[int, ...]] = participant.get_pytorch_shapes(model=pytorch_model)
    assert shapes == [(6, 10), (6,), (2, 6), (2,)]


def test_get_set_pytorch_weights(  # pylint: disable=redefined-outer-name
    participant: Participant, pytorch_model: Module
) -> None:
    """Test the getting and setting of pytorch weights.

    Args:
        participant (nxain_sdk.participant.Participant): A test participant.
        pytorch_model (~torch.nn.Module): A test pytorch model.
    """

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


def test_update_metrics(
    participant: Participant,  # pylint: disable=redefined-outer-name
) -> None:
    """Test the metrics updating.

    Args:
        participant (nxain_sdk.participant.Participant): A test participant.
    """

    metrics: Dict = {
        "metricS1": 0.5,
        "metricS2": 42,
        "metricL1": [0.5, -0.7],
        "metricL2": [[0.1, 0.8, 0.3], [0.4, 0.9], 0.0],
        "metricA1": np.array(0.5),
        "metricA2": np.array([0.5, -0.7]),
        "metricA3": np.array([[0.1, 0.8, 0.3], [0.4, 0.9, 0.0]]),
    }
    assert participant.metrics == {}
    participant.update_metrics(epoch=0, epoch_base=0, **metrics)
    current_time: int = participant.metrics[("metricS1", 0)]["time"]
    assert participant.metrics == {
        ("metricS1", 0): {
            "measurement": "metricS1",
            "time": current_time,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metricS1": 0.5},
        },
        ("metricS2", 0): {
            "measurement": "metricS2",
            "time": current_time,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metricS2": 42.0},
        },
        ("metricL1", 0): {
            "measurement": "metricL1",
            "time": current_time,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metricL1_0": 0.5, "metricL1_1": -0.7},
        },
        ("metricL2", 0): {
            "measurement": "metricL2",
            "time": current_time,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {
                "metricL2_0_0": 0.1,
                "metricL2_0_1": 0.8,
                "metricL2_0_2": 0.3,
                "metricL2_1_0": 0.4,
                "metricL2_1_1": 0.9,
                "metricL2_2": 0.0,
            },
        },
        ("metricA1", 0): {
            "measurement": "metricA1",
            "time": current_time,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metricA1": 0.5},
        },
        ("metricA2", 0): {
            "measurement": "metricA2",
            "time": current_time,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metricA2_0": 0.5, "metricA2_1": -0.7},
        },
        ("metricA3", 0): {
            "measurement": "metricA3",
            "time": current_time,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {
                "metricA3_0_0": 0.1,
                "metricA3_0_1": 0.8,
                "metricA3_0_2": 0.3,
                "metricA3_1_0": 0.4,
                "metricA3_1_1": 0.9,
                "metricA3_1_2": 0.0,
            },
        },
    }


def test_update_metrics_overwrite(
    participant: Participant,  # pylint: disable=redefined-outer-name
) -> None:
    """Test the metrics updating with overwriting.

    Args:
        participant (nxain_sdk.participant.Participant): A test participant.
    """

    # update metrics for epoch 0
    assert participant.metrics == {}
    participant.update_metrics(epoch=0, epoch_base=0, metric1=0.5, metric2=0.7)
    current_time_u1: int = participant.metrics[("metric1", 0)]["time"]
    assert participant.metrics == {
        ("metric1", 0): {
            "measurement": "metric1",
            "time": current_time_u1,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metric1": 0.5},
        },
        ("metric2", 0): {
            "measurement": "metric2",
            "time": current_time_u1,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metric2": 0.7},
        },
    }

    # overwrite metrics for epoch 0
    participant.update_metrics(epoch=0, epoch_base=0, metric1=[0.1, -0.3])
    current_time_ow: int = participant.metrics[("metric1", 0)]["time"]
    assert participant.metrics == {
        ("metric1", 0): {
            "measurement": "metric1",
            "time": current_time_ow,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metric1_0": 0.1, "metric1_1": -0.3},
        },
        ("metric2", 0): {
            "measurement": "metric2",
            "time": current_time_u1,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metric2": 0.7},
        },
    }

    # update metrics for epoch 1
    participant.update_metrics(epoch=1, epoch_base=0, metric1=1.0)
    current_time_u2: int = participant.metrics[("metric1", 1)]["time"]
    assert participant.metrics == {
        ("metric1", 0): {
            "measurement": "metric1",
            "time": current_time_ow,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metric1_0": 0.1, "metric1_1": -0.3},
        },
        ("metric2", 0): {
            "measurement": "metric2",
            "time": current_time_u1,
            "tags": {"id": participant.dummy_id, "epoch_global": "0"},
            "fields": {"metric2": 0.7},
        },
        ("metric1", 1): {
            "measurement": "metric1",
            "time": current_time_u2,
            "tags": {"id": participant.dummy_id, "epoch_global": "1"},
            "fields": {"metric1": 1.0},
        },
    }
