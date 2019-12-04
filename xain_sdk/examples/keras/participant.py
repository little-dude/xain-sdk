"""Keras/Tensorflow example for the SDK Participant implementation."""

from typing import Dict, List, Tuple

import numpy as np
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from xain_sdk.sdk.participant import Participant as ABCParticipant


class Participant(ABCParticipant):
    """An example of a Keras/Tensorflow implementation of a participant for federated learning.

    The attributes for the model and the datasets are only for convenience, they might as well be
    loaded in the `train_round()` method on the fly.

    Attributes:
        model: The model to be trained.
        trainset: A dataset for training.
        valset: A dataset for validation.
        testset: A dataset for testing.
    """

    def __init__(self):
        """Initialize the custom participant.

        The model and the datasets are defined here only for convenience, they might as well be
        loaded in the `train_round()` method on the fly. Due to the nature of this example, the
        model is a simple dense neural network and the datasets are randomly generated.
        """

        super(Participant, self).__init__()

        # define or load a model to be trained
        input_layer: Tensor = Input(shape=(10,), dtype="float32")
        hidden_layer: Tensor = Dense(
            units=6,
            activation="relu",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )(inputs=input_layer)
        output_layer: Tensor = Dense(
            units=2,
            activation="softmax",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )(inputs=hidden_layer)
        self.model: Model = Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        # define or load data to be trained on
        self.trainset: Dataset = (
            Dataset.from_tensor_slices(
                tensors=(
                    np.ones(shape=(80, 10), dtype=np.float32),
                    np.eye(N=80, M=10, dtype=np.float32),
                )
            )
            .shuffle(buffer_size=80)
            .batch(batch_size=10)
        )
        self.valset: Dataset = Dataset.from_tensor_slices(
            tensors=(
                np.ones(shape=(10, 10), dtype=np.float32),
                np.eye(N=10, M=10, dtype=np.float32),
            )
        ).batch(batch_size=10)
        self.testset: Dataset = Dataset.from_tensor_slices(
            tensors=(
                np.ones(shape=(10, 10), dtype=np.float32),
                np.eye(N=10, M=10, dtype=np.float32),
            )
        ).batch(batch_size=10)

    def train_round(  # pylint: disable=unused-argument
        self, weights: List[np.ndarray], epochs: int, epoch_base: int
    ) -> Tuple[List[np.ndarray], Dict[str, List[np.ndarray]]]:
        # pylint: disable=line-too-long
        """Train the model in a federated learning round.

        A global model is given in terms of its `weights` and it is trained on local data for a
        number of `epochs`. The weights of the updated local model are returned together with a set
        of metrics.

        Args:
            weights (~typing.List[~numpy.ndarray]): The weights of the global model.
            epochs (int): The number of epochs to be trained.
            epoch_base (int): The epoch base number for the optimizer state (in case of epoch
                dependent optimizer parameters).

        Returns:
            ~typing.Tuple[~typing.List[~numpy.ndarray], ~typing.Dict[str, ~typing.List[~numpy.ndarray]]]:
                The updated model weights and the gathered metrics.
        """
        # pylint: enable=line-too-long

        # load the weights of the global model into the local model
        self.model.set_weights(weights)

        # train the local model for the specified number of epochs and gather the metrics
        metrics: Dict[str, List[np.ndarray]] = {
            metric_name: [] for metric_name in self.model.metrics_names
        }
        for _ in range(epochs):
            self.model.fit(x=self.trainset, verbose=2, shuffle=False)
            for metric_name, metric in zip(
                self.model.metrics_names, self.model.evaluate(x=self.valset, verbose=0)
            ):
                metrics[metric_name].append(metric)

        # return the updated weights of the local model and the gathered metrics
        return self.model.get_weights(), metrics
