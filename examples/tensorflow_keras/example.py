"""Keras/Tensorflow example for the SDK Participant implementation."""

from typing import Dict, List, Tuple

import numpy as np
from tensorflow import Tensor
from tensorflow.data import Dataset  # pylint: disable=import-error
from tensorflow.keras import Input, Model  # pylint: disable=import-error
from tensorflow.keras.layers import Dense  # pylint: disable=import-error

from xain_sdk.participant import Participant as ABCParticipant
from xain_sdk.participant_state_machine import start_participant


class Participant(ABCParticipant):
    """An example of a Keras/Tensorflow implementation of a participant for federated learning.

    The attributes for the model and the datasets are only for convenience, they might as well be
    loaded in the `train_round()` method on the fly.

    Attributes:
        model: The model to be trained.
        trainset: A dataset for training.
        valset: A dataset for validation.
        testset: A dataset for testing.
        number_samples: The number of samples in the training dataset.
    """

    def __init__(self) -> None:
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
            optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"],
        )
        # define the number of samples in the training dataset
        self.number_samples: int = 80

        # define or load data to be trained on
        self.trainset: Dataset = (
            Dataset.from_tensor_slices(
                tensors=(
                    np.ones(shape=(80, 10), dtype=np.float32),
                    np.concatenate([np.eye(N=2, M=2, dtype=np.float32) for _ in range(40)]),
                )
            )
            .shuffle(buffer_size=80)
            .batch(batch_size=10)
        )
        self.valset: Dataset = Dataset.from_tensor_slices(
            tensors=(
                np.ones(shape=(10, 10), dtype=np.float32),
                np.concatenate([np.eye(N=2, M=2, dtype=np.float32) for _ in range(5)]),
            )
        ).batch(batch_size=10)
        self.testset: Dataset = Dataset.from_tensor_slices(
            tensors=(
                np.ones(shape=(10, 10), dtype=np.float32),
                np.concatenate([np.eye(N=2, M=2, dtype=np.float32) for _ in range(5)]),
            )
        ).batch(batch_size=10)

    def train_round(  # pylint: disable=unused-argument
        self, weights: List[np.ndarray], epochs: int, epoch_base: int
    ) -> Tuple[List[np.ndarray], int, Dict[str, np.ndarray]]:
        """Train the model in a federated learning round.

        A global model is given in terms of its `weights` and it is trained on local data for a
        number of `epochs`. The weights of the updated local model are returned together with the
        number of samples in the training dataset and a set of metrics.

        Args:
            weights (~typing.List[~numpy.ndarray]): The weights of the global model.
            epochs (int): The number of epochs to be trained.
            epoch_base (int): The epoch base number for the optimizer state (in case of epoch
                dependent optimizer parameters).

        Returns:
            ~typing.Tuple[~typing.List[~numpy.ndarray], int, ~typing.Dict[str, ~numpy.ndarray]]: The
                updated model weights, the number of training samples and the gathered metrics.
        """

        # load the weights of the global model into the local model
        self.model.set_weights(weights)

        # train the local model for the specified number of epochs and gather the metrics
        metrics_per_epoch: List[List[np.ndarray]] = []
        for _ in range(epochs):
            self.model.fit(x=self.trainset, verbose=2, shuffle=False)
            metrics_per_epoch.append(self.model.evaluate(x=self.valset, verbose=0))
        metrics: Dict[str, np.ndarray] = {
            name: np.stack(np.atleast_1d(*metric))
            for name, metric in zip(self.model.metrics_names, zip(*metrics_per_epoch))
        }

        # return the updated model weights, the number of train samples and the gathered metrics
        return self.model.get_weights(), self.number_samples, metrics


def main() -> None:
    """Entry point to start a participant."""

    participant: Participant = Participant()
    start_participant(participant, "127.0.0.1:50051")


if __name__ == "__main__":
    main()
