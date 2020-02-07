"""Tensorflow Keras example for the SDK Participant implementation."""

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from tensorflow import Tensor
from tensorflow.data import Dataset  # pylint: disable=import-error
from tensorflow.keras import Input, Model  # pylint: disable=import-error
from tensorflow.keras.layers import Dense  # pylint: disable=import-error

from xain_sdk.config import Config, InvalidConfig
from xain_sdk.logger import StructLogger, get_logger
from xain_sdk.participant import Participant as ABCParticipant
from xain_sdk.participant_state_machine import start_participant

logger: StructLogger = get_logger(__name__)


class Participant(ABCParticipant):
    """An example of a TF Keras implementation of a participant for federated learning.

    The attributes for the model and the datasets are only for convenience, they might
    as well be loaded elsewhere.

    Attributes:
        model: The model to be trained.
        model_shapes: The shapes of the model weights.
        trainset: A dataset for training.
        valset: A dataset for validation.
        testset: A dataset for testing.
    """

    def __init__(self) -> None:
        """Initialize the custom participant.

        The model and the datasets are defined here only for convenience, they might as
        well be loaded elsewhere. Due to the nature of this example, the model is a
        simple dense neural network and the datasets are randomly generated.
        """

        super(Participant, self).__init__()

        # define or load a model to be trained
        self.init_model()

        # define or load datasets to be trained on
        self.init_datasets()

    def train_round(  # pylint: disable=unused-argument
        self, weights: Optional[np.ndarray], epochs: int, epoch_base: int
    ) -> Tuple[np.ndarray, int, Dict[str, np.ndarray]]:
        """Train a model in a federated learning round.

        A model is given in terms of its weights and the model is trained on the
        participant's dataset for a number of epochs. The weights of the updated model
        are returned in combination with the number of samples of the train dataset and
        some gathered metrics.

        If the weights given are None, then the participant is expected to initialize
        the weights according to its model definition and return them without training.

        Args:
            weights (~typing.Optional[~numpy.ndarray]): The weights of the model to be
                trained.
            epochs (int): The number of epochs to be trained.
            epoch_base: The global training epoch number.

        Returns:
            ~typing.Tuple[~numpy.ndarray, int, ~typing.Dict[str, ~numpy.ndarray]]: The
                updated model weights, the number of training samples and the gathered
                metrics.
        """

        number_samples: int
        metrics: Dict[str, np.ndarray]

        if weights is not None:
            # load the weights of the global model into the local model
            self.set_tensorflow_weights(
                weights=weights, shapes=self.model_shapes, model=self.model
            )

            # train the local model for the specified no. of epochs and gather metrics
            number_samples = 80
            metrics_per_epoch: List[List[np.ndarray]] = []
            for _ in range(epochs):
                self.model.fit(x=self.trainset, verbose=2, shuffle=False)
                metrics_per_epoch.append(self.model.evaluate(x=self.valset, verbose=0))
            metrics = {
                name: np.stack(np.atleast_1d(*metric))
                for name, metric in zip(
                    self.model.metrics_names, zip(*metrics_per_epoch)
                )
            }

        else:
            # initialize the weights of the local model
            self.init_model()
            number_samples = 0
            metrics = {}

        # return the updated model weights, the number of train samples and the metrics
        weights = self.get_tensorflow_weights(model=self.model)
        return weights, number_samples, metrics

    def init_model(self) -> None:
        """Initialize a model."""

        # define model layers and compile the model
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

        # get the shapes of the model weights
        self.model_shapes: List[Tuple[int, ...]] = self.get_tensorflow_shapes(
            model=self.model
        )

    def init_datasets(self) -> None:
        """Initialize datasets."""

        self.trainset: Dataset = (
            Dataset.from_tensor_slices(
                tensors=(
                    np.ones(shape=(80, 10), dtype=np.float32),
                    np.concatenate(
                        [np.eye(N=2, M=2, dtype=np.float32) for _ in range(40)]
                    ),
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


def main() -> None:
    """Entry point to start a participant."""

    participant: Participant = Participant()

    try:
        config = Config.load("config.toml")
    except InvalidConfig as err:
        logger.error("Invalid config", error=str(err))
        sys.exit(1)

    start_participant(participant, config)


if __name__ == "__main__":
    main()
