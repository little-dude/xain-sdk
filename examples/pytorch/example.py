"""PyTorch example for the SDK Participant implementation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import utils
from torchvision import datasets, transforms

from cnn_class import Net
from xain_sdk.participant import Participant as ABCParticipant
from xain_sdk.participant_state_machine import start_participant
from xain_sdk.store import S3StorageConfig


class Participant(ABCParticipant):
    """An example of a PyTorch implementation of a participant for federated learning.

    The attributes for the model and the datasets are only for convenience, they might
    as well be loaded elsewhere.

    Attributes:
        model: The model to be trained.
        trainset: A dataset for training.
        testset: A dataset for testing.
        trainloader: A pytorch data loader obtained from train data set.
        testloader: A pytorch data loader obtained from test data set.
        flattened: A flattened vector of models weights.
        shape: CNN model architecture.
        indices: Indices of split points in the flattened vector.
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
            epoch_base (int): The epoch base number for the optimizer state (in case of
                epoch dependent optimizer parameters).

        Returns:
            ~typing.Tuple[~numpy.ndarray, int, ~typing.Dict[str, ~numpy.ndarray]]: The
                updated model weights, the number of training samples and the gathered
                metrics.
        """

        number_samples: int
        metrics: Dict[str, np.ndarray]

        if weights is not None:
            # load the weights of the global model into the local model
            self.set_pytorch_weights(
                weights=weights, shapes=self.model_shapes, model=self.model
            )

            # train the local model for the specified no. of epochs and gather metrics
            number_samples = len(self.trainloader)
            self.model.train_n_epochs(self.trainloader, epochs)
            metrics = {}  # TODO: return metric values from `train_n_epochs`

        else:
            # initialize the weights of the local model
            self.init_model()
            number_samples = 0
            metrics = {}

        # return the updated model weights, the number of train samples and the metrics
        weights = self.get_pytorch_weights(model=self.model)
        return weights, number_samples, metrics

    def init_model(self) -> None:
        """Initialize a model."""

        # define model layers
        self.model: Net = Net()

        # get the shapes of the model weights
        self.model.forward(torch.zeros((4, 3, 32, 32)))  # pylint: disable=no-member
        self.model_shapes: List[Tuple[int, ...]] = [
            tuple(weight.shape) for weight in self.model.state_dict().values()
        ]

    def init_datasets(self) -> None:
        """Initialize datasets."""

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        self.trainloader = utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=2
        )
        self.testset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        self.testloader = utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=2
        )


def main() -> None:
    """Entry point to start a participant."""

    participant: Participant = Participant()
    storage_config: S3StorageConfig = S3StorageConfig(
        endpoint_url="http://localhost:9000",
        access_key_id="minio",
        secret_access_key="minio123",
        bucket="xain-fl-temporary-weights",
    )
    start_participant(
        participant=participant,
        coordinator_url="localhost:50051",
        storage_config=storage_config,
    )


if __name__ == "__main__":
    main()
