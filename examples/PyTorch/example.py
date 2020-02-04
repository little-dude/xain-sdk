"""PyTorch example for the SDK Participant implementation."""

from typing import Dict, List, Tuple

import numpy as np
from torch import utils
from torchvision import datasets, transforms

from cnn_class import Net
from xain_sdk.participant import Participant as ABCParticipant
from xain_sdk.participant_state_machine import start_participant


class Participant(ABCParticipant):
    """An example of a PyTorch implementation of a participant for federated learning.
    The attributes for the model and the datasets are only for convenience, they might as well be
    loaded in the `train_round()` method on the fly.
    Attributes:
        model: The model to be trained.
        trainset: A dataset for training.
        testset: A dataset for testing.
        trainloader:  A pytorch data loader obtained from  train data set
        testloader: A pytorch data loader obtained from  test data set
        number_samples: The number of samples in the training dataset.
        flattened: flattened vector of models weights
        shape: CNN model   architecture
        indices: indices of split points in the flattened vector
    """

    def __init__(self) -> None:
        """Initialize the custom participant.
        The model and the datasets are defined here only for convenience, they might as well be
        loaded in the `train_round()` method on the fly. Due to the nature of this example, the
        model is a simple dense neural network and the datasets are randomly generated.
        """

        super(Participant, self).__init__()
        # define or load a model to be trained

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
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
        self.model = Net()
        self.flattened, self.shapes, self.indices = self.model.flatten_weights()
        self.number_samples: int = len(self.trainloader)

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

        self.model.read_from_vector(self.indices, weights, self.shapes)
        self.model.train_n_epochs(self.trainloader, epochs)
        self.flattened, self.shapes, self.indices = self.model.flatten_weights()

        # TODO: return metric values from `train_n_epochs` and update the dict accordingly
        metrics: Dict[str, np.ndarray] = {}

        return self.flattened, self.number_samples, metrics


def main() -> None:
    """Entry point to start a participant."""

    participant: Participant = Participant()
    start_participant(participant, coordinator_url="localhost:50051")


if __name__ == "__main__":
    main()
