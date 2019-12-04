"""Provides participant API"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from xain_sdk.sdk.use_case import UseCase


def start(coordinator_url: str, use_case: UseCase) -> None:
    """Starts a participant which will connect to coordinator_url and
    work on use_case

    Args:
        coordinator_url (str): URL of the coordinator to connect to
        use_case (UseCase): Instance of UseCase class
    """

    raise NotImplementedError


class Participant(ABC):
    """An abstract participant for federated learning."""

    @abstractmethod
    def train_round(
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
