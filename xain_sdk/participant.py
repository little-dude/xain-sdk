"""Provides participant API"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from numpy import ndarray

from xain_sdk.store import AbstractStore


class Participant(ABC):
    """An abstract participant for federated learning."""

    @abstractmethod
    def train_round(
        self, weights: List[ndarray], epochs: int, epoch_base: int
    ) -> Tuple[List[ndarray], int, Dict[str, ndarray]]:
        """Train a model in a federated learning round.

        A model is given in terms of its weights and the model is trained on the participant's
        dataset for a number of epochs. The weights of the updated model are returned in combination
        with the number of samples of the train dataset and some gathered metrics.

        If no weights are given (i.e. an empty list of weights), then the participant is expected to
        initialize the weights according to its model definition and return them without training.

        Args:
            weights: The weights of the model to be trained.
            epochs: The number of epochs to be trained.
            epoch_base: The epoch base number for the optimizer state (in case of epoch
                dependent optimizer parameters).

        Returns:
            The updated model weights, the number of training samples
            and the gathered metrics.

        """


class InternalParticipant:
    """Internal representation of a participant that encapsulates the
    user-defined Participant class.

    Args:

        participant: user provided implementation of a participant
        store: client for a storage service

    """

    def __init__(self, participant: Participant, store: AbstractStore):
        self.participant = participant
        self.store = store

    def train_round(
        self, weights: List[ndarray], epochs: int, epoch_base: int
    ) -> Tuple[List[ndarray], int, Dict[str, ndarray]]:
        """:py:meth:`~xain_sdk.participant.Participant.train_round` wrapper

        """
        return self.participant.train_round(weights, epochs, epoch_base)

    def write_weights(self, round: int, weights: List[ndarray]) -> None:
        """:py:meth:`~xain_sdk.store.AbstractStore.write_weights` wrapper"""
        return self.store.write_weights(round, weights)
