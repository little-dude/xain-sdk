"""Participant API"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TypeVar, cast

import numpy as np
from numpy import ndarray

from xain_sdk.store import AbstractStore

# Currently, the combination of sphinx_autodoc_typehints and typing.TYPE_CHECKING
# crashes, see https://github.com/agronholm/sphinx-autodoc-typehints/issues/22.
# Therefor, the workaround introduces a descriptive generic type alias which gets casted
# to a locally imported type.
TensorflowKerasModel = TypeVar("TensorflowKerasModel")  # for tensorflow.keras.Model
TorchNNModule = TypeVar("TorchNNModule")  # for torch.nn.Module


class Participant(ABC):
    """An abstract participant for federated learning."""

    @abstractmethod
    def train_round(
        self, weights: Optional[ndarray], epochs: int, epoch_base: int
    ) -> Tuple[ndarray, int, Dict[str, ndarray]]:
        """Train a model in a federated learning round.

        A model is given in terms of its weights and the model is trained on the
        participant's dataset for a number of epochs. The weights of the updated model
        are returned in combination with the number of samples of the train dataset and
        some gathered metrics.

        If the weights given are None, then the participant is expected to initialize
        the weights according to its model definition and return them without training.

        Args:
            weights: The weights of the model to be trained.
            epochs: The number of epochs to be trained.
            epoch_base: The epoch base number for the optimizer state (in case of epoch
                dependent optimizer parameters).

        Returns:
            The updated model weights, the number of training samples and the gathered
                metrics.
        """

    @staticmethod
    def get_tensorflow_weights(model: TensorflowKerasModel) -> ndarray:
        """Get the flattened weights vector from a tensorflow model.

        Args:
            model (~tensorflow.keras.Model): A tensorflow model.

        Returns:
            ~numpy.ndarray: The vector of the flattened model weights.
        """

        # tensorflow must be imported locally for sdk framework agnosticity
        from tensorflow.keras import Model  # pylint: disable=import-error

        return np.concatenate(cast(Model, model).get_weights(), axis=None)

    @staticmethod
    def set_tensorflow_weights(
        weights: ndarray, shapes: List[Tuple[int, ...]], model: TensorflowKerasModel
    ) -> None:
        """Set the weights of a tensorflow model.

        Args:
            weights (~numpy.ndarray): A vector of flat model weights.
            shapes (~typing.List[~typing.Tuple[int, ...]]): The original shapes of the
                tensorflow model weights.
            model (~tensorflow.keras.Model): A tensorflow model.
        """

        # tensorflow must be imported locally for sdk framework agnosticity
        from tensorflow.keras import Model  # pylint: disable=import-error

        # expand the flat weights
        indices: ndarray = np.cumsum([np.prod(shape) for shape in shapes])
        tensorflow_weights: List[ndarray] = np.split(
            weights, indices_or_sections=indices
        )
        tensorflow_weights = [
            np.reshape(weight, newshape=shape)
            for weight, shape in zip(tensorflow_weights, shapes)
        ]

        # apply the weights to the tensorflow model
        cast(Model, model).set_weights(tensorflow_weights)

    @staticmethod
    def get_pytorch_weights(model: TorchNNModule) -> ndarray:
        """Get the flattened weights vector from a pytorch model.

        Note:
            This will only work with models which already did a forward pass at least
            once.

        Args:
            model (~torch.nn.Module): A pytorch model.

        Returns:
            ~numpy.ndarray: The vector of the flattened model weights.
        """

        # pytorch must be imported locally for sdk framework agnosticity
        from torch.nn import Module

        return np.concatenate(
            list(cast(Module, model).state_dict().values()), axis=None
        )

    @staticmethod
    def set_pytorch_weights(
        weights: ndarray, shapes: List[Tuple[int, ...]], model: TorchNNModule
    ) -> None:
        """Set the weights of a pytorch model.

        Args:
            weights (~numpy.ndarray): A vector of flat model weights.
            shapes (~typing.List[~typing.Tuple[int, ...]]): The original shapes of the
                pytorch model weights.
            model (~torch.nn.Module): A pytorch model.
        """

        # pytorch must be imported locally for sdk framework agnosticity
        import torch
        from torch.nn import Module

        # expand the flat weights
        indices: ndarray = np.cumsum([np.prod(shape) for shape in shapes])
        pytorch_weights: List[ndarray] = np.split(weights, indices_or_sections=indices)
        pytorch_weights = [
            np.reshape(weight, newshape=shape)
            for weight, shape in zip(pytorch_weights, shapes)
        ]

        # apply the weights to the pytorch model
        state_dict: Dict = {
            layer: torch.from_numpy(weight)  # pylint: disable=no-member
            for layer, weight in zip(
                cast(Module, model).state_dict().keys(), pytorch_weights
            )
        }
        cast(Module, model).load_state_dict(state_dict)


class InternalParticipant:
    """Internal representation of a participant that encapsulates the
    user-defined Participant class.

    Args:
        participant: A user provided implementation of a participant.
        store: A client for a storage service.
    """

    def __init__(self, participant: Participant, store: AbstractStore):
        """Initialize the internal representation of a participant.

        Args:
            participant: A user provided implementation of a participant.
            store: A client for a storage service.
        """

        self.participant: Participant = participant
        self.store: AbstractStore = store

    def train_round(
        self, weights: Optional[ndarray], epochs: int, epoch_base: int
    ) -> Tuple[ndarray, int, Dict[str, ndarray]]:
        """A wrapper for :py:meth:`~xain_sdk.participant.Participant.train_round`."""

        return self.participant.train_round(
            weights=weights, epochs=epochs, epoch_base=epoch_base
        )

    def write_weights(self, round: int, weights: ndarray) -> None:
        """A wrapper for :py:meth:`~xain_sdk.store.AbstractStore.write_weights`."""

        self.store.write_weights(round=round, weights=weights)
