"""Participant API"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple, TypeVar, cast
import uuid

import numpy as np
from numpy import ndarray

from xain_sdk.store import AbstractStore, NullObjectStore

# Currently, the combination of sphinx_autodoc_typehints and typing.TYPE_CHECKING
# crashes, see https://github.com/agronholm/sphinx-autodoc-typehints/issues/22.
# Therefore, the workaround introduces a descriptive generic type alias which gets
# casted to a locally imported type.
TensorflowKerasModel = TypeVar("TensorflowKerasModel")  # for tensorflow.keras.Model
TorchNnModule = TypeVar("TorchNnModule")  # for torch.nn.Module


class Participant(ABC):
    """An abstract participant for federated learning.

    Args:
        metrics: A dictionary to gather the metrics of the current training round.
        dummy_id: A fake id for the participant. Will be replaced later on.
    """

    def __init__(self) -> None:
        """Initialize a participant."""

        super(Participant, self).__init__()
        self.metrics: Dict[Tuple[str, int], Dict] = {}
        self.dummy_id: str = str(uuid.uuid4())

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
            epoch_base: The global training epoch number.

        Returns:
            The updated model weights, the number of training samples and the gathered
                metrics.
        """

    @staticmethod
    def get_tensorflow_shapes(model: TensorflowKerasModel) -> List[Tuple[int, ...]]:
        """Get the shapes of the weights of a tensorflow model.

        Args:
            model (~tensorflow.keras.Model): A tensorflow model.

        Returns:
            ~typing.List[~typing.Tuple[int, ...]]: The shapes of the model weights per
                layer.
        """

        # tensorflow must be imported locally for sdk framework agnosticity
        from tensorflow.keras import Model  # pylint: disable=import-error

        return [weight.shape for weight in cast(Model, model).get_weights()]

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
    def get_pytorch_shapes(model: TorchNnModule) -> List[Tuple[int, ...]]:
        """Get the shapes of the weights of a pytorch model.

        Note:
            This will only work with models which already did a forward pass at least
            once.

        Args:
            model (~torch.nn.Module): A pytorch model.

        Returns:
            ~typing.List[~typing.Tuple[int, ...]]: The shapes of the model weights per
                layer.
        """

        # pytorch must be imported locally for sdk framework agnosticity
        from torch.nn import Module

        return [
            tuple(weight.shape) for weight in cast(Module, model).state_dict().values()
        ]

    @staticmethod
    def get_pytorch_weights(model: TorchNnModule) -> ndarray:
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
        weights: ndarray, shapes: List[Tuple[int, ...]], model: TorchNnModule
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

    def update_metrics(self, epoch: int, epoch_base: int, **kwargs: Any) -> None:
        """Update the metrics for the current training epoch.

        Metrics are expected as key=value pairs where the key is a name for the metric
        and the value is any value of the metric from the current epoch. Values can be
        scalars/lists/arrays of numerical values and must be convertible to floats. If
        a metric is already present for the current epoch, then its values will be
        overwritten.

        Examples:
            update_metrics(0, 0, Accuracy=0.8, Accuracy_per_Category=[0.8, 0.7, 0.9])
            update_metrics(0, 0, F1_per_Category=np.ndarray([0.85, 0.9, 0.95]))

        Args:
            epoch (int): The local training epoch number.
            epoch_base (int): The global training epoch number.
            kwargs (~typing.Any): The metrics names and values.
        """

        def get_fields(fields: Dict[str, float], **kwargs: Any) -> Dict[str, float]:
            """Get all fields for the metric update.

            The key-value pairs are traversed recursively through all dimensions of the
            values until scalar values can be returned. The key naming corresponds to
            the original key name followed by the indices of the original value array.

            Args:
                fields (~typing.Dict[str, float]): The fields to be updated.
                kwargs (~typing.Any): The metrics names and values.

            Returns:
                ~typing.Dict[str, float]: The updated fields.
            """

            for key, value in kwargs.items():
                # get the metrics fields recursively from value arrays
                try:
                    fields = get_fields(
                        fields,
                        **{f"{key}_{idx}": val for idx, val in enumerate(iter(value))},
                    )

                # add a metric field for scalar values
                except TypeError:
                    fields[key] = float(value)

            return fields

        # update each metric for the current epoch
        current_time: int = int(time.time())
        epoch_global: str = str(epoch + epoch_base)
        for key, value in kwargs.items():
            self.metrics[(key, epoch)] = {
                "measurement": key,
                "time": current_time,
                "tags": {"id": self.dummy_id, "epoch_global": epoch_global},
                "fields": get_fields({}, **{key: value}),
            }


class InternalParticipant:
    """Internal representation of a participant that encapsulates the
    user-defined Participant class.

    Args:
        participant: A user provided implementation of a participant.
        store: A client for a storage service.
    """

    def __init__(
        self, participant: Participant, store: AbstractStore = NullObjectStore()
    ):
        """Initialize the internal representation of a participant.

        Args:
            participant: A user provided implementation of a participant.
            store: A client for a storage service. Defaults to `NullObjectStore()`.
        """

        self.participant: Participant = participant
        self.store: AbstractStore = store

    def train_round(
        self, weights: Optional[ndarray], epochs: int, epoch_base: int
    ) -> Tuple[ndarray, int, Dict[str, ndarray]]:
        """A wrapper for :py:meth:`~xain_sdk.participant.Participant.train_round`."""

        # reset the metrics
        self.participant.metrics = {}

        # execute local training
        aggregation_data: int
        metrics: Dict[str, ndarray]
        weights, aggregation_data, metrics = self.participant.train_round(
            weights=weights, epochs=epochs, epoch_base=epoch_base
        )

        # finalize the metrics
        # TODO(PB-394): make use of this as a JSON return value
        unused_json_metrics: List[Dict] = [
            metric for metric in self.participant.metrics.values()
        ]

        return weights, aggregation_data, metrics

    def write_weights(self, round: int, weights: ndarray) -> None:
        """A wrapper for :py:meth:`~xain_sdk.store.AbstractStore.write_weights`."""

        self.store.write_weights(round=round, weights=weights)
