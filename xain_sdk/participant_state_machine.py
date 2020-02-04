"""Module implementing the networked Participant using gRPC."""

from enum import Enum, auto
import threading
import time
from typing import Dict, Optional, Tuple

from grpc import Channel, insecure_channel
from numpy import ndarray
from xain_proto.fl.coordinator_pb2 import (
    EndTrainingRoundRequest,
    EndTrainingRoundResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    RendezvousReply,
    RendezvousRequest,
    RendezvousResponse,
    StartTrainingRoundRequest,
    StartTrainingRoundResponse,
    State,
)
from xain_proto.fl.coordinator_pb2_grpc import CoordinatorStub
from xain_proto.np import ndarray_to_proto, proto_to_ndarray
from xain_proto.np.ndarray_pb2 import NDArray as pndarray

from xain_sdk.logger import get_logger
from xain_sdk.participant import InternalParticipant, Participant
from xain_sdk.store import AbstractStore, DummyStore, S3StorageConfig, S3Store

logger = get_logger(__name__)


# timings in seconds
RETRY_TIMEOUT: int = 5
HEARTBEAT_TIME: int = 10


class ParState(Enum):
    """Enumeration of Participant states."""

    WAITING: auto = auto()
    TRAINING: auto = auto()
    DONE: auto = auto()


def rendezvous(channel: Channel) -> None:
    """Start a rendezvous exchange with a coordinator.

    Args:
        channel: A gRPC channel to the coordinator.
    """

    coordinator: CoordinatorStub = CoordinatorStub(channel=channel)

    reply: RendezvousReply = RendezvousReply.LATER
    response: RendezvousResponse
    while reply == RendezvousReply.LATER:
        response = coordinator.Rendezvous(request=RendezvousRequest())
        if response.reply == RendezvousReply.ACCEPT:
            logger.info("Participant received: ACCEPT")
        elif response.reply == RendezvousReply.LATER:
            logger.info(
                "Participant received: LATER. Retrying...", retry_timeout=RETRY_TIMEOUT
            )
            time.sleep(RETRY_TIMEOUT)

        reply = response.reply


def start_training_round(channel: Channel) -> Tuple[ndarray, int, int]:
    """Start a training round initiation exchange with a coordinator.

    The decoded contents of the response from the coordinator are returned.

    Args:
        channel: A gRPC channel to the coordinator.

    Returns:
        A tuple ``(weights, epochs, epoch_base)`` where ``weights`` is
        the weights of a global model to train on, ``epochs`` is the
        number of epochs to train, and ``epoch_base`` is the epoch base
        of the global model.
    """

    coordinator: CoordinatorStub = CoordinatorStub(channel=channel)

    # send request to start training
    response: StartTrainingRoundResponse = coordinator.StartTrainingRound(
        request=StartTrainingRoundRequest()
    )
    logger.info("Participant received", response_type=type(response))

    # unpack the response
    weights: ndarray = proto_to_ndarray(response.weights)
    epochs: int = response.epochs
    epoch_base: int = response.epoch_base

    return weights, epochs, epoch_base


def end_training_round(
    channel: Channel, weights: ndarray, number_samples: int, metrics: Dict[str, ndarray]
) -> None:
    """Start a training round completion exchange with a coordinator.

    The locally trained model weights, the number of samples and the gathered metrics
    are sent.

    Args:
        channel: A gRPC channel to the coordinator.
        weights: The weights of the locally trained model.
        number_samples: The number of samples in the training dataset.
        metrics: Metrics data.
    """

    coordinator: CoordinatorStub = CoordinatorStub(channel=channel)

    # model weight arrays as protobuf message
    weights_proto: pndarray = ndarray_to_proto(weights)

    # metric data containing the metric names mapped to arrays as protobuf message
    metrics_proto: Dict[str, pndarray] = {
        key: ndarray_to_proto(value) for key, value in metrics.items()
    }

    # assembling a request with the update of the weights and the metrics
    request: EndTrainingRoundRequest = EndTrainingRoundRequest(
        weights=weights_proto, number_samples=number_samples, metrics=metrics_proto
    )
    response: EndTrainingRoundResponse = coordinator.EndTrainingRound(request=request)
    logger.info("Participant received", response_type=type(response))


def training_round(
    channel: Channel, participant: InternalParticipant, round: int
) -> None:
    """Initiate a training round exchange with a coordinator.

    Begins with `start_training_round`. Then performs local training computation using
    the `participant`. Finally, completes with `end_training_round`.

    In case of empty weights from the coordinator (i.e. a 0th round for weights
    initialization) the aggregation meta data and metrics from the participant are
    ignored.

    Args:
        channel: A gRPC channel to the coordinator.
        participant: The local participant.
        round: round number.

    Raises:
        TypeError: If the model weights received from the participant's local training
            round are not of type `ndarray`.
        TypeError: If the aggregation meta data received from the participant's local
            training round is not of type `int`.
        ValueError: If the aggregation meta data received from the participant's local
            training round is negative.
        TypeError: If the metrics received from the participant's local training round
            are not of type `Dict[str, ndarray]`.
    """

    # retreive global weights, epochs and epoch base from the coordinator
    global_weights: Optional[ndarray]
    epochs: int
    epoch_base: int
    global_weights, epochs, epoch_base = start_training_round(channel=channel)
    if not global_weights.size:
        global_weights = None

    # start a local training round of the participant
    local_weights: ndarray
    if global_weights is not None:  # ith training round
        number_samples: int
        metrics: Dict[str, ndarray]
        local_weights, number_samples, metrics = participant.train_round(
            weights=global_weights, epochs=epochs, epoch_base=epoch_base
        )

        # data validation
        if not isinstance(local_weights, ndarray):
            raise TypeError("Model weights must be of type `ndarray`!")
        if not isinstance(number_samples, int):
            raise TypeError("Aggregation meta data must be of type `int`!")
        if number_samples < 0:
            raise ValueError("Aggregation meta data must be nonnegative!")
        if (
            not isinstance(metrics, Dict)
            or not all(isinstance(key, str) for key in metrics.keys())
            or not all(isinstance(value, ndarray) for value in metrics.values())
        ):
            raise TypeError("Metrics must be of type `Dict[str, ndarray]`!")

        logger.info("Storing weights", round=round)
        participant.write_weights(round, local_weights)

        # return updated weights, no. of training samples and metrics to the coordinator
        end_training_round(
            channel=channel,
            weights=local_weights,
            number_samples=number_samples,
            metrics=metrics,
        )

    else:  # 0th training round
        local_weights, _, _ = participant.train_round(
            weights=global_weights, epochs=epochs, epoch_base=epoch_base
        )

        # data validation
        if not isinstance(local_weights, ndarray):
            raise TypeError("Model weights must be of type `ndarray`!")

        logger.info("Storing weights", round=round)
        participant.write_weights(round, local_weights)

        # return initialized weights
        end_training_round(
            channel=channel, weights=local_weights, number_samples=0, metrics={}
        )


class StateRecord:
    """Thread-safe record of a participant's state and round number."""

    def __init__(  # pylint: disable=redefined-builtin
        self, state: ParState = ParState.WAITING, round: int = -1
    ) -> None:
        """Initialize the state record.

        Args:
            state: The initial state. Defaults to WAITING.
            round: The initial training round. Defaults to -1.
        """

        self.cond: threading.Condition = threading.Condition()
        self.round: int = round
        self.state: ParState = state

    def lookup(self) -> Tuple[ParState, int]:
        """Get the state and round number.

        Returns:
            The state and round number.
        """

        with self.cond:
            return self.state, self.round

    def update(self, state: ParState) -> None:
        """Update the state.

        Args:
            state: The state to update to.
        """

        with self.cond:
            self.state = state
            self.cond.notify()

    def wait_until_selected_or_done(self) -> ParState:
        """Wait until the participant was selected for training or is done.

        Returns:
            The new state the participant is in.
        """

        with self.cond:
            self.cond.wait_for(lambda: self.state in {ParState.TRAINING, ParState.DONE})
            return self.state


def transit(state_record: StateRecord, heartbeat_response: HeartbeatResponse) -> None:
    """Participant state transition function on a heartbeat response.

    Updates the state record.

    Args:
        state_record: The updatable state record of the participant.
        heartbeat_response: The heartbeat response from the coordinator.
    """

    msg: State = heartbeat_response.state
    global_round: int = heartbeat_response.round
    with state_record.cond:
        if state_record.state == ParState.WAITING:
            if msg == State.ROUND and global_round > state_record.round:
                state_record.state = ParState.TRAINING
                state_record.round = global_round
                state_record.cond.notify()
                logger.info(
                    "Transition to training state",
                    local_state=state_record.state,
                    local_round=state_record.round,
                )
            elif msg == State.FINISHED:
                state_record.state = ParState.DONE
                state_record.cond.notify()
                logger.info(
                    "Transition to finished state", local_state=state_record.state
                )
            elif (
                msg == State.STANDBY
                or msg == State.ROUND
                and global_round == state_record.round
            ):
                logger.debug(
                    "Continue in waiting state",
                    local_round=state_record.round,
                    heartbeat_message=msg,
                    heartbeat_round=global_round,
                )
            else:
                logger.warn(
                    "Unexpected heartbeat response",
                    local_round=state_record.round,
                    heartbeat_message=msg,
                    heartbeat_round=global_round,
                )


def message_loop(
    channel: Channel, state_record: StateRecord, terminate: threading.Event
) -> None:
    """Periodically send (and handle) heartbeat messages in a loop.

    Args:
        channel: A gRPC channel to the coordinator.
        state_record: The participant's state record.
        terminate: An event to terminate the message loop.
    """

    coordinator: CoordinatorStub = CoordinatorStub(channel=channel)
    while not terminate.is_set():
        request = HeartbeatRequest()
        transit(
            state_record=state_record,
            heartbeat_response=coordinator.Heartbeat(request=request),
        )
        time.sleep(HEARTBEAT_TIME)


def start_participant(
    participant: Participant,
    coordinator_url: str,
    storage_config: Optional[S3StorageConfig] = None,
) -> None:
    """Top-level function for the participant's state machine.

    After rendezvous and heartbeat initiation, the Participant is WAITING. If
    selected to train for the current round, it moves to TRAINING, otherwise it
    remains in WAITING. After training is complete for the round, it moves back
    to WAITING. When there is no more training to be done, it moves to the
    terminal state DONE.

    Args:
        participant: The participant for local training.
        storage_config: The storage configuration
        coordinator_url: The URL of the coordinator to connect to.
    """

    # FIXME(XP-515): the storage feature is highly experimental. In
    # order to avoid breaking existing setups, we use a dummy store
    # unless the user provides a valid configuration for using an S3
    # store.
    store: AbstractStore = DummyStore()
    if isinstance(storage_config, S3StorageConfig):
        store = S3Store(storage_config)

    internal_participant: InternalParticipant = InternalParticipant(participant, store)

    # use insecure channel for now
    with insecure_channel(target=coordinator_url) as channel:  # thread-safe
        rendezvous(channel=channel)

        state_record: StateRecord = StateRecord()
        terminate: threading.Event = threading.Event()
        msg_loop = threading.Thread(
            target=message_loop, args=(channel, state_record, terminate)
        )
        msg_loop.start()

        # in WAITING state
        logger.info("rendezvous successful, begin WAITING...")
        begin_waiting(state_record, channel, internal_participant)

        # in DONE state
        logger.info("shutting down participant...")
        terminate.set()
        msg_loop.join()


def begin_waiting(
    state_record: StateRecord, channel: Channel, participant: InternalParticipant
) -> None:
    """"Perform actions in the Participant state WAITING.

    Args:
        state_record: The participant's state record.
        channel: A gRPC channel to the coordinator.
        participant: The participant for local training.
    """

    state: ParState = state_record.wait_until_selected_or_done()
    if state == ParState.TRAINING:  # selected
        logger.debug("received TRAINING signal")
        begin_training(state_record, channel, participant)
    elif state == ParState.DONE:
        logger.info("received DONE signal")
    else:
        logger.warn("received unknown signal", state_signal=state)


def begin_training(
    state_record: StateRecord, channel: Channel, participant: InternalParticipant
) -> None:
    """Perform actions in the Participant state TRAINING.

    Args:
        state_record: The participant's state record.
        channel: A gRPC channel to the coordinator.
        participant: The participant for local training.
    """

    _, local_round = state_record.lookup()
    # perform training comms and computation
    training_round(channel, participant, local_round)

    # internal transition back to WAITING
    logger.debug("trained round, going back to WAITING...")
    state_record.update(ParState.WAITING)
    begin_waiting(state_record, channel, participant)
