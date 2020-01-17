"""Module implementing the networked Participant using gRPC."""

from enum import Enum, auto
import threading
import time
from typing import Dict, List, Tuple

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
from xain_proto.numproto import ndarray_to_proto, proto_to_ndarray
from xain_proto.numproto.ndarray_pb2 import NDArray as pndarray

from xain_sdk.logger import get_logger
from xain_sdk.participant import Participant

logger = get_logger(__name__)


# timings in seconds
RETRY_TIMEOUT: int = 5
HEARTBEAT_TIME: int = 10


class ParState(Enum):
    """Enumeration of Participant states."""

    WAITING_FOR_SELECTION: auto = auto()
    TRAINING: auto = auto()
    POST_TRAINING: auto = auto()
    DONE: auto = auto()


def rendezvous(channel: Channel) -> None:
    """Start a rendezvous exchange with a coordinator.

    Args:
        channel (~grpc.Channel): A gRPC channel to the coordinator.
    """

    coordinator: CoordinatorStub = CoordinatorStub(channel=channel)

    reply: RendezvousReply = RendezvousReply.LATER
    response: RendezvousResponse
    while reply == RendezvousReply.LATER:
        response = coordinator.Rendezvous(request=RendezvousRequest())
        if response.reply == RendezvousReply.ACCEPT:
            logger.info("Participant received: ACCEPT")
        elif response.reply == RendezvousReply.LATER:
            logger.info("Participant received: LATER. Retrying...", retry_timeout=RETRY_TIMEOUT)
            time.sleep(RETRY_TIMEOUT)

        reply = response.reply


def start_training_round(channel: Channel) -> Tuple[List[ndarray], int, int]:
    """Start a training round initiation exchange with a coordinator.

    The decoded contents of the response from the coordinator are returned.

    Args:
        channel (~grpc.Channel): A gRPC channel to the coordinator.

    Returns:
        ~typing.List[~numpy.ndarray]: The weights of a global model to train on.
        int: The number of epochs to train.
        int: The epoch base of the global model.
    """

    coordinator: CoordinatorStub = CoordinatorStub(channel=channel)

    # send request to start training
    response: StartTrainingRoundResponse = coordinator.StartTrainingRound(
        request=StartTrainingRoundRequest()
    )
    logger.info("Participant received", response_type=type(response))

    weights: List[ndarray] = [proto_to_ndarray(weight) for weight in response.weights]
    epochs: int = response.epochs
    epoch_base: int = response.epoch_base

    return weights, epochs, epoch_base


def end_training_round(
    channel: Channel, weights: List[ndarray], number_samples: int, metrics: Dict[str, ndarray],
) -> None:
    """Start a training round completion exchange with a coordinator.

    The locally trained model weights, the number of samples and the gathered metrics are sent.

    Args:
        channel (~grpc.Channel): A gRPC channel to the coordinator.
        weights (~typing.List[~numpy.ndarray]): The weights of the locally trained model.
        number_samples (int): The number of samples in the training dataset.
        metrics (~typing.Dict[str, ~numpy.ndarray]): Metrics data.
    """

    coordinator: CoordinatorStub = CoordinatorStub(channel=channel)

    # model weight arrays as protobuf message
    weights_proto: List[pndarray] = [ndarray_to_proto(weight) for weight in weights]

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


def training_round(channel: Channel, participant: Participant) -> None:
    """Initiate a training round exchange with a coordinator.

    Begins with `start_training_round`. Then performs local training computation using the
    `participant`. Finally, completes with `end_training_round`.

    Args:
        channel (~grpc.Channel): A gRPC channel to the coordinator.
        participant (~xain_sdk.participant.Participant): The local participant.

    Raises:
        TypeError: If the model weights received from the participant's local training round are not
            of type ~typing.List[~numpy.ndarray].
        TypeError: If the metrics received from the participant's local training round are not of
            type ~typing.Dict[str, ~numpy.ndarray].
    """

    # retreive global weights, epochs and epoch base from the coordinator
    weights: List[ndarray]
    epochs: int
    epoch_base: int
    weights, epochs, epoch_base = start_training_round(channel=channel)

    # start a local training round of the participant
    number_samples: int
    metrics: Dict[str, ndarray]
    weights, number_samples, metrics = participant.train_round(
        weights=weights, epochs=epochs, epoch_base=epoch_base
    )

    # data validation
    is_not_list_weights: bool = not isinstance(weights, List)
    is_not_ndarray_weights: bool = not all(isinstance(weight, ndarray) for weight in weights)
    if is_not_list_weights or is_not_ndarray_weights:
        raise TypeError("Model weights must be of type `List[ndarray]`!")
    is_not_dict_metrics: bool = not isinstance(metrics, Dict)
    is_not_str_metrics: bool = not all(isinstance(key, str) for key in metrics.keys())
    is_not_ndarray_metrics: bool = not all(isinstance(value, ndarray) for value in metrics.values())
    if is_not_dict_metrics or is_not_str_metrics or is_not_ndarray_metrics:
        raise TypeError("Metrics must be of type `Dict[str, ndarray]`!")

    # return updated weights, number of training samples and metrics to the coordinator
    end_training_round(
        channel=channel, weights=weights, number_samples=number_samples, metrics=metrics
    )


class StateRecord:
    """Thread-safe record of a participant's state and round number."""

    def __init__(  # pylint: disable=redefined-builtin
        self, state: ParState = ParState.WAITING_FOR_SELECTION, round: int = 0
    ) -> None:
        """Initialize the state record.

        Args:
            state (~xain_sdk.participant_state_machine.ParState): The initial state. Defaults to
                WAITING_FOR_SELECTION.
            round (int): The initial training round. Defaults to 0.
        """

        self.cond: threading.Condition = threading.Condition()
        self.round: int = round
        self.state: ParState = state

    def lookup(self) -> Tuple[ParState, int]:
        """Get the state and round number.

        Returns:
            ~typing.Tuple[~xain_sdk.participant_state_machine.ParState, int]: The state and round
                number.
        """

        with self.cond:
            return self.state, self.round

    def update(self, state: ParState) -> None:
        """Update the state.

        Args:
            state (~xain_sdk.participant_state_machine.ParState): The state to update to.
        """

        with self.cond:
            self.state = state
            self.cond.notify()

    def wait_until_selected_or_done(self) -> ParState:
        """Wait until the participant was selected for training or is done.

        Returns:
            ~xain_sdk.participant_state_machine.ParState: The new state the participant is in.
        """

        with self.cond:
            self.cond.wait_for(lambda: self.state in {ParState.TRAINING, ParState.DONE})
            return self.state

    def wait_until_next_round(self) -> ParState:
        """Wait until the participant can start into the next round of training.

        Returns:
            ~xain_sdk.participant_state_machine.ParState: The new state the participant is in.
        """

        with self.cond:
            self.cond.wait_for(
                lambda: self.state
                in {ParState.TRAINING, ParState.WAITING_FOR_SELECTION, ParState.DONE}
            )
            return self.state


def transit(state_record: StateRecord, heartbeat_response: HeartbeatResponse) -> None:
    """Participant state transition function on a heartbeat response. Updates the state record.

    Args:
        state_record (~xain_sdk.participant_state_machine.StateRecord): The updatable state record
            of the participant.
        heartbeat_response (~xain_proto.fl.coordinator_pb2.HeartbeatResponse): The heartbeat
            response from the coordinator.
    """

    state: State = heartbeat_response.state
    round: int = heartbeat_response.round  # pylint: disable=redefined-builtin
    with state_record.cond:
        if state_record.state == ParState.WAITING_FOR_SELECTION:
            if state == State.ROUND:
                state_record.state = ParState.TRAINING
                state_record.round = round
                state_record.cond.notify()
            elif state == State.FINISHED:
                state_record.state = ParState.DONE
                state_record.cond.notify()
        elif state_record.state == ParState.POST_TRAINING:
            if state == State.STANDBY:
                # not selected
                state_record.state = ParState.WAITING_FOR_SELECTION
                # prob ok to keep state_record.round as it is
                state_record.cond.notify()
            elif state == State.ROUND and round == state_record.round + 1:
                state_record.state = ParState.TRAINING
                state_record.round = round
                state_record.cond.notify()
            elif state == State.FINISHED:
                state_record.state = ParState.DONE
                state_record.cond.notify()


def message_loop(channel: Channel, state_record: StateRecord, terminate: threading.Event) -> None:
    """Periodically send (and handle) heartbeat messages in a loop.

    Args:
        channel (~grpc.Channel): A gRPC channel to the coordinator.
        state_record (~xain_sdk.participant_state_machine.StateRecord): The participant's state
            record.
        terminate (~threading.Event): An event to terminate the message loop.
    """

    coordinator: CoordinatorStub = CoordinatorStub(channel=channel)
    while not terminate.is_set():
        request = HeartbeatRequest()
        transit(
            state_record=state_record, heartbeat_response=coordinator.Heartbeat(request=request),
        )
        time.sleep(HEARTBEAT_TIME)


def start_participant(participant: Participant, coordinator_url: str) -> None:
    """Top-level function for the participant's state machine.

    After rendezvous and heartbeat initiation, the Participant is WAITING_FOR_SELECTION. When
    selected, it moves to TRAINING followed by POST_TRAINING. If selected again for the next round,
    it moves back to TRAINING, otherwise it is back to WAITING_FOR_SELECTION.

    Args:
        participant (~xain_sdk.participant.Participant): The participant for local training.
        coordinator_url (str): The URL of the coordinator to connect to.
    """

    # use insecure channel for now

    with insecure_channel(target=coordinator_url) as channel:  # thread-safe
        rendezvous(channel=channel)

        state_record: StateRecord = StateRecord()
        terminate: threading.Event = threading.Event()
        msg_loop = threading.Thread(target=message_loop, args=(channel, state_record, terminate))
        msg_loop.start()

        # in WAITING_FOR_SELECTION state
        begin_selection_wait(state_record=state_record, channel=channel, participant=participant)

        # possibly several training rounds later... in DONE state
        terminate.set()
        msg_loop.join()


def begin_selection_wait(
    state_record: StateRecord, channel: Channel, participant: Participant
) -> None:
    """Perform actions in Participant state WAITING_FOR_SELECTION.

    Args:
        state_record (~xain_sdk.participant_state_machine.StateRecord): The participant's state
            record.
        channel (~grpc.Channel): A gRPC channel to the coordinator.
        participant (~xain_sdk.participant.Participant): The participant for local training.
    """

    state: ParState = state_record.wait_until_selected_or_done()
    if state == ParState.TRAINING:
        # selected
        begin_training(state_record=state_record, channel=channel, participant=participant)
    elif state == ParState.DONE:
        pass


def begin_training(state_record: StateRecord, channel: Channel, participant: Participant) -> None:
    """Perform actions in Participant state TRAINING and POST_TRAINING.

    Args:
        state_record (~xain_sdk.participant_state_machine.StateRecord): The participant's state
            record.
        channel (~grpc.Channel): A gRPC channel to the coordinator.
        participant (~xain_sdk.participant.Participant): The participant for local training.
    """

    # perform the training procedures
    training_round(channel=channel, participant=participant)

    # move to POST_TRAINING state
    state_record.update(state=ParState.POST_TRAINING)
    begin_post_training(state_record, channel, participant)


def begin_post_training(
    state_record: StateRecord, channel: Channel, participant: Participant
) -> None:
    """Perform actions in the Participant state POST_TRAINING.

    Args:
        state_record (~xain_sdk.participant_state_machine.StateRecord): The participant's state
            record.
        channel (~grpc.Channel): A gRPC channel to the coordinator.
        participant (~xain_sdk.participant.Participant): The participant for local training.
    """

    state: ParState = state_record.wait_until_next_round()
    if state == ParState.TRAINING:
        # selected again
        begin_training(state_record=state_record, channel=channel, participant=participant)
    elif state == ParState.WAITING_FOR_SELECTION:
        # not this time
        begin_selection_wait(state_record=state_record, channel=channel, participant=participant)
    elif state == ParState.DONE:
        # that was the last round
        pass
