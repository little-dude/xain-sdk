"""Module implementing the networked Participant using gRPC."""

import threading
import time
from enum import Enum, auto
from typing import Dict, List, Tuple

import grpc
from numproto import ndarray_to_proto, proto_to_ndarray
from numpy import ndarray

from xain_sdk.grpc import coordinator_pb2
from xain_sdk.grpc.coordinator_pb2 import EndTrainingRequest, StartTrainingRequest
from xain_sdk.grpc.coordinator_pb2_grpc import CoordinatorStub
from xain_sdk.participant import Participant

RETRY_TIMEOUT: int = 5
HEARTBEAT_TIME: int = 10


# ATTENTION
# https://github.com/grpc/grpc/blob/master/doc/statuscodes.md


class ParState(Enum):
    """Enumeration of Participant states."""

    WAITING_FOR_SELECTION = auto()
    TRAINING = auto()
    POST_TRAINING = auto()
    DONE = auto()


def log_error(grpcError: grpc.RpcError) -> Tuple[str, str, Tuple[int, str]]:
    # print the gRPC error message set by server using
    # context.set_details(msg)
    msg = grpcError.details()
    status_code = grpcError.code()

    print(f"Participant received: `{msg}` with status code {status_code.name}.")

    return msg, status_code.name, status_code.value


def rendezvous(channel: grpc.Channel) -> None:
    """Starts a rendezvous exchange with Coordinator.

    Args:
        channel: gRPC channel to Coordinator.
    """
    stub: CoordinatorStub = CoordinatorStub(channel)  # type: ignore

    response = coordinator_pb2.RendezvousResponse.LATER

    while response == coordinator_pb2.RendezvousResponse.LATER:
        try:
            reply = stub.Rendezvous(coordinator_pb2.RendezvousRequest())
            if reply.response == coordinator_pb2.RendezvousResponse.ACCEPT:
                print("Participant received: ACCEPT")
            elif reply.response == coordinator_pb2.RendezvousResponse.LATER:
                print(f"Participant received: LATER. Retrying in {RETRY_TIMEOUT}s")
                time.sleep(RETRY_TIMEOUT)

            response = reply.response
        except grpc.RpcError as e:
            log_error(e)
            # In case we want to handle some of the status code differently
            # we can do it this way:
            # if grpc.StatusCode.INVALID_ARGUMENT == status_code:
            #     pass
            print(f"Participant received: ERROR. Retrying in {RETRY_TIMEOUT}s")
            break


def start_training(channel: grpc.Channel) -> Tuple[List[ndarray], int, int]:
    """Starts a training initiation exchange with Coordinator. Returns the decoded
    contents of the response from Coordinator.

    Args:
    channel: gRPC channel to Coordinator.

    Returns:
    obj:`List[~numpy.ndarray]`: Global model to train on.
    obj:`int`: Number of epochs.
    obj:`int`: Epoch base.
    """

    stub: CoordinatorStub = CoordinatorStub(channel)  # type: ignore
    request: StartTrainingRequest = StartTrainingRequest()
    # send request to start training
    reply = stub.StartTraining(request)
    print(f"Participant received: {type(reply)}")

    weights: List[ndarray]
    epochs: int
    epoch_base: int
    weights, epochs, epoch_base = reply.weights, reply.epochs, reply.epoch_base

    return [proto_to_ndarray(pnda) for pnda in weights], epochs, epoch_base


def end_training(
    channel: grpc.Channel,
    weights: List[ndarray],
    number_samples: int,
    metrics: Dict[str, List[ndarray]],
) -> None:
    """Starts a training completion exchange with the coordinator.

    Sends locally trained weights and the number of samples as update of the weights and metadata
    about metrics by starting a training completion exchange with the coordinator.

    Args:
    channel: gRPC channel to Coordinator.
    weight_update (obj:`Tuple[List[ndarray], int]`): Weights of the locally trained model and
    the number of samples.
    number_samples (obj: `int`): The number of samples in the training dataset.
    metrics (obj:`Dict[str, ~numpy.ndarray]`): Metrics metadata.
    """

    stub: CoordinatorStub = CoordinatorStub(channel)  # type: ignore
    # build request starting with weight update
    weights_proto = [ndarray_to_proto(nda) for nda in weights]

    # metric data containing the metric names mapped to Metrics as protobuf message
    metric: Dict[str, EndTrainingRequest.Metrics] = {
        key: EndTrainingRequest.Metrics(
            metrics=[ndarray_to_proto(value) for value in values]
        )
        for key, values in metrics.items()
    }

    # assembling a request with the update of the weights and the metrics
    request: EndTrainingRequest = EndTrainingRequest(
        weights=weights_proto, number_samples=number_samples, metrics=metric
    )
    reply = stub.EndTraining(request)
    print(f"Participant received: {type(reply)}")


def training_round(channel: grpc.Channel, participant: Participant) -> None:
    """Initiates training round exchange with Coordinator.
    Begins with `start_training`. Then performs local training computation using
    `participant`. Finally, completes with `end_training`.

    Args:
        channel: gRPC channel to Coordinator.
        participant (obj:`Participant`): Local Participant.
    """

    # retreiving global weights, epochs and base epochs from the coordinator response
    weights_global: List[ndarray]
    epochs: int
    epoch_base: int
    weights_global, epochs, epoch_base = start_training(channel)

    # starting a local training round of the participant
    weights: List[ndarray]
    number_samples: int
    metrics: Dict[str, List[ndarray]]
    weights, number_samples, metrics = participant.train_round(
        weights_global, epochs, epoch_base
    )
    end_training(channel, weights, number_samples, metrics)


class StateRecord:
    """Thread-safe record of Participant state and round number.
    """

    # pylint: disable=W0622
    def __init__(
        self, state: ParState = ParState.WAITING_FOR_SELECTION, round: int = 0
    ) -> None:
        self.cv = threading.Condition()  # pylint: disable=invalid-name
        self.round = round
        self.state = state

    def lookup(self) -> Tuple[ParState, int]:
        """Looks up the state and round number.

        Returns:
            :obj:`Tuple[ParState, int]`: State and round number
        """
        with self.cv:
            return self.state, self.round

    def update(self, state: ParState) -> None:
        """Updates state.

        Args:
            state (:obj:`ParState`): State to update to.
        """
        with self.cv:
            self.state = state
            self.cv.notify()

    def wait_until_selected_or_done(self) -> ParState:
        """Waits until Participant is in the state of having been selected for training
        (or is completely done).

        Returns:
            :obj:`ParState`: New state Participant is in.
        """
        with self.cv:
            self.cv.wait_for(lambda: self.state in {ParState.TRAINING, ParState.DONE})
            # which one was it?
            return self.state

    def wait_until_next_round(self) -> ParState:
        """Waits until Participant is in a state indicating the start of the next round
        of training.

        Returns:
            :obj:`ParState`: New state Participant is in.
        """
        with self.cv:
            self.cv.wait_for(
                lambda: self.state
                in {ParState.TRAINING, ParState.WAITING_FOR_SELECTION, ParState.DONE}
            )
            # which one was it?
            return self.state


def transit(  # pylint: disable=invalid-name
    st: StateRecord, beat_reply: coordinator_pb2.HeartbeatReply
) -> None:
    """Participant state transition function on a heartbeat response. Updates the
    state record `st`.

    Args:
        st (obj:`StateRecord`): Participant state record to update.
        beat_reply (obj:`coordinator_pb2.HeartbeatReply`): Heartbeat from Coordinator.
    """
    msg, r = beat_reply.state, beat_reply.round  # pylint: disable=invalid-name

    with st.cv:
        if st.state == ParState.WAITING_FOR_SELECTION:
            # We are currenly in WAITING_FOR_SELECTION

            if msg == coordinator_pb2.State.ROUND:
                # Server transitioned to ROUND
                # => we transition to TRAINING
                st.state = ParState.TRAINING
                st.round = r
                st.cv.notify()
            elif msg == coordinator_pb2.State.FINISHED:
                # Server transitioned to FINISHED
                # => we transition to DONE
                st.state = ParState.DONE
                st.cv.notify()

        elif st.state == ParState.POST_TRAINING:
            # We are currently in POST_TRANING

            if msg == coordinator_pb2.State.STANDBY:
                # Server transitioned to STANDBY
                # => we transition to WAITING_FOR_SELECTION

                # not selected
                st.state = ParState.WAITING_FOR_SELECTION
                # prob ok to keep st.round as it is
                st.cv.notify()

            elif msg == coordinator_pb2.State.ROUND and r == st.round + 1:
                # Server transitioned to ROUND (next round)
                # => we transition to TRAINING and update current round
                st.state = ParState.TRAINING
                st.round = r
                st.cv.notify()
            elif msg == coordinator_pb2.State.FINISHED:
                # Server transitioned to FINISHED
                # => we transition to DONE
                st.state = ParState.DONE
                st.cv.notify()


def message_loop(  # pylint: disable=invalid-name
    chan: grpc.Channel, st: StateRecord, terminate: threading.Event
) -> None:
    """Periodically sends (and handles) heartbeat messages in a loop.

    Args:
        chan: gRPC channel to Coordinator.
        st (obj:`StateRecord`): Participant state record.
        terminate (obj:`threading.Event`): Event to terminate message loop.
    """
    coord = CoordinatorStub(chan)  # type: ignore
    while not terminate.is_set():
        try:
            req = coordinator_pb2.HeartbeatRequest()
            reply = coord.Heartbeat(req)

            transit(st, reply)
            time.sleep(HEARTBEAT_TIME)
        except grpc.RpcError as e:
            log_error(e)
            # Suggestion:
            # Failing heartbeats should be handled differently based
            # on the status code. E.g. status code UNAVAILABLE might be
            # temporary and its reasonable to do a certain amount of
            # retries here. This should be discussed more throughly
            # for each potential error case.

            # This probably should be solved more intelligently
            # as even a simple network error might invalidate training
            # results.
            # As a sensible default we will just transition to WAITING_FOR_SELECTION
            st.state = ParState.WAITING_FOR_SELECTION
            st.cv.notify()


def begin_selection_wait(  # pylint: disable=invalid-name
    st: StateRecord, chan: grpc.Channel, part: Participant
) -> None:
    """Perform actions in Participant state WAITING_FOR_SELECTION.

    Args:
        st (obj:`StateRecord`): Participant state record.
        chan: gRPC channel to Coordinator.
        part (obj:`Participant`): Participant object for training computation.
    """
    ps = st.wait_until_selected_or_done()
    if ps == ParState.TRAINING:
        # selected
        begin_training(st, chan, part)
    elif ps == ParState.DONE:
        pass


def begin_training(  # pylint: disable=invalid-name
    st: StateRecord, chan: grpc.Channel, part: Participant
) -> None:
    """Perform actions in Participant state TRAINING and POST_TRAINING.

    Args:
        st (obj:`StateRecord`): Participant state record.
        chan: gRPC channel to Coordinator.
        part (obj:`Participant`): Participant object for training computation.
    """
    # perform the training procedures
    training_round(chan, part)
    # move to POST_TRAINING state
    st.update(ParState.POST_TRAINING)
    ps = st.wait_until_next_round()  # pylint: disable=invalid-name
    if ps == ParState.TRAINING:
        # selected again
        begin_training(st, chan, part)
    elif ps == ParState.WAITING_FOR_SELECTION:
        # not this time
        begin_selection_wait(st, chan, part)
    elif ps == ParState.DONE:
        # that was the last round
        pass


def start_participant(
    part: Participant, coordinator_url: str
) -> None:  # pylint: disable=invalid-name
    """Top-level function for the Participant state machine.
    After rendezvous and heartbeat initiation, the Participant is
    WAITING_FOR_SELECTION. When selected, it moves to TRAINING followed by
    POST_TRAINING. If selected again for the next round, it moves back to
    TRAINING, otherwise it is back to WAITING_FOR_SELECTION.

    Args:
        part (obj:`Participant`): Participant object for training computation.
        coordinator_url (obj:`str`): The URL of the coordinator to connect to.
    """
    # use insecure channel for now
    with grpc.insecure_channel(target=coordinator_url) as chan:  # thread-safe
        rendezvous(chan)

        st = StateRecord()  # pylint: disable=invalid-name
        terminate = threading.Event()
        ml = threading.Thread(  # pylint: disable=invalid-name
            target=message_loop, args=(chan, st, terminate)
        )
        ml.start()

        try:
            # in WAITING_FOR_SELECTION state
            begin_selection_wait(st, chan, part)
        except grpc.RpcError as e:
            log_error(e)

        # possibly several training rounds later...
        # in DONE state
        terminate.set()
        ml.join()
