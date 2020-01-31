"""Tests for the participant state machine."""

from unittest import mock

import pytest
from xain_proto.fl.coordinator_pb2 import HeartbeatResponse, State

from xain_sdk.participant_state_machine import (
    ParState,
    StateRecord,
    begin_training,
    begin_waiting,
    transit,
)


def test_from_start() -> None:
    """Test start."""

    state_record: StateRecord = StateRecord()
    assert state_record.lookup() == (ParState.WAITING, -1)

    heartbeat_response: HeartbeatResponse = HeartbeatResponse(state=State.ROUND)
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (ParState.TRAINING, 0)

    # should return immediately
    assert state_record.wait_until_selected_or_done() == ParState.TRAINING


def test_waiting_to_training() -> None:
    """Test waiting to training."""

    state_record: StateRecord = StateRecord(state=ParState.WAITING, round=1)
    # trained round 1, selected for round 3
    heartbeat_response: HeartbeatResponse = HeartbeatResponse(state=State.ROUND, round=3)
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (ParState.TRAINING, 3)

    # should return immediately
    assert state_record.wait_until_selected_or_done() == ParState.TRAINING


def test_waiting_to_done() -> None:
    """Test waiting to done."""

    state_record = StateRecord(state=ParState.WAITING, round=2)
    heartbeat_response: HeartbeatResponse = HeartbeatResponse(state=State.FINISHED)
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (ParState.DONE, 2)

    # should return immediately
    assert state_record.wait_until_selected_or_done() == ParState.DONE


def test_waiting_to_waiting() -> None:
    """Test waiting to waiting."""

    # not selected
    state_record: StateRecord = StateRecord(state=ParState.WAITING, round=3)
    heartbeat_response: HeartbeatResponse = HeartbeatResponse(state=State.STANDBY)
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (ParState.WAITING, 3)

    # round still open
    heartbeat_response.state = State.ROUND
    heartbeat_response.round = 3
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (ParState.WAITING, 3)

    # selected for past round(!)
    heartbeat_response.state = State.ROUND
    heartbeat_response.round = 1
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (ParState.WAITING, 3)


def test_training_to_training() -> None:
    """Test training to training."""

    state_record: StateRecord = StateRecord(state=ParState.TRAINING, round=4)
    start_state, round_num = state_record.lookup()
    assert isinstance(start_state, ParState)
    # heartbeats essentially get ignored in training state...

    heartbeat_response: HeartbeatResponse = HeartbeatResponse(state=State.STANDBY)
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (start_state, round_num)

    heartbeat_response.state = State.ROUND
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (start_state, round_num)

    heartbeat_response.state = State.FINISHED
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (start_state, round_num)


def test_dropout_scenario() -> None:
    """Test the scenario of participant drop-outs."""

    # P is done with training round 5
    state_record: StateRecord = StateRecord(ParState.WAITING, 5)

    # some other participants drop out => coordinator back to STANDBY
    heartbeat_response: HeartbeatResponse = HeartbeatResponse(state=State.STANDBY)
    transit(state_record=state_record, heartbeat_response=heartbeat_response)
    assert state_record.lookup() == (ParState.WAITING, 5)

    # enough participants replace dropouts => coordinator back to ROUND 5
    heartbeat_response.state = State.ROUND
    heartbeat_response.round = 5
    transit(state_record=state_record, heartbeat_response=heartbeat_response)

    # P just continues to wait
    assert state_record.lookup() == (ParState.WAITING, 5)


@mock.patch("xain_sdk.participant_state_machine.begin_training")
def test_selection_wait_done(mock_begin_training: mock.Mock) -> None:
    """Tests that WAITING goes to DONE if state record says so."""

    state_rec = StateRecord(state=ParState.DONE)
    _chan, _part = mock.MagicMock(), mock.MagicMock()
    begin_waiting(state_rec, _chan, _part)
    par_state, _ = state_rec.lookup()

    assert par_state == ParState.DONE
    # DONE means done! should *not* begin training
    mock_begin_training.assert_not_called()


@mock.patch("xain_sdk.participant_state_machine.begin_training")
def test_selection_wait_training(mock_begin_training: mock.Mock) -> None:
    """Tests that WAITING goes to TRAINING if state record says so."""

    state_rec = StateRecord(state=ParState.TRAINING)
    _chan, _part = mock.MagicMock(), mock.MagicMock()
    begin_waiting(state_rec, _chan, _part)

    # training should have begun
    mock_begin_training.assert_called_once_with(state_rec, _chan, _part)


@pytest.mark.parametrize("parstate", [ParState.TRAINING, ParState.DONE, ParState.WAITING])
@mock.patch("xain_sdk.participant_state_machine.training_round")
@mock.patch("xain_sdk.participant_state_machine.begin_waiting")
def test_begin_training(
    mock_begin_waiting: mock.Mock, mock_training_round: mock.Mock, parstate: ParState
) -> None:
    """Tests that TRAINING goes to WAITING independent of what state record says."""

    # state record says TRAINING
    state_rec = StateRecord(state=parstate)
    _chan, _part = mock.MagicMock(), mock.MagicMock()
    begin_training(state_rec, _chan, _part)
    par_state, round = state_rec.lookup()

    assert par_state == ParState.WAITING
    mock_training_round.assert_called_once_with(_chan, _part, round)
    mock_begin_waiting.assert_called_once_with(state_rec, _chan, _part)
