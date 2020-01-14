"""Tests for GRPC Participant."""

from unittest import mock

from xain_proto.fl.coordinator_pb2 import HeartbeatReply, State

from xain_sdk.participant_state_machine import (
    ParState,
    StateRecord,
    begin_post_training,
    begin_selection_wait,
    transit,
)


def test_from_start() -> None:
    """Test start."""

    state_record: StateRecord = StateRecord()
    assert state_record.lookup() == (ParState.WAITING_FOR_SELECTION, 0)

    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.ROUND)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (ParState.TRAINING, 0)

    # should return immediately
    assert state_record.wait_until_selected_or_done() == ParState.TRAINING


def test_waiting_to_training_i() -> None:
    """Test waiting to training."""

    state_record: StateRecord = StateRecord(state=ParState.WAITING_FOR_SELECTION)
    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.ROUND, round=1)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (ParState.TRAINING, 1)

    # should return immediately
    assert state_record.wait_until_selected_or_done() == ParState.TRAINING


def test_waiting_to_done() -> None:
    """Test waiting to done."""

    state_record = StateRecord(state=ParState.WAITING_FOR_SELECTION, round=2)
    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.FINISHED)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (ParState.DONE, 2)

    # should return immediately
    assert state_record.wait_until_selected_or_done() == ParState.DONE


def test_waiting_to_waiting() -> None:
    """Test waiting to waiting."""

    state_record: StateRecord = StateRecord(state=ParState.WAITING_FOR_SELECTION, round=3)
    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.STANDBY)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (ParState.WAITING_FOR_SELECTION, 3)


def test_training_to_training() -> None:
    """Test training to training."""

    state_record: StateRecord = StateRecord(state=ParState.TRAINING, round=4)
    start_state, round_num = state_record.lookup()
    assert isinstance(start_state, ParState)
    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.STANDBY)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (start_state, round_num)

    heartbeat_reply.state = State.ROUND
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (start_state, round_num)

    heartbeat_reply.state = State.FINISHED
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (start_state, round_num)


def test_posttraining_to_training() -> None:
    """Test postraining to training."""

    state_record: StateRecord = StateRecord(state=ParState.POST_TRAINING, round=5)
    start_state, round_num = state_record.lookup()
    assert isinstance(start_state, ParState)
    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.ROUND, round=5)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (start_state, round_num)

    # old round? shouldn't affect me...
    heartbeat_reply.round = 0
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (start_state, round_num)

    # NOTE a "future" round e.g. 7 would be unexpected under current assumptions
    # it should be preceded by a STANDBY to indicate nonselection for round 6

    # selected for next round
    heartbeat_reply.round = 6
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (ParState.TRAINING, 6)

    # should return immediately
    assert state_record.wait_until_next_round() == ParState.TRAINING


def test_posttraining_to_done() -> None:
    """Test posttraining to done."""

    state_record: StateRecord = StateRecord(state=ParState.POST_TRAINING, round=6)
    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.FINISHED)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (ParState.DONE, 6)
    # should return immediately
    assert state_record.wait_until_next_round() == ParState.DONE


def test_posttraining_to_waiting() -> None:
    """Test posttraining to waiting."""

    state_record: StateRecord = StateRecord(state=ParState.POST_TRAINING, round=7)
    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.STANDBY)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (ParState.WAITING_FOR_SELECTION, 7)
    # should return immediately
    assert state_record.wait_until_next_round() == ParState.WAITING_FOR_SELECTION


def test_restart_round() -> None:
    """Test restart."""

    # participant has done its training for round 8
    state_record: StateRecord = StateRecord(state=ParState.POST_TRAINING, round=8)
    # it's told to go into waiting
    heartbeat_reply: HeartbeatReply = HeartbeatReply(state=State.STANDBY)
    transit(state_record=state_record, heartbeat_reply=heartbeat_reply)
    assert state_record.lookup() == (ParState.WAITING_FOR_SELECTION, 8)

    # and back again to training...
    heartbeat_reply.state = State.ROUND
    heartbeat_reply.round = 8  # but still in round 8!
    # => interpret this as "round restarted" e.g. original theta was corrupt or something
    transit(state_record, heartbeat_reply)
    # => re-do the training...
    assert state_record.lookup() == (ParState.TRAINING, 8)


@mock.patch("xain_sdk.participant_state_machine.begin_training")
def test_selection_wait_done(mock_begin_training):
    """Test effect of selection-waiting until DONE state."""

    state_rec = StateRecord(state=ParState.DONE)
    chan, part = mock.MagicMock(), mock.MagicMock()
    begin_selection_wait(state_rec, chan, part)
    p_st, _ = state_rec.lookup()
    assert p_st == ParState.DONE
    mock_begin_training.assert_not_called()


@mock.patch("xain_sdk.participant_state_machine.begin_training")
def test_selection_wait_training(mock_begin_training):
    """Test effect of selection-waiting until TRAINING state."""

    state_rec = StateRecord(state=ParState.TRAINING)
    chan, part = mock.MagicMock(), mock.MagicMock()
    begin_selection_wait(state_rec, chan, part)
    mock_begin_training.assert_called_once_with(
        state_record=state_rec, channel=chan, participant=part
    )


@mock.patch("xain_sdk.participant_state_machine.begin_training")
@mock.patch("xain_sdk.participant_state_machine.begin_selection_wait")
def test_post_training_training(mock_begin_selection_wait, mock_begin_training):
    """Test effect of post-train-waiting until TRAINING state."""

    state_rec = StateRecord(state=ParState.TRAINING)
    chan, part = mock.MagicMock(), mock.MagicMock()
    begin_post_training(state_rec, chan, part)
    mock_begin_training.assert_called_once_with(
        state_record=state_rec, channel=chan, participant=part
    )
    mock_begin_selection_wait.assert_not_called()


@mock.patch("xain_sdk.participant_state_machine.begin_training")
@mock.patch("xain_sdk.participant_state_machine.begin_selection_wait")
def test_post_training_waiting(mock_begin_selection_wait, mock_begin_training):
    """Test effect of post-train-waiting until WAITING_FOR_SELECTION state."""

    state_rec = StateRecord(state=ParState.WAITING_FOR_SELECTION)
    chan, part = mock.MagicMock(), mock.MagicMock()
    begin_post_training(state_rec, chan, part)
    mock_begin_selection_wait.assert_called_once_with(
        state_record=state_rec, channel=chan, participant=part
    )
    mock_begin_training.assert_not_called()


@mock.patch("xain_sdk.participant_state_machine.begin_training")
@mock.patch("xain_sdk.participant_state_machine.begin_selection_wait")
def test_post_training_done(mock_begin_selection_wait, mock_begin_training):
    """Test effect of post-train-waiting until DONE state."""

    state_rec = StateRecord(state=ParState.DONE)
    chan, part = mock.MagicMock(), mock.MagicMock()
    begin_post_training(state_rec, chan, part)
    mock_begin_training.assert_not_called()
    mock_begin_selection_wait.assert_not_called()
