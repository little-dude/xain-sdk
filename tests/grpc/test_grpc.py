import sys
import threading
from unittest import mock

import pytest

from xain_sdk.grpc.participant import StateRecord, message_loop

# Some grpc tests fail on macos.
# `pytestmark` when defined on a module will mark all tests in that module.
# For more information check
# http://doc.pytest.org/en/latest/skipping.html#skip-all-test-functions-of-a-class-or-module
if sys.platform == "darwin":
    pytestmark = pytest.mark.xfail(reason="some grpc tests fail on macos")


@mock.patch("threading.Event.is_set", side_effect=[False, False, True])
@mock.patch("time.sleep", return_value=None)
@mock.patch("xain_sdk.grpc.coordinator_pb2.HeartbeatRequest")
def test_participant_heartbeat(mock_heartbeat_request, _mock_sleep, _mock_event):
    channel = mock.MagicMock()
    terminate_event = threading.Event()
    st = StateRecord()

    message_loop(channel, st, terminate_event)

    # check that the heartbeat is sent exactly twice
    mock_heartbeat_request.assert_has_calls([mock.call(), mock.call()])
