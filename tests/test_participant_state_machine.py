"""Tests for GRPC Participant."""

import time
from concurrent import futures
from typing import Any
from unittest import mock

import grpc
import grpc_testing
import pytest
from google.protobuf import descriptor as _descriptor

import xain_sdk.cproto.coordinator_pb2_grpc
import xain_sdk.participant_state_machine
from xain_sdk.cproto.coordinator_pb2 import (
    DESCRIPTOR,
    RendezvousReply,
    RendezvousRequest,
    RendezvousResponse,
)
from xain_sdk.participant_state_machine import rendezvous


def take_method_description_by_name(name):
    methods = [
        md
        for md in DESCRIPTOR.services_by_name["Coordinator"].methods
        if md.name == name
    ]

    return methods[0]


def send_rendezvous_reply(channel, msg, status_code, details=""):
    invocation_metadata, request, rpc = channel.take_unary_unary(
        take_method_description_by_name("Rendezvous")
    )

    rpc.send_initial_metadata(())
    rpc.terminate(msg, (), status_code, details)


def test_rendezvous_OK(client_thread_pool, monkeypatch):
    # Reduce timeout to zero to speedup test
    monkeypatch.setattr(xain_sdk.participant_state_machine, "RETRY_TIMEOUT", 0)

    real_time = grpc_testing.strict_real_time()
    real_time_channel = grpc_testing.channel(
        DESCRIPTOR.services_by_name.values(), real_time
    )

    future = client_thread_pool.submit(rendezvous, channel=real_time_channel)

    send_rendezvous_reply(
        channel=real_time_channel,
        msg=RendezvousReply(response=RendezvousResponse.LATER),
        status_code=grpc.StatusCode.OK,
    )
    send_rendezvous_reply(
        channel=real_time_channel,
        msg=RendezvousReply(response=RendezvousResponse.ACCEPT),
        status_code=grpc.StatusCode.OK,
    )

    result = future.result()
    assert result is None


def test_rendezvous_INTERNAL(client_thread_pool, monkeypatch):
    # Reduce timeout to zero to speedup test
    monkeypatch.setattr(xain_sdk.participant_state_machine, "RETRY_TIMEOUT", 0)

    real_time = grpc_testing.strict_real_time()
    real_time_channel = grpc_testing.channel(
        DESCRIPTOR.services_by_name.values(), real_time
    )

    future = client_thread_pool.submit(rendezvous, channel=real_time_channel)

    send_rendezvous_reply(
        channel=real_time_channel, msg=None, status_code=grpc.StatusCode.INTERNAL,
    )

    with pytest.raises(grpc.RpcError):
        future.result()
