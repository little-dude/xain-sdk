"""
Tests for the `xain_sdk.config.Config` class.
"""

import re
from types import TracebackType
from typing import Optional, Pattern

import pytest

from xain_sdk.config import (
    Config,
    CoordinatorConfig,
    InvalidConfig,
    LoggingConfig,
    StorageConfig,
)


@pytest.fixture
def coordinator_sample() -> dict:
    """
    Return a valid "coordinator" section
    """
    return {
        "host": "localhost",
        "port": 50051,
        "grpc_options": {
            "grpc.max_receive_message_length": -1,
            "grpc.max_send_message_length": -1,
        },
    }


@pytest.fixture
def storage_sample() -> dict:
    """
    Return a valid "storage" section
    """
    return {
        "enable": False,
        "endpoint": "http://localhost:9000",
        "bucket": "aggregated_weights",
        "secret_access_key": "my-secret",
        "access_key_id": "my-key-id",
    }


@pytest.fixture
def logging_sample() -> dict:
    """
    Return a valid "logging" section
    """
    return {
        "level": "debug",
    }


@pytest.fixture
def config_sample(  # pylint: disable=redefined-outer-name
    coordinator_sample: dict, storage_sample: dict, logging_sample: dict,
) -> dict:
    """
    Return a valid config
    """
    return {
        "coordinator": coordinator_sample,
        "storage": storage_sample,
        "logging": logging_sample,
    }


def test_default_logging_config(  # pylint: disable=redefined-outer-name
    config_sample: dict,
) -> None:
    """Check that the config loads if the [logging] section is not
    specified

    """
    del config_sample["logging"]
    config = Config.from_unchecked_dict(config_sample)
    config_logging: LoggingConfig = config.logging
    assert config_logging.level == "info"

    config_sample["logging"] = {}
    config = Config.from_unchecked_dict(config_sample)
    config_logging = config.logging
    assert config_logging.level == "info"


def test_invalid_logging_config(
    config_sample: dict,  # pylint: disable=redefined-outer-name
) -> None:
    """Various negative cases for the [logging] section"""
    config_sample["logging"] = {"level": "invalid"}

    with AssertInvalid() as err:
        Config.from_unchecked_dict(config_sample)

    err.check_other(
        re.compile(
            "`logging.level`: value must be one of: notset, debug, info, warning, error, critical"
        )
    )


def test_load_valid_config(
    # pylint: disable=redefined-outer-name
    config_sample: dict,
) -> None:
    """
    Check that a valid config is loaded correctly
    """
    config = Config.from_unchecked_dict(config_sample)

    config_coord: CoordinatorConfig = config.coordinator
    assert config_coord.host == "localhost"
    assert config_coord.port == 50051

    assert config_coord.grpc_options == [
        ("grpc.max_receive_message_length", -1),
        ("grpc.max_send_message_length", -1),
    ]

    config_storage: StorageConfig = config.storage
    assert config_storage.enable is False
    assert config_storage.endpoint == "http://localhost:9000"
    assert config_storage.bucket == "aggregated_weights"
    assert config_storage.secret_access_key == "my-secret"
    assert config_storage.access_key_id == "my-key-id"


def test_coordinator_config_ip_address(  # pylint: disable=redefined-outer-name
    config_sample: dict, coordinator_sample: dict,
) -> None:
    """Check that the config is loaded correctly when the `coordinator.host`
    key is an IP address

    """
    # Ipv4 host
    coordinator_sample["host"] = "1.2.3.4"
    config_sample["coordinator"] = coordinator_sample
    config = Config.from_unchecked_dict(config_sample)
    config_coord: CoordinatorConfig = config.coordinator
    assert config_coord.host == coordinator_sample["host"]

    # Ipv6 host
    coordinator_sample["host"] = "::1"
    config_sample["coordinator"] = coordinator_sample
    config = Config.from_unchecked_dict(config_sample)
    config_coord = config.coordinator
    assert config_coord.host == coordinator_sample["host"]


def test_coordinator_config_extra_key(  # pylint: disable=redefined-outer-name
    config_sample: dict, coordinator_sample: dict,
) -> None:
    """Check that the config is rejected when the coordinator section contains
    an extra key

    """
    coordinator_sample["extra-key"] = "foo"
    config_sample["coordinator"] = coordinator_sample

    with AssertInvalid() as err:
        Config.from_unchecked_dict(config_sample)

    err.check_section("coordinator")
    err.check_extra_key("extra-key")


def test_coordinator_config_invalid_host(  # pylint: disable=redefined-outer-name
    config_sample: dict, coordinator_sample: dict,
) -> None:
    """Check that the config is rejected when the `coordinator.host` key is
    invalid.

    """
    coordinator_sample["host"] = 1.0
    config_sample["coordinator"] = coordinator_sample

    with AssertInvalid() as err:
        Config.from_unchecked_dict(config_sample)

    err.check_other(
        re.compile(
            "Invalid `coordinator.host`: value must be a valid domain name or IP address"
        )
    )


def test_server_config_valid_ipv6(  # pylint: disable=redefined-outer-name
    config_sample: dict, coordinator_sample: dict,
) -> None:
    """Check some edge cases with IPv6 `server.host` key"""
    coordinator_sample["host"] = "::"
    config_sample["coordinator"] = coordinator_sample
    config = Config.from_unchecked_dict(config_sample)
    config_coord: CoordinatorConfig = config.coordinator
    assert config_coord.host == coordinator_sample["host"]

    coordinator_sample["host"] = "fe80::"
    config_sample["coordinator"] = coordinator_sample
    config = Config.from_unchecked_dict(config_sample)
    config_coord = config.coordinator
    assert config_coord.host == coordinator_sample["host"]


# Adapted from unittest's assertRaises
class AssertInvalid:
    """A context manager that checks that an `xainfl.config.InvalidConfig`
    exception is raised, and provides helpers to perform checks on the
    exception.

    """

    def __init__(self) -> None:
        self.message: Optional[str] = None

    def __enter__(self) -> "AssertInvalid":
        return self

    def __exit__(self, exc_type: type, exc_value: None, _tb: TracebackType) -> bool:
        if exc_type is None:
            raise Exception("Did not get an exception")
        if not isinstance(exc_value, InvalidConfig):
            # let this unexpected exception be re-raised
            return False

        self.message = str(exc_value)

        return True

    def check_section(self, section: str) -> None:
        """Check that the error message mentions the given section of the
        configuration file.

        """

        needle: Pattern = re.compile(f"Key '{section}' error:")
        assert re.search(needle, self.message)

    def check_extra_key(self, key: str) -> None:
        """Check that the error mentions the given configuration key"""
        needle: Pattern = re.compile(f"Wrong key '{key}' in")
        assert re.search(needle, self.message)

    def check_other(self, needle: Pattern) -> None:
        """Check that the error message matches the given pattern.

        """
        assert re.search(needle, self.message)
