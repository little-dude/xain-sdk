"""This module provides helpers for reading and validating the TOML
configuration.

"""

# Implementation notes:
# =====================
#
# We use the `schema` library to validate the configuration provided
# by the user. However, `schema` works mainly with dictionaries, which
# are not particularly convenient for us:
#
# - we cannot generate documentation for all the dictionary keys
# - the syntax for accessing values is quite verbose, especially if
#   the dictionary has nested values
#
# Thus, although we use schemas to validate configurations (COORDINATOR_SCHEMA
# STORAGE_SCHEMA, and LOGGING_SCHEMA), we don't expose them. Instead,
# we use them to dynamically generate classes where attributes are the
# schema keys: CoordinatorConfig, StorageConfig, and LoggingConfig. This
# hackery happens in create_class_from_schema(). It is not
# perfect. For instance, we cannot document the type of each
# attribute. But it is arguably better than using plain dictionaries.

from collections import namedtuple
import ipaddress
from typing import Any, Mapping, NamedTuple, Type, TypeVar, Union
import urllib.error
import urllib.parse
import uuid

import idna
from schema import And, Optional, Or, Schema, SchemaError, Use
import toml

from xain_sdk.logger import StructLogger, get_logger

logger: StructLogger = get_logger(__name__)


def error(key: str, description: str) -> str:
    """Return an error message for the given configuration item and
    description of the expected value type.

    Args:

        key (str): key of the configuration item
        description (str): description of the expected type of value
            for this configuration item
    """
    return f"Invalid `{key}`: value must be {description}"


def non_negative_integer(
    key: str, expected_value: str = "a positive integer"
) -> Schema:
    """Return a non-negative integer validator for the given configuration
    item.

    Args:

        key: key of the configuration item
        expected_value: description of the expected type of
            value for this configuration item

    """
    return And(int, lambda value: value >= 0, error=error(key, expected_value))


def url(key: str, expected_value: str = "a valid URL") -> Schema:
    """Return a URL validator for the given configuration item.

    Args:

        key: key of the configuration item
        expected_value: description of the expected type of
            value for this configuration item

    """

    def is_valid_url(value: str) -> Union[bool, And]:
        try:
            parsed = urllib.parse.urlparse(value)
        except (ValueError, urllib.error.URLError):
            return False
        # A URL is considered valid if it has at least a scheme and a
        # network location.
        return all([parsed.scheme, parsed.netloc])

    return And(str, is_valid_url, error=error(key, expected_value))


def is_valid_hostname(value: str) -> bool:
    """Return whether the given string is a valid hostname

    Args:

        value: string to check

    Returns:

        `True` if the given value is a valid hostname, `False`
        otherwise
    """
    try:
        idna.encode(value)
    except idna.IDNAError:
        return False
    return True


def is_valid_ip_address(value: str) -> bool:
    """Return whether the given string is a valid IP address

    Args:

        value: string to check

    Returns:

        `True` if the given value is a valid IP address, `False`
        otherwise
    """
    try:
        ipaddress.ip_address(value)
    except ipaddress.AddressValueError:
        return False
    return True


def hostname_or_ip_address(
    key: str, expected_value: str = "a valid domain name or IP address"
) -> Schema:
    """Return a hostname or IP address validator for the given
    configuration item.

    Args:

        key: key of the configuration item
        expected_value: description of the expected type of
            value for this configuration item

    """
    return And(
        str,
        Or(is_valid_hostname, is_valid_ip_address),
        error=error(key, expected_value),
    )


COORDINATOR_SCHEMA = Schema(
    {
        Optional("host", default="localhost"): hostname_or_ip_address(
            "coordinator.host"
        ),
        Optional("port", default=50051): non_negative_integer("coordinator.port"),
        Optional("grpc_options", default=dict): Use(
            lambda opt: list(opt.items()),
            error=error("coordinator.grpc_options", "valid gRPC options"),
        ),
    }
)

STORAGE_SCHEMA = Schema(
    {
        Optional("enable", default=False): Use(
            bool, error=error("storage.enable", "a boolean")
        ),
        "endpoint": And(str, url, error=error("storage.endpoint", "a valid URL")),
        "bucket": Use(str, error=error("storage.endpoint", "an S3 bucket name")),
        "secret_access_key": Use(
            str, error=error("storage.secret_access_key", "a valid utf-8 string")
        ),
        "access_key_id": Use(
            str, error=error("storage.access_key_id", "a valid utf-8 string")
        ),
        # FIXME(XP-515): each participant should write their data
        # under a unique key in the bucket. This key should come from
        # the coordinator but this part of the infrastructure is not
        # implemented, so when we create the storage configuration, we
        # generate a random key.
        Optional("directory", default=uuid.uuid4()): Use(
            uuid.UUID, error=error("storage.directory", "a unique directory ID")
        ),
    }
)


def log_level(key: str) -> Schema:
    """Return a validator for logging levels

    Args:

        key: key of the configuration item
    """
    log_levels = ["notset", "debug", "info", "warning", "error", "critical"]
    error_message = "one of: " + ", ".join(log_levels)
    log_level_validator = lambda value: value in log_levels
    return And(str, log_level_validator, error=error(key, error_message))


LOGGING_SCHEMA = Schema(
    {Optional("level", default="info"): log_level("logging.level"),}
)

CONFIG_SCHEMA = Schema(
    {
        Optional(
            "coordinator", default=COORDINATOR_SCHEMA.validate({})
        ): COORDINATOR_SCHEMA,
        "storage": STORAGE_SCHEMA,
        Optional("logging", default=LOGGING_SCHEMA.validate({})): LOGGING_SCHEMA,
    }
)


def create_class_from_schema(class_name: str, schema: Schema) -> Any:

    """Create a class named `class_name` from the given `schema`, where
    the attributes of the new class are the schema's keys.

    Args:

        class_name: name of the class to create
        schema: schema from which to create the class

    Returns:

        A new class where attributes are the given schema's keys
    """
    # pylint: disable=protected-access
    keys = schema._schema.keys()
    attributes = list(
        map(lambda key: key._schema if isinstance(key, Schema) else key, keys)  # type: ignore
    )
    return namedtuple(class_name, attributes)


# pylint: disable=invalid-name
CoordinatorConfig = create_class_from_schema("CoordinatorConfig", COORDINATOR_SCHEMA)
CoordinatorConfig.__doc__ = (
    "The coordinator configuration: TLS, addresses for incoming connections, etc."
)

StorageConfig = create_class_from_schema("StorageConfig", STORAGE_SCHEMA)
StorageConfig.__doc__ = (
    "Storage related configuration: storage endpoints and credentials, etc."
)

LoggingConfig = create_class_from_schema("LoggingConfig", LOGGING_SCHEMA)
LoggingConfig.__doc__ = "Logging related configuration: log level, colors, etc."

T = TypeVar("T", bound="Config")


class Config:
    """The coordinator configuration.

    Configuration is split in three sections: `Config.coordinator` for the
    coordinator to connect to, `Config.storage` for storage related
    items, and `Config.logging` for logging related configuration.

    The configuration is usually loaded from a dictionary the `Config`
    attributes map to the dictionary keys.

    Args:

        coordinator: the configuration corresponding to the `[coordinator]`
            section of the toml config file

        storage: the configuration corresponding to the `[storage]`
            section of the toml config file

        logging: the configuration corresponding to the `[logging]`
            section of the toml config file
    """

    def __init__(
        self, coordinator: NamedTuple, storage: NamedTuple, logging: NamedTuple,
    ):
        self.coordinator = coordinator
        self.storage = storage
        self.logging = logging

    @classmethod
    def from_unchecked_dict(cls: Type[T], dictionary: Mapping[str, Any]) -> T:
        """Check if the given dictionary contains a valid configuration, and
        if so, create a `Config` instance from it.

        Args:

            dictionary: a dictionary containing the configuration
        """
        try:
            valid_config = CONFIG_SCHEMA.validate(dictionary)
        except SchemaError as err:
            raise InvalidConfig(err.code) from err
        return cls.from_valid_dict(valid_config)

    @classmethod
    def from_valid_dict(cls: Type[T], dictionary: Mapping[str, Any]) -> T:
        """Create a `Config` instance for the given dictionary, assuming it
        contains a valid configuration

        Args:

            dictionary: a dictionary containing the configuration

        """
        return cls(
            CoordinatorConfig(**dictionary["coordinator"]),
            StorageConfig(**dictionary["storage"]),
            LoggingConfig(**dictionary["logging"]),
        )

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """Read the config file from the given path, check that it contains a
        valid configuration, and return the corresponding `Config`
        instance.

        Args:

            path: path to a configuration file
        """
        try:
            with open(path, "r") as f:
                raw_config = toml.load(f)
        except IsADirectoryError as err:
            raise InvalidConfig(f"{path} is a directory") from err
        except FileNotFoundError as err:
            raise InvalidConfig(f"{path} not found") from err
        except PermissionError as err:
            raise InvalidConfig(
                f"failed to read {path}: insufficient permissions"
            ) from err
        except toml.TomlDecodeError as err:
            raise InvalidConfig(f"failed to decode {path}: {err}") from err
        except OSError as err:
            raise InvalidConfig(str(err)) from err
        return cls.from_unchecked_dict(raw_config)


class InvalidConfig(ValueError):
    """
    Exception raised upon trying to load an invalid configuration
    """
