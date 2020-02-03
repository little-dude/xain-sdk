"""This module provides classes for weights storage.

It currently only works with services that provides the AWS S3 APIs.
"""

import abc
import pickle
from typing import List
import uuid

import boto3
from numpy import ndarray


class AbstractStore(abc.ABC):
    """An abstract class that defines the API a store must implement."""

    # TODO(XP-515): in the future, this method's parameters will
    # differ. For instance it needs to take
    @abc.abstractmethod
    def write_weights(self, round: int, weights: ndarray) -> None:
        """Store the given `weights`, corresponding to the given `round`.

        Args:
            round: The round number the weights correspond to.
            weights: The weights to store.
        """


# FIXME(XP-515): this class is temporary. The storage information
# should come from the coordinator.
class S3StorageConfig:
    """Storage service configuration.

    Args:
        endpoint_url: An URL of the storage service.
        access_key_id: An AWS access key ID for the storage service.
        secret_access_key: An AWS secret access key for the storage service.
        bucket: A name of the bucket to store the weights into.
        directory: A unique directory ID.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
    ) -> None:
        """Initialize the S3 storage configuration.

        Args:
            endpoint_url: An URL of the storage service.
            access_key_id: An AWS access key ID for the storage service.
            secret_access_key: An AWS secret access key for the storage service.
            bucket: A name of the bucket to store the weights into.
        """

        self.endpoint_url: str = endpoint_url
        self.access_key_id: str = access_key_id
        self.secret_access_key: str = secret_access_key
        self.bucket: str = bucket

        # FIXME(XP-515): each participant should write their data
        # under a unique key in the bucket. This key should come from
        # the coordinator but this part of the infrastructure is not
        # implemented, so when we create the storage configuration, we
        # generate a random key.
        self.directory: uuid.UUID = uuid.uuid4()


class S3Store(AbstractStore):
    """A store for services that offer the AWS S3 API.

    Args:
        config: The storage configuration (endpoint URL, credentials, etc.).
        s3: The S3 bucket.
    """

    def __init__(self, config: S3StorageConfig):
        """Initialize the S3 storage.

        Args:
            config: The storage configuration.
        """

        self.config: S3StorageConfig = config
        self.s3 = boto3.resource(  # pylint: disable=invalid-name
            "s3",
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            region_name="dummy",  # FIXME(XP-515): not sure what this should be for now
        )

    def write_weights(self, round: int, weights: ndarray) -> None:
        """Store the given `weights`, corresponding to the given `round`.

        Args:
            round: The round number the weights correspond to.
            weights: The weights to store.
        """

        bucket = self.s3.Bucket(self.config.bucket)
        bucket.put_object(
            Body=pickle.dumps(weights), Key=f"{self.config.directory}/{round}"
        )


# FIXME(XP-515): Storage is a highly experimental feature so we do not
# want to enable by default. Therefore, we provide this dummy class
# that can be used by participants that does not want to use a real
# storage service.
class DummyStore(AbstractStore):
    """A dummy store that does not do anything."""

    def write_weights(self, round: int, weights: List[ndarray]) -> None:
        """Return without doing anything.

        Args:
            round: The round number the weights correspond to (unused).
            weights: The weights to store (unused).
        """
