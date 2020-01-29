"""This module provides classes for weights storage. It currently only
works with services that provides the AWS S3 APIs.

"""
import abc
import pickle
from typing import List
import uuid

import boto3
from numpy import ndarray


class AbstractStore(abc.ABC):

    """An abstract class that defines the API a store must implement.

    """

    # TODO(XP-515): in the future, this method's parameters will
    # differ. For instance it needs to take
    @abc.abstractmethod
    def write_weights(self, round: int, weights: List[ndarray]) -> None:
        """Store the given `weights`, corresponding to the given `round`.

        Args:

            round: round number the weights correspond to
            weights: weights to store

        """


# FIXME(XP-515): this class is temporary. The storage information
# should come from the coordinator.
#
# pylint: disable=too-few-public-methods
class S3StorageConfig:
    """
    Storage service configuration

    Args:

        endpoint_url: URL of the storage service
        secret_access_key: AWS secret access key for the storage service
        access_key_id: AWS access key ID for the storage service
        bucket: Name of the bucket to store the weights into
    """

    def __init__(
        self, endpoint_url: str, access_key_id: str, secret_access_key: str, bucket: str,
    ):
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket = bucket

        # FIXME(XP-515): each participant should write their data
        # under a unique key in the bucket. This key should come from
        # the coordinator but this part of the infrastructure is not
        # implemented, so when we create the storage configuration, we
        # generate a random key.
        self.directory = uuid.uuid4()


class S3Store(AbstractStore):
    """A store for services that offer the AWS S3 API.

    Args:

        config: the storage configuration (endpoint URL, credentials,
            etc.)

    """

    def __init__(self, config: S3StorageConfig):
        self.config = config
        # pylint: disable=invalid-name
        self.s3 = boto3.resource(
            "s3",
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            # FIXME(XP-515): not sure what this should be for now
            region_name="dummy",
        )

    def write_weights(self, round: int, weights: List[ndarray]) -> None:
        """Store the given `weights`, corresponding to the given `round`.

        Args:

            round: round number the weights correspond to
            weights: weights to store

        """
        bucket = self.s3.Bucket(self.config.bucket)
        bucket.put_object(Body=pickle.dumps(weights), Key=f"{self.config.directory}/{round}")


# FIXME(XP-515): Storage is a highly experimental feature so we do not
# want to enable by default. Therefore, we provide this dummy class
# that can be used by participants that does not want to use a real
# storage service.
class DummyStore(AbstractStore):
    """A dummy store that does not do anything

    """

    def write_weights(self, _round: int, _weights: List[ndarray]) -> None:
        """Return without doing anything.

        Args:

            _round: round number the weights correspond to (unused)
            _weights: weights to store (unused)

        """
