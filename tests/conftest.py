from concurrent import futures
from typing import Generator

import pytest


@pytest.fixture
def client_thread_pool() -> Generator:
    pool = futures.ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=False)
