import os

import pytest


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    """Define the docker compose file.

    Args:
        pytestconfig: The pytest config.

    Returns:
        The docker compose file.
    """
    return os.path.join(
        str(pytestconfig.rootdir),
        "test",
        "integration",
        "docker-compose.yaml",
    )
