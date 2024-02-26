from requests.exceptions import ConnectionError

from urllib.parse import urljoin

import pytest
import requests


def is_responsive(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
    except ConnectionError:
        return False


@pytest.fixture(scope="session")
def pytorch_service(docker_ip, docker_services):
    """Ensure that HTTP service is up and responsive."""

    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("customservice", 8080)
    url = "http://{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive(url)
    )
    return url


model_name = "kserve-demo-model"


class TestPytorchRuntime:
    @pytest.mark.parametrize(
        "url, expected_response",
        [
            (
                f"/v2/models/{model_name}",
                dict(
                    name=model_name,
                    versions=None,
                    platform="",
                    inputs=[],
                    outputs=[],
                ),
            ),
            ("/v2/health/ready", dict(ready=True)),
            ("/v2/health/live", dict(live=True)),
            (
                f"/v2/models/{model_name}/ready",
                dict(name=model_name, ready=True),
            ),
        ],
    )
    def test_ready_endpoint(
        self, url: str, expected_response: dict, pytorch_service: str
    ) -> None:
        """
        Test all the health endpoints are working.
        1. Model metadata (v2/models/<model_name>)
        2. Server Ready (v2/health/ready)
        3. Server Live (v2/health/live)
        4. Server Metadata (v2)
        4. Model Ready (v2/models/<model_name>/ready)
        Args:
            pytorch_service: Pytorch service URL in docker-compose.

        Returns:
            None
        """
        health_url = urljoin(pytorch_service, url)
        response = requests.get(health_url)
        assert response.status_code == 200
        assert response.json() == expected_response
