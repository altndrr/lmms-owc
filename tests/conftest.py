import dotenv
import pytest


@pytest.fixture(scope="session", autouse=True)
def load_dotenv() -> None:
    """Load the environment variables from the .env file."""
    dotenv.load_dotenv(dotenv.find_dotenv())


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options for running tests."""
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow unit tests")


def pytest_configure(config: pytest.Config) -> None:
    """Add custom markers for tests."""
    config.addinivalue_line("markers", "slow: mark as slow, skipping them by default")
