import pytest
import os
import json
from pathlib import Path


def pytest_configure(config):
    """Configure pytest for integration tests."""
    # Mark integration tests
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly requested."""
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(
            reason="need --integration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add integration test option."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )


@pytest.fixture(scope="session")
def aws_credentials():
    """Check for AWS credentials."""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_DEFAULT_REGION'
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing AWS credentials: {', '.join(missing)}")


@pytest.fixture(scope="session")
def test_data_dir():
    """Get path to test data directory."""
    data_dir = Path(__file__).parent / 'test_data'
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture(scope="session")
def integration_config(test_data_dir):
    """Load integration test configuration."""
    config_path = test_data_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def pytest_sessionstart(session):
    """Set up integration test session."""
    # Create test results directory
    results_dir = Path("test_results")
    if not results_dir.exists():
        results_dir.mkdir()


def pytest_sessionfinish(session, exitstatus):
    """Clean up after integration test session."""
    # Archive test results
    results_dir = Path("test_results")
    if results_dir.exists():
        # Keep only last 5 results
        results = sorted(results_dir.glob('*.json'))[:-5]
        for old_result in results:
            old_result.unlink()
