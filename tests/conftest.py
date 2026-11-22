import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-model", action="store_true", help="Allow real-model tests and downloads"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-model"):
        for item in items:
            if "model" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason="Requires explicit --run-model")
                )
