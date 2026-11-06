import os

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)
