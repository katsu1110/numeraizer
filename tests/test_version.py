from src import version

def test_version_text():
    assert not version.__version__ is None