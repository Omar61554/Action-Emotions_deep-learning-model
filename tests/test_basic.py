from src import main


def test_main_runs():
    # basic smoke test: ensure main callable exists
    assert hasattr(main, 'main')
