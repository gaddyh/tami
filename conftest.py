import pytest

from run_tracker import record_test_end, record_test_start, write_report


def pytest_configure(config):
    config.addinivalue_line("markers", "hebrew: Hebrew language test examples")


def pytest_runtest_setup(item):
    record_test_start(item.name)
    lang = "he" if any(m.name == "hebrew" for m in item.iter_markers()) else "en"
    from run_tracker import record_test_language
    record_test_language(lang)


def pytest_runtest_makereport(item, call):
    if call.when == "call":
        outcome = "passed" if call.excinfo is None else "failed"
        error = str(call.excinfo.value) if call.excinfo else None
        record_test_end(outcome, error)
    elif call.when == "setup" and call.excinfo is not None:
        record_test_end("failed", str(call.excinfo.value))


def pytest_sessionfinish(session, exitstatus):
    write_report()
