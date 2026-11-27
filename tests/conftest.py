"""Select CI test groups without changing the default full test run."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        file = Path(item.path).name
        if file == "test_viewers.py":
            item.add_marker(pytest.mark.browser)
        elif (file == "test_examples.py" and item.name != "test_lesson_page_is_the_example") or (
            file == "test_gallery.py" and "scene" in item.fixturenames
        ):
            item.add_marker(pytest.mark.gallery)
        elif file == "test_docs.py" and hasattr(item, "callspec"):
            doc = item.callspec.params.get("doc")
            if doc and "gallery.fetch(" in doc.read_text():
                item.add_marker(pytest.mark.gallery)
        elif file == "test_copies.py":
            item.add_marker(pytest.mark.copies)
        elif file == "test_engine.py" and item.name in {
            "test_one_simulated_second_holds_home_at_real_time", "test_doctor_passes",
        }:
            item.add_marker(pytest.mark.performance)
