"""A stalled CDP command must not leave the viewer loop waiting indefinitely."""

import json
import threading

import browser
import pytest


def test_command_timeout_names_the_stalled_method():
    class Socket:
        def send(self, message):
            assert json.loads(message)["method"] == "Page.captureScreenshot"

        def recv(self, *, timeout):
            assert 0 < timeout <= browser.CDP_TIMEOUT
            raise TimeoutError("no reply")

    client = browser.Browser.__new__(browser.Browser)
    client._lock = threading.Lock()
    client._id = 0
    client._ws = Socket()
    with pytest.raises(TimeoutError, match="Page.captureScreenshot"):
        client.call("Page.captureScreenshot")


def test_events_do_not_restart_the_command_deadline(monkeypatch):
    ticks = iter((0.0, 1.0, 31.0))
    monkeypatch.setattr(browser.time, "monotonic", lambda: next(ticks))

    class Socket:
        def send(self, message):
            pass

        def recv(self, *, timeout):
            return json.dumps({"method": "Runtime.consoleAPICalled", "params": {}})

    client = browser.Browser.__new__(browser.Browser)
    client._lock = threading.Lock()
    client._id = 0
    client._ws = Socket()
    client.log = []
    with pytest.raises(TimeoutError, match="Runtime.evaluate"):
        client.call("Runtime.evaluate")
    assert len(client.log) == 1
