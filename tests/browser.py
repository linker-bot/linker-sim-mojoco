"""Headless Chrome for the viewer tests, driven over the Chrome DevTools Protocol.

The viewers are web pages; these tests open them as a user would: load the page, press
and hold keys, click at screen positions, read the page and take screenshots, and collect
every console error. The browser draws with software WebGL (SwiftShader), so no GPU or
display is needed.

Chrome is found as ``$CHROME``, else ``google-chrome``, ``chromium`` or the macOS
application; tests that need it skip without it.
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

CDP_TIMEOUT = 30.0  # total time for one command, including intervening events

SIZE = (1600, 1000)
"""Window size [px]."""


def chrome() -> str | None:
    """The Chrome executable, or None."""
    candidates = [
        os.environ.get("CHROME"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


KEY_CODES = {"tab": ("Tab", "Tab", 9), "shift": ("Shift", "ShiftLeft", 16), "space": (" ", "Space", 32)}


class Browser:
    """One headless Chrome with one tab."""

    def __init__(self):
        executable = chrome()
        if executable is None:
            raise RuntimeError("Chrome not found; set CHROME")
        self._profile = tempfile.mkdtemp(prefix="linker-sim-chrome-")
        port = free_port()
        self._process = subprocess.Popen(
            [
                executable, "--headless=new", f"--remote-debugging-port={port}",
                f"--window-size={SIZE[0]},{SIZE[1]}", "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--no-first-run",
                "--no-default-browser-check", f"--user-data-dir={self._profile}", "about:blank",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        self._ws = None
        try:
            for _ in range(100):
                try:
                    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1))
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Chrome did not start")
            page = next(t for t in tabs if t["type"] == "page")
            self._ws = connect(page["webSocketDebuggerUrl"], max_size=2**28)
            self._id = 0
            self.log: list[dict] = []
            """Console messages, exceptions and log entries, as the protocol reports them."""
            self._lock = threading.Lock()
            for domain in ("Runtime", "Log", "Page"):
                self.call(f"{domain}.enable")
            self.modifiers = 0
        except BaseException:
            self.close()
            raise

    def call(self, method: str, **params) -> dict:
        with self._lock:
            self._id += 1
            mine = self._id
            self._ws.send(json.dumps({"id": mine, "method": method, "params": params}))
            deadline = time.monotonic() + CDP_TIMEOUT
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Chrome did not answer {method} within {CDP_TIMEOUT:g} s")
                try:
                    message = json.loads(self._ws.recv(timeout=remaining))
                except TimeoutError as error:
                    raise TimeoutError(
                        f"Chrome did not answer {method} within {CDP_TIMEOUT:g} s"
                    ) from error
                if message.get("id") == mine:
                    if "error" in message:
                        raise RuntimeError(f"{method}: {message['error']}")
                    return message.get("result", {})
                self._record(message)

    def _record(self, message: dict) -> None:
        if message.get("method") in ("Runtime.consoleAPICalled", "Runtime.exceptionThrown", "Log.entryAdded"):
            self.log.append(message)

    def wait(self, seconds: float) -> None:
        """Wait, collecting console messages."""
        end = time.time() + seconds
        while time.time() < end:
            with self._lock:
                try:
                    message = json.loads(self._ws.recv(timeout=min(0.1, max(0.0, end - time.time()))))
                except TimeoutError:
                    continue
                self._record(message)

    def errors(self) -> list[str]:
        """The console errors and uncaught exceptions so far, one line each."""
        found = []
        for m in self.log:
            p = m["params"]
            if m["method"] == "Runtime.exceptionThrown":
                found.append(p["exceptionDetails"].get("exception", {}).get("description") or p["exceptionDetails"]["text"])
            elif m["method"] == "Runtime.consoleAPICalled" and p["type"] == "error":
                found.append(" ".join(str(a.get("value", a.get("description", ""))) for a in p["args"]))
            elif m["method"] == "Log.entryAdded" and p["entry"]["level"] == "error":
                found.append(p["entry"]["text"])
        return found

    def open(self, url: str) -> None:
        self.call("Page.navigate", url=url)

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def until(self, expression: str, timeout: float, poll: float = 0.5):
        """Wait until ``expression`` is truthy in the page; returns its value (None on timeout)."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                value = self.evaluate(expression)
            except RuntimeError:
                value = None
            if value:
                return value
            self.wait(poll)
        return None

    def key(self, key: str, down: bool) -> None:
        """Press (``down``) or release one key: a letter, a digit, ``tab``, ``space``, ``shift``."""
        text, code, vk = KEY_CODES.get(key) or (
            key, f"Digit{key}" if key.isdigit() else f"Key{key.upper()}", ord(key.upper())
        )
        if key == "shift":
            self.modifiers = 8 if down else 0
        elif self.modifiers & 8 and key.isalpha():
            text = key.upper()
        event = {"type": "keyDown" if down else "keyUp", "key": text, "code": code,
                 "windowsVirtualKeyCode": vk, "modifiers": self.modifiers}  # fmt: skip
        if down and len(text) == 1 and key != "space":
            event["text"] = text
        self.call("Input.dispatchKeyEvent", **event)

    def press(self, key: str, hold: float = 0.05) -> None:
        self.key(key, True)
        self.wait(hold)
        self.key(key, False)

    def click(self, x: float, y: float, hold: float = 0.3) -> None:
        """A left click at (x, y) [px]: move there, press, release after ``hold`` seconds.

        A page that draws a frame only every few hundred milliseconds (software WebGL on a
        busy computer) can miss a release that comes within the same frame as the press, and
        then takes later mouse moves for a drag; a human click lasts about this long too."""
        self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="none", buttons=0)
        self.wait(0.3)
        for kind, buttons in (("mousePressed", 1), ("mouseReleased", 0)):
            self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", buttons=buttons, clickCount=1)
            self.wait(hold)

    def screenshot(self, path: Path | None = None) -> bytes:
        png = base64.b64decode(self.call("Page.captureScreenshot", format="png")["data"])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
        return png

    def close(self) -> None:
        """Quit Chrome (again: nothing); the log stays readable."""
        try:
            if self._ws is not None:
                self._ws.close()
        finally:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(10)
            shutil.rmtree(self._profile, ignore_errors=True)


def serve_page(html: str, headers: dict[str, str] | None = None) -> tuple[str, http.server.HTTPServer]:
    """Serve ``html`` at ``http://127.0.0.1:<port>/`` (another origin than the viewer), with
    extra response ``headers``; e.g. a page that embeds a viewer in an ``<iframe>``."""
    body = html.encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/", server
