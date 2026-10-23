"""Ordered, session-bound transports to an isolated native model runtime."""

import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path


class SubprocessTransport:
    """JSON lines on stdout; runtime diagnostics belong on stderr."""

    def __init__(self, command, timeout_seconds=1800):
        self.command = command
        self.timeout = timeout_seconds
        self.process = None
        self.session = uuid.uuid4().hex
        self.sequence = 0

    def request(self, op, **payload):
        """Wait for a completed operation, independent of audio time."""
        if self.process is None:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.lines = queue.Queue()

            def read():
                for line in self.process.stdout:
                    self.lines.put(line)
                self.lines.put(None)

            threading.Thread(target=read, daemon=True).start()
        self.sequence += 1
        request = dict(payload, op=op, session_id=self.session, sequence=self.sequence)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError(f"Native runtime {op} exceeded {self.timeout}s") from exc
        if line is None:
            raise RuntimeError("Native runtime exited before operation completed")
        return validate_response(json.loads(line), self.session, self.sequence)

    def close(self):
        """Terminate and reap the isolated process."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None


def validate_response(response, session, sequence):
    """Reject stale replies and propagate runtime failures."""
    if response.get("session_id") != session or response.get("sequence") != sequence:
        raise RuntimeError("Native runtime response identity mismatch")
    if "error" in response:
        raise RuntimeError(f"Native runtime failed: {response['error']}")
    return response["result"]


class MailboxTransport:
    """Atomic shared-directory exchange with a remote Merlin worker."""

    def __init__(self, root, timeout_seconds=1800):
        self.root = Path(root)
        self.session = uuid.uuid4().hex
        self.directory = self.root / self.session
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        self.timeout = timeout_seconds

    def request(self, op, **payload):
        """Publish one request and wait for its matching committed response."""
        if (self.directory / "closed").exists():
            if op != "connect":
                raise RuntimeError("Mailbox session is closed")
            self.session = uuid.uuid4().hex
            self.directory = self.root / self.session
            self.directory.mkdir(exist_ok=False)
            self.sequence = 0
        self.sequence += 1
        stem = f"{self.sequence:08d}"
        temporary = self.directory / (stem + ".tmp")
        temporary.write_text(
            json.dumps(
                dict(payload, op=op, session_id=self.session, sequence=self.sequence)
            )
        )
        temporary.replace(self.directory / (stem + ".request.json"))
        response = self.directory / (stem + ".response.json")
        deadline = time.monotonic() + self.timeout
        while not response.exists():
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError(
                    f"Merlin runtime {op} exceeded {self.timeout}s; session {self.session}"
                )
            time.sleep(0.05)
        return validate_response(
            json.loads(response.read_text()), self.session, self.sequence
        )

    def close(self):
        """Mark abandoned sessions so the worker releases model state."""
        (self.directory / "closed").touch()
