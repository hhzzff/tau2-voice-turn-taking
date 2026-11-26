"""Persistent SALMONN session for the native Python 3.10 worker.

This module deliberately does not import tau2 or the GPU runtime at import time.
"""

import os
from pathlib import Path


class SalmonnWorker:
    """Consume only the current 80 ms input and retain native model state."""

    def connect(self, config: dict, system_prompt: str) -> dict:
        """Load the selected checkpoint with its own training configuration."""
        checkpoint = Path(config.get("checkpoint", ""))
        if not checkpoint.is_dir() or not (checkpoint.parent / "config.json").is_file():
            raise ValueError(
                "SALMONN requires a checkpoint with adjacent training config"
            )
        if config.get("settings", {}).get("chunk_ms", 80) != 80:
            raise ValueError("SALMONN requires an 80 ms tick")
        if getattr(self, "session", None) is not None:
            raise RuntimeError("SALMONN worker is already connected")
        os.environ.setdefault("USE_TF", "0")
        from salmonn_session import NativeSession

        session = NativeSession.__new__(NativeSession)
        try:
            session.__init__(config, system_prompt)
        except Exception:
            # Constructor may fail after installing an instance override.
            if hasattr(session, "override"):
                session.close()
            raise
        self.session = session
        return dict(session.provenance)

    def step(self, pcm16: bytes) -> dict:
        """Consume one tick of PCM16, returning only audio available now."""
        if not getattr(self, "session", None):
            raise RuntimeError("SALMONN worker is not connected")
        return self.session.step(pcm16)

    def disconnect(self) -> None:
        """Release a session and all acoustic, decoder and playback state."""
        session = getattr(self, "session", None)
        self.session = None
        if session is not None:
            session.close()
