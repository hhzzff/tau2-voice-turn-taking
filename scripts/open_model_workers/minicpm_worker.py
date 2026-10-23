"""MiniCPM-o-4.5 persistent runtime, compatible with native Python 3.10."""

import base64
import importlib
import sys
from pathlib import Path


class MiniCPMWorker:
    """Own one mutable duplex session and its KV/audio caches."""

    def connect(self, config, system_prompt):
        """Load the configured native processor and prepare one conversation."""
        checkpoint = Path(config.get("checkpoint", "")).resolve()
        supported = Path(
            "/mnt/bn/audio-visual-llm-data6/base_models/MiniCPM-o-4_5"
        ).resolve()
        if (
            checkpoint != supported
            or config.get("assets", {}).get("pt_path") is not None
        ):
            raise ValueError(
                "MiniCPM-o-4.5 MVP requires its approved base checkpoint without pt_path overrides"
            )
        import numpy as np

        self.np = np
        root = Path(config["assets"]["package_root"]).resolve()
        sys.path.insert(0, str(root))
        processors = importlib.import_module("core.processors")
        schemas = importlib.import_module("core.schemas")
        for module in (processors, schemas):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError("MiniCPM module origin mismatch")
        settings = config.get("settings", {})
        if settings.get("chunk_ms", 1000) != 1000:
            raise ValueError("MiniCPM requires native 1000ms input")
        duplex_config = schemas.DuplexConfig(
            force_listen_count=0,
            chunk_ms=1000,
            sample_rate=16000,
            generate_audio=True,
            decode_mode=settings.get("decode_mode", "sampling"),
            temperature=settings.get("temperature", 0.7),
            tts_temperature=settings.get("tts_temperature", 0.8),
        )
        self.unified = processors.UnifiedProcessor(
            model_path=str(checkpoint),
            pt_path=config["assets"].get("pt_path"),
            duplex_config=duplex_config,
        )
        self.duplex = self.unified.set_duplex_mode()
        self.duplex.config = duplex_config
        self.duplex.prepare(
            system_prompt_text=system_prompt,
            ref_audio_path=config["assets"]["reference_audio"],
        )
        return {
            "model": "MiniCPM-o-4.5",
            "checkpoint": str(checkpoint),
            "sample_rate": 24000,
            "python": sys.executable,
            "processors": processors.__file__,
            "settings": settings,
        }

    def step(self, pcm16):
        """Ingest only current input, finish native generation, preserve cache."""
        if len(pcm16) != 32000:
            raise ValueError("MiniCPM input must be 16000 PCM16 samples")
        audio = self.np.frombuffer(pcm16, dtype="<i2").astype(self.np.float32) / 32768
        self.duplex.prefill(audio_waveform=audio)
        result = self.duplex.generate(force_listen=False)
        self.duplex.finalize()
        waveform = (
            self.np.frombuffer(base64.b64decode(result.audio_data), dtype="<f4")
            if result.audio_data
            else self.np.empty(0)
        )
        if not self.np.isfinite(waveform).all():
            raise RuntimeError("MiniCPM generated nonfinite audio")
        pcm = (self.np.clip(waveform, -1, 1) * 32767).astype("<i2").tobytes()
        return {
            "audio_pcm16_b64": base64.b64encode(pcm).decode(),
            "sample_rate": 24000,
            "text": result.text or "",
            "events": [],
        }

    def disconnect(self):
        """Discard session state; never force a late output flush."""
        if getattr(self, "duplex", None) is not None:
            try:
                self.duplex.stop()
            finally:
                self.duplex.cleanup()
                self.duplex = None
                self.unified = None
