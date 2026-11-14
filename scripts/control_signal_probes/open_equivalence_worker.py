"""Isolated native runtimes for the four-model scheduler audit.

Moshi/Freeze use the previously audited reference controller, not a new provider
protocol. MiniCPM direct uses the SDK; scheduled uses the production worker.
"""

import argparse
import ast
import audioop
import base64
import contextlib
import hashlib
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/open_model_workers"))


def seed():
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class Runtime:
    def __init__(self, model, mode, config, prompt):
        import torch

        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed()
        self.model, self.mode = model, mode
        self.resample = None
        self.native_input_rate = 16000
        if model == "minicpm" and mode == "direct":
            sys.path.insert(0, config["assets"]["package_root"])
            from core.processors import UnifiedProcessor
            from core.schemas import DuplexConfig

            s = config["settings"]
            options = DuplexConfig(
                force_listen_count=0,
                chunk_ms=1000,
                sample_rate=16000,
                generate_audio=True,
                decode_mode=s.get("decode_mode", "sampling"),
                temperature=s.get("temperature", 0.7),
                tts_temperature=s.get("tts_temperature", 0.8),
            )
            self.processor = UnifiedProcessor(
                model_path=config["checkpoint"], pt_path=None, duplex_config=options
            )
            self.native = self.processor.set_duplex_mode()
            self.native.config = options
            self.native.prepare(
                system_prompt_text=prompt,
                ref_audio_path=config["assets"]["reference_audio"],
            )
            self.rate = 24000
        elif model == "minicpm":
            from minicpm_worker import MiniCPMWorker

            self.native = MiniCPMWorker()
            self.native.connect(config, prompt)
            self.rate = 24000
        elif model == "salmonn":
            if mode == "direct":
                from salmonn_session import NativeSession

                self.native = NativeSession(config, prompt)
            else:
                from salmonn_worker import SalmonnWorker

                self.native = SalmonnWorker()
                self.native.connect(config, prompt)
            self.rate = 16000
        else:
            import native

            native.PROMPT = prompt
            cls = native.MoshiRuntime if model == "moshi" else native.FreezeRuntime
            self.native = cls(model, config)
            self.rate = self.native.rate
            self.native_input_rate = self.native.input_rate
            if model == "moshi" and config["settings"].get("audit_server_warmup"):
                # This installed server starts its CLI at import time. Compile
                # only its unmodified warmup method, avoiding server startup.
                source = Path(importlib.util.find_spec("moshi.server").origin)
                tree = ast.parse(source.read_text(), filename=str(source))
                cls_node = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == "ServerState"
                )
                method = next(
                    n
                    for n in cls_node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "warmup"
                )
                scope = {"torch": torch}
                exec(
                    compile(
                        ast.Module(body=[method], type_ignores=[]), str(source), "exec"
                    ),
                    scope,
                )
                scope["warmup"](self.native.state)
                self.native.state.mimi.reset_streaming()
                self.native.state.lm_gen.reset_streaming()
                self.native.first = True
        seed()
        torch.cuda.synchronize()
        self.provenance = {
            "model": model,
            "mode": mode,
            "checkpoint": config["checkpoint"],
            "sample_rate": self.rate,
            "input_sample_rate": self.native_input_rate,
            "seed": 42,
            "server_warmup": config["settings"].get("audit_server_warmup", False),
            "gpu": torch.cuda.get_device_name(),
            "controller": "native_no_forced_listen"
            if model == "minicpm"
            else "existing_native_reference",
        }

    def step(self, pcm):
        import numpy as np
        import torch

        started = time.monotonic()
        if self.native_input_rate != 16000:
            pcm, self.resample = audioop.ratecv(
                pcm, 2, 1, 16000, self.native_input_rate, self.resample
            )
            expected = 1920 * 2  # Moshi's 80ms at 24kHz.
            if len(pcm) == expected - 2:
                pcm += pcm[-2:]
            if len(pcm) != expected:
                raise ValueError("Moshi native frame size mismatch")
        if self.model == "minicpm" and self.mode == "direct":
            self.native.prefill(
                audio_waveform=np.frombuffer(pcm, dtype="<i2").astype(np.float32)
                / 32768
            )
            r = self.native.generate(force_listen=False)
            torch.cuda.synchronize()
            ready = time.monotonic()
            wave = (
                np.frombuffer(base64.b64decode(r.audio_data), dtype="<f4")
                if r.audio_data
                else np.empty(0)
            )
            raw = (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()
            event = {
                "sample_rate": 24000,
                "text": r.text or "",
                "is_listen": bool(r.is_listen),
                "end_of_turn": bool(r.end_of_turn),
                "events": [],
            }
            self.native.finalize()
        elif self.model in {"salmonn", "minicpm"}:
            event = self.native.step(pcm)
            raw = base64.b64decode(event.pop("audio_pcm16_b64"))
            torch.cuda.synchronize()
            ready = time.monotonic()
        else:
            raw, event = self.native.step(pcm)
            event["sample_rate"] = self.rate
            event["text"] = event.get("text_piece", "")
            torch.cuda.synchronize()
            ready = time.monotonic()
        event["audio_pcm16_b64"] = base64.b64encode(raw).decode()
        event["audit"] = {
            "started_monotonic": started,
            "ready_monotonic": ready,
            "finished_monotonic": time.monotonic(),
            "input_sha256": hashlib.sha256(pcm).hexdigest(),
            "output_sha256": hashlib.sha256(raw).hexdigest(),
        }
        return event

    def close(self):
        if self.model == "minicpm" and self.mode == "direct":
            self.native.stop()
            self.native.cleanup()
        elif hasattr(self.native, "disconnect"):
            self.native.disconnect()
        else:
            self.native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    runtime = None
    for line in sys.stdin:
        request = json.loads(line)
        response = {k: request[k] for k in ("session_id", "sequence")}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if request["op"] == "connect":
                    runtime = Runtime(
                        args.model,
                        args.mode,
                        request["config"],
                        request["system_prompt"],
                    )
                    result = runtime.provenance
                elif request["op"] == "step":
                    result = runtime.step(base64.b64decode(request["audio_pcm16_b64"]))
                else:
                    runtime.close()
                    result = {}
            response["result"] = result
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response["error"] = str(exc)
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
