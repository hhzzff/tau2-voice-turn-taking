"""Standalone native worker: stdio or ordered shared-filesystem mailboxes."""

import argparse
import base64
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path


def serve(model, mailbox=None, idle_timeout=1800):
    """Run isolated model sessions; protocol stdout never includes model logs."""
    if model == "minicpm":
        from minicpm_worker import MiniCPMWorker

        factory = MiniCPMWorker
    else:
        from salmonn_worker import SalmonnWorker

        factory = SalmonnWorker
    worker = None
    session = None
    sequence = 0
    output = sys.stdout

    def dispatch(request):
        nonlocal worker, session, sequence
        response = {
            "session_id": request["session_id"],
            "sequence": request["sequence"],
        }
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if request["op"] == "connect":
                    if worker is not None:
                        raise RuntimeError("Worker already owns an active session")
                    session, sequence = request["session_id"], request["sequence"]
                    worker = factory()
                    result = worker.connect(request["config"], request["system_prompt"])
                else:
                    if (
                        session != request["session_id"]
                        or request["sequence"] != sequence + 1
                    ):
                        raise RuntimeError("Session or sequence mismatch")
                    sequence = request["sequence"]
                    if request["op"] == "step":
                        result = worker.step(
                            base64.b64decode(request["audio_pcm16_b64"], validate=True)
                        )
                    elif request["op"] == "disconnect":
                        worker.disconnect()
                        worker = None
                        result = {}
                    else:
                        raise ValueError("Unknown operation")
                response["result"] = result
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response["error"] = f"{type(exc).__name__}: {exc}"
            if worker is not None:
                with contextlib.suppress(Exception):
                    worker.disconnect()
                worker = None
        return response

    try:
        if mailbox is None:
            for line in sys.stdin:
                output.write(json.dumps(dispatch(json.loads(line))) + "\n")
                output.flush()
        else:
            root = Path(mailbox)
            root.mkdir(parents=True, exist_ok=True)
            (root / "ready.json").write_text(
                json.dumps({"model": model, "python": sys.executable})
            )
            last = time.monotonic()
            while not (root / "STOP").exists():
                if time.monotonic() - last > idle_timeout:
                    break
                if (
                    session
                    and (root / session / "closed").exists()
                    and worker is not None
                ):
                    worker.disconnect()
                    worker = None
                for path in sorted(root.glob("*/*.request.json")):
                    target = path.with_name(
                        path.name.replace(".request.", ".response.")
                    )
                    if target.exists() or (path.parent / "closed").exists():
                        continue
                    if worker is not None and path.parent.name != session:
                        continue
                    response = dispatch(json.loads(path.read_text()))
                    temporary = target.with_suffix(".tmp")
                    temporary.write_text(json.dumps(response))
                    temporary.replace(target)
                    last = time.monotonic()
                time.sleep(0.05)
    finally:
        if worker is not None:
            with contextlib.redirect_stdout(sys.stderr):
                worker.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("minicpm", "salmonn"))
    parser.add_argument("--mailbox")
    parser.add_argument("--idle-timeout", type=float, default=1800)
    args = parser.parse_args()
    serve(args.model, args.mailbox, args.idle_timeout)
