from __future__ import annotations

import json, time, os
from functools import wraps
from typing import Callable, Any

LOG_DIR = os.environ.get("SGR_LOG_DIR", "logs")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def trace(stage: str):
    def deco(fn: Callable[..., Any]):
        @wraps(fn)
        def inner(*args, **kwargs):
            trace_id = kwargs.get("trace_id") or (args[0] if args else "no-trace")
            t0 = time.time()
            out = None
            error = None
            try:
                out = fn(*args, **kwargs)
                return out
            except Exception as e:
                error = str(e)
                raise
            finally:
                ms = int((time.time() - t0) * 1000)
                d = {
                    "stage": stage,
                    "latency_ms": ms,
                    "error": error,
                }
                dir_ = f"{LOG_DIR}/{trace_id}"
                ensure_dir(dir_)
                with open(f"{dir_}/{stage}.json", "w", encoding="utf-8") as f:
                    json.dump(d if out is None else {**d, "output": out}, f, ensure_ascii=False, indent=2)

        return inner

    return deco
