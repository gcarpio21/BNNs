from __future__ import annotations

import os
import random
import time

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def checkpoint_exists(checkpoint_path: str | None) -> bool:
    return bool(checkpoint_path) and os.path.exists(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    map_location: torch.device | str | None = None,
    weights_only: bool = False,
):
    """Load checkpoints with compatibility across PyTorch versions.

    PyTorch 2.6 changed torch.load default `weights_only=True`, which breaks
    object checkpoints (e.g., Laplace objects) unless explicitly disabled.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except TypeError:
        try:
            return torch.load(checkpoint_path, map_location=map_location)
        except Exception as e:
            last_exc = e
    except Exception as e:
        last_exc = e

    try:
        msg = str(last_exc)
    except Exception:
        msg = repr(last_exc)

    corrupt_indicators = [
        'PytorchStreamReader failed locating file data.pkl',
        'miniz error',
        'file not found',
        'UnpicklingError',
        'pickle.UnpicklingError',
    ]
    if any(ind.lower() in msg.lower() for ind in corrupt_indicators):
        try:
            corrupt_path = checkpoint_path + '.corrupt'
            os.replace(checkpoint_path, corrupt_path)
        except Exception:
            pass
        print(f"Warning: checkpoint appears corrupted. Renamed to: {corrupt_path}")
        return None

    raise last_exc


def save_checkpoint(payload: dict, checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def _sanitize(obj_dict: dict) -> dict:
        safe = {}
        for k, v in (obj_dict or {}).items():
            if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
                try:
                    safe_key = k if k.endswith("_state_dict") else f"{k}_state_dict"
                    safe[safe_key] = v.state_dict()
                    continue
                except Exception:
                    continue
            try:
                import pickle as _pickle
                _pickle.dumps(v)
                safe[k] = v
            except Exception:
                continue
        return safe

    def _torch_save_with_retry(obj, path, retries=4, delay=1.0):
        last_exc = None
        for attempt in range(retries):
            try:
                torch.save(obj, path)
                return
            except Exception as exc:
                last_exc = exc
                try:
                    os.remove(path)
                except Exception:
                    pass
                if attempt < retries - 1:
                    time.sleep(delay)
        raise last_exc

    safe_payload = _sanitize(payload)
    tmp_path = checkpoint_path + ".tmp"
    try:
        _torch_save_with_retry(safe_payload, tmp_path)
        try:
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            os.remove(tmp_path) if os.path.exists(tmp_path) else None
            raise
    except Exception as e:
        try:
            pruned = {}
            for k, v in safe_payload.items():
                if isinstance(v, (torch.Tensor, int, float, str, dict, list, tuple, bool, type(None))):
                    pruned[k] = v
            _torch_save_with_retry(pruned, tmp_path)
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            raise e
