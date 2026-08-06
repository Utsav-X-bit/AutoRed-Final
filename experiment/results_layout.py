"""Results directory layout rules for AutoRed-Final.

Owns the mapping from (mode, model_id, characteristics) to on-disk paths.
All functions are pure except ``runs_root``, which creates directories.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Characters safe to keep verbatim in a single directory name.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _slug_segment(value: str) -> str:
    """Return a single filesystem-safe directory segment.

    Slashes become ``--`` (preserving HF org/name readability); any other
    unsafe character becomes ``_``; surrounding separators are stripped.
    """
    value = value.strip().strip("/")
    value = value.replace("/", "--")
    value = _SAFE.sub("_", value)
    value = value.strip(".-_")
    return value or "unknown"


def slugify_model_id(model_id: str) -> str:
    """Slugify a HuggingFace model id or bare name into one directory name.

    ``org/name`` -> ``org--name``; ``gpt2`` -> ``gpt2``.
    """
    return _slug_segment(model_id)


def resolve_model_id(victim_model_id: str | None, load_path: str | None = None) -> str:
    """Resolve the model id directory segment.

    Priority:
      1. explicit ``victim_model_id`` (from --victim-model-id),
      2. a HuggingFace cache dir of the form ``.../models--org--name/snapshots/...``,
      3. basename of ``load_path`` + 6-char hash (collision-safe for local checkpoints),
      4. ``"unknown"``.
    """
    if victim_model_id:
        return slugify_model_id(victim_model_id)

    if load_path:
        # HF cache layout: models--org--name
        m = re.search(r"models--([^/]+)", str(load_path))
        if m:
            tail = m.group(1)
            if "--" in tail:
                # org--name form
                return _slug_segment(tail.replace("--", "/", 1))
            return _slug_segment(tail)
        # Local checkpoint path -> basename + hash
        base = Path(load_path).name or "model"
        h = hashlib.sha256(str(load_path).encode()).hexdigest()[:6]
        return f"{_slug_segment(base)}_{h}"

    return "unknown"
