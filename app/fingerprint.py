"""Deterministic fingerprints for inputs and rule versions.

A job fingerprint is the SHA-256 of canonical JSON over

* the input data digest (points, GCPs, correspondences as imported),
* the plan configuration (model, weights, disabled groups, locks, splits),
* and the rule versions that govern solving and rejection.

Replaying the same job returns the stored job and does not emit a second
audit event.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

RULE_VERSION = "rules-2026-09-20-v1"


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(payload: Dict[str, Any]) -> str:
    return sha256_text(canonical_json(payload))


def job_fingerprint(input_digest: str, config: Dict[str, Any]) -> str:
    return fingerprint({
        "input_digest": input_digest,
        "config": config,
        "rule_version": RULE_VERSION,
    })
