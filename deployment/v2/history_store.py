# -*- coding: utf-8 -*-
"""
history_store.py — durable detection history for the dashboard's History
panel (Single File / 3-Mic / Multi-Drone results).

Backed by Firestore when available (Cloud Run: the attached service
account authenticates automatically, no credentials file needed) so
history survives restarts/redeploys and is consistent across however
many Cloud Run instances are handling traffic. Falls back to a local
JSONL file — same degrade-gracefully pattern used throughout this
codebase for optional remote dependencies (WebDAV, remotezip, etc.) —
when Firestore isn't reachable, e.g. local dev with no GCP credentials,
or the Firestore API not enabled on the project.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("drone_v2.history_store")

_COLLECTION = "detection_history"

_fallback_path = Path(__file__).parent / "logs" / "detection_history.jsonl"
_fallback_lock = threading.Lock()

_client = None
_client_init_attempted = False


def _get_client():
    """Lazily construct (and cache) a Firestore client. Returns None if
    unavailable, so callers fall back to the local JSONL file."""
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client
    _client_init_attempted = True
    try:
        from google.cloud import firestore
        _client = firestore.Client()
        # Cheap round-trip to confirm the API is actually enabled/reachable
        # rather than only checking that the client object constructed —
        # firestore.Client() succeeds even with no credentials/API access.
        list(_client.collection(_COLLECTION).limit(1).stream())
        log.info("history_store: using Firestore")
    except Exception as exc:
        log.info("history_store: Firestore unavailable (%s) — using local JSONL fallback", exc)
        _client = None
    return _client


def _fallback_append(record: Dict[str, Any]) -> None:
    _fallback_path.parent.mkdir(parents=True, exist_ok=True)
    with _fallback_lock:
        with open(_fallback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _fallback_list(limit: int, detection_type: Optional[str]) -> List[Dict[str, Any]]:
    if not _fallback_path.exists():
        return []
    with _fallback_lock:
        lines = _fallback_path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if detection_type and rec.get("type") != detection_type:
            continue
        records.append(rec)
        if len(records) >= limit:
            break
    return records


def _fallback_get(record_id: str) -> Optional[Dict[str, Any]]:
    if not _fallback_path.exists():
        return None
    with _fallback_lock:
        lines = _fallback_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("id") == record_id:
            return rec
    return None


# Firestore rejects arrays that directly contain arrays (e.g. track
# "positions": [[x, y], ...]) and documents over 1 MiB. full_result is
# therefore stored as one JSON string, trimmed to fit if necessary.
_MAX_FULL_RESULT_BYTES = 900_000
_TRIMMABLE_LISTS = ("frames", "results")   # repository frame log, batch windows


def _json_default(obj):
    if hasattr(obj, "tolist"):   # numpy arrays and scalars
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


def _encode_full_result(full_result: Dict[str, Any]) -> str:
    """JSON-encode, dropping the oldest frames/windows until it fits."""
    encoded = _dumps(full_result)
    if len(encoded.encode("utf-8")) <= _MAX_FULL_RESULT_BYTES:
        return encoded
    trimmed = dict(full_result)
    for key in _TRIMMABLE_LISTS:
        items = trimmed.get(key)
        if not isinstance(items, list):
            continue
        original = len(items)
        while len(items) > 1 and len(encoded.encode("utf-8")) > _MAX_FULL_RESULT_BYTES:
            items = items[len(items) // 2:]   # keep the most recent half
            trimmed[key] = items
            trimmed["history_trimmed"] = {"key": key, "kept": len(items), "original": original}
            if key == "frames":
                trimmed["frame_log_truncated"] = True
            encoded = _dumps(trimmed)
    return encoded


def _decode(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Restore full_result from its JSON string (older records hold a dict)."""
    if record and "full_result_json" in record:
        record["full_result"] = json.loads(record.pop("full_result_json"))
    return record


def log_detection(detection_type: str, summary: Dict[str, Any], full_result: Dict[str, Any]) -> None:
    """
    Save one detection event.

    detection_type : "single" | "3mic" | "multi" | "batch" | "repository"
    summary         : small fields shown in the History list (source label,
                      detected, probability, position, timestamp)
    full_result     : the complete JSON response, replayed later by feeding
                       it straight back into the same render*Result() JS
                       function that displayed it originally
    """
    try:
        record = {
            "id":        uuid.uuid4().hex[:16],
            "type":      detection_type,
            "timestamp": time.time(),
            **json.loads(_dumps(summary)),
            "full_result_json": _encode_full_result(full_result),
        }
    except Exception:
        log.exception("history_store: could not serialise %s record", detection_type)
        return
    try:
        client = _get_client()
        if client is not None:
            client.collection(_COLLECTION).document(record["id"]).set(record)
            return
    except Exception:
        log.exception("history_store: Firestore write failed, falling back to local log")
    try:
        _fallback_append(record)
    except Exception:
        log.exception("history_store: local fallback write failed")


# Summary fields returned by list_history(). Projecting onto these keeps the
# (potentially ~1 MB) full_result out of the History list query.
_SUMMARY_FIELDS = ["id", "type", "timestamp", "source", "detected",
                   "probability", "position", "metrics"]
# How many recent records to scan when filtering by type. Filtering in
# Python avoids needing a Firestore composite index on (type, timestamp).
_FILTER_SCAN_LIMIT = 1000


def list_history(limit: int = 50, detection_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Most recent first. Returns summary fields only (no full_result)."""
    limit = max(1, min(limit, 200))
    try:
        client = _get_client()
        if client is not None:
            from google.cloud import firestore
            q = (client.collection(_COLLECTION)
                 .select(_SUMMARY_FIELDS)
                 .order_by("timestamp", direction=firestore.Query.DESCENDING)
                 .limit(_FILTER_SCAN_LIMIT if detection_type else limit))
            out = []
            for d in q.stream():
                rec = d.to_dict()
                if detection_type and rec.get("type") != detection_type:
                    continue
                out.append(rec)
                if len(out) >= limit:
                    break
            return out
    except Exception:
        log.exception("history_store: Firestore read failed, falling back to local log")

    records = _fallback_list(limit, detection_type)
    for r in records:
        r.pop("full_result", None)
        r.pop("full_result_json", None)
    return records


def get_detection(record_id: str) -> Optional[Dict[str, Any]]:
    """Full record (including full_result) for replay."""
    try:
        client = _get_client()
        if client is not None:
            doc = client.collection(_COLLECTION).document(record_id).get()
            if doc.exists:
                return _decode(doc.to_dict())
            return None
    except Exception:
        log.exception("history_store: Firestore read failed, falling back to local log")

    return _decode(_fallback_get(record_id))
