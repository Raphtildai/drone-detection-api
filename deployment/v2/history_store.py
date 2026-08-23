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


def log_detection(detection_type: str, summary: Dict[str, Any], full_result: Dict[str, Any]) -> None:
    """
    Save one detection event.

    detection_type : "single" | "3mic" | "multi"
    summary         : small fields shown in the History list (source label,
                      detected, probability, position, timestamp)
    full_result     : the complete JSON response, replayed later by feeding
                       it straight back into the same render*Result() JS
                       function that displayed it originally
    """
    record = {
        "id":        uuid.uuid4().hex[:16],
        "type":      detection_type,
        "timestamp": time.time(),
        **summary,
        "full_result": full_result,
    }
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


def list_history(limit: int = 50, detection_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Most recent first. Returns summary fields only (no full_result)."""
    limit = max(1, min(limit, 200))
    try:
        client = _get_client()
        if client is not None:
            from google.cloud import firestore
            q = client.collection(_COLLECTION).order_by(
                "timestamp", direction=firestore.Query.DESCENDING
            )
            if detection_type:
                q = q.where("type", "==", detection_type)
            docs = q.limit(limit).stream()
            out = []
            for d in docs:
                rec = d.to_dict()
                rec.pop("full_result", None)
                out.append(rec)
            return out
    except Exception:
        log.exception("history_store: Firestore read failed, falling back to local log")

    records = _fallback_list(limit, detection_type)
    for r in records:
        r.pop("full_result", None)
    return records


def get_detection(record_id: str) -> Optional[Dict[str, Any]]:
    """Full record (including full_result) for replay."""
    try:
        client = _get_client()
        if client is not None:
            doc = client.collection(_COLLECTION).document(record_id).get()
            if doc.exists:
                return doc.to_dict()
            return None
    except Exception:
        log.exception("history_store: Firestore read failed, falling back to local log")

    return _fallback_get(record_id)
