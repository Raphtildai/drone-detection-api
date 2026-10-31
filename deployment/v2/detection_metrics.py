# -*- coding: utf-8 -*-
"""
detection_metrics.py — ground-truth evaluation shared by Batch Scan and
Repository sessions: detection confusion matrix / precision / recall / F1,
and localization error summaries.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional


def compute_detection_metrics(results: Iterable[dict]) -> dict:
    """
    Precision/recall/F1 for the *detection* step (is there a drone in this
    window at all), evaluated against ground truth — separate from the
    localization error, and unaffected by the array-geometry issue: this only
    checks the detected/not-detected verdict, never the estimated position.

    Each item needs "detected" (bool) and "gt_presence":
      "present" — ground truth confirms a drone (counted as TP or FN)
      "absent"  — ground truth confirms no drone (counted as FP or TN)
      anything else — no ground truth either way, not counted

    Precision needs at least one confirmed-absent window: with none, fp is
    trivially 0 and tp/(tp+fp) would read as a misleadingly perfect 1.0
    rather than "not measurable". Repository datasets contain drone-present
    segments only (UaVirBASE ambient recordings are skipped by the loader),
    so for them only recall is measurable. n_confirmed_* are always included
    so a metric based on very few samples reads as exactly that.
    """
    tp = fn = fp = tn = 0
    for r in results:
        presence = r.get("gt_presence")
        if presence == "present":
            if r["detected"]:
                tp += 1
            else:
                fn += 1
        elif presence == "absent":
            if r["detected"]:
                fp += 1
            else:
                tn += 1

    n_present = tp + fn
    n_absent  = fp + tn
    precision = tp / (tp + fp) if n_absent > 0 and (tp + fp) > 0 else None
    recall    = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    accuracy = (tp + tn) / (n_present + n_absent) if n_present > 0 and n_absent > 0 else None
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_confirmed_present": n_present,
        "n_confirmed_absent":  n_absent,
        "precision": round(precision, 3) if precision is not None else None,
        "recall":    round(recall, 3) if recall is not None else None,
        "f1":        round(f1, 3) if f1 is not None else None,
        "accuracy":  round(accuracy, 3) if accuracy is not None else None,
    }


def summarise_errors(values: List[float]) -> Optional[dict]:
    """MAE / median / RMSE of absolute errors, or None when there are none."""
    v = sorted(abs(float(x)) for x in values if x is not None and math.isfinite(float(x)))
    if not v:
        return None
    n = len(v)
    median = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    return {
        "n":      n,
        "mae":    round(sum(v) / n, 3),
        "median": round(median, 3),
        "rmse":   round(math.sqrt(sum(x * x for x in v) / n), 3),
    }


def azimuth_error_deg(est_deg: float, true_deg: float) -> float:
    """Absolute angular difference, wrapped to [0, 180]."""
    return abs((float(est_deg) - float(true_deg) + 180.0) % 360.0 - 180.0)
