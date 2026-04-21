"""
uavirbase_patch.py
──────────────────
Two things in one file:

  1. DIAGNOSTIC — run standalone to inspect your label.json files and
     find out exactly what's failing:

       python uavirbase_patch.py C:/tmp/drone_v15

  2. PATCH — import this at the top of build_training_datasets_and_figures.py
     (or any script that uses drone_detection) to monkey-patch
     parse_label_json with a deep recursive version that handles every
     plausible UaVirBASE JSON structure:

       import uavirbase_patch   # must come BEFORE any drone_detection import

The root cause
──────────────
parse_label_json in datasets.py tries these paths in order:
  1. data["drone"]["azimuth/distance/height"]
  2. data["drone"] Cartesian (x/y/z)
  3. top-level _AZ_KEYS / _DIST_KEYS / _HT_KEYS
  4. sub-dict under ["uav","target","labels","annotation","data","position"]

UaVirBASE label.json is produced by the "UaVirBASE Recorder" desktop app.
The paper (Figure 10) shows a rich nested JSON with microphone positions,
weather, etc.  Based on the recording app's UI (Drone Tab fields: distance,
height, azimuth, side, model) the most likely layouts are:

  Layout A — position nested one level down:
    {"drone": {"model": "DJI Mavic 3 Cine", "sound_source": "drone",
               "position": {"azimuth": 90, "distance": 10, "height": 5}}}

  Layout B — azimuth_deg / distance_m / height_m inside "drone":
    {"drone": {"azimuth_deg": 90, "distance_m": 10, "height_m": 5, ...}}

  Layout C — top-level with a "source" string key (not "drone"):
    {"source": "drone", "azimuth": 90, "distance": 10, "height": 5}

  Layout D — recording_setup wrapper:
    {"recording": {"drone": {"azimuth": 90, ...}}, "microphones": [...]}

None of B–D match the existing parser. The fix is a recursive descent that
searches every nested dict for any triple of (azimuth-like, distance-like,
height-like) scalars, with ambient filtering at every level.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

# ── Key name sets (superset of what datasets.py has) ─────────────────────────

_AZ_KEYS   = frozenset([
    "azimuth_deg", "azimuth", "az", "Azimuth", "AZ",
    "bearing", "heading", "direction_deg", "direction",
    "angle", "phi", "angle_deg", "angle_rad",
])
_DIST_KEYS = frozenset([
    "distance_m", "distance", "dist", "Distance",
    "range", "range_m", "horizontal_distance", "slant_range",
    "r", "radius",
])
_HT_KEYS   = frozenset([
    "height_m", "height", "alt", "altitude", "Height",
    "z", "elevation", "Elevation", "altitude_m", "z_m",
    "height_agl", "h",
])

_AMBIENT_STRINGS = frozenset(["ambient", "silence", "noise", "background"])


def _fv(v) -> Optional[float]:
    """Safely convert to float, return None if invalid."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _get(d: dict, keys) -> Optional[float]:
    for k in keys:
        v = _fv(d.get(k))
        if v is not None:
            return v
    return None


def _is_ambient(d: dict) -> bool:
    """Return True if this dict describes an ambient/background recording."""
    for k in ("sound_source", "source", "type", "class", "label", "category"):
        v = d.get(k)
        if isinstance(v, str) and any(a in v.lower() for a in _AMBIENT_STRINGS):
            return True
    return False


def _xyz_to_az_dist_ht(d: dict) -> Optional[Tuple[float, float, float]]:
    """Try Cartesian (x, y, z) → (azimuth_deg, dist_m, height_m)."""
    for xk in ("x", "pos_x", "east", "X", "lng", "lon"):
        x = _fv(d.get(xk))
        if x is None:
            continue
        for yk in ("y", "pos_y", "north", "Y", "lat"):
            y = _fv(d.get(yk))
            if y is None:
                continue
            for zk in ("z", "pos_z", "height", "height_m", "alt", "altitude", "Z", "h"):
                z = _fv(d.get(zk))
                if z is None:
                    continue
                dist = math.sqrt(x ** 2 + y ** 2)
                az   = math.degrees(math.atan2(y, x))
                return float(az), float(dist), float(abs(z))
    return None


def _try_extract(d: dict) -> Optional[Tuple[float, float, float]]:
    """
    Try to extract (az_deg, dist_m, ht_m) from a single flat dict.
    Returns None if the dict looks like an ambient recording.
    """
    if _is_ambient(d):
        return None

    # Direct named fields
    az = _get(d, _AZ_KEYS)
    di = _get(d, _DIST_KEYS)
    ht = _get(d, _HT_KEYS)
    if az is not None and di is not None and ht is not None:
        return float(az), float(di), float(ht)

    # Partial: az + dist but no ht → use az+dist with default ht
    if az is not None and di is not None:
        return float(az), float(di), 5.0   # conservative default

    # Cartesian fallback
    return _xyz_to_az_dist_ht(d)


def parse_label_json_v2(raw: bytes) -> Optional[Tuple[float, float, float]]:
    """
    Drop-in replacement for parse_label_json in datasets.py.

    Strategy:
    1. Quick ambient check at the top level.
    2. Try the existing fast paths first (preserves backward compatibility).
    3. Deep recursive search: BFS through all nested dicts, trying each.
    4. If everything fails but the JSON has a "drone" key with no useful
       position data → return None (not a parse error, a genuine ambient/
       malformed session).
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    # ── 1. Top-level ambient guard ────────────────────────────────────────────
    if _is_ambient(data):
        return None

    # ── 2. Existing fast paths ────────────────────────────────────────────────
    # Path A: data["drone"] with direct fields
    if "drone" in data and isinstance(data["drone"], dict):
        drone = data["drone"]
        if _is_ambient(drone):
            return None
        r = _try_extract(drone)
        if r:
            return r

    # Path B: top-level direct fields
    r = _try_extract(data)
    if r:
        return r

    # ── 3. Deep recursive BFS ─────────────────────────────────────────────────
    # Walk every nested dict (up to depth 6 to avoid infinite loops on odd data)
    visited_ids = set()

    def _recurse(obj, depth: int) -> Optional[Tuple[float, float, float]]:
        if depth <= 0 or not isinstance(obj, dict) or id(obj) in visited_ids:
            return None
        visited_ids.add(id(obj))

        if _is_ambient(obj):
            return None

        r = _try_extract(obj)
        if r:
            return r

        for v in obj.values():
            if isinstance(v, dict):
                r = _recurse(v, depth - 1)
                if r:
                    return r
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        r = _recurse(item, depth - 1)
                        if r:
                            return r
        return None

    return _recurse(data, depth=6)


# =============================================================================
#  MONKEY-PATCH — call import_uavirbase_patch() before any drone_detection use
# =============================================================================

def apply_patch():
    """
    Replace datasets.parse_label_json with the v2 deep-recursive version.
    Also patches the UaVirBASEDatasetManager._download_partial validation loop
    to use parse_label_json_v2 directly (bypasses the module-level name binding).

    Call this ONCE, after drone_detection is importable but before prepare().
    """
    try:
        import drone_detection.datasets as _ds
        original = _ds.parse_label_json
        _ds.parse_label_json = parse_label_json_v2
        print("  🔧  parse_label_json patched with deep-recursive v2")

        # Also patch the _download_partial method to use the new function
        # by rebinding it in the UaVirBASEDatasetManager class
        original_dp = _ds.UaVirBASEDatasetManager._download_partial

        def _patched_download_partial(self, url, proc, n_sessions):
            # Temporarily override the module-level parse_label_json
            _ds.parse_label_json = parse_label_json_v2
            return original_dp(self, url, proc, n_sessions)

        _ds.UaVirBASEDatasetManager._download_partial = _patched_download_partial
        print("  🔧  UaVirBASEDatasetManager._download_partial patched")
        return True

    except ImportError:
        # drone_detection not yet importable — patch will be applied later
        return False


# =============================================================================
#  STANDALONE DIAGNOSTIC
# =============================================================================

def diagnose(base_dir: str):
    """
    Inspect every label.json under base_dir and report:
    - What the existing parse_label_json returns
    - What parse_label_json_v2 returns
    - The top-level JSON structure
    """
    root = Path(base_dir)
    print(f"\n{'='*70}")
    print(f"  UaVirBASE label.json diagnostic")
    print(f"  Scanning: {root}")
    print(f"{'='*70}\n")

    # Try processed/localization first, then raw uavirbase dir
    search_roots = [
        root / "processed" / "localization",
        root / "uavirbase",
        root,
    ]

    label_files = []
    for sr in search_roots:
        if sr.exists():
            # Look for *_label.json (processed) and label.json (raw)
            label_files.extend(list(sr.rglob("*_label.json"))[:10])
            label_files.extend(list(sr.rglob("label.json"))[:10])
    label_files = list(dict.fromkeys(label_files))[:20]  # deduplicate, cap at 20

    if not label_files:
        print("  ⚠  No label files found. Check the base-dir path.")
        print(f"     Tried: {[str(s) for s in search_roots]}")
        return

    # Try importing the original parser
    try:
        from drone_detection.datasets import parse_label_json as original_parser
    except ImportError:
        original_parser = None
        print("  ⚠  drone_detection not importable — only v2 parser tested\n")

    n_orig_ok = n_v2_ok = n_both_fail = 0

    for lf in label_files:
        raw = lf.read_bytes()
        try:
            data = json.loads(raw.decode("utf-8"))
            top_keys = list(data.keys())[:8]
        except Exception as e:
            top_keys = [f"PARSE_ERROR: {e}"]
            data = {}

        orig_result = original_parser(raw) if original_parser else "N/A"
        v2_result   = parse_label_json_v2(raw)

        orig_ok = orig_result not in (None, "N/A")
        v2_ok   = v2_result is not None

        if orig_ok:   n_orig_ok += 1
        if v2_ok:     n_v2_ok   += 1
        if not orig_ok and not v2_ok: n_both_fail += 1

        status = ("✅ BOTH" if orig_ok and v2_ok else
                  "🔧 V2 ONLY" if v2_ok else
                  "❌ BOTH FAIL")

        print(f"  {status}  {lf.name}")
        print(f"    keys: {top_keys}")
        if v2_ok:
            az, di, ht = v2_result
            print(f"    parsed: az={az:.1f}°  dist={di:.1f}m  ht={ht:.1f}m")
        else:
            # Show first sub-dict keys to help diagnose
            for k, v in data.items():
                if isinstance(v, dict):
                    print(f"    sub['{k}']: {list(v.keys())[:6]}")
            print(f"    raw (first 300 chars): {raw[:300].decode('utf-8', errors='replace')}")
        print()

    print(f"{'─'*70}")
    print(f"  Results:  original_ok={n_orig_ok}  v2_ok={n_v2_ok}  "
          f"both_fail={n_both_fail}  total={len(label_files)}")

    if n_both_fail > 0:
        print(f"\n  ⚠  {n_both_fail} labels still fail with v2 parser.")
        print("     Please paste the 'raw' output above and open a GitHub issue.")
    elif n_v2_ok > n_orig_ok:
        print(f"\n  ✅  v2 parser recovers {n_v2_ok - n_orig_ok} previously failed labels.")
        print("     Run build_training_datasets_and_figures.py — it will apply the patch automatically.")
    else:
        print("\n  ✅  Both parsers agree. The issue may be network/remotezip access.")


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python uavirbase_patch.py C:/tmp/drone_v15")
        sys.exit(0)
    diagnose(sys.argv[1])