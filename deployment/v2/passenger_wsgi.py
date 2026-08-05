# -*- coding: utf-8 -*-
"""
passenger_wsgi.py — cPanel "Setup Python App" (Passenger) entry point.

Passenger requires a file with exactly this name at the app root, exposing
a WSGI callable named `application`. cPanel's "Application startup file" /
"Application Entry point" fields only take effect at the moment you click
Save on that page — if this file is later deleted or overwritten (e.g. by
a git pull), Passenger falls back to its own generic placeholder ("It
works!") instead of re-deriving the wiring from those fields. So this file
must stay physically present and tracked in git.

This intentionally does NOT use socketio.run() / Flask-SocketIO's own
server — Passenger serves the plain Flask `app` object directly. That
means the Real-Time (WebSocket) panel will not function under this entry
point; Single File detection and Batch Scan (both plain HTTP POST
endpoints) are unaffected.
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from app_v2 import app as application  # noqa: E402,F401
