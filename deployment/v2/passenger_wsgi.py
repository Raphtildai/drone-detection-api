# -*- coding: utf-8 -*-
"""
passenger_wsgi.py — cPanel "Setup Python App" (Passenger) entry point.

cPanel's Python App tool looks for a module named exactly this, at the
app's root directory, exposing a WSGI callable named `application`.

This intentionally does NOT use socketio.run() / Flask-SocketIO's own
server — Passenger serves the plain Flask WSGI `app` object directly.
That means the Real-Time (WebSocket) panel will not function under this
entry point; Single File detection and Batch Scan (both plain HTTP POST
endpoints) are unaffected.
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from app_v2 import app as application  # noqa: E402,F401
