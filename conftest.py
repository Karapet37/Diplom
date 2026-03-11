from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROACHES_ROOT = ROOT / "roaches_viz"

for path in (ROOT, ROACHES_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

