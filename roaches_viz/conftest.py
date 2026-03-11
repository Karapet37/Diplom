from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

for path in (ROOT, HERE):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

