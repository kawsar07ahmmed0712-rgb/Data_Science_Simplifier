from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    streamlit_file = Path("ui") / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_file)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())