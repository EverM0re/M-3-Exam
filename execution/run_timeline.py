#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from m3exam.functions.timeline.generate_timeline import main as _main

    _main()


if __name__ == "__main__":
    main()
