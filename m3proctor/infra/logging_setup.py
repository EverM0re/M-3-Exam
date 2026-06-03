from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


_LOGGER_NAME = "m3proctor"


def setup_logger(
    log_dir: Path,
    *,
    dataset_name: str = "",
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{dataset_name}" if dataset_name else ""
    log_file = log_dir / f"run{suffix}_{ts}.log"

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    logger.info("=== logging started -> %s ===", log_file)
    return logger, log_file


class _Tee:
    def __init__(self, primary, log_path: Path):
        self.primary = primary
        self.log_path = log_path
        self._fp = open(log_path, "a", encoding="utf-8")

    def write(self, s):
        try:
            self.primary.write(s)
        except Exception:
            pass
        try:
            self._fp.write(s)
            self._fp.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.primary.flush()
        except Exception:
            pass
        try:
            self._fp.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()
