# for MemVerse / RAG-Anything evaluation: whether to reusedataset directory underalready have LightRAG workspace (default reuse, only invalid/force rebuild / drop store) . 

from __future__ import annotations

import os
import shutil
from pathlib import Path


def reuse_lightrag_index_requested() -> bool:
    v = os.environ.get("MULTIMODEL_REUSE_LIGHTRAG_INDEX", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return True


def force_rebuild_lightrag_index_requested() -> bool:
    v = os.environ.get("MULTIMODEL_FORCE_REBUILD_LIGHTRAG_INDEX", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def lightrag_mmkg_all_vdb_nonempty(workdir: Path) -> bool:
    base = Path(workdir)
    return all(
        lightrag_vdb_chunks_nonempty(base / "MMKG" / k)
        for k in ("core", "episodic", "semantic")
    )


def lightrag_vdb_chunks_nonempty(workdir: Path) -> bool:
    p = Path(workdir) / "vdb_chunks.json"
    try:
        return p.is_file() and p.stat().st_size > 64
    except OSError:
        return False


def mmkg_triplet_progress_detected(workdir: Path):
    mmkg = Path(workdir) / "MMKG"
    if not mmkg.is_dir():
        return (False, False, False)
    return (
        lightrag_vdb_chunks_nonempty(mmkg / "core"),
        lightrag_vdb_chunks_nonempty(mmkg / "episodic"),
        lightrag_vdb_chunks_nonempty(mmkg / "semantic"),
    )


def wipe_mmkg_if_triplet_inconsistent(workdir: Path) -> bool:
    if (
        os.environ.get("MULTIMODEL_LIGHTRAG_KEEP_PARTIAL_MMKG", "")
        .strip()
        .lower()
        in ("1", "true", "yes", "on")
    ):
        return False
    root = Path(workdir)
    mmkg = root / "MMKG"
    if not mmkg.is_dir():
        return False
    states = mmkg_triplet_progress_detected(root)
    if not any(states):
        return False
    if all(states):
        return False
    shutil.rmtree(mmkg, ignore_errors=True)
    return True
