from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set


_ROUND_RE = re.compile(r"[A-Za-z]\d+:\d+")


def _round_has_media(dlg: dict) -> bool:
    for key in ("image_file", "img_file", "pdf_file"):
        vals = dlg.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        if any(str(v).strip() for v in vals):
            return True
    return False


def _build_media_round_index(sessions: List[dict]) -> Set[str]:
    media_rounds: Set[str] = set()
    for sess in sessions:
        for dlg in sess.get("dialogues", []):
            rid = str(dlg.get("round", "")).strip()
            if rid and _round_has_media(dlg):
                media_rounds.add(rid)
    return media_rounds


def analyse(dataset_dir: str) -> Dict[str, dict]:
    sessions = json.load(open(os.path.join(dataset_dir, "sessions.json")))
    questions = json.load(open(os.path.join(dataset_dir, "question.json")))
    media_rounds = _build_media_round_index(sessions)

    per_type: Dict[str, dict] = defaultdict(
        lambda: {"n": 0, "mm": 0, "sf_img": 0}
    )

    for q in questions:
        qt = str(q.get("type", "?")).lower()
        sf = q.get("supporting_facts", "") or ""
        cited = _ROUND_RE.findall(sf)
        has_media_sf = any(rid in media_rounds for rid in cited)

        per_type[qt]["n"] += 1
        if has_media_sf:
            per_type[qt]["mm"] += 1
            per_type[qt]["sf_img"] += 1

    out: Dict[str, dict] = {}
    for qt, c in per_type.items():
        n = c["n"]
        out[qt] = {
            "n": n,
            "mm_ratio": c["mm"] / n if n else 0.0,
            "sf_img_ratio": c["sf_img"] / n if n else 0.0,
        }
    return out


def _print_table(stats: Dict[str, dict], title: str) -> None:
    order = ["ss", "ms", "tr", "fj", "mr", "fm", "th", "ii"]
    keys = [k for k in order if k in stats] + [
        k for k in stats if k not in order
    ]
    print(f"\n=== {title} ===")
    print(f"{'type':<6}{'n':>6}{'mm_ratio':>11}{'sf_img_ratio':>14}")
    tot_n = tot_mm = tot_sf = 0
    for k in keys:
        s = stats[k]
        tot_n += s["n"]
        tot_mm += round(s["mm_ratio"] * s["n"])
        tot_sf += round(s["sf_img_ratio"] * s["n"])
        print(f"{k.upper():<6}{s['n']:>6}{s['mm_ratio']:>11.3f}{s['sf_img_ratio']:>14.3f}")
    if tot_n:
        print(f"{'ALL':<6}{tot_n:>6}{tot_mm/tot_n:>11.3f}{tot_sf/tot_n:>14.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="Directory with sessions.json and question.json")
    args = ap.parse_args()
    stats = analyse(args.dataset_dir)
    _print_table(stats, os.path.basename(args.dataset_dir.rstrip("/")))


if __name__ == "__main__":
    main()
