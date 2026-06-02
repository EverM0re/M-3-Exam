#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from m3exam.config.config_loader import cfg

# Round IDs of the form <Letter><digits>:<digits> e.g. "D2:5"
_ROUND_RE = re.compile(r"[A-Za-z]\d+:\d+")

# PDF filenames inside supporting_facts strings.
_PDF_FILE_RE = re.compile(r"[^\s,]+?\.pdf", re.IGNORECASE)

_IMG_EXT_RE = re.compile(r"\.(png|jpg|jpeg|webp|gif|bmp)$", re.IGNORECASE)


def _normalise_path_list(raw: Any) -> List[Path]:
    if raw in (None, "", []):
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw if str(x) and str(x).strip()]
    else:
        s = str(raw).strip()
        if not s:
            return []
        items = [t.strip() for t in s.split(",") if t.strip()] if "," in s else [s]
    return [Path(p) for p in items]


def _find_image(filename: str, search_dirs: List[Path]) -> Optional[Path]:
    for d in search_dirs:
        p = d / filename
        if p.is_file():
            return p
    return None


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r",\s*([\]\}])", r"\1", text)
        data = json.loads(text)
        if isinstance(data, list):
            return data
        print(f"  [WARN] {path} did not contain a JSON array - ignored")
        return []
    except Exception as e:
        print(f"  [WARN] Could not parse {path}: {e}")
        return []


def _attach_image_dirs(sessions: List[Dict[str, Any]], dirs: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sess in sessions:
        ns = dict(sess)
        ns["_image_dirs"] = [str(p) for p in dirs]
        out.append(ns)
    return out


def _resolve_base_sessions_path(input_dir: Path) -> Path:
    direct = input_dir / "sessions.json"
    if direct.is_file():
        return direct

    nested = sorted(input_dir.glob("*/sessions.json"))
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        print(
            f"  [WARN] multiple */sessions.json found under {input_dir}, "
            f"using the first:\n         " + "\n         ".join(str(c) for c in nested)
        )
        return nested[0]
    return direct


def _collect_base_question_paths(input_dir: Path) -> List[Path]:
    found: List[Path] = []
    for sub in sorted(input_dir.glob("questions_*/*/question.json")):
        if sub.is_file():
            found.append(sub)
    if not found:
        flat = input_dir / "question.json"
        if flat.is_file():
            found.append(flat)
    return found


def _find_timeline_file(input_dir: Path) -> Optional[Path]:
    timeline_root = input_dir / "timeline"
    if timeline_root.is_dir():
        named = sorted(timeline_root.glob("*/timeline_*.json"))
        if named:
            return named[0]
        plain = sorted(timeline_root.glob("*/timeline.json"))
        if plain:
            return plain[0]
        named_flat = sorted(timeline_root.glob("timeline_*.json"))
        if named_flat:
            return named_flat[0]
        plain_flat = timeline_root / "timeline.json"
        if plain_flat.is_file():
            return plain_flat

    flat_named = sorted(input_dir.glob("timeline_*.json"))
    if flat_named:
        return flat_named[0]
    flat_plain = input_dir / "timeline.json"
    if flat_plain.is_file():
        return flat_plain
    return None


def _load_base(input_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path, List[Path]]:
    sessions_path = _resolve_base_sessions_path(input_dir)
    if not sessions_path.is_file():
        print(f"[ERROR] base sessions.json not found under {input_dir}", file=sys.stderr)
        print(
            f"        looked at:\n          {input_dir / 'sessions.json'}\n"
            f"          {input_dir / '*' / 'sessions.json'}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(sessions_path, encoding="utf-8") as f:
        sessions = json.load(f)
    if not isinstance(sessions, list):
        print(f"[ERROR] {sessions_path} must contain a JSON array.", file=sys.stderr)
        sys.exit(1)

    img_dirs: List[Path] = []
    if (sessions_path.parent / "images").is_dir():
        img_dirs.append(sessions_path.parent / "images")
    img_dirs.append(sessions_path.parent)
    if (input_dir / "images").is_dir() and (input_dir / "images") not in img_dirs:
        img_dirs.append(input_dir / "images")
    sessions = _attach_image_dirs(sessions, img_dirs)

    pdf_dirs: List[Path] = []
    if (sessions_path.parent / "pdfs").is_dir():
        pdf_dirs.append(sessions_path.parent / "pdfs")
    if (input_dir / "pdfs").is_dir() and (input_dir / "pdfs") not in pdf_dirs:
        pdf_dirs.append(input_dir / "pdfs")
    if pdf_dirs:
        for sess in sessions:
            sess["_pdf_dirs"] = [str(p) for p in pdf_dirs]

    question_paths = _collect_base_question_paths(input_dir)
    questions: List[Dict[str, Any]] = []
    for qp in question_paths:
        questions.extend(_load_questions(qp))

    return sessions, questions, sessions_path, question_paths


_FIELD_REMAP = {
    "image_file": "img_file",
    "image_id":   "img_id",
    "image_description": "img_description",
}


def _unify_dialogue_keys(dlg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in dlg.items():
        out[_FIELD_REMAP.get(k, k)] = v
    return out


def _renumber_segment(
    sessions: List[Dict[str, Any]],
    start_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    round_map: Dict[str, str] = {}
    new_sessions: List[Dict[str, Any]] = []

    for offset, sess in enumerate(sessions):
        new_sid = f"D{start_idx + offset}"
        new_dialogues: List[Dict[str, Any]] = []

        for j, dlg in enumerate(sess.get("dialogues", []), start=1):
            old_round = str(dlg.get("round", "")).strip()
            new_round = f"{new_sid}:{j}"
            if old_round:
                if old_round in round_map and round_map[old_round] != new_round:
                    print(
                        f"  [WARN] Duplicate old round id '{old_round}' "
                        f"(was '{round_map[old_round]}', now '{new_round}'). "
                        "Keeping latest mapping."
                    )
                round_map[old_round] = new_round

            new_dlg = _unify_dialogue_keys(dlg)
            new_dlg["round"] = new_round
            new_dialogues.append(new_dlg)

        new_sess = dict(sess)
        new_sess["session_id"] = new_sid
        new_sess["dialogues"] = new_dialogues
        new_sessions.append(new_sess)

    return new_sessions, round_map


def _ext_for(name: str) -> str:
    m = _IMG_EXT_RE.search(name)
    return m.group(0).lower() if m else ".jpg"


def collect_and_rename_images(
    sessions: List[Dict[str, Any]],
) -> Tuple[List[Tuple[Path, str]], int]:
    next_idx = 1
    by_path: Dict[str, str] = {}
    copy_plan: List[Tuple[Path, str]] = []
    dropped = 0

    for sess in sessions:
        search_dirs = [Path(p) for p in sess.get("_image_dirs", [])]

        for dlg in sess.get("dialogues", []):
            files = dlg.get("img_file")
            if not isinstance(files, list) or not files:
                for k in ("img_file", "img_id", "img_description"):
                    if dlg.get(k) == [] or dlg.get(k) == "":
                        dlg.pop(k, None)
                continue

            new_files: List[str] = []
            kept_indices: List[int] = []

            for i, raw in enumerate(files):
                fn = Path(str(raw).strip()).name
                if not fn:
                    dropped += 1
                    continue

                src = _find_image(fn, search_dirs)
                if src is None:
                    # Silently drop the reference; counter unaffected so the
                    # next found image gets this slot (numbering shifts forward).
                    dropped += 1
                    continue

                key = str(src.resolve())
                if key not in by_path:
                    new_name = f"img_{next_idx}{_ext_for(fn)}"
                    by_path[key] = new_name
                    copy_plan.append((src, new_name))
                    next_idx += 1
                new_files.append(by_path[key])
                kept_indices.append(i)

            if new_files:
                dlg["img_file"] = new_files
                dlg["img_id"] = [
                    int(re.search(r"img_(\d+)", n).group(1))
                    for n in new_files
                    if re.search(r"img_(\d+)", n)
                ]
                desc = dlg.get("img_description")
                if isinstance(desc, list) and len(desc) == len(files):
                    dlg["img_description"] = [desc[i] for i in kept_indices]
            else:
                for k in ("img_file", "img_id", "img_description"):
                    dlg.pop(k, None)

    return copy_plan, dropped


def copy_planned_images(copy_plan: List[Tuple[Path, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, new_name in copy_plan:
        try:
            shutil.copy2(src, out_dir / new_name)
            copied += 1
        except Exception as e:
            print(f"  [WARN] Failed to copy {src} -> {new_name}: {e}")
    print(f"  Images copied   : {copied}")


def _find_pdf(filename: str, search_dirs: List[Path]) -> Optional[Path]:
    for d in search_dirs:
        p = d / filename
        if p.is_file():
            return p
    return None


def collect_and_rename_pdfs(
    sessions: List[Dict[str, Any]],
) -> Tuple[List[Tuple[Path, str]], Dict[str, str], int]:
    next_idx = 1
    by_path: Dict[str, str] = {}
    basename_map: Dict[str, str] = {}
    copy_plan: List[Tuple[Path, str]] = []
    dropped = 0

    for sess in sessions:
        search_dirs = [Path(p) for p in sess.get("_pdf_dirs", [])]
        if not search_dirs:
            continue

        original_source_pdf = str(sess.get("source_pdf", "")).strip()

        for dlg in sess.get("dialogues", []):
            files = dlg.get("pdf_file")
            if not isinstance(files, list) or not files:
                if dlg.get("pdf_file") in ([], ""):
                    dlg.pop("pdf_file", None)
                continue

            new_files: List[str] = []
            kept_indices: List[int] = []

            for i, raw in enumerate(files):
                fn = Path(str(raw).strip()).name
                if not fn:
                    dropped += 1
                    continue

                src = _find_pdf(fn, search_dirs)
                if src is None:
                    dropped += 1
                    continue

                key = str(src.resolve())
                if key not in by_path:
                    new_name = f"file_{next_idx}.pdf"
                    by_path[key] = new_name
                    basename_map[fn.lower()] = new_name
                    copy_plan.append((src, new_name))
                    next_idx += 1
                else:
                    basename_map.setdefault(fn.lower(), by_path[key])

                new_files.append(by_path[key])
                kept_indices.append(i)

            if new_files:
                dlg["pdf_file"] = new_files
                if isinstance(dlg.get("pdf_id"), list):
                    dlg["pdf_id"] = [
                        int(re.search(r"file_(\d+)", n).group(1))
                        for n in new_files
                        if re.search(r"file_(\d+)", n)
                    ]
                desc = dlg.get("pdf_description")
                if isinstance(desc, list) and len(desc) == len(files):
                    dlg["pdf_description"] = [desc[i] for i in kept_indices]
            else:
                for k in ("pdf_file", "pdf_id", "pdf_description"):
                    dlg.pop(k, None)

        if original_source_pdf:
            new_src = basename_map.get(original_source_pdf.lower())
            if new_src:
                sess["source_pdf"] = new_src

    return copy_plan, basename_map, dropped


def copy_planned_pdfs(copy_plan: List[Tuple[Path, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, new_name in copy_plan:
        try:
            shutil.copy2(src, out_dir / new_name)
            copied += 1
        except Exception as e:
            print(f"  [WARN] Failed to copy {src} -> {new_name}: {e}")
    print(f"  PDFs copied     : {copied}")


def _update_supporting_facts(
    sf: str,
    round_map: Dict[str, str],
    pdf_map: Optional[Dict[str, str]] = None,
) -> str:
    if not sf:
        return sf

    pdf_map = pdf_map or {}
    unmatched: List[str] = []

    def _replace_pdf(m: re.Match) -> str:
        old = m.group(0)
        new = pdf_map.get(old.lower())
        if new is None:
            unmatched.append(old)
            return old
        return new
    result = _PDF_FILE_RE.sub(_replace_pdf, sf)

    def _replace_round(m: re.Match) -> str:
        old = m.group(0)
        new = round_map.get(old)
        if new is None:
            unmatched.append(old)
            return old
        return new
    result = _ROUND_RE.sub(_replace_round, result)

    for u in unmatched:
        print(f"  [WARN] supporting_facts token '{u}' has no mapping - kept as-is")
    return result


def remap_questions(
    questions: List[Dict[str, Any]],
    round_map: Dict[str, str],
    pdf_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for q in questions:
        nq = dict(q)
        sf = str(nq.get("supporting_facts", ""))
        nq["supporting_facts"] = _update_supporting_facts(sf, round_map, pdf_map)
        out.append(nq)
    return out


def write_questions_compact(questions: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, item in enumerate(questions):
            line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            comma = "," if i < len(questions) - 1 else ""
            f.write(f"  {line}{comma}\n")
        f.write("]\n")


def _strip_internal(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sess in sessions:
        ns = {k: v for k, v in sess.items() if not str(k).startswith("_")}
        out.append(ns)
    return out


def main() -> None:
    base_dir_str   = cfg("finalize_dataset", "finalize_input_dir")  or ""
    output_dir_str = cfg("finalize_dataset", "finalize_output_dir") or ""

    base_dir   = Path(base_dir_str)
    output_dir = Path(output_dir_str)

    if not base_dir_str or not base_dir.is_dir():
        print(f"[ERROR] finalize_input_dir not found: '{base_dir}'", file=sys.stderr)
        sys.exit(1)
    if not output_dir_str:
        print("[ERROR] finalize_output_dir is not configured.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Finalize Dataset")
    print("=" * 64)
    print(f"  Base input : {base_dir}")
    print(f"  Output     : {output_dir}")
    print("=" * 64)

    print("\n[1] Loading inputs ...")
    base_sessions, base_questions, base_sessions_path, base_question_paths = _load_base(base_dir)
    print(f"      sessions.json : {base_sessions_path}")
    print(f"      sessions      : {len(base_sessions)}")
    if base_question_paths:
        print(f"      questions     : {len(base_questions)} item(s) merged from")
        for qp in base_question_paths:
            print(f"        - {qp}")
    else:
        print(f"      questions     : (none found under questions_*/*/question.json)")

    print("\n[2] Renumbering sessions/rounds (D1, D2, ...) and unifying schema ...")
    merged_sessions, round_map = _renumber_segment(base_sessions, start_idx=1)
    print(f"      sessions after renumber : {len(merged_sessions)}")

    print("\n[3] Renaming + copying images ...")
    copy_plan, dropped = collect_and_rename_images(merged_sessions)
    print(f"      unique images planned : {len(copy_plan)}")
    if dropped:
        print(
            f"      dropped missing refs  : {dropped} "
            "(referenced in sessions but no source file on disk - sessions updated)"
        )
    copy_planned_images(copy_plan, output_dir / "images")

    print("\n[4] Renaming + copying PDFs ...")
    pdf_copy_plan, pdf_basename_map, pdf_dropped = collect_and_rename_pdfs(merged_sessions)
    print(f"      unique PDFs planned   : {len(pdf_copy_plan)}")
    if pdf_dropped:
        print(
            f"      dropped missing refs  : {pdf_dropped} "
            "(referenced in sessions but no source file on disk - sessions updated)"
        )
    if pdf_copy_plan:
        copy_planned_pdfs(pdf_copy_plan, output_dir / "pdfs")

    merged_questions = remap_questions(base_questions, round_map, pdf_basename_map)

    print("\n[5] Writing sessions.json ...")
    out_sessions = output_dir / "sessions.json"
    with open(out_sessions, "w", encoding="utf-8") as f:
        json.dump(_strip_internal(merged_sessions), f, indent=2, ensure_ascii=False)
    print(f"      -> {out_sessions}")

    print("\n[6] Writing question.json ...")
    out_questions = output_dir / "question.json"
    if merged_questions:
        write_questions_compact(merged_questions, out_questions)
        print(f"      -> {out_questions}  ({len(merged_questions)} item(s))")
    else:
        print("      no questions across inputs - skipping")

    print("\n[7] Copying timeline ...")
    timeline_src = _find_timeline_file(base_dir)
    if timeline_src is None:
        print(f"      no timeline_*.json found under {base_dir}/timeline/ - skipping")
    else:
        timeline_dst = output_dir / timeline_src.name
        try:
            shutil.copy2(timeline_src, timeline_dst)
            print(f"      -> {timeline_dst}")
            print(f"      (source: {timeline_src})")
        except Exception as e:
            print(f"      [WARN] failed to copy timeline {timeline_src} -> {timeline_dst}: {e}")

    print(f"\n{'=' * 64}")
    print(f"  Done.  Output: {output_dir}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
