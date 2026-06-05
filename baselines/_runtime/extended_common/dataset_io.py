from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_sessions(data_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    direct = data_dir / "sessions.json"
    if direct.is_file():
        path = direct
    else:
        cand = sorted(data_dir.glob("*/sessions.json"))
        if not cand:
            raise FileNotFoundError(f"sessions.json not found in {data_dir}")
        path = cand[0]
    with open(path, encoding="utf-8") as f:
        sessions = json.load(f)
    return sessions, path


def load_questions(data_dir: Path) -> List[Dict[str, Any]]:
    qp = data_dir / "question.json"
    if not qp.is_file():
        raise FileNotFoundError(f"question.json not found: {qp}")
    raw = qp.read_text(encoding="utf-8")
    raw = re.sub(r",\s*([\]\}])", r"\1", raw)
    return json.loads(raw)


def _img_block_for_dialogue(dlg: Dict[str, Any]) -> str:
    files = dlg.get("img_file") or []
    descs = dlg.get("img_description") or []
    if not files:
        return ""
    parts: List[str] = []
    for i, fn in enumerate(files):
        d = descs[i] if i < len(descs) else ""
        if d:
            parts.append(f'{fn} ("{d}")')
        else:
            parts.append(str(fn))
    return f"[Images attached: {', '.join(parts)}]"


def _pdf_block_for_dialogue(dlg: Dict[str, Any]) -> str:
    files = dlg.get("pdf_file") or []
    descs = dlg.get("pdf_description") or []
    if not files:
        return ""
    parts: List[str] = []
    for i, fn in enumerate(files):
        d = descs[i] if i < len(descs) else ""
        if d:
            parts.append(f'{fn} ("{d}")')
        else:
            parts.append(str(fn))
    return f"[PDF attached: {', '.join(parts)}]"


def dialogue_to_text(dlg: Dict[str, Any], session_id: str = "") -> str:
    rnd = dlg.get("round", "")
    user = dlg.get("user", "")
    asst = dlg.get("assistant", "")
    media: List[str] = []
    img_blk = _img_block_for_dialogue(dlg)
    if img_blk:
        media.append(img_blk)
    pdf_blk = _pdf_block_for_dialogue(dlg)
    if pdf_blk:
        media.append(pdf_blk)
    media_text = ("\n" + "\n".join(media)) if media else ""
    return f"[{rnd}]\nUser: {user}\nAssistant: {asst}{media_text}"


def session_to_text(session: Dict[str, Any]) -> str:
    sid = session.get("session_id", "?")
    date = session.get("date", "")
    header = f"=== Session {sid}" + (f" ({date})" if date else "") + " ==="
    lines = [header]
    for dlg in session.get("dialogues", []):
        lines.append("")
        lines.append(dialogue_to_text(dlg, sid))
    return "\n".join(lines)


def iter_dialogues(
    sessions: List[Dict[str, Any]],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for sess in sessions:
        sid = str(sess.get("session_id", "?"))
        for dlg in sess.get("dialogues", []):
            rnd = str(dlg.get("round", ""))
            out.append((sid, rnd, dlg))
    return out


def collect_dialogue_images(
    dlg: Dict[str, Any], images_dir: Path
) -> List[str]:
    out: List[str] = []
    for fn in dlg.get("img_file") or []:
        p = images_dir / str(fn).strip()
        if p.is_file():
            out.append(str(p))
    return out


def collect_dialogue_pdfs(
    dlg: Dict[str, Any], data_dir: Path
) -> List[str]:
    out: List[str] = []
    for fn in dlg.get("pdf_file") or []:
        name = str(fn).strip()
        for cand in (data_dir / "pdfs" / name, data_dir / name, data_dir / "images" / name):
            if cand.is_file():
                out.append(str(cand))
                break
    return out


def parse_session_ids_from_supporting_facts(sf: str) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for sid in re.findall(r"([A-Z]\d+):\d+", sf or ""):
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def parse_round_ids_from_supporting_facts(sf: str) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for r in re.findall(r"([A-Z]\d+:\d+)", sf or ""):
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out
