# FM / image questions: not using supporting_facts  given that, as much as possible「retrieve context → round → disk image」chain together. 
# LightRAG/MMKG fragmentmaynotcontain standard D10:4, so addmultiple-form literal match + based on corpus and question vocabulary overlap. 

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from m3exam.baselines._runtime.multimodel_common.eval_metrics import parsing_gold_rounds

def _needs_image_gallery(question: str) -> bool:
    q = (question or "").lower()
    if "img_<number>" in q or "image file name" in q or "file name" in q:
        return True
    return bool(
        re.search(r"\b(image|photo|picture|visual|image|photo|image)\b", question or "", re.I)
    )


def question_wants_image_gallery(question: str, question_type: str = "") -> bool:
    qt = str(question_type or "").strip().lower()
    if qt in ("fm", "mr"):
        return True
    return _needs_image_gallery(question)


def rounds_from_lightrag_context(text: str) -> List[str]:
    if not text:
        return []
    patterns = (
        re.compile(r"\b([Dd]\d+\s*:\s*\d+)\b"),
        re.compile(r"\[\s*([Dd]\d+\s*:\s*\d+)\s*\]"),
        re.compile(r"(?i)round\s+([Dd]\d+\s*:\s*\d+)"),
        re.compile(r"\|\s*Round\s+([Dd]\d+\s*:\s*\d+)\s*\|"),
    )
    seen: Set[str] = set()
    out: List[str] = []
    for pat in patterns:
        for m in pat.finditer(text):
            raw = m.group(1).replace(" ", "")
            if raw[:1] == "d":
                raw = "D" + raw[1:]
            if raw not in seen:
                seen.add(raw)
                out.append(raw)
    return out


def _tokens(s: str) -> Set[str]:
    t = (s or "").lower()
    for ch in string.punctuation:
        t = t.replace(ch, " ")
    return {w for w in t.split() if len(w) > 2}


def dialogue_image_rel_paths(dlg: Dict[str, Any]) -> List[str]:
    raw = (
        dlg.get("img_file")
        or dlg.get("image_file")
        or dlg.get("img_files")
        or dlg.get("images")
        or []
    )
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, list):
            continue
        s = str(item).strip().replace("\\", "/")
        if s:
            out.append(s)
    return out


def dialogue_image_trace_basenames(dlg: Dict[str, Any]) -> List[str]:
    return [Path(r).name for r in dialogue_image_rel_paths(dlg)]


def resolve_dialogue_image_paths(
    dlg: Dict[str, Any],
    images_dir: Path,
    *,
    max_images: int = 10,
) -> List[str]:
    paths: List[str] = []
    for rel in dialogue_image_rel_paths(dlg):
        p = (images_dir / rel).resolve()
        if p.is_file():
            s = str(p)
            if s not in paths:
                paths.append(s)
        if len(paths) >= max_images:
            break
    return paths


def image_paths_for_rounds(
    rounds: List[str],
    sessions: List[Dict[str, Any]],
    images_dir: Path,
) -> List[str]:
    want = {str(r).replace(" ", "") for r in rounds}
    paths: List[str] = []
    for sess in sessions:
        for dlg in sess.get("dialogues") or []:
            rnd = str(dlg.get("round", "")).replace(" ", "")
            if rnd not in want:
                continue
            for s in resolve_dialogue_image_paths(dlg, images_dir, max_images=24):
                if s not in paths:
                    paths.append(s)
    return paths


def merge_supporting_facts_images_first(
    supporting_facts: str,
    sessions: List[Dict[str, Any]],
    images_dir: Path,
    paths: List[str],
    *,
    max_gold_paths: int = 32,
) -> List[str]:
    gold = sorted(parsing_gold_rounds(supporting_facts))
    if not gold:
        return list(paths)
    gold_paths = collect_paths_for_rounds_ordered(
        gold, sessions, images_dir, max_paths=max_gold_paths
    )
    seen: Set[str] = set()
    out: List[str] = []
    for p in gold_paths + list(paths):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def collect_paths_for_rounds_ordered(
    rounds: List[str],
    sessions: List[Dict[str, Any]],
    images_dir: Path,
    *,
    max_paths: int,
) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for rnd in rounds:
        for p in image_paths_for_rounds([rnd], sessions, images_dir):
            if p not in seen:
                seen.add(p)
                out.append(p)
            if len(out) >= max_paths:
                return out
    return out


def _meta_overlap_score(question: str, meta: Dict[str, Any]) -> float:
    blob = " ".join(
        str(meta.get(k) or "") for k in ("user", "assistant", "dialogue_vis")
    )
    qt, mt = _tokens(question), _tokens(blob)
    if not qt:
        return 0.0
    return len(qt & mt) / max(len(qt), 1)


def corpus_text_overlap_round_ids(
    question: str,
    corpus_meta: List[Dict[str, Any]],
    *,
    budget_rounds: int,
) -> List[str]:
    scored: List[tuple[float, str]] = []
    for meta in corpus_meta:
        rnd = str(meta.get("round", "")).replace(" ", "")
        if not rnd:
            continue
        scored.append((_meta_overlap_score(question, meta), rnd))
    scored.sort(key=lambda x: (-x[0], x[1]))
    seen: Set[str] = set()
    out: List[str] = []
    for sc, rnd in scored:
        if sc <= 0 and out:
            break
        if rnd in seen:
            continue
        seen.add(rnd)
        out.append(rnd)
        if len(out) >= budget_rounds:
            break
    if not out:
        for meta in corpus_meta:
            rnd = str(meta.get("round", "")).replace(" ", "")
            if not rnd or rnd in seen:
                continue
            seen.add(rnd)
            out.append(rnd)
            if len(out) >= min(8, budget_rounds):
                break
    return out


def fm_overlap_round_ids(
    question: str,
    corpus_meta: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    images_dir: Path,
    *,
    budget_rounds: int,
) -> List[str]:
    scored: List[tuple[float, str]] = []
    for meta in corpus_meta:
        rnd = str(meta.get("round", "")).replace(" ", "")
        if not rnd:
            continue
        if not image_paths_for_rounds([rnd], sessions, images_dir):
            continue
        scored.append((_meta_overlap_score(question, meta), rnd))
    scored.sort(key=lambda x: (-x[0], x[1]))
    seen: Set[str] = set()
    out: List[str] = []
    for _sc, rnd in scored:
        if rnd in seen:
            continue
        seen.add(rnd)
        out.append(rnd)
        if len(out) >= budget_rounds:
            break
    min_fill = min(8, budget_rounds)
    if len(out) < min_fill:
        for meta in corpus_meta:
            rnd = str(meta.get("round", "")).replace(" ", "")
            if not rnd or rnd in seen:
                continue
            if not image_paths_for_rounds([rnd], sessions, images_dir):
                continue
            seen.add(rnd)
            out.append(rnd)
            if len(out) >= budget_rounds:
                break
    return out


def _normalize_eval_answer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    for sep in (". ", ".\n", "!\n", "?\n", "\n"):
        if sep in t:
            t = t.split(sep, 1)[0].strip()
            break
    lower = t.lower()
    for prefix in (
        "the answer is ",
        "answer: ",
        "based on the memories, ",
        "based on the conversation, ",
        "the recommended ",
        "it is ",
        "it would be ",
    ):
        if lower.startswith(prefix):
            t = t[len(prefix) :].strip()
            lower = t.lower()
    t = t.strip(".,;:!?\"'")
    if len(t) > 120:
        t = t[:120].rsplit(" ", 1)[0]
    return t.strip()


def fm_resolve_image_paths_traced(
    question: str,
    ctx: str,
    corpus_meta: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    images_dir: Path,
    *,
    top_k: int,
    hint_rounds: Optional[List[str]] = None,
    question_type: str = "",
    fm_min_images: int = 10,
) -> tuple[List[str], List[Dict[str, Any]]]:
    needs_gallery = question_wants_image_gallery(question, question_type)
    min_gallery = max(10, int(fm_min_images or 10))
    max_paths = (
        max(top_k * 2, top_k + 2, min_gallery)
        if needs_gallery
        else max(top_k * 3, 12)
    )
    trace: List[Dict[str, Any]] = []
    seen_paths: Set[str] = set()

    def _append_rounds(rounds: List[str], source: str) -> None:
        for rnd in rounds:
            rr = str(rnd).replace(" ", "")
            if not rr:
                continue
            for p in image_paths_for_rounds([rr], sessions, images_dir):
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                trace.append(
                    {
                        "image_file": Path(p).name,
                        "image_path": p,
                        "round_id": rr,
                        "resolve_source": source,
                    }
                )
                if len(trace) >= max_paths:
                    return

    hint = [str(r).replace(" ", "") for r in (hint_rounds or []) if str(r).strip()]
    if hint:
        _append_rounds(hint, "retrieve_round")
    if trace and needs_gallery:
        return [t["image_path"] for t in trace], trace

    rounds_ctx = rounds_from_lightrag_context(ctx)
    _append_rounds(rounds_ctx, "context_round")
    if trace and needs_gallery:
        return [t["image_path"] for t in trace], trace

    if not needs_gallery:
        return [t["image_path"] for t in trace], trace

    tail_rounds = [str(m.get("round", "")).replace(" ", "") for m in corpus_meta[-top_k * 3 :]]
    tail_rounds = [r for r in tail_rounds if r]
    _append_rounds(tail_rounds, "corpus_tail")
    if trace:
        return [t["image_path"] for t in trace], trace

    overlap_rounds = fm_overlap_round_ids(
        question,
        corpus_meta,
        sessions,
        images_dir,
        budget_rounds=max(top_k * 8, 16),
    )
    _append_rounds(overlap_rounds, "overlap")
    return [t["image_path"] for t in trace], trace


def fm_resolve_image_paths(
    question: str,
    ctx: str,
    corpus_meta: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    images_dir: Path,
    *,
    top_k: int,
    hint_rounds: Optional[List[str]] = None,
    question_type: str = "",
    fm_min_images: int = 10,
) -> List[str]:
    paths, _trace = fm_resolve_image_paths_traced(
        question,
        ctx,
        corpus_meta,
        sessions,
        images_dir,
        top_k=top_k,
        hint_rounds=hint_rounds,
        question_type=question_type,
        fm_min_images=fm_min_images,
    )
    return paths
