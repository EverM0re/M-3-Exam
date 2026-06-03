from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from m3exam.m3proctor.core.retriever import RetrievedItem


_VISUAL_REFERRING_RE = re.compile(
    r"\b("
    r"image[s]?|images?\s+(of|showing)|photo[s]?|picture[s]?|figure[s]?|"
    r"diagram[s]?|screenshot[s]?|visual(?:ly)?|"
    r"shown|depict(?:ed|s|ing)?|displayed|appears?\s+in|"
    r"file\s*name[s]?|filename[s]?|"
    r"which\s+(one|image|photo|picture|figure)|"
    r"\bimg[_-]?\d+\b"
    r")\b",
    re.IGNORECASE,
)
_FILENAME_INTENT_RE = re.compile(
    r"(file\s*name|filename|"
    r"which\s+(image|photo|picture|figure|img|file)|"
    r"matching\s+image[s]?|identify\s+the\s+image|name\s+the\s+image|"
    r"return.*image\s+file|among\s+.*\s+images?\s+(shows?|which))",
    re.IGNORECASE,
)
_MULTI_IMG_RE = re.compile(r"\bimg[_-]?\d+\b", re.IGNORECASE)

_PDF_REFERRING_RE = re.compile(
    r"\b(paper|report|document|pdf|file_\d+\.pdf|article|study|publication)\b",
    re.IGNORECASE,
)

_CHART_QUESTION_RE = re.compile(
    r"\b(chart|graph|plot|bar\s*chart|pie\s*chart|histogram|figure|"
    r"percentage|percent|how\s+many|how\s+much|"
    r"axis|legend|x[-_\s]?axis|y[-_\s]?axis|"
    r"percentage\s+points?|share\s+of|rate\s+of)\b|%",
    re.IGNORECASE,
)


def _collect_context_text(items: List[RetrievedItem], max_chars: int = 12000) -> str:
    parts = [it.chunk.content for it in items]
    s = "\n\n--CHUNK--\n".join(parts)
    return s[:max_chars]


def _collect_images(
    items: List[RetrievedItem], *, max_images: int
) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for it in items:
        for ip in it.chunk.image_paths:
            if ip not in seen and ip:
                seen.add(ip)
                out.append(ip)
                if len(out) >= max_images:
                    return out
    return out


def _collect_pdfs(items: List[RetrievedItem]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for it in items:
        for pp in it.chunk.pdf_paths:
            if pp not in seen and pp:
                seen.add(pp)
                out.append(pp)
    return out


def _collect_hit_pdf_pages(items: List[RetrievedItem]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    seen: set = set()
    for it in items:
        if it.chunk.kind != "pdf_page":
            continue
        for pp in it.chunk.pdf_paths:
            key = (pp, int(it.chunk.pdf_page_index))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _maybe_render_pdf_pages(
    pdf_paths: List[str],
    *,
    cache_dir: Path,
    dpi: int = 144,
    max_pages_per_pdf: int = 3,
    total_cap: int = 6,
) -> List[str]:
    from m3exam.m3proctor.infra.pdf_render import render_pdf_pages

    out: List[str] = []
    for pp in pdf_paths:
        try:
            pages = render_pdf_pages(pp, cache_dir, dpi=dpi, max_pages=max_pages_per_pdf)
        except Exception as e:
            print(f"[answerer] pdf render failed: {e}")
            pages = []
        for p in pages:
            out.append(p)
            if len(out) >= total_cap:
                return out
    return out


def _render_specific_pdf_pages(
    pdf_page_keys: List[Tuple[str, int]],
    *,
    cache_dir: Path,
    dpi: int = 144,
    total_cap: int = 6,
) -> List[str]:
    if not pdf_page_keys:
        return []
    try:
        import fitz
    except ImportError:
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: List[str] = []
    from collections import defaultdict
    grouped: Dict[str, List[int]] = defaultdict(list)
    for pp, pi in pdf_page_keys:
        grouped[pp].append(pi)
    for pp, pages in grouped.items():
        try:
            doc = fitz.open(pp)
        except Exception as e:
            print(f"[answerer] open pdf failed: {pp}: {e}")
            continue
        try:
            stem = Path(pp).stem
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for pi in pages:
                if 0 <= pi < doc.page_count:
                    out_path = cache_dir / f"{stem}_p{pi+1:02d}.png"
                    if not out_path.is_file():
                        page = doc.load_page(pi)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        pix.save(str(out_path))
                    out.append(str(out_path))
                    if len(out) >= total_cap:
                        return out
        finally:
            doc.close()
    return out


def build_messages(
    question: str,
    items: List[RetrievedItem],
    *,
    image_paths: List[str],
    pdf_page_paths: List[str],
    fm_image_filenames: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    context = _collect_context_text(items)
    prompt = (
        "You are answering a memory-QA question. Use ONLY the chunks below "
        "(and any attached images / rendered PDF pages).\n\n"
        "=== CHUNKS ===\n"
        f"{context}\n"
        "=== END ===\n\n"
        f"Question: {question}\n\n"
        "Give a short, direct answer (a few words or a brief phrase). "
        "Do not add explanations or repeat the question."
    )
    if fm_image_filenames:
        prompt += (
            "\n\nThe candidate images attached below come from the retrieved rounds. "
            "If the question asks for image file name(s), reply with one to three matching "
            f"file names from this list (comma-separated): {', '.join(fm_image_filenames)}. "
            "Do not invent file names that are not in this list. "
            "Do not respond with the placeholder img_<number>."
        )

    from m3exam.m3proctor.infra.llm_client import _image_data_url

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for pv in pdf_page_paths:
        url = _image_data_url(pv)
        if not url:
            continue
        content.append({"type": "text", "text": f"[PDF page] {Path(pv).name}"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    for ip in image_paths:
        url = _image_data_url(ip)
        if not url:
            continue
        content.append({"type": "text", "text": f"[image] {Path(ip).name}"})
        content.append({"type": "image_url", "image_url": {"url": url}})

    if len(content) == 1:
        return [{"role": "user", "content": prompt}]
    return [{"role": "user", "content": content}]


def _visual_score(question: str, items: List[RetrievedItem], flags: Dict[str, bool]) -> float:
    if not items:
        retrieval_evidence = 0.0
        chart_evidence = 0.0
    else:
        retrieval_evidence = sum(1 for it in items if it.chunk.has_image) / len(items)
        chart_evidence = sum(1 for it in items if it.chunk.has_chart) / len(items)
    classifier_signal = 1.0 if flags.get("needs_image") else 0.0
    surface_signal = 1.0 if _VISUAL_REFERRING_RE.search(question or "") else 0.0
    chart_signal = 1.0 if flags.get("needs_chart") else 0.0
    return (
        0.5 * retrieval_evidence
        + 0.35 * classifier_signal
        + 0.35 * surface_signal
        + 0.35 * chart_evidence
        + 0.30 * chart_signal
    )


def _pdf_score(question: str, items: List[RetrievedItem], flags: Dict[str, bool]) -> float:
    if not items:
        retrieval_evidence = 0.0
    else:
        retrieval_evidence = sum(1 for it in items if it.chunk.has_pdf) / len(items)
    classifier_signal = 1.0 if flags.get("needs_pdf") else 0.0
    surface_signal = 1.0 if _PDF_REFERRING_RE.search(question or "") else 0.0
    return 0.5 * retrieval_evidence + 0.35 * classifier_signal + 0.35 * surface_signal


def _chart_score(question: str, items: List[RetrievedItem], flags: Dict[str, bool]) -> float:
    if not items:
        retrieval_evidence = 0.0
    else:
        retrieval_evidence = sum(1 for it in items if it.chunk.has_chart) / len(items)
    classifier_signal = 1.0 if flags.get("needs_chart") else 0.0
    surface_signal = 1.0 if _CHART_QUESTION_RE.search(question or "") else 0.0
    return 0.55 * retrieval_evidence + 0.35 * classifier_signal + 0.3 * surface_signal


def _needs_filename_answer(question: str, items: List[RetrievedItem]) -> bool:
    if not _FILENAME_INTENT_RE.search(question or ""):
        return False
    return any(it.chunk.image_paths for it in items)


def _is_multi_image_compose(question: str) -> bool:
    return len(_MULTI_IMG_RE.findall(question or "")) >= 2


_EVASIVE_RE = re.compile(
    r"\b(i\s+don'?t\s+know|i\s+do\s+not\s+know|"
    r"based\s+on\s+the\s+context|"
    r"no\s+specific|no\s+information|"
    r"not\s+(provided|mentioned|specified|available|stated|present|given)|"
    r"unable\s+to\s+(answer|determine|find|provide|tell)|"
    r"cannot\s+(answer|determine|find)|can'?t\s+(answer|determine|tell)|"
    r"insufficient\s+(information|context)|"
    r"the\s+(provided|given)\s+context\s+does\s+not|"
    r"unclear\s+from\s+the\s+context|"
    r"is\s+not\s+visible|not\s+shown\s+in\s+the\s+text)\b",
    re.IGNORECASE,
)


def stage1_is_low_confidence(
    answer: str,
    items: List[RetrievedItem],
    *,
    visual_score: float,
    pdf_score: float,
    filename_intent: bool,
    chart_score: float = 0.0,
    chart_numeric_min: float = 0.55,
    halluc_modal_min: float = 0.72,
) -> Tuple[bool, str]:
    a = (answer or "").strip()
    if not a:
        return True, "empty"
    if _EVASIVE_RE.search(a):
        return True, "evasive"
    if filename_intent and not re.search(r"\bimg[_-]?\d+", a, re.IGNORECASE):
        return True, "filename_missing"
    if chart_score >= chart_numeric_min and re.search(r"\d", a):
        return True, "chart_numeric_verify"
    if visual_score >= halluc_modal_min or pdf_score >= halluc_modal_min:
        a_tokens = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", a)}
        if not a_tokens:
            if not re.search(r"\d", a):
                return True, "strong_modal_no_tokens"
        else:
            ctx_blob = " ".join(it.chunk.content for it in items).lower()
            hits = sum(1 for t in a_tokens if t in ctx_blob)
            if hits == 0:
                return True, "answer_not_in_context"
    return False, "ok"


def answer_question(
    question: str,
    q_type: str,
    items: List[RetrievedItem],
    modality_flags: Dict[str, bool],
    llm,
    *,
    max_images_per_chat: int = 10,
    pdf_vision_max_pages: int = 3,
    pdf_render_dpi: int = 144,
    pdf_cache_dir: Optional[Path] = None,
    pdf_total_cap: int = 6,
    visual_score_threshold: float = 0.40,
    pdf_score_threshold: float = 0.40,
    enable_two_stage: bool = True,
    cascade_chart_numeric_min: float = 0.55,
    cascade_hallucination_modal_min: float = 0.72,
    forced_image_paths: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    debug: Dict[str, Any] = {}

    vscore = _visual_score(question, items, modality_flags)
    pscore = _pdf_score(question, items, modality_flags)
    cscore = _chart_score(question, items, modality_flags)
    filename_intent = _needs_filename_answer(question, items)
    multi_img_compose = _is_multi_image_compose(question)

    # Stage 1: text-only answer.
    s1_messages = build_messages(
        question,
        items,
        image_paths=[],
        pdf_page_paths=[],
        fm_image_filenames=None,
    )
    text1, usage1 = llm.chat(s1_messages, temperature=0.2)
    text1 = (text1 or "").strip()
    p1 = int(usage1.get("prompt_tokens", 0) or 0)
    c1 = int(usage1.get("completion_tokens", 0) or 0)

    debug["stage1_answer"] = text1
    debug["stage1_prompt_tokens"] = p1

    upgrade = False
    upgrade_reason = "ok"
    if not enable_two_stage:
        upgrade_reason = "two_stage_disabled"
    elif enable_two_stage:
        if multi_img_compose:
            upgrade = True
            upgrade_reason = "multi_img_compose"
        else:
            low, upgrade_reason = stage1_is_low_confidence(
                text1, items,
                visual_score=vscore, pdf_score=pscore, filename_intent=filename_intent,
                chart_score=cscore,
                chart_numeric_min=cascade_chart_numeric_min,
                halluc_modal_min=cascade_hallucination_modal_min,
            )
            upgrade = low
    debug["upgrade_to_stage2"] = upgrade
    debug["upgrade_reason"] = upgrade_reason
    debug["multi_img_compose"] = multi_img_compose
    debug["chart_score"] = round(cscore, 3)

    if not upgrade:
        debug.update(
            {
                "n_images_attached": 0,
                "n_pdf_pages_attached": 0,
                "modality_flags": modality_flags,
                "visual_score": round(vscore, 3),
                "pdf_score": round(pscore, 3),
                "chart_score": round(cscore, 3),
                "filename_intent": filename_intent,
                "candidate_filenames": [],
                "stage": 1,
            }
        )
        return text1, dict(usage1), debug

    # Stage 2: escalate to multimodal evidence.
    image_paths: List[str] = []
    pdf_pages: List[str] = []
    candidate_filenames: Optional[List[str]] = None

    if (
        multi_img_compose
        or filename_intent
        or vscore >= visual_score_threshold
        or cscore >= visual_score_threshold
    ):
        forced = [p for p in (forced_image_paths or []) if p]
        retrieved = _collect_images(items, max_images=max_images_per_chat)
        seen: set = set()
        merged: List[str] = []
        for ip in forced + retrieved:
            if ip not in seen:
                seen.add(ip)
                merged.append(ip)
        image_paths = merged[:max_images_per_chat]
        if filename_intent and not multi_img_compose:
            candidate_filenames = [Path(p).name for p in image_paths]

    if pscore >= pdf_score_threshold and pdf_cache_dir is not None:
        hit_keys = _collect_hit_pdf_pages(items)
        if hit_keys:
            pdf_pages = _render_specific_pdf_pages(
                hit_keys,
                cache_dir=pdf_cache_dir,
                dpi=pdf_render_dpi,
                total_cap=pdf_total_cap,
            )
        else:
            pdfs = _collect_pdfs(items)
            if pdfs:
                pdf_pages = _maybe_render_pdf_pages(
                    pdfs,
                    cache_dir=pdf_cache_dir,
                    dpi=pdf_render_dpi,
                    max_pages_per_pdf=pdf_vision_max_pages,
                    total_cap=pdf_total_cap,
                )

    total = len(image_paths) + len(pdf_pages)
    if total > max_images_per_chat:
        keep_img = max(0, max_images_per_chat - len(pdf_pages))
        image_paths = image_paths[:keep_img]

    s2_messages = build_messages(
        question,
        items,
        image_paths=image_paths,
        pdf_page_paths=pdf_pages,
        fm_image_filenames=candidate_filenames,
    )
    text2, usage2 = llm.chat(s2_messages, temperature=0.2)
    text2 = (text2 or "").strip()

    debug["stage2_answer"] = text2
    debug["stage2_image_paths"] = list(image_paths)
    debug["stage2_pdf_render_paths"] = list(pdf_pages)

    merged_usage = {
        "prompt_tokens": p1 + int(usage2.get("prompt_tokens", 0) or 0),
        "completion_tokens": c1 + int(usage2.get("completion_tokens", 0) or 0),
        "total_tokens": p1 + c1
        + int(usage2.get("prompt_tokens", 0) or 0)
        + int(usage2.get("completion_tokens", 0) or 0),
    }

    debug.update(
        {
            "stage": 2,
            "n_images_attached": len(image_paths),
            "n_pdf_pages_attached": len(pdf_pages),
            "modality_flags": modality_flags,
            "visual_score": round(vscore, 3),
            "pdf_score": round(pscore, 3),
            "chart_score": round(cscore, 3),
            "filename_intent": filename_intent,
            "candidate_filenames": candidate_filenames or [],
            "stage2_used_hit_pages": bool(_collect_hit_pdf_pages(items)),
        }
    )
    return text2, merged_usage, debug
