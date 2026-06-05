from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _truncate_multimodal_content_to_max_images(
    content: List[Dict[str, Any]], cap: int
) -> List[Dict[str, Any]]:
    if cap < 1 or len(content) <= 1:
        return content
    idx_img = [i for i, c in enumerate(content) if c.get("type") == "image_url"]
    if len(idx_img) <= cap:
        return content
    drop: set[int] = set()
    for j in idx_img[cap:]:
        drop.add(j)
        if j > 0 and content[j - 1].get("type") == "text":
            drop.add(j - 1)
    return [c for i, c in enumerate(content) if i not in drop]


def max_image_urls_per_chat() -> int:
    raw = os.environ.get("MULTIMODEL_CHAT_MAX_IMAGES", "10").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 10
    return max(1, min(n, 32))


def effective_max_image_urls_per_chat() -> int:
    u = max_image_urls_per_chat()
    raw = os.environ.get("MULTIMODEL_GATEWAY_MAX_IMAGES", "10").strip()
    try:
        g = int(raw)
    except ValueError:
        g = 10
    g = max(1, min(g, 64))
    return max(1, min(u, g))


def _image_data_url(path: str) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    try:
        data = base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        return None
    return f"data:{mime};base64,{data}"


def is_image_filename_question(question: str) -> bool:
    q = (question or "").lower()
    return "img_<number>" in q or "image file name" in q or "file name" in q


def build_answer_messages(
    question: str,
    context: str,
    image_paths: Optional[List[str]] = None,
    *,
    vision_pdf_page_paths: Optional[List[str]] = None,
    vision_pdf_page_labels: Optional[List[str]] = None,
    pdf_answer_mode: str = "text_only",
    image_attach_labels: Optional[List[str]] = None,
    cite_source_rounds: bool = False,
    question_type: str = "",
) -> List[Dict[str, Any]]:
    qtype = (question_type or "").strip().lower()
    prompt = (
        "You are an AI assistant with access to a multi-session conversation history, "
        "optional excerpts from PDF documents attached to some sessions, and optional "
        "labeled images.\n"
        "Answer the following question based ONLY on this material.\n"
        "\n"
        "IMPORTANT:\n"
        "- Each session header may show a calendar date like (2023-09-06). "
        "For time-related questions, use ONLY those session dates from the history, "
        "NOT today's date and NOT the year 2026.\n"
        "- Prefer the dialogue turn marked with [TURN_ID] / [D#:#] that directly answers "
        "the question; ignore unrelated sessions.\n"
        "\n"
        "=== CONVERSATION HISTORY ===\n"
        f"{context}\n"
        "=== END OF CONVERSATION ===\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        "Give a short, direct answer. Do not repeat the question or output nonsense."
    )
    if qtype == "ss":
        prompt += (
            "\nThis is a single-session factual question: find the assistant's answer in the "
            "most relevant [D#:#] turn and reply with the shortest phrase that uses the SAME "
            "key terms the assistant used (e.g. if the assistant said 'macro lens', answer "
            "'macro lens', not a paraphrase). One phrase only; no preamble."
        )
    elif qtype == "ii":
        prompt += (
            "\nThis is an implicit-inference question: the answer is NOT stated outright in "
            "one turn — synthesize evidence across MANY sessions (dates, activities, tone). "
            "Reply with ONE short noun phrase describing the implied trait "
            "(e.g. 'fall semester', 'undergraduate botany major', 'methodically systematic'). "
            "Do not name a random technical topic from a single PDF chart."
        )
    elif qtype == "tr":
        prompt += (
            "\nThis is a temporal reasoning question: extract the exact date or event "
            "order from the session dates and [D#:#] turn order in the history. "
            "Reply in one short phrase."
        )
    elif qtype == "th":
        prompt += (
            "\nThis is a thematic synthesis question asking for an end-to-end plan or protocol "
            "assembled across multiple sessions. Start directly with 3–6 bullet points "
            "(lines beginning with '- '). No introduction like 'Here are the key steps'. "
            "Cover the main stages the assistant gave; use only facts from the history."
        )
    elif qtype not in ("fm", "fj"):
        prompt += " Prefer a brief phrase (a few words) without long preamble."
    if cite_source_rounds:
        from eval_metrics import ROUND_CITATION_PROMPT

        prompt += "\n" + ROUND_CITATION_PROMPT

    pdf_vis = [str(p) for p in (vision_pdf_page_paths or []) if p]
    pdf_labels = list(vision_pdf_page_labels or [])
    cap = effective_max_image_urls_per_chat()
    # retrieved images + PDF page imageshared cap; in vision_pages mode prefer keeping PDF pages ((truncate dialogue images)) , avoid 400「At most N image(s)」
    if pdf_answer_mode == "vision_pages" and pdf_vis:
        pdf_use = pdf_vis[:cap]
        paths = [str(p) for p in (image_paths or []) if p][: max(0, cap - len(pdf_use))]
    else:
        pdf_use = []
        paths = [str(p) for p in (image_paths or []) if p][:cap]

    if pdf_answer_mode == "vision_pages" and pdf_use:
        prompt += (
            "\n\nAdditional raster images below are rendered pages from PDF files "
            "(see [ROUND_REF] / [PDF_PATH] / [PDF_PAGE] in the conversation text when present; "
            "ROUND_REF matches supporting_facts like D17:1). "
            "Use them together with the text excerpts when answering."
        )
    if not paths and not pdf_use:
        return [{"role": "user", "content": prompt}]

    candidate_names = [Path(p).name for p in paths]
    if paths and is_image_filename_question(question):
        prompt += (
            "\n\nThe images below are labeled by their exact file names. "
            "Pick the ONE image that best matches the question. "
            f"Valid file names in this gallery: {', '.join(candidate_names)}. "
            "Reply with ONLY that single file name (e.g. img_5). "
            "No explanation, no commas, no extra words. "
            "If the question explicitly allows NONE when no image matches, you may reply NONE. "
            "Do not use ordinals like 'image 4'. "
            "Do not reply with the placeholder img_<number>."
        )
    elif paths:
        prompt += "\n\nUse the candidate images below when visual evidence is needed."

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    # first attach rendered PDF page, then attachretrieved images; iftriggertail truncation, preferentially dropdialogue images. 
    for j, pv in enumerate(pdf_use, start=1):
        pth = Path(pv)
        data_url = _image_data_url(pv)
        if not data_url:
            continue
        lbl = (
            pdf_labels[j - 1]
            if j - 1 < len(pdf_labels) and pdf_labels[j - 1]
            else f"PDF rendered page {j}: {pth.name}"
        )
        content.append({"type": "text", "text": lbl})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    for idx, path in enumerate(paths, start=1):
        p = Path(path)
        data_url = _image_data_url(path)
        if not data_url:
            continue
        if image_attach_labels and idx - 1 < len(image_attach_labels):
            lbl = f"Image {idx} {image_attach_labels[idx - 1]}"
        else:
            lbl = f"Image {idx} exact file name: {p.name}"
        content.append({"type": "text", "text": lbl})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    if len(content) == 1:
        return [{"role": "user", "content": prompt}]
    content = _truncate_multimodal_content_to_max_images(content, cap)
    return [{"role": "user", "content": content}]
