from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Heuristic chart-signal regex used to flag chunks that look chart-like.
_CHART_HINT_RE = re.compile(
    r"\b(chart|figure|fig\.|graph|plot|table|axis|legend|bar|"
    r"pie|histogram|percentage|percent|%|trend|distribution|"
    r"scatter|line\s+chart|x[-_ ]?axis|y[-_ ]?axis)\b",
    re.IGNORECASE,
)


@dataclass
class IndexedChunk:
    chunk_id: str
    kind: str
    session_id: str
    round_id: str
    content: str
    has_image: bool = False
    has_pdf: bool = False
    has_chart: bool = False
    n_images: int = 0
    image_paths: List[str] = field(default_factory=list)
    pdf_paths: List[str] = field(default_factory=list)
    img_filenames: List[str] = field(default_factory=list)
    pdf_file: str = ""
    pdf_page_index: int = 0
    pdf_digest: str = ""


def _dialogue_round_id(d: Dict[str, Any]) -> str:
    return str(d.get("round", "")).strip()


def _safe_id(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_") or "x"


def _has_chart_signal(text: str) -> bool:
    return bool(_CHART_HINT_RE.search(text or ""))


def caption_image_with_vlm(
    image_path: str, llm, *, max_chars: int = 280
) -> str:
    from m3exam.m3proctor.infra.llm_client import _image_data_url

    url = _image_data_url(image_path)
    if not url:
        return ""
    prompt = (
        "Describe this image in one sentence (<= 35 words). "
        "Focus on factual content: objects, text, chart elements, numbers."
    )
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    try:
        text, _usage = llm.chat(
            [{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=120,
        )
    except Exception as e:
        print(f"[indexer] vlm-caption failed for {image_path}: {e}")
        return ""
    return (text or "").strip().replace("\n", " ")[:max_chars]


# Stricter chart detector used during caption pre-screening: avoids "table"
# false positives such as "puppy on a table" by requiring chart-specific
# phrases.
_CHART_DESC_RE = re.compile(
    r"\b(chart|figure|fig\.|graph|plot|diagram|"
    r"bar\s*chart|pie\s*chart|line\s*chart|histogram|scatter|"
    r"trend|distribution|percentage|percent|"
    r"x[-\s]?axis|y[-\s]?axis|legend|"
    r"data\s+table|table\s+(show|show s|of|comparing)|"
    r"share\s+of|over\s+time|"
    r"by\s+(\w+\s+){0,2}(part(y|isan)|rac\w*|ethnic\w*|age|income|educat\w*|year|gender|region|demographic\w*|attainment))\b",
    re.IGNORECASE,
)


def looks_like_chart(filename: str, description: str) -> bool:
    return bool(_CHART_DESC_RE.search(description or "")) or bool(
        _CHART_DESC_RE.search(filename or "")
    )


def rich_caption_chart_with_vlm(
    image_path: str, llm, *, short_desc: str = "", max_chars: int = 900
) -> str:
    from m3exam.m3proctor.infra.llm_client import _image_data_url

    url = _image_data_url(image_path)
    if not url:
        return ""
    hint = f" The chart is about: {short_desc}." if short_desc else ""
    prompt = (
        "This image is a chart/figure/table.{hint}\n"
        "Transcribe ALL quantitative content as compact text so a reader who cannot see "
        "the image could still answer questions about it. Include:\n"
        "- the title / subtitle if any\n"
        "- every series / category label\n"
        "- EVERY numeric value with its label and (if present) time point, "
        "e.g. 'Democrats May 2022 = 48%, June 2023 = 62%'\n"
        "- axis labels and units\n"
        "- any legend entries\n"
        "Be exhaustive about numbers. Do NOT summarize away the values. "
        "Output as semicolon-separated label=value facts, no prose intro."
    ).format(hint=hint)
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    try:
        text, _usage = llm.chat(
            [{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as e:
        print(f"[indexer] rich-chart-caption failed for {image_path}: {e}")
        return ""
    return (text or "").strip().replace("\n", " ")[:max_chars]


def extract_pdf_text_chunks(
    pdf_path: str, *, max_chars: int = 4000
) -> str:
    try:
        import fitz
    except ImportError:
        return ""
    p = Path(pdf_path)
    if not p.is_file():
        return ""
    try:
        doc = fitz.open(p)
    except Exception:
        return ""
    try:
        parts: List[str] = []
        total = 0
        for i, page in enumerate(doc):
            t = (page.get_text("text") or "").strip()
            if not t:
                continue
            parts.append(f"[p{i+1}] {t}")
            total += len(t)
            if max_chars and total >= max_chars:
                break
        out = "\n".join(parts)
        return out[:max_chars] if max_chars else out
    finally:
        doc.close()


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    try:
        import fitz
    except ImportError:
        return []
    p = Path(pdf_path)
    if not p.is_file():
        return []
    try:
        doc = fitz.open(p)
    except Exception:
        return []
    try:
        out: List[Dict[str, Any]] = []
        for i, page in enumerate(doc):
            t = (page.get_text("text") or "").strip()
            if not t:
                continue
            out.append({"page_index": i, "text": t})
        return out
    finally:
        doc.close()


def digest_pdf_page_with_llm(
    page_text: str, llm, *, pdf_name: str = "", page_index: int = 0
) -> str:
    if not page_text:
        return ""
    snippet = page_text[:2400]
    prompt = (
        f"This is page {page_index + 1} of `{pdf_name}`. "
        "Write ONE sentence (<= 30 words) that captures the page's most concrete content. "
        "Mention named entities (taxa, methods, tools, figure numbers, percentages, gene names). "
        "Do NOT begin with 'This page...' / 'The page discusses...'. State the content directly.\n\n"
        f"PAGE TEXT:\n{snippet}\n\nONE-SENTENCE DIGEST:"
    )
    try:
        text, _u = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80,
        )
        return (text or "").strip().replace("\n", " ")[:280]
    except Exception as e:
        print(f"[indexer] pdf-digest failed for {pdf_name} p{page_index+1}: {e}")
        return page_text[:200].replace("\n", " ")


def load_pdf_digest_cache(path: Path) -> Dict[str, str]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_pdf_digest_cache(path: Path, cache: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pdf_page_chunks(
    pdf_path: str,
    *,
    pdf_filename: str,
    session_id: str,
    round_id: str,
    llm,
    digest_cache: Optional[Dict[str, str]] = None,
    page_text_max_chars: int = 1800,
) -> List[IndexedChunk]:
    pages = extract_pdf_pages(pdf_path)
    chunks: List[IndexedChunk] = []
    for p in pages:
        pi = p["page_index"]
        full_text = p["text"]
        cache_key = f"{pdf_filename}::p{pi+1}"
        digest = ""
        if digest_cache is not None and cache_key in digest_cache:
            digest = digest_cache[cache_key]
        else:
            digest = digest_pdf_page_with_llm(full_text, llm, pdf_name=pdf_filename, page_index=pi)
            if digest_cache is not None:
                digest_cache[cache_key] = digest

        text_for_chunk = full_text[:page_text_max_chars] if page_text_max_chars else full_text
        content = (
            f"[PDF_PAGE {pdf_filename} p{pi+1}]\n"
            f"[SOURCE_SESSION] {session_id}  [SOURCE_ROUND] {round_id}\n"
            f"[DIGEST] {digest}\n"
            f"[TEXT] {text_for_chunk}"
        )
        chunks.append(
            IndexedChunk(
                chunk_id=f"pdf_{_safe_id(pdf_filename)}_p{pi+1}_{_safe_id(round_id)}",
                kind="pdf_page",
                session_id=session_id,
                round_id=round_id,
                content=content,
                has_image=False,
                has_pdf=True,
                has_chart=_has_chart_signal(full_text),
                n_images=0,
                pdf_paths=[pdf_path],
                pdf_file=pdf_filename,
                pdf_page_index=pi,
                pdf_digest=digest,
            )
        )
    return chunks


def summarize_session_with_llm(
    session: Dict[str, Any], llm, *, max_chars: int = 1600
) -> str:
    sid = session.get("session_id", "?")
    date = session.get("date", "")
    bullets: List[str] = []
    for d in session.get("dialogues", []):
        rnd = d.get("round", "")
        user = (d.get("user") or "")[:200]
        asst = (d.get("assistant") or "")[:300]
        media = []
        if d.get("img_file"):
            media.append(f"{len(d['img_file'])} img(s)")
        if d.get("pdf_file"):
            media.append(f"pdf={d['pdf_file']}")
        m = (" [" + ", ".join(media) + "]") if media else ""
        bullets.append(f"- {rnd}: U: {user} | A: {asst}{m}")
    transcript = "\n".join(bullets)

    prompt = (
        f"Below are the dialogues of session {sid} (date {date}). "
        "Extract a MINI TIMELINE of atomic events from this session. "
        "Rules:\n"
        "- One line per event, formatted EXACTLY as: `[ROUND_ID] <event sentence <= 25 words>`\n"
        "- An event is a single action / decision / question asked / key fact / media reference\n"
        "- A round may produce 0, 1, or multiple events - split when it covers distinct topics\n"
        "- Include named entities (people, products, plants, papers, places) in the sentence\n"
        "- If a round references images, mention what the images depict (e.g. 'photographed roadside thistles')\n"
        "- If a round references a PDF, mention the topic of the PDF\n"
        "- Do NOT invent details, do not paraphrase generically (avoid 'discussed something')\n"
        "- Do NOT add headers, prose, bullets - only `[ROUND_ID] event` lines\n"
        "- Max 25 lines total\n\n"
        f"DIALOGUES:\n{transcript}\n\nTIMELINE:"
    )
    try:
        text, _u = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
    except Exception as e:
        print(f"[indexer] session-summary failed for {sid}: {e}")
        text = "\n".join(
            f"[{d.get('round','')}] {((d.get('user') or '')[:80]).strip()} -> {((d.get('assistant') or '')[:100]).strip()}"
            for d in session.get("dialogues", [])
        )[:max_chars]
    return (text or "").strip()[:max_chars]


def build_round_chunk(
    session: Dict[str, Any],
    dlg: Dict[str, Any],
    images_dir: Path,
    data_dir: Path,
    llm,
    *,
    caption_missing_img: bool = True,
    pdf_max_chars: int = 4000,
    img_caption_cache: Optional[Dict[str, str]] = None,
    rich_chart_caption: bool = True,
    chart_caption_cache: Optional[Dict[str, str]] = None,
) -> IndexedChunk:
    sid = str(session.get("session_id", "?"))
    date = str(session.get("date", ""))
    rnd = _dialogue_round_id(dlg)
    user = dlg.get("user", "") or ""
    asst = dlg.get("assistant", "") or ""

    img_files: List[str] = list(dlg.get("img_file") or [])
    img_descs: List[str] = list(dlg.get("img_description") or [])
    pdf_files: List[str] = list(dlg.get("pdf_file") or [])

    img_paths: List[str] = []
    img_lines: List[str] = []
    round_has_chart = False
    for i, fn in enumerate(img_files):
        ip = images_dir / fn
        if ip.is_file():
            img_paths.append(str(ip))
        desc = img_descs[i] if i < len(img_descs) else ""
        if not desc and caption_missing_img and ip.is_file():
            cache_key = str(ip)
            if img_caption_cache is not None and cache_key in img_caption_cache:
                desc = img_caption_cache[cache_key]
            else:
                desc = caption_image_with_vlm(str(ip), llm)
                if img_caption_cache is not None:
                    img_caption_cache[cache_key] = desc
            if desc:
                desc = f"(vlm-caption) {desc}"
        if desc:
            img_lines.append(f"[IMG_DESC {fn}] {desc}")
        else:
            img_lines.append(f"[IMG_DESC {fn}] (no description available)")

        if rich_chart_caption and ip.is_file() and looks_like_chart(fn, desc):
            round_has_chart = True
            ckey = str(ip)
            chart_data = ""
            if chart_caption_cache is not None and ckey in chart_caption_cache:
                chart_data = chart_caption_cache[ckey]
            else:
                chart_data = rich_caption_chart_with_vlm(str(ip), llm, short_desc=desc)
                if chart_caption_cache is not None:
                    chart_caption_cache[ckey] = chart_data
            if chart_data:
                img_lines.append(f"[CHART_DATA {fn}] {chart_data}")

    pdf_paths: List[str] = []
    pdf_lines: List[str] = []
    for pn in pdf_files:
        cand = None
        for c in (data_dir / "pdfs" / pn, data_dir / pn, data_dir / "images" / pn):
            if c.is_file():
                cand = c
                break
        if cand is None:
            pdf_lines.append(f"[PDF_REF {pn}] (file not found)")
            continue
        pdf_paths.append(str(cand))
        preview = extract_pdf_text_chunks(str(cand), max_chars=240).replace("\n", " ")
        if preview:
            pdf_lines.append(f"[PDF_REF {pn}] {preview[:240]}")
        else:
            pdf_lines.append(f"[PDF_REF {pn}] (no extractable text layer)")

    has_image = bool(img_paths)
    has_pdf = bool(pdf_paths)
    chart_blob = " ".join(img_descs) + " " + " ".join(pdf_lines) + " " + user + " " + asst
    has_chart = round_has_chart or _has_chart_signal(chart_blob)

    parts = [
        f"[ROUND_TAG] {rnd}",
        f"[SESSION] {sid} (date: {date})",
        f"User: {user}",
        f"Assistant: {asst}",
    ]
    parts.extend(img_lines)
    parts.extend(pdf_lines)
    if has_image or has_pdf or has_chart:
        tags = []
        if has_image:
            tags.append("image")
        if has_pdf:
            tags.append("pdf")
        if has_chart:
            tags.append("chart")
        parts.append(f"[MODALITY] {', '.join(tags)}")

    content = "\n".join(parts)
    return IndexedChunk(
        chunk_id=f"rnd_{_safe_id(rnd)}",
        kind="round",
        session_id=sid,
        round_id=rnd,
        content=content,
        has_image=has_image,
        has_pdf=has_pdf,
        has_chart=has_chart,
        n_images=len(img_paths),
        image_paths=img_paths,
        pdf_paths=pdf_paths,
        img_filenames=[Path(p).name for p in img_paths],
    )


def build_session_summary_chunk(
    session: Dict[str, Any], llm, *, pdf_max_chars: int = 4000
) -> IndexedChunk:
    sid = str(session.get("session_id", "?"))
    date = str(session.get("date", ""))
    rounds_present = [d.get("round", "") for d in session.get("dialogues", [])]
    timeline = summarize_session_with_llm(session, llm)
    has_image = any((d.get("img_file") or []) for d in session.get("dialogues", []))
    has_pdf = any((d.get("pdf_file") or []) for d in session.get("dialogues", []))
    has_chart = _has_chart_signal(timeline)

    header = (
        f"[SESSION_TIMELINE] {sid} (date: {date}; rounds: "
        f"{rounds_present[0] if rounds_present else ''}..{rounds_present[-1] if rounds_present else ''})"
    )
    content = f"{header}\n{timeline}"
    return IndexedChunk(
        chunk_id=f"sum_{_safe_id(sid)}",
        kind="summary",
        session_id=sid,
        round_id="",
        content=content,
        has_image=has_image,
        has_pdf=has_pdf,
        has_chart=has_chart,
        n_images=0,
    )


def load_caption_cache(path: Path) -> Dict[str, str]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_caption_cache(path: Path, cache: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
