from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

PdfPolicy = Literal["native_only", "native_then_ocr", "ocr_pages"]

# answering stagerender PDF pageas PNG (vision_pages) when default DPI
PDF_VISION_RENDER_DPI_DEFAULT = 120

# the four evaluation adapters「only auto, and LLM substringnotmatched 」when default indexing policy (based onthisbundled in repo upstream README, 
# no live web crawling; can be regarded asdocumentation static decision, see multimodel_config.yaml comment) . 
# - raganything: RAG-Anything-main/README explicitly「Universal Document Support」covers PDF and other formats; 
#   evaluation adapter layerstill mostly uses PyMuPDF text layer, , so the default prefers native_only (text suffices, otherwise skips OCR) . 
# - ngm / universalrag / memverse: the README does not advertisededicated PDF pipeline (NGM byimage/retrieveasprimary; UniversalRAG
#   focuses onmultimodal routingandpreprocessedcorpus; MemVerse focuses onmemoryand graph) , default native_then_ocr is safer. 
README_BACKED_ADAPTER_PDF_POLICY_DEFAULTS: Dict[str, PdfPolicy] = {
    "raganything": "native_only",
    "ngm": "native_then_ocr",
    "universalrag": "native_then_ocr",
    "memverse": "native_only",
    "mirix": "native_then_ocr",
}


def collect_session_pdf_origins(session: dict, pdfs_dir: Path) -> List[Tuple[Path, str]]:
    sid = str(session.get("session_id") or "").strip() or "?"
    ordered: List[Tuple[str, str]] = []

    sp = session.get("source_pdf")
    if sp:
        fn = str(sp).strip()
        if fn:
            ordered.append((fn, f"{sid}:source"))

    for i, dlg in enumerate(session.get("dialogues") or []):
        rnd = str(dlg.get("round") or "").strip()
        if not rnd:
            rnd = f"{sid}:{i + 1}"
        for pf in dlg.get("pdf_file") or []:
            fn = str(pf).strip()
            if fn:
                ordered.append((fn, rnd))

    seen: set[str] = set()
    out: List[Tuple[Path, str]] = []
    for fn, rnd in ordered:
        if not fn or fn in seen:
            continue
        seen.add(fn)
        p = pdfs_dir / fn
        if p.is_file():
            out.append((p, rnd))
    return out


def collect_session_pdf_paths(session: dict, pdfs_dir: Path) -> List[Path]:
    return [p for p, _ in collect_session_pdf_origins(session, pdfs_dir)]


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars // 2] + "\n...[truncated]...\n" + s[-max_chars // 2 :]


def _should_strip_pdf_structure() -> bool:
    v = os.environ.get("MULTIMODEL_PDF_STRIP_STRUCT", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def open_pdf_normalized(path: Path | str):
    import fitz  # pymupdf

    p = Path(path)
    silent = os.environ.get("MULTIMODEL_PDF_MUPDF_SILENT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    prev_err = True
    prev_warn = False
    if silent:
        prev_err = fitz.TOOLS.mupdf_display_errors()
        prev_warn = fitz.TOOLS.mupdf_display_warnings()
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
    try:
        src = fitz.open(p)
        if not _should_strip_pdf_structure():
            return src
        if src.page_count == 0:
            return src
        dst = fitz.open()
        try:
            dst.insert_pdf(src, from_page=0, to_page=src.page_count - 1)
        except Exception:
            dst.close()
            return src
        src.close()
        return dst
    finally:
        if silent:
            fitz.TOOLS.mupdf_display_errors(prev_err)
            fitz.TOOLS.mupdf_display_warnings(prev_warn)


def extract_pdf_text_native(path: Path, max_pages: int = 100) -> str:
    doc = open_pdf_normalized(path)
    parts: List[str] = []
    n = min(doc.page_count, max_pages)
    for i in range(n):
        parts.append(str(doc.load_page(i).get_text("text") or ""))
    doc.close()
    return "\n".join(parts).strip()


def ocr_pdf_pages(
    path: Path,
    max_pages: int = 15,
    dpi: int = 150,
) -> str:
    import fitz  # pymupdf
    from PIL import Image
    import pytesseract  # type: ignore[import-untyped]

    doc = open_pdf_normalized(path)
    texts: List[str] = []
    n = min(doc.page_count, max_pages)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(n):
        pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        texts.append(pytesseract.image_to_string(img) or "")
    doc.close()
    return "\n".join(texts).strip()


def _native_page_texts(path: Path, max_pages: int) -> List[str]:
    doc = open_pdf_normalized(path)
    n = min(doc.page_count, max_pages)
    out: List[str] = []
    for i in range(n):
        out.append(str(doc.load_page(i).get_text("text") or "").strip())
    doc.close()
    return out


def _ocr_single_page(path: Path, page_0based: int, dpi: int = 150) -> str:
    import fitz  # pymupdf
    from PIL import Image
    import pytesseract  # type: ignore[import-untyped]

    doc = open_pdf_normalized(path)
    if page_0based < 0 or page_0based >= doc.page_count:
        doc.close()
        return ""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = doc.load_page(page_0based).get_pixmap(matrix=mat, alpha=False)
    doc.close()
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return (pytesseract.image_to_string(img) or "").strip()


def extract_pdf_pages_for_eval(
    path: Path,
    policy: PdfPolicy,
    min_native_chars: int = 400,
    native_max_pages: int = 100,
    ocr_max_pages: int = 15,
) -> List[str]:
    if policy == "ocr_pages":
        import fitz  # pymupdf
        from PIL import Image
        import pytesseract  # type: ignore[import-untyped]

        dpi = 150
        doc = open_pdf_normalized(path)
        n = min(doc.page_count, native_max_pages)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        ocr_all: List[str] = []
        for i in range(n):
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            ocr_all.append((pytesseract.image_to_string(img) or "").strip())
        doc.close()
        return ocr_all

    native_pages = _native_page_texts(path, native_max_pages)
    if policy == "native_only":
        return native_pages
    total_native = sum(len(p) for p in native_pages)
    if total_native >= min_native_chars:
        return native_pages
    n = min(len(native_pages), ocr_max_pages) if native_pages else 0
    if not native_pages:
        doc = open_pdf_normalized(path)
        n = min(doc.page_count, ocr_max_pages)
        doc.close()
    ocr_pages: List[str] = []
    for i in range(n):
        ocr_pages.append(_ocr_single_page(path, i))
    return ocr_pages


def extract_pdf_for_eval(
    path: Path,
    policy: PdfPolicy,
    min_native_chars: int = 400,
    native_max_pages: int = 100,
    ocr_max_pages: int = 15,
) -> str:
    pages = extract_pdf_pages_for_eval(
        path,
        policy,
        min_native_chars=min_native_chars,
        native_max_pages=native_max_pages,
        ocr_max_pages=ocr_max_pages,
    )
    return "\n".join(pages).strip()


def _format_traced_pdf_sections(
    pdf_origins: List[Tuple[Path, str]],
    policy: PdfPolicy,
    min_native_chars: int,
) -> str:
    sections: List[str] = []
    for p, round_ref in pdf_origins:
        try:
            page_texts = extract_pdf_pages_for_eval(
                p, policy, min_native_chars=min_native_chars
            )
        except Exception as exc:
            sections.append(
                f"[PDF_PATH] {p.name}\n[ROUND_REF] {round_ref}\n[PDF_PAGE] -\n"
                f"[PDF read error: {exc}]"
            )
            continue
        for pi, txt in enumerate(page_texts, start=1):
            body = txt.strip() if txt else ""
            sections.append(
                f"[PDF_PATH] {p.name}\n[ROUND_REF] {round_ref}\n[PDF_PAGE] {pi}\n{body}".rstrip()
            )
    return "\n\n".join(s for s in sections if s)


def session_ids_from_formatted_context(context: str) -> List[str]:
    return re.findall(r"=== Session\s+([^\s(]+)", context or "")


_PDF_PAGE_LINE_RE = re.compile(r"\[PDF_PAGE\]\s*(\d+)", re.IGNORECASE)


_ROUND_REF_LINE_RE = re.compile(r"\[ROUND_REF\]\s*(\S+)", re.IGNORECASE)


def pdf_trace_markers_in_retrieved_context(context: str) -> Dict[str, Any]:
    txt = context or ""
    has_path = "[PDF_PATH]" in txt
    has_page = "[PDF_PAGE]" in txt
    pages: List[int] = []
    for m in _PDF_PAGE_LINE_RE.finditer(txt):
        try:
            pages.append(int(m.group(1)))
        except ValueError:
            continue
    round_refs: List[str] = []
    for m in _ROUND_REF_LINE_RE.finditer(txt):
        round_refs.append(m.group(1).strip())
    return {
        "pdf_context_has_path_marker": has_path,
        "pdf_context_has_page_marker": has_page,
        "pdf_context_pdf_pages_found": pages[:32],
        "pdf_context_round_refs_found": round_refs[:48],
    }


def session_ids_from_round_ids(
    round_ids: List[str],
    corpus_meta: List[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    meta_by_round: Dict[str, str] = {}
    for meta in corpus_meta:
        rnd = str(meta.get("round") or "").replace(" ", "")
        sid = str(meta.get("session_id") or "")
        if rnd and sid and rnd not in meta_by_round:
            meta_by_round[rnd] = sid
    for rid in round_ids:
        rr = str(rid).replace(" ", "")
        sid = meta_by_round.get(rr, "")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def render_pdf_pages_for_rounds(
    sessions: List[dict],
    pdfs_dir: Path,
    round_ids: List[str],
    corpus_meta: List[Dict[str, Any]],
    *,
    max_pages: int = 3,
    dpi: int = PDF_VISION_RENDER_DPI_DEFAULT,
    out_dir: Path,
    tag: str = "0",
) -> Tuple[List[str], int, List[Dict[str, Any]]]:
    order = session_ids_from_round_ids(round_ids, corpus_meta)
    if not order:
        return [], 0, []
    fake_ctx = "\n".join(f"=== Session {sid} ===" for sid in order)
    return render_pdf_pages_for_vision(
        sessions,
        pdfs_dir,
        fake_ctx,
        max_pages=max_pages,
        dpi=dpi,
        out_dir=out_dir,
        tag=tag,
    )


def render_pdf_pages_for_priority_rounds(
    sessions: List[dict],
    pdfs_dir: Path,
    round_ids: List[str],
    corpus_meta: List[Dict[str, Any]],
    *,
    max_pages: int = 3,
    dpi: int = PDF_VISION_RENDER_DPI_DEFAULT,
    out_dir: Path,
    tag: str = "0",
) -> Tuple[List[str], int, List[Dict[str, Any]]]:
    if not round_ids:
        return [], 0, []
    return render_pdf_pages_for_rounds(
        sessions,
        pdfs_dir,
        list(round_ids),
        corpus_meta,
        max_pages=max_pages,
        dpi=dpi,
        out_dir=out_dir,
        tag=tag,
    )


def render_pdf_pages_for_vision(
    sessions: List[dict],
    pdfs_dir: Path,
    context: str,
    *,
    max_pages: int = 3,
    dpi: int = PDF_VISION_RENDER_DPI_DEFAULT,
    out_dir: Path,
    tag: str = "0",
) -> Tuple[List[str], int, List[Dict[str, Any]]]:
    import fitz  # pymupdf

    order = session_ids_from_formatted_context(context)
    by_id = {str(s.get("session_id", "")): s for s in sessions}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_out: List[str] = []
    meta: List[Dict[str, Any]] = []
    remaining = max(1, min(max_pages, 50))

    for sid in order:
        if remaining <= 0:
            break
        sess = by_id.get(sid)
        if not sess:
            continue
        origins = collect_session_pdf_origins(sess, pdfs_dir)
        for pdf_path, round_ref in origins:
            if remaining <= 0:
                break
            try:
                doc = open_pdf_normalized(pdf_path)
                n_doc = doc.page_count
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                for page_i in range(min(n_doc, remaining)):
                    pix = doc.load_page(page_i).get_pixmap(matrix=mat, alpha=False)
                    fname = f"pdfvis_{tag}_{pdf_path.stem}_p{page_i + 1}.png"
                    fpath = out_dir / fname
                    fpath.write_bytes(pix.tobytes("png"))
                    paths_out.append(str(fpath.resolve()))
                    meta.append(
                        {
                            "session_id": sid,
                            "round_ref": round_ref,
                            "pdf_name": pdf_path.name,
                            "page": page_i + 1,
                            "path": str(fpath.resolve()),
                        }
                    )
                    remaining -= 1
                    if remaining <= 0:
                        break
                doc.close()
            except Exception:
                continue

    return paths_out, len(paths_out), meta


def _sessions_progress_iter(
    sessions: List[dict],
    policy: PdfPolicy,
    *,
    desc: str,
):
    if policy != "ocr_pages":
        return sessions
    try:
        from tqdm import tqdm

        if sys.stderr.isatty():
            return tqdm(
                sessions,
                desc=desc,
                unit="sess",
                leave=False,
                ncols=90,
                dynamic_ncols=False,
            )
    except ImportError:
        pass
    return sessions


def build_session_pdf_snippets(
    sessions: List[dict],
    pdfs_dir: Path,
    policy: PdfPolicy,
    embed_max_chars: int = 8000,
    min_native_chars: int = 400,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for sess in _sessions_progress_iter(
        sessions, policy, desc="PDF ocr_pages (snippet)"
    ):
        sid = str(sess.get("session_id", ""))
        origins = collect_session_pdf_origins(sess, pdfs_dir)
        if not origins:
            continue
        try:
            merged = _format_traced_pdf_sections(origins, policy, min_native_chars)
        except Exception as exc:
            merged = f"[PDF_PATH] -\n[PDF_PAGE] -\n[PDF session error: {exc}]"
        if merged:
            out[sid] = _truncate(merged, embed_max_chars)
    return out


def build_session_pdf_context_blocks(
    sessions: List[dict],
    pdfs_dir: Path,
    policy: PdfPolicy,
    context_max_chars: int = 12000,
    min_native_chars: int = 400,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for sess in _sessions_progress_iter(
        sessions, policy, desc="PDF ocr_pages (context)"
    ):
        sid = str(sess.get("session_id", ""))
        origins = collect_session_pdf_origins(sess, pdfs_dir)
        if not origins:
            continue
        try:
            merged = _format_traced_pdf_sections(origins, policy, min_native_chars)
        except Exception as exc:
            merged = f"[PDF_PATH] -\n[PDF_PAGE] -\n[PDF session error: {exc}]"
        if merged:
            out[sid] = _truncate(merged, context_max_chars)
    return out


def dialogue_image_description(dlg: dict) -> str:
    desc = dlg.get("img_description")
    if desc is None:
        return ""
    if isinstance(desc, list):
        return " ".join(str(x) for x in desc if x)
    return str(desc)


def _pdf_policy_llm_model(cfg: Dict[str, Any], llm_model: Optional[str]) -> str:
    s = (llm_model or "").strip()
    if s:
        return s
    raw = cfg.get("llm")
    llm_b = raw if isinstance(raw, dict) else {}
    return str(llm_b.get("model") or "").strip()


def _llm_matches_pdf_extract_substrings(model: str, substrings: Any) -> bool:
    if not model or not isinstance(substrings, list):
        return False
    m = model.lower()
    for item in substrings:
        if not isinstance(item, str):
            continue
        if item.lower() in m:
            return True
    return False


def _adapter_fallback_pdf_policy(model_name: str, cfg: Dict[str, Any]) -> PdfPolicy:
    models_block = cfg.get("models")
    if not isinstance(models_block, dict):
        return "native_then_ocr"
    custom = models_block.get("pdf_policy_auto_adapter_defaults")
    if isinstance(custom, dict):
        v = custom.get(model_name)
        if v in ("native_only", "native_then_ocr", "ocr_pages"):
            return v  # type: ignore[return-value]
    native_list = models_block.get("pdf_native_capable_models")
    if native_list is None:
        native_list = ["raganything"]
    if isinstance(native_list, list) and model_name in native_list:
        return "native_only"
    return README_BACKED_ADAPTER_PDF_POLICY_DEFAULTS.get(model_name, "native_then_ocr")


def _fallback_pdf_policy_from_probe(cfg: Dict[str, Any]) -> PdfPolicy:
    pb = cfg.get("pdf_capability_probe")
    if not isinstance(pb, dict):
        return "native_then_ocr"
    v = pb.get("fallback_pdf_policy", "native_then_ocr")
    if v in ("native_only", "native_then_ocr", "ocr_pages"):
        return v  # type: ignore[return-value]
    return "native_then_ocr"


def resolve_pdf_policy(
    model_name: str,
    cfg: Dict[str, Any],
    *,
    llm_model: Optional[str] = None,
    probe_native_pdf: Optional[bool] = None,
) -> PdfPolicy:
    models_block = cfg.get("models")
    native_capable: List[str] = []
    explicit_pol: Optional[PdfPolicy] = None
    if isinstance(models_block, dict):
        raw_native = models_block.get("pdf_native_capable_models")
        if isinstance(raw_native, list):
            native_capable = [str(x) for x in raw_native if x]
        sub = models_block.get(model_name)
        if isinstance(sub, dict):
            raw_pol = sub.get("pdf_policy")
            if raw_pol in ("native_only", "native_then_ocr", "ocr_pages"):
                explicit_pol = raw_pol  # type: ignore[assignment]

    # native PDF capabilitymodel (such as mirix ingest_pdf_native) keepper-block pdf_policy, not eval.pdf_policy globally OCR override. 
    if explicit_pol and model_name in native_capable:
        return explicit_pol

    eval_b = cfg.get("eval")
    if isinstance(eval_b, dict):
        forced = eval_b.get("pdf_policy")
        if forced in ("native_only", "native_then_ocr", "ocr_pages"):
            return forced  # type: ignore[return-value]

    if not isinstance(models_block, dict):
        return "native_then_ocr"
    sub = models_block.get(model_name)
    if isinstance(sub, dict):
        raw_pol = sub.get("pdf_policy")
        if raw_pol in ("native_only", "native_then_ocr", "ocr_pages"):
            return raw_pol  # type: ignore[return-value]
        if str(raw_pol or "").strip().lower() == "auto":
            if probe_native_pdf is True:
                return "native_only"
            if probe_native_pdf is False:
                return _fallback_pdf_policy_from_probe(cfg)
        # auto but no probe, missingorother values → use LLM andsubstring listinference

    raw_llm = cfg.get("llm")
    llm_b: Dict[str, Any] = raw_llm if isinstance(raw_llm, dict) else {}
    lm = _pdf_policy_llm_model(cfg, llm_model)

    then_ocr_sub = llm_b.get("pdf_extract_native_then_ocr_substrings")
    if then_ocr_sub is None:
        then_ocr_sub = ["qwen2.5-vl"]
    native_only_sub = llm_b.get("pdf_extract_native_only_substrings")
    if native_only_sub is None:
        native_only_sub = []

    if _llm_matches_pdf_extract_substrings(lm, then_ocr_sub):
        return "native_then_ocr"
    if _llm_matches_pdf_extract_substrings(lm, native_only_sub):
        return "native_only"

    return _adapter_fallback_pdf_policy(model_name, cfg)
