import os
import re
import json
import shutil
import logging
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from m3exam.config.config_loader import cfg
from m3exam.functions.common.conversation_utils import (
    THEMATIC_USER_PROMPT_INIT,
    THEMATIC_USER_PROMPT,
    THEMATIC_AGENT_PROMPT_INIT,
    THEMATIC_AGENT_PROMPT,
    DIALOGUE_SUMMARY_PROMPT,
    DIALOGUE_SELFCHECK_CHUNK_PROMPT,
    DIALOGUE_SELFCHECK_CROSS_PROMPT,
    DIALOGUE_REPAIR_PROMPT,
    PDF_USER_PROMPT_INIT,
    PDF_USER_PROMPT,
    PDF_FOLLOWUP_USER_PROMPT,
    PDF_AGENT_PROMPT,
)
from m3exam.functions.common.json_llm import (
    extract_clean_text,
    extract_json_objects,
    parse_json_array,
    parse_json_object,
)
from m3exam.functions.common.summary_utils import normalize_summary_to_string
from m3exam.functions.common.vision_llm import call_vision_llm as _call_vision_llm
from m3exam.global_methods import run_chatgpt

logger = logging.getLogger(__name__)


def _extract_single_date(ts: str) -> str:
    m = re.search(r"\d{4}-\d{2}-\d{2}", ts)
    return m.group(0) if m else ts


def _crawl_and_rename_images(keyword: str, images_dir: str, max_count: int) -> Tuple[List[str], List[str]]:
    os.makedirs(images_dir, exist_ok=True)
    try:
        from icrawler.builtin import BingImageCrawler
        with tempfile.TemporaryDirectory() as tmp_dir:
            crawler = BingImageCrawler(
                storage={"root_dir": tmp_dir},
                log_level=logging.ERROR,
            )
            crawler.crawl(keyword=keyword, max_num=max_count, file_idx_offset=0)

            raw_files = sorted(
                f for f in os.listdir(tmp_dir)
                if os.path.isfile(os.path.join(tmp_dir, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"))
            )

            renamed_files: List[str] = []
            renamed_paths: List[str] = []
            for idx, f in enumerate(raw_files, start=1):
                src = os.path.join(tmp_dir, f)
                ext = os.path.splitext(f)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png"):
                    ext = ".jpg"
                new_name = f"{idx}{ext}"
                dst      = os.path.join(images_dir, new_name)
                shutil.copy2(src, dst)
                renamed_files.append(new_name)
                renamed_paths.append(dst)
                logger.info("Crawled  %s  ->  %s", f, dst)

            return renamed_files, renamed_paths

    except ImportError:
        logger.error("icrawler not installed - run: pip install icrawler")
        return [], []
    except Exception as exc:
        logger.error("Image crawl failed for keyword '%s': %s", keyword, exc)
        return [], []


def _copy_and_rename_images(
    graph_route: str,
    included_graphs: List[str],
    images_dir: str,
) -> Tuple[List[str], List[str]]:
    os.makedirs(images_dir, exist_ok=True)
    renamed_files: List[str] = []
    renamed_paths: List[str] = []
    for idx, name in enumerate(included_graphs, start=1):
        src = os.path.join(graph_route, name.strip())
        if not os.path.exists(src):
            logger.warning("Local image not found: %s", src)
            continue
        ext      = os.path.splitext(name)[1]
        new_name = f"{idx}{ext}"
        dst      = os.path.join(images_dir, new_name)
        shutil.copy2(src, dst)
        renamed_files.append(new_name)
        renamed_paths.append(dst)
        logger.info("Copied  %s  ->  %s", src, dst)
    return renamed_files, renamed_paths


def _get_images_for_round(
    keyword: str,
    round_images_dir: str,
    max_count: int,
    automatic_mode: bool,
    graph_route: str,
    included_graphs: List[str],
) -> Tuple[List[str], List[str]]:
    if automatic_mode:
        return _crawl_and_rename_images(keyword, round_images_dir, max_count)
    return _copy_and_rename_images(graph_route, included_graphs, round_images_dir)


_PDF_EXTS = (".pdf",)


def _resolve_included_pdfs(pdf_route: str, included: str) -> List[str]:
    raw = (included or "").strip()
    if not pdf_route or not os.path.isdir(pdf_route):
        return []
    all_pdfs = sorted(
        f for f in os.listdir(pdf_route)
        if f.lower().endswith(_PDF_EXTS)
        and os.path.isfile(os.path.join(pdf_route, f))
    )
    if not raw or raw.lower() in ("auto", "*", "all"):
        return all_pdfs
    asked = [s.strip() for s in raw.split(",") if s.strip()]
    return [n for n in asked if (os.path.isfile(os.path.join(pdf_route, n)))]


def _render_pdf_pages(
    pdf_path: str,
    dst_dir: str,
    *,
    dpi: int,
    max_pages: int,
    name_prefix: str = "pdf",
) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required when thematic_subset.included_pdf is set. "
            "Install with: pip install pymupdf"
        ) from exc

    os.makedirs(dst_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        n = min(doc.page_count, max(1, int(max_pages)))
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        out: List[str] = []
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            bn = f"{name_prefix}_{i + 1}.png"
            pix.save(os.path.join(dst_dir, bn))
            out.append(bn)
    finally:
        doc.close()
    return out


def _split_pdf_into_fragments(n_pages: int, per_fragment: int) -> List[Tuple[int, int]]:
    per_fragment = max(1, per_fragment)
    out: List[Tuple[int, int]] = []
    lo = 0
    while lo < n_pages:
        out.append((lo, min(lo + per_fragment, n_pages)))
        lo += per_fragment
    return out


def _load_or_generate_timeline(th_cfg: dict, out_dir: str) -> List[dict]:
    do_generate = bool(th_cfg.get("generate_timeline", True))
    timeline_route = str(th_cfg.get("timeline_route", os.path.join(out_dir, "timeline.json"))).strip()
    user_persona = str(th_cfg.get("user_persona", "")).strip()
    core_event = str(th_cfg.get("core_event", "")).strip()

    if not timeline_route.lower().endswith(".json"):
        timeline_route += ".json"

    if do_generate:
        from m3exam.functions.timeline.generate_timeline import (
            generate_timeline as _gen,
            TIMELINE_EVENT_TARGET,
        )

        event_count = int(th_cfg.get("event_count", TIMELINE_EVENT_TARGET) or TIMELINE_EVENT_TARGET)
        logger.info("Generating timeline -> %s  (%d events)", timeline_route, event_count)
        timeline = _gen(user_persona, core_event, timeline_route, event_count=event_count)
        # Also copy to session out_dir for traceability
        ref_path = os.path.join(out_dir, "timeline.json")
        if os.path.abspath(timeline_route) != os.path.abspath(ref_path):
            try:
                shutil.copy2(timeline_route, ref_path)
            except Exception:
                pass
        return timeline
    else:
        if not os.path.exists(timeline_route):
            logger.error("Timeline file not found: %s", timeline_route)
            return []
        with open(timeline_route, "r", encoding="utf-8") as fh:
            timeline = json.load(fh)
        logger.info("Loaded timeline: %s  (%d events)", timeline_route, len(timeline))
        return timeline


def _flat_to_rounds(dialog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rounds: List[Dict[str, Any]] = []
    i = 0
    while i + 1 < len(dialog):
        u = dialog[i]
        a = dialog[i + 1]
        if (u.get("speaker") == "User"
                and a.get("speaker") == "Agent"
                and u.get("dia_id")):
            rounds.append({
                "round_id":   str(u.get("dia_id")),
                "user_idx":   i,
                "agent_idx":  i + 1,
                "user_text":  str(u.get("clean_text", "")),
                "agent_text": str(a.get("clean_text", "")),
                "has_pdf":    bool(u.get("pdf_file")),
                "has_image":  bool(u.get("img_file")),
            })
            i += 2
        else:
            i += 1
    return rounds


def _format_round_chunk(chunk: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for r in chunk:
        lines.append(f"[{r['round_id']}] User: {r['user_text']}")
        lines.append(f"[{r['round_id']}] Assistant: {r['agent_text']}")
    return "\n".join(lines)


def _format_dialog_overview(rounds: List[Dict[str, Any]], max_chars: int = 120) -> str:
    return "\n".join(
        f"[{r['round_id']}] {r['user_text'][:max_chars]}"
        for r in rounds
    )


def _audit_chunk(
    persona: str,
    core_event: str,
    timeline_summary: str,
    chunk: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = DIALOGUE_SELFCHECK_CHUNK_PROMPT % (
        persona, core_event, timeline_summary, _format_round_chunk(chunk),
    )
    try:
        raw = run_chatgpt(prompt, 1, 1500, "chatgpt").strip()
    except Exception as exc:
        logger.warning("Dialogue self-check (chunk) LLM call failed: %s", exc)
        return {}
    report = parse_json_object(raw, warn_prefix="dialogue self-check chunk: ")
    if not report:
        return {}
    for k in ("persona_violations", "core_event_violations", "hallucinations", "internal_contradictions"):
        report.setdefault(k, [])
    return report


def _audit_cross_chunk(
    persona: str,
    core_event: str,
    rounds: List[Dict[str, Any]],
    per_chunk_findings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    findings_str = json.dumps(per_chunk_findings, ensure_ascii=False)
    prompt = DIALOGUE_SELFCHECK_CROSS_PROMPT % (
        persona, core_event, findings_str, _format_dialog_overview(rounds),
    )
    try:
        raw = run_chatgpt(prompt, 1, 1500, "chatgpt").strip()
    except Exception as exc:
        logger.warning("Dialogue self-check (cross) LLM call failed: %s", exc)
        return {}
    report = parse_json_object(raw, warn_prefix="dialogue self-check cross: ")
    if not report:
        return {}
    for k in ("cross_chunk_contradictions", "global_persona_violations", "global_core_event_violations"):
        report.setdefault(k, [])
    return report


def _collect_flagged_round_ids(
    per_chunk_findings: List[Dict[str, Any]],
    cross_report: Dict[str, Any],
) -> List[str]:
    seen: Dict[str, None] = {}

    def _add(rid: Any) -> None:
        if rid is None:
            return
        rid_s = str(rid).strip()
        if rid_s and rid_s not in seen:
            seen[rid_s] = None

    for report in per_chunk_findings:
        for k in ("persona_violations", "core_event_violations", "hallucinations"):
            for item in report.get(k, []) or []:
                _add(item.get("round"))
        for item in report.get("internal_contradictions", []) or []:
            for r in item.get("rounds", []) or []:
                _add(r)
    for k in ("global_persona_violations", "global_core_event_violations"):
        for item in cross_report.get(k, []) or []:
            _add(item.get("round"))
    for item in cross_report.get("cross_chunk_contradictions", []) or []:
        for r in item.get("rounds", []) or []:
            _add(r)
    return list(seen.keys())


def _collect_issues_for_round(
    round_id: str,
    per_chunk_findings: List[Dict[str, Any]],
    cross_report: Dict[str, Any],
) -> List[str]:
    issues: List[str] = []
    for report in per_chunk_findings:
        for k in ("persona_violations", "core_event_violations", "hallucinations"):
            for item in report.get(k, []) or []:
                if str(item.get("round", "")).strip() == round_id and item.get("issue"):
                    issues.append(f"[{k}] {item['issue']}")
        for item in report.get("internal_contradictions", []) or []:
            if round_id in [str(r).strip() for r in item.get("rounds", []) or []]:
                others = [str(r) for r in item.get("rounds", []) if str(r) != round_id]
                issues.append(
                    f"[internal_contradiction] vs {','.join(others) or '?'}: {item.get('issue', '')}"
                )
    for k in ("global_persona_violations", "global_core_event_violations"):
        for item in cross_report.get(k, []) or []:
            if str(item.get("round", "")).strip() == round_id and item.get("issue"):
                issues.append(f"[{k}] {item['issue']}")
    for item in cross_report.get("cross_chunk_contradictions", []) or []:
        if round_id in [str(r).strip() for r in item.get("rounds", []) or []]:
            others = [str(r) for r in item.get("rounds", []) if str(r) != round_id]
            issues.append(
                f"[cross_chunk_contradiction] vs {','.join(others) or '?'}: {item.get('issue', '')}"
            )
    return issues


def _repair_round(
    persona: str,
    core_event: str,
    rounds: List[Dict[str, Any]],
    target_idx: int,
    issues: List[str],
    window: int = 2,
) -> Optional[Tuple[str, str]]:
    lo = max(0, target_idx - window)
    hi = min(len(rounds), target_idx + window + 1)
    window_lines: List[str] = []
    for i in range(lo, hi):
        marker = "  >>> TARGET >>>" if i == target_idx else ""
        r = rounds[i]
        window_lines.append(f"[{r['round_id']}]{marker} User: {r['user_text']}")
        window_lines.append(f"[{r['round_id']}] Assistant: {r['agent_text']}")
    window_text = "\n".join(window_lines)
    issues_text = "\n".join(f"- {it}" for it in issues) or "- (no specific issue text)"

    prompt = DIALOGUE_REPAIR_PROMPT % (persona, core_event, window_text, issues_text)
    try:
        raw = run_chatgpt(prompt, 1, 800, "chatgpt").strip()
    except Exception as exc:
        logger.warning("Dialogue repair LLM call failed: %s", exc)
        return None
    obj = parse_json_object(raw, warn_prefix="dialogue repair: ")
    if not obj:
        return None
    new_user = str(obj.get("user", "")).strip().strip('"')
    new_agent = str(obj.get("assistant", "")).strip().strip('"')
    if not new_user or not new_agent:
        return None
    return new_user, new_agent


def run_dialogue_self_check(
    dialog: List[Dict[str, Any]],
    persona: str,
    core_event: str,
    timeline: List[dict],
    *,
    chunk_size: int = 8,
    repair: bool = True,
) -> Tuple[Dict[str, Any], int]:
    rounds = _flat_to_rounds(dialog)
    if not rounds:
        return ({"ok": True, "note": "no rounds parsed"}, 0)

    timeline_summary = "\n".join(
        f"- [{e.get('Event_Time', 'N/A')}] {e.get('Event_Description', '')}"
        for e in (timeline or [])
    ) or "(no timeline available)"

    chunks: List[List[Dict[str, Any]]] = [
        rounds[i:i + chunk_size]
        for i in range(0, len(rounds), max(1, chunk_size))
    ]
    per_chunk_findings: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        logger.info(
            "Dialogue self-check chunk %d/%d (rounds %s..%s)",
            idx, len(chunks), chunk[0]["round_id"], chunk[-1]["round_id"],
        )
        per_chunk_findings.append(_audit_chunk(persona, core_event, timeline_summary, chunk))

    logger.info("Dialogue self-check cross-chunk pass over %d rounds", len(rounds))
    cross_report = _audit_cross_chunk(persona, core_event, rounds, per_chunk_findings)

    flagged_ids = _collect_flagged_round_ids(per_chunk_findings, cross_report)
    n_repaired = 0
    repairs: List[Dict[str, Any]] = []

    if repair and flagged_ids:
        round_by_id = {r["round_id"]: (i, r) for i, r in enumerate(rounds)}
        for rid in flagged_ids:
            if rid not in round_by_id:
                logger.info("Self-check flagged unknown round %s - skipping", rid)
                continue
            target_idx, _r = round_by_id[rid]
            issues = _collect_issues_for_round(rid, per_chunk_findings, cross_report)
            out = _repair_round(persona, core_event, rounds, target_idx, issues)
            if out is None:
                repairs.append({"round": rid, "status": "repair_failed", "issues": issues})
                continue
            new_user, new_agent = out
            # Mutate the underlying dialog list at the recorded user/agent indices.
            r = rounds[target_idx]
            dialog[r["user_idx"]]["clean_text"] = new_user
            dialog[r["agent_idx"]]["clean_text"] = new_agent
            # Keep the round view in sync so later iterations see the rewrite.
            rounds[target_idx]["user_text"]  = new_user
            rounds[target_idx]["agent_text"] = new_agent
            n_repaired += 1
            repairs.append({
                "round": rid,
                "status": "repaired",
                "issues": issues,
                "old_user":      _r["user_text"],
                "old_assistant": _r["agent_text"],
                "new_user":      new_user,
                "new_assistant": new_agent,
            })

    report = {
        "stage_a_per_chunk":  per_chunk_findings,
        "stage_b_cross":      cross_report,
        "flagged_round_ids":  flagged_ids,
        "repairs":            repairs,
        "n_flagged":          len(flagged_ids),
        "n_repaired":         n_repaired,
        "ok":                 not flagged_ids,
    }
    return report, n_repaired


def run_thematic_pipeline(out_dir: str) -> None:

    th_cfg         = cfg("thematic_subset") or {}
    user_persona   = str(th_cfg.get("user_persona", "")).strip()
    automatic_mode = bool(th_cfg.get("automatic_mode", True))
    max_images     = int(th_cfg.get("max_images_per_turn", 5))
    graph_route    = str(th_cfg.get("graph_route",     ".")).strip()
    included_raw   = str(th_cfg.get("included_graph",  ""))
    included_graphs = [g.strip() for g in included_raw.split(",") if g.strip()]
    max_turns      = int(th_cfg.get("max_turns_per_session_thematic", 40))

    if not user_persona:
        logger.error("thematic_subset.user_persona is empty – aborting.")
        return

    logger.info(
        "Thematic pipeline  |  automatic=%s  max_images=%d  max_turns=%d  out_dir=%s",
        automatic_mode, max_images, max_turns, out_dir,
    )

    images_base    = os.path.join(out_dir, "images")
    dialog_file    = os.path.join(out_dir, "dialog.json")
    summaries_file = os.path.join(out_dir, "summaries.json")
    os.makedirs(images_base, exist_ok=True)

    timeline = _load_or_generate_timeline(th_cfg, out_dir)
    if not timeline:
        return
    num_events = len(timeline)

    dialog:        List[Dict[str, Any]] = []
    all_summaries: Dict[str, Any]       = {}  # { "round_1": {...}, "round_2": {...}, … }

    global_turn: int = 0
    round_num:   int = 0
    event_ptr:   int = 0   # index of the "newest" event used in each round
    last_summary: dict = {}

    while global_turn < max_turns and event_ptr < num_events:
        round_num += 1

        if round_num == 1:
            current_event     = timeline[0]
            events_for_prompt = json.dumps(current_event, ensure_ascii=False, indent=2)
            keyword = str(current_event.get("Keyword", "")).strip()
        else:
            newer_event = timeline[event_ptr]
            older_event = timeline[event_ptr - 1] if event_ptr > 0 else newer_event
            two_events  = [older_event, newer_event] if older_event is not newer_event else [newer_event]
            events_for_prompt = json.dumps(two_events, ensure_ascii=False, indent=2)
            keyword = str(newer_event.get("Keyword", "")).strip()

        round_images_dir = os.path.join(images_base, f"round_{round_num}")
        img_files, img_paths = _get_images_for_round(
            keyword, round_images_dir, max_images,
            automatic_mode, graph_route, included_graphs,
        )
        img_ids    = list(range(1, len(img_files) + 1))
        image_desc = f"Images of {keyword}" if keyword else ", ".join(img_files)

        logger.info(
            "[Round %d] keyword='%s'  images=%d  event_ptr=%d",
            round_num, keyword, len(img_files), event_ptr,
        )

        if round_num == 1:
            turn_u = global_turn + 1
            turn_a = global_turn + 2

            # User
            user_prompt = THEMATIC_USER_PROMPT_INIT % (user_persona, events_for_prompt, image_desc)
            logger.info("[Round 1 | Turn %d] User (init) …", turn_u)
            user_raw  = _call_vision_llm(user_prompt, img_paths, max_tokens=300)
            user_obj   = parse_json_object(user_raw)
            user_text  = user_obj.get("clean_text", user_raw).strip('"')
            user_ts    = _extract_single_date(str(user_obj.get("timestamp", "")))
            user_img_desc = user_obj.get("image_description", "")

            user_record: Dict[str, Any] = {
                "speaker":           user_obj.get("speaker", "User"),
                "timestamp":         user_ts,
                "clean_text":        user_text,
                "image_description": user_img_desc,
                "img_file":          img_files,
                "img_id":            img_ids,
                "dia_id":            f"D1:{turn_u}",
            }
            dialog.append(user_record)
            print(f"\nUser  [D1:{turn_u}]: {user_text}")
            global_turn += 1

            if global_turn >= max_turns:
                logger.info("Reached max_turns after User init turn – stopping.")
                break

            # Agent
            agent_prompt = THEMATIC_AGENT_PROMPT_INIT % (user_text, image_desc)
            logger.info("[Round 1 | Turn %d] Agent (init) …", turn_a)
            agent_raw  = _call_vision_llm(agent_prompt, img_paths, max_tokens=400)
            agent_obj  = parse_json_object(agent_raw)
            agent_text = agent_obj.get("clean_text", agent_raw).strip('"')

            agent_record: Dict[str, Any] = {
                "speaker":    "Agent",
                "clean_text": agent_text,
                "dia_id":     f"D1:{turn_a}",
            }
            dialog.append(agent_record)
            print(f"Agent [D1:{turn_a}]: {agent_text}")
            global_turn += 1

            # Summary
            conv_text      = f"User: {user_text}\nAgent: {agent_text}"
            summary_prompt = DIALOGUE_SUMMARY_PROMPT % conv_text
            logger.info("[Round 1] Generating summary …")
            summary_raw  = run_chatgpt(summary_prompt, 1, 500, "chatgpt").strip()
            summary_obj  = parse_json_object(summary_raw)
            last_summary = normalize_summary_to_string(
                summary_obj if summary_obj else None,
                summary_raw,
            )

            all_summaries["round_1"] = {
                "dia_ids": [f"D1:{turn_u}", f"D1:{turn_a}"],
                "summary": last_summary,
            }
            with open(summaries_file, "w", encoding="utf-8") as fh:
                json.dump(all_summaries, fh, indent=2, ensure_ascii=False)
            logger.info("Summaries updated: %s  (round 1)", summaries_file)
            print(f"Summary [Round 1]: {last_summary[:100]}…\n{'─'*60}")

            event_ptr = 1

        else:
            summary_str = json.dumps(last_summary, ensure_ascii=False)

            turn_u1 = global_turn + 1
            turn_a1 = global_turn + 2
            turn_u2 = global_turn + 3
            turn_a2 = global_turn + 4

            # User (2 questions)
            user_prompt = THEMATIC_USER_PROMPT % (user_persona, events_for_prompt, summary_str, image_desc)
            logger.info("[Round %d | Turns %d,%d] User (follow-up) …", round_num, turn_u1, turn_u2)
            user_raw  = _call_vision_llm(user_prompt, img_paths, max_tokens=500)
            user_objs = extract_json_objects(user_raw)
            while len(user_objs) < 2:
                user_objs.append({})

            u1_text     = user_objs[0].get("clean_text", "").strip('"')
            u1_ts       = _extract_single_date(str(user_objs[0].get("timestamp", "")))
            u2_text     = user_objs[1].get("clean_text", "").strip('"')
            u2_ts       = _extract_single_date(str(user_objs[1].get("timestamp", "")))
            u2_img_desc = user_objs[1].get("image_description", "")

            # Agent (2 answers)
            questions_json = json.dumps(
                {"QUESTION_1": u1_text, "QUESTION_2": u2_text},
                ensure_ascii=False, indent=2,
            )
            agent_prompt = THEMATIC_AGENT_PROMPT % (questions_json, image_desc)
            logger.info("[Round %d | Turns %d,%d] Agent (follow-up) …", round_num, turn_a1, turn_a2)
            agent_raw  = _call_vision_llm(agent_prompt, img_paths, max_tokens=600)
            agent_objs = extract_json_objects(agent_raw)
            while len(agent_objs) < 2:
                agent_objs.append({})

            a1_text = agent_objs[0].get("clean_text", "").strip('"')
            a2_text = agent_objs[1].get("clean_text", "").strip('"')

            # Build records
            user1_record: Dict[str, Any] = {
                "speaker":    user_objs[0].get("speaker", "User"),
                "timestamp":  u1_ts,
                "clean_text": u1_text,
                "dia_id":     f"D1:{turn_u1}",
            }
            agent1_record: Dict[str, Any] = {
                "speaker":    "Agent",
                "clean_text": a1_text,
                "dia_id":     f"D1:{turn_a1}",
            }
            user2_record: Dict[str, Any] = {
                "speaker":           user_objs[1].get("speaker", "User"),
                "timestamp":         u2_ts,
                "clean_text":        u2_text,
                "image_description": u2_img_desc,
                "img_file":          img_files,
                "img_id":            img_ids,
                "dia_id":            f"D1:{turn_u2}",
            }
            agent2_record: Dict[str, Any] = {
                "speaker":    "Agent",
                "clean_text": a2_text,
                "dia_id":     f"D1:{turn_a2}",
            }

            # Append in order: user1, agent1, user2, agent2; respect max_turns
            new_records = [user1_record, agent1_record, user2_record, agent2_record]
            added = 0
            for rec in new_records:
                dialog.append(rec)
                global_turn += 1
                added += 1
                if global_turn >= max_turns:
                    logger.info("Reached max_turns (%d) mid-round – stopping.", max_turns)
                    break

            print(f"\nUser1  [D1:{turn_u1}]: {u1_text}")
            if added >= 2:
                print(f"Agent1 [D1:{turn_a1}]: {a1_text}")
            if added >= 3:
                print(f"User2  [D1:{turn_u2}]: {u2_text}")
            if added >= 4:
                print(f"Agent2 [D1:{turn_a2}]: {a2_text}")

            # Summary (always generated for completed portion of this round)
            conv_text = (
                f"User: {u1_text}\nAgent: {a1_text}\n"
                f"User: {u2_text}\nAgent: {a2_text}"
            )
            summary_prompt = DIALOGUE_SUMMARY_PROMPT % conv_text
            logger.info("[Round %d] Generating summary …", round_num)
            summary_raw  = run_chatgpt(summary_prompt, 1, 500, "chatgpt").strip()
            summary_obj  = parse_json_object(summary_raw)
            last_summary = normalize_summary_to_string(
                summary_obj if summary_obj else None,
                summary_raw,
            )

            round_key = f"round_{round_num}"
            all_summaries[round_key] = {
                "dia_ids": [
                    f"D1:{turn_u1}", f"D1:{turn_a1}",
                    f"D1:{turn_u2}", f"D1:{turn_a2}",
                ],
                "summary": last_summary,
            }
            with open(summaries_file, "w", encoding="utf-8") as fh:
                json.dump(all_summaries, fh, indent=2, ensure_ascii=False)
            logger.info("Summaries updated: %s  (round %d)", summaries_file, round_num)
            print(f"Summary [Round {round_num}]: {last_summary[:100]}…\n{'─'*60}")

            event_ptr += 1

            if global_turn >= max_turns:
                break

    # Generate fragment-based primary+followup PDF rounds for every entry in
    # thematic_subset.included_pdf.  Rounds are appended to the same dialog.json
    # with continuing dia_id D1:N and round_num; PDF user records carry
    # pdf_file=[<basename>.pdf] so downstream MR question generation can detect
    # a PDF context.
    pdf_route    = str(th_cfg.get("pdf_route", "")).strip()
    pdf_included = str(th_cfg.get("included_pdf", "")).strip()
    pdf_pdfs     = _resolve_included_pdfs(pdf_route, pdf_included) if pdf_route else []
    pdfs_kept    = 0

    if pdf_pdfs:
        pages_per_fragment    = max(1, int(th_cfg.get("pdf_pages_per_fragment", 3) or 3))
        max_rounds_per_pdf    = max(2, int(th_cfg.get("pdf_max_rounds_per_pdf", 40) or 40))
        pdf_dpi               = int(th_cfg.get("pdf_dpi", 144) or 144)
        max_pages_per_pdf     = max(1, int(th_cfg.get("pdf_max_pages_per_pdf", 60) or 60))
        keep_page_images      = bool(th_cfg.get("pdf_keep_page_images", False))

        pdfs_out_dir = os.path.join(out_dir, "pdfs")
        os.makedirs(pdfs_out_dir, exist_ok=True)

        logger.info(
            "PDF tail   |  pdfs=%d  pages_per_fragment=%d  max_rounds_per_pdf=%d  dpi=%d",
            len(pdf_pdfs), pages_per_fragment, max_rounds_per_pdf, pdf_dpi,
        )

        for pdf_name in pdf_pdfs:
            src_pdf = os.path.join(pdf_route, pdf_name)
            try:
                shutil.copy2(src_pdf, os.path.join(pdfs_out_dir, pdf_name))
            except Exception as exc:
                logger.warning("Could not copy source PDF %s: %s", pdf_name, exc)

            # Render pages into a temporary folder so we can pass paths to the
            # vision LLM; the rendered PNGs are deleted at the end of this loop
            # unless ``pdf_keep_page_images`` is true.
            doc_title = os.path.splitext(pdf_name)[0]
            tmp_pages_dir = os.path.join(pdfs_out_dir, f"_pages_{doc_title}")
            try:
                page_basenames = _render_pdf_pages(
                    src_pdf, tmp_pages_dir,
                    dpi=pdf_dpi, max_pages=max_pages_per_pdf,
                    name_prefix=f"{doc_title}_p",
                )
            except Exception as exc:
                logger.error("Failed to render PDF %s: %s", pdf_name, exc)
                continue
            if not page_basenames:
                logger.warning("PDF %s produced 0 page renders — skipping.", pdf_name)
                continue
            page_paths = [os.path.join(tmp_pages_dir, b) for b in page_basenames]
            n_pages = len(page_basenames)

            fragments = _split_pdf_into_fragments(n_pages, pages_per_fragment)
            # Each fragment uses 2 user turns (primary + followup) -> 4 utterances.
            max_fragments = max(1, max_rounds_per_pdf // 2)
            if len(fragments) > max_fragments:
                fragments = fragments[:max_fragments]

            revealed = 0
            last_assistant_text = ""
            pdf_session_summaries: Dict[str, Any] = {}

            for frag_idx, (lo, hi) in enumerate(fragments, start=1):
                new_count = max(0, hi - revealed)
                revealed  = max(revealed, hi)
                cumulative_paths = page_paths[:revealed]

                round_num += 1
                turn_u = global_turn + 1
                turn_a = global_turn + 2
                summaries_str = json.dumps(pdf_session_summaries, ensure_ascii=False)
                if frag_idx == 1:
                    user_prompt = PDF_USER_PROMPT_INIT.format(
                        n_pages=new_count, doc_title=doc_title, extra="None.",
                    )
                else:
                    user_prompt = PDF_USER_PROMPT.format(
                        doc_title=doc_title, revealed=revealed, total=n_pages,
                        summaries=summaries_str, n_new=new_count,
                        extra="Avoid repeating phrasing or topic of earlier turns.",
                    )

                logger.info(
                    "[PDF %s | Frag %d/%d | Turn %d] User (primary)",
                    pdf_name, frag_idx, len(fragments), turn_u,
                )
                user_raw = _call_vision_llm(user_prompt, cumulative_paths, max_tokens=600)
                user_text = extract_clean_text(user_raw, default=user_raw.strip().strip('"'))

                user_record = {
                    "speaker":    "User",
                    "timestamp":  "",
                    "clean_text": user_text,
                    "img_file":   [],
                    "img_id":     [],
                    "pdf_file":   [pdf_name],
                    "dia_id":     f"D1:{turn_u}",
                }
                dialog.append(user_record)
                global_turn += 1
                if global_turn >= max_turns:
                    break

                agent_prompt = PDF_AGENT_PROMPT.format(
                    doc_title=doc_title, revealed=revealed, user_text=user_text,
                )
                agent_raw  = _call_vision_llm(agent_prompt, cumulative_paths, max_tokens=900)
                agent_text = extract_clean_text(agent_raw, default=agent_raw.strip().strip('"'))
                dialog.append({
                    "speaker":    "Agent",
                    "clean_text": agent_text,
                    "dia_id":     f"D1:{turn_a}",
                })
                global_turn += 1
                last_assistant_text = agent_text

                # Summary so the next-fragment user prompt has context.
                conv_text = f"User: {user_text}\nAgent: {agent_text}"
                summary_raw = run_chatgpt(DIALOGUE_SUMMARY_PROMPT % conv_text, 1, 500, "chatgpt").strip()
                summary_obj = parse_json_object(summary_raw)
                last_summary = normalize_summary_to_string(summary_obj if summary_obj else None, summary_raw)
                round_key = f"round_{round_num}"
                all_summaries[round_key] = {
                    "dia_ids": [f"D1:{turn_u}", f"D1:{turn_a}"],
                    "summary": last_summary,
                }
                pdf_session_summaries[round_key] = {"summary": last_summary}
                with open(summaries_file, "w", encoding="utf-8") as fh:
                    json.dump(all_summaries, fh, indent=2, ensure_ascii=False)

                if global_turn >= max_turns:
                    break

                round_num += 1
                turn_u = global_turn + 1
                turn_a = global_turn + 2
                fu_user_prompt = PDF_FOLLOWUP_USER_PROMPT.format(
                    doc_title=doc_title, last_assistant=last_assistant_text,
                )
                try:
                    fu_user_raw = run_chatgpt(fu_user_prompt, 1, 400, "chatgpt")
                except Exception as exc:
                    logger.warning(
                        "PDF %s followup user (text LLM) failed: %s — falling back to vision",
                        pdf_name, exc,
                    )
                    fu_user_raw = _call_vision_llm(fu_user_prompt, cumulative_paths, max_tokens=400)
                fu_user_text = extract_clean_text(fu_user_raw, default=fu_user_raw.strip().strip('"'))

                dialog.append({
                    "speaker":    "User",
                    "timestamp":  "",
                    "clean_text": fu_user_text,
                    "dia_id":     f"D1:{turn_u}",
                    "is_followup": True,
                })
                global_turn += 1
                if global_turn >= max_turns:
                    break

                fu_agent_prompt = PDF_AGENT_PROMPT.format(
                    doc_title=doc_title, revealed=revealed, user_text=fu_user_text,
                )
                fu_agent_raw  = _call_vision_llm(fu_agent_prompt, cumulative_paths, max_tokens=900)
                fu_agent_text = extract_clean_text(fu_agent_raw, default=fu_agent_raw.strip().strip('"'))
                dialog.append({
                    "speaker":    "Agent",
                    "clean_text": fu_agent_text,
                    "dia_id":     f"D1:{turn_a}",
                })
                global_turn += 1
                last_assistant_text = fu_agent_text

                conv_text = f"User (follow-up): {fu_user_text}\nAgent: {fu_agent_text}"
                summary_raw = run_chatgpt(DIALOGUE_SUMMARY_PROMPT % conv_text, 1, 500, "chatgpt").strip()
                summary_obj = parse_json_object(summary_raw)
                last_summary = normalize_summary_to_string(summary_obj if summary_obj else None, summary_raw)
                round_key = f"round_{round_num}"
                all_summaries[round_key] = {
                    "dia_ids": [f"D1:{turn_u}", f"D1:{turn_a}"],
                    "summary": last_summary,
                    "is_followup": True,
                }
                pdf_session_summaries[round_key] = {"summary": last_summary}
                with open(summaries_file, "w", encoding="utf-8") as fh:
                    json.dump(all_summaries, fh, indent=2, ensure_ascii=False)

                if global_turn >= max_turns:
                    break

            # Cleanup rendered PNGs unless we were told to keep them.
            if not keep_page_images:
                try:
                    shutil.rmtree(tmp_pages_dir)
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", tmp_pages_dir, exc)
            pdfs_kept += 1

            if global_turn >= max_turns:
                logger.info(
                    "PDF tail hit max_turns (%d) — stopping after PDF %s.",
                    max_turns, pdf_name,
                )
                break

    dialog_check_enabled = bool(th_cfg.get("dialogue_self_check", False))
    dialog_check_report: Dict[str, Any] = {}
    dialog_n_repaired = 0
    if dialog_check_enabled and dialog:
        chunk_size = max(1, int(th_cfg.get("dialogue_self_check_chunk_size", 8) or 8))
        repair_on  = bool(th_cfg.get("dialogue_self_check_repair", True))
        logger.info(
            "Dialogue self-check  |  chunk_size=%d  repair=%s",
            chunk_size, repair_on,
        )
        dialog_check_report, dialog_n_repaired = run_dialogue_self_check(
            dialog,
            user_persona,
            str(th_cfg.get("core_event", "")).strip(),
            timeline,
            chunk_size=chunk_size,
            repair=repair_on,
        )
        check_path = os.path.join(out_dir, "dialog_check.json")
        with open(check_path, "w", encoding="utf-8") as fh:
            json.dump(dialog_check_report, fh, indent=2, ensure_ascii=False)
        logger.info(
            "Dialog self-check report saved: %s  (flagged=%d, repaired=%d)",
            check_path,
            dialog_check_report.get("n_flagged", 0),
            dialog_n_repaired,
        )

    with open(dialog_file, "w", encoding="utf-8") as fh:
        json.dump(dialog, fh, indent=2, ensure_ascii=False)
    logger.info("Dialog saved: %s  (%d turns)", dialog_file, len(dialog))

    print(
        f"\n{'═' * 60}\n"
        f"Thematic pipeline complete\n"
        f"  Turns     : {global_turn}\n"
        f"  Rounds    : {round_num}\n"
        f"  Events    : {num_events} in timeline  (used up to index {event_ptr - 1})\n"
        + (f"  PDFs      : {pdfs_kept}\n" if pdfs_kept else "")
        + (f"  Self-check: flagged={dialog_check_report.get('n_flagged', 0)} "
           f"repaired={dialog_n_repaired}\n" if dialog_check_enabled else "")
        + f"  Dialog    : {dialog_file}\n"
        f"  Images    : {images_base}\n"
        f"  Summaries : {summaries_file}\n"
        f"{'═' * 60}"
    )
