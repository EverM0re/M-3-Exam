import os
import re
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from m3exam.config.config_loader import cfg
from m3exam.functions.common.conversation_utils import (
    DISTRACTOR_TIMELINE_PROMPT,
    TIMELINE_GENERATION_PROMPT,
    TIMELINE_REPAIR_PROMPT,
    TIMELINE_SELFCHECK_PROMPT,
)
from m3exam.functions.common.json_llm import (
    parse_json_array,
    parse_json_object,
    parse_timeline_events,
)
from m3exam.global_methods import run_chatgpt, set_openai_key

logger = logging.getLogger(__name__)

TIMELINE_EVENT_TARGET = 25


def _timeline_retry_suffix(event_count: int) -> str:
    return (
        "\n\nCRITICAL — your previous answer was invalid. Reply with ONLY a JSON array.\n"
        f"- EXACTLY {event_count} objects; Event_Index must be 1..{event_count} in order.\n"
        '- Keys per object IN ORDER: "Event_Index", "Event_Description", "Query_Description", "Event_Time", "Keyword".\n'
        "- Every Event_Description, Query_Description, Event_Time, Keyword must be a non-empty string.\n"
        "- Query_Description must use the protagonist name from Persona (after \"Your name is\"), not Carl: "
        "[Name] shares photos/screenshots of [concrete visuals] and asks [specific question].\n"
    )


def normalize_timeline_events_export(
    events: List[dict], event_count: int = TIMELINE_EVENT_TARGET
) -> List[dict]:
    raw: List[dict] = [e for e in events if isinstance(e, dict)]
    if len(raw) < event_count:
        logger.error(
            "Timeline parse yielded %d dict event(s); need at least %d to normalize.",
            len(raw),
            event_count,
        )
        return []
    raw = raw[:event_count]

    out: List[dict] = []
    for i, ev in enumerate(raw, start=1):
        desc = str(ev.get("Event_Description", "")).strip()
        query = str(ev.get("Query_Description", "")).strip()
        et = str(ev.get("Event_Time", "")).strip()
        kw = str(ev.get("Keyword", "")).strip()

        if not desc:
            stem = kw or f"event {i}"
            desc = f"A story beat on {et or 'the scheduled day'}: {stem[:100]}."
            logger.debug("Event %d: repaired empty Event_Description", i)

        if not query:
            query = (
                f"The user shares photos illustrating: {desc[:120]} "
                f"and asks the assistant one specific, visually grounded question about that moment."
            )
            logger.debug("Event %d: repaired empty Query_Description", i)

        if not et:
            et = "2023-01-01"
            logger.debug("Event %d: repaired missing Event_Time", i)

        if not kw:
            alpha = re.sub(r"[^a-zA-Z0-9 ]+", " ", desc)
            bits = [w for w in alpha.split() if len(w) > 2][:4]
            kw = " ".join(bits).lower() if bits else f"life moment {i}"
            logger.debug("Event %d: repaired empty Keyword", i)

        out.append(
            {
                "Event_Index": i,
                "Event_Description": desc,
                "Query_Description": query,
                "Event_Time": et,
                "Keyword": kw,
            }
        )

    return out


def normalize_timeline_json_path(route: str) -> str:
    route = (route or "").strip()
    if not route.lower().endswith(".json"):
        route += ".json"
    return route


def is_usable_timeline_file(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list) or len(data) == 0:
            return False
        return all(isinstance(x, dict) for x in data)
    except Exception:
        return False


def _default_outputs_dir() -> str:
    paths_cfg = cfg("paths") or {}
    base = str(paths_cfg.get("out_dir", "./outputs")).strip() or "./outputs"
    project_root = Path(__file__).resolve().parent.parent
    if not os.path.isabs(base):
        return str((project_root / base).resolve())
    return os.path.abspath(base)


def resolve_thematic_output_dir(tlg: dict) -> str:
    custom = str((tlg or {}).get("thematic_output_dir", "")).strip()
    if custom:
        return os.path.abspath(custom)
    base = _default_outputs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, f"thematic_dataset_{ts}")


def _generate_mainline(
    persona: str, core_event: str, event_count: int = TIMELINE_EVENT_TARGET
) -> List[dict]:
    base_prompt = TIMELINE_GENERATION_PROMPT % (persona, core_event, event_count)
    for attempt in range(2):
        prompt = base_prompt if attempt == 0 else base_prompt + _timeline_retry_suffix(event_count)
        raw = run_chatgpt(prompt, 1, 4096, "chatgpt").strip()
        parsed = parse_timeline_events(raw)
        if not parsed:
            logger.error(
                "Mainline timeline returned no events (attempt %d).\n"
                "Raw LLM output (first 400 chars):\n%s",
                attempt + 1,
                raw[:400],
            )
            continue
        normalized = normalize_timeline_events_export(parsed, event_count)
        if len(normalized) == event_count:
            return normalized
        logger.warning(
            "Mainline attempt %d: after normalize got %d events (need %d)",
            attempt + 1,
            len(normalized),
            event_count,
        )
    return []


def _format_mainline_for_context(events: List[dict]) -> str:
    return "\n".join(
        f"- [{e.get('Event_Time', 'N/A')}] {e.get('Event_Description', '')}"
        for e in events
    )


def _generate_distractors(
    persona: str,
    core_event: str,
    mainline: List[dict],
    n: int,
) -> List[dict]:
    if n <= 0:
        return []
    prompt = DISTRACTOR_TIMELINE_PROMPT % (
        persona,
        core_event,
        _format_mainline_for_context(mainline),
        n,
    )
    raw = run_chatgpt(prompt, 1, 2048, "chatgpt").strip()
    parsed = parse_json_array(raw, warn_prefix="distractor: ")
    cleaned: List[dict] = []
    for ev in parsed:
        if not isinstance(ev, dict):
            continue
        desc = str(ev.get("Event_Description", "")).strip()
        query = str(ev.get("Query_Description", "")).strip()
        et = str(ev.get("Event_Time", "")).strip()
        kw = str(ev.get("Keyword", "")).strip()
        if not (desc and query and et and kw):
            continue
        cleaned.append(
            {
                "Event_Description": desc,
                "Query_Description": query,
                "Event_Time": et,
                "Keyword": kw,
                "is_distractor": True,
            }
        )
        if len(cleaned) >= n:
            break
    if len(cleaned) < n:
        logger.warning(
            "Distractor generation produced %d/%d usable events.",
            len(cleaned),
            n,
        )
    return cleaned


def _interleave_and_renumber(
    mainline: List[dict],
    distractors: List[dict],
) -> List[dict]:
    if not distractors:
        # Pure mainline path — keep the original 1..25 indices, no extra key.
        return [dict(e) for e in mainline]

    tagged: List[Tuple[str, int, dict]] = []  # (date, kind_rank, event)
    for ev in mainline:
        tagged.append((str(ev.get("Event_Time", "")), 0, ev))
    for ev in distractors:
        tagged.append((str(ev.get("Event_Time", "")), 1, ev))
    tagged.sort(key=lambda t: (t[0], t[1]))

    merged: List[dict] = []
    for new_idx, (_, _, ev) in enumerate(tagged, start=1):
        out = {"Event_Index": new_idx}
        for key in ("Event_Description", "Query_Description", "Event_Time", "Keyword"):
            out[key] = ev.get(key, "")
        if ev.get("is_distractor"):
            out["is_distractor"] = True
        merged.append(out)
    return merged


def _selfcheck_timeline(
    persona: str,
    core_event: str,
    timeline: List[dict],
) -> dict:
    prompt = TIMELINE_SELFCHECK_PROMPT % (
        persona,
        core_event,
        json.dumps(timeline, ensure_ascii=False),
    )
    raw = run_chatgpt(prompt, 1, 1500, "chatgpt").strip()
    report = parse_json_object(raw, warn_prefix="timeline self-check: ")
    if not report:
        logger.warning("Self-check returned unparseable response; treating as ok.")
        return {"ok": True, "persona_violations": [], "core_event_violations": [], "contradictions": []}
    # Make sure expected keys exist.
    for k in ("persona_violations", "core_event_violations", "contradictions"):
        report.setdefault(k, [])
    if "ok" not in report:
        report["ok"] = not any(report[k] for k in ("persona_violations", "core_event_violations", "contradictions"))
    return report


def _repair_timeline(
    persona: str,
    core_event: str,
    timeline: List[dict],
    report: dict,
) -> List[dict]:
    prompt = TIMELINE_REPAIR_PROMPT % (
        json.dumps(timeline, ensure_ascii=False),
        json.dumps(report, ensure_ascii=False),
        persona,
        core_event,
    )
    raw = run_chatgpt(prompt, 1, 4096, "chatgpt").strip()
    parsed = parse_json_array(raw, warn_prefix="timeline repair: ")
    if len(parsed) != len(timeline):
        logger.warning(
            "Repair LLM returned %d events but timeline has %d — discarding repair.",
            len(parsed),
            len(timeline),
        )
        return []
    # Preserve Event_Index ordering and the is_distractor flags from the input
    # timeline (the repair prompt asked the model to keep non-flagged events
    # byte-identical, but we re-enforce structure to be safe).
    repaired: List[dict] = []
    for original, candidate in zip(timeline, parsed):
        if not isinstance(candidate, dict):
            return []
        out = dict(original)  # start from the original to lock in unmodified keys
        for key in ("Event_Description", "Query_Description", "Event_Time", "Keyword"):
            v = candidate.get(key, "")
            if isinstance(v, str) and v.strip():
                out[key] = v.strip()
        repaired.append(out)
    return repaired


def generate_timeline(
    persona: str,
    core_event: str,
    output_path: str,
    *,
    event_count: int = TIMELINE_EVENT_TARGET,
    add_distractors: bool = False,
    distractor_count: int = 0,
    self_check: bool = False,
    self_check_max_passes: int = 2,
) -> List[dict]:
    logger.info(
        "Generating timeline  |  output=%s  add_distractors=%s "
        "distractor_count=%d  self_check=%s",
        output_path,
        add_distractors,
        distractor_count if add_distractors else 0,
        self_check,
    )

    mainline = _generate_mainline(persona, core_event, event_count)
    if not mainline:
        logger.error(
            "Timeline generation failed after retries (need %d valid main-line events).",
            event_count,
        )
        return []

    distractors: List[dict] = []
    if add_distractors and distractor_count > 0:
        distractors = _generate_distractors(persona, core_event, mainline, distractor_count)
    timeline = _interleave_and_renumber(mainline, distractors)

    final_report: dict = {}
    if self_check:
        passes = max(1, int(self_check_max_passes))
        for pass_idx in range(1, passes + 1):
            report = _selfcheck_timeline(persona, core_event, timeline)
            final_report = report
            if report.get("ok"):
                logger.info("Self-check pass %d: ok.", pass_idx)
                break
            n_p = len(report.get("persona_violations", []))
            n_c = len(report.get("core_event_violations", []))
            n_x = len(report.get("contradictions", []))
            logger.warning(
                "Self-check pass %d found issues: persona=%d, core_event=%d, contradictions=%d",
                pass_idx, n_p, n_c, n_x,
            )
            if pass_idx >= passes:
                logger.warning("Self-check budget exhausted; keeping last timeline as-is.")
                break
            repaired = _repair_timeline(persona, core_event, timeline, report)
            if not repaired:
                logger.warning("Repair pass returned nothing usable; aborting further passes.")
                break
            timeline = repaired

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(timeline, fh, indent=2, ensure_ascii=False)
    logger.info("Timeline saved: %s  (%d events)", output_path, len(timeline))

    if self_check and final_report:
        check_path = os.path.join(out_dir, "timeline_check.json") if out_dir else "timeline_check.json"
        with open(check_path, "w", encoding="utf-8") as fh:
            json.dump(final_report, fh, indent=2, ensure_ascii=False)
        logger.info("Self-check report saved: %s", check_path)

    return timeline


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    set_openai_key()

    tl_cfg = cfg("timeline_generation") or {}
    persona = str(tl_cfg.get("user_persona", "")).strip()
    core_event = str(tl_cfg.get("core_event", "")).strip()
    timeline_route = normalize_timeline_json_path(str(tl_cfg.get("timeline_route", "./timeline.json")).strip())
    continue_dialogue = bool(tl_cfg.get("continue_with_thematic_dialogue", False))

    def maybe_run_thematic() -> None:
        if not continue_dialogue:
            return
        out_dir = resolve_thematic_output_dir(tl_cfg)
        os.makedirs(out_dir, exist_ok=True)
        logger.info("continue_with_thematic_dialogue=true -> run_thematic_pipeline(%s)", out_dir)
        from m3exam.functions.thematic.thematic_pipeline import run_thematic_pipeline

        run_thematic_pipeline(out_dir)

    # Chaining: use existing timeline JSON + thematic only (no LLM overwrite).
    if continue_dialogue and is_usable_timeline_file(timeline_route):
        with open(timeline_route, encoding="utf-8") as fh:
            n_existing = len(json.load(fh))
        logger.info(
            "Existing timeline at %s (%d events); skip LLM, run thematic only.",
            timeline_route,
            n_existing,
        )
        maybe_run_thematic()
        print(f"Thematic dialogue using existing timeline: {timeline_route}  ({n_existing} events)")
        return

    if not persona:
        print("ERROR: timeline_generation.user_persona is empty in config.yaml")
        sys.exit(1)
    if not core_event:
        print("ERROR: timeline_generation.core_event is empty in config.yaml")
        sys.exit(1)

    event_count = int(tl_cfg.get("event_count", TIMELINE_EVENT_TARGET) or TIMELINE_EVENT_TARGET)
    add_distractors = bool(tl_cfg.get("add_distractors", False))
    distractor_count = int(tl_cfg.get("distractor_count", 0) or 0)
    self_check = bool(tl_cfg.get("self_check", False))
    self_check_max_passes = int(tl_cfg.get("self_check_max_passes", 2) or 2)

    events = generate_timeline(
        persona,
        core_event,
        timeline_route,
        event_count=event_count,
        add_distractors=add_distractors,
        distractor_count=distractor_count,
        self_check=self_check,
        self_check_max_passes=self_check_max_passes,
    )
    if events:
        n_distract = sum(1 for e in events if e.get("is_distractor"))
        if add_distractors and n_distract:
            print(
                f"Timeline generated: {timeline_route}  "
                f"({len(events)} events, {n_distract} distractor)"
            )
        else:
            print(f"Timeline generated: {timeline_route}  ({len(events)} events)")
    else:
        print("ERROR: Timeline generation failed.")
        sys.exit(1)

    maybe_run_thematic()


if __name__ == "__main__":
    main()
