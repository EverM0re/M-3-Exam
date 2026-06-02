from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def clean_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()


def fix_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


# JSON only allows backslash followed by one of: " \ / b f n r t  or uXXXX.
# Anything else is an "Invalid \\escape" - most often the model wrote LaTeX
# inside a string (\(P(s)\), \sigma, \pi, ...). Turn those rogue single
# backslashes into double-backslashes so json.loads can read the text
# verbatim (consumers see the original LaTeX after parsing).
_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def fix_invalid_escapes(s: str) -> str:
    return _INVALID_ESCAPE_RE.sub(r"\\\\", s)


def _sanitise_for_json(s: str) -> str:
    return fix_invalid_escapes(fix_trailing_commas(s))


def _loads_lenient(s: str):
    return json.loads(s, strict=False)


def parse_json_object(text: str, *, warn_prefix: str = "") -> Dict[str, Any]:
    text = clean_markdown_fences(text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        logger.warning("%sNo JSON object found: %.120s", warn_prefix, text)
        return {}
    json_str = _sanitise_for_json(text[start : end + 1])
    try:
        out = _loads_lenient(json_str)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError as exc:
        logger.warning("%sJSON decode error (%s): %.120s", warn_prefix, exc, json_str)
        return {}


def parse_json_array(text: str, *, warn_prefix: str = "") -> List[Any]:
    text = clean_markdown_fences(text.strip())
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end <= start:
        logger.warning("%sNo JSON array found: %.120s", warn_prefix, text)
        return []
    chunk = _sanitise_for_json(text[start:end])
    try:
        data = _loads_lenient(chunk)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as exc:
        logger.warning("%sJSON array decode error (%s): %.120s", warn_prefix, exc, chunk)
        # Last-ditch: try item-by-item recovery so one bad object does not
        # nuke the whole batch.
        return _recover_array_items(chunk, warn_prefix=warn_prefix)


def _recover_array_items(chunk: str, *, warn_prefix: str = "") -> List[Any]:
    items: List[Any] = []
    i = 0
    n = len(chunk)
    while i < n:
        start = chunk.find("{", i)
        if start == -1:
            break
        depth = 0
        j = start
        in_string = False
        escape_next = False
        while j < n:
            c = chunk[j]
            if escape_next:
                escape_next = False
            elif c == "\\" and in_string:
                escape_next = True
            elif c == '"':
                in_string = not in_string
            elif not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        piece = _sanitise_for_json(chunk[start : j + 1])
                        try:
                            items.append(_loads_lenient(piece))
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                "%sSkipping malformed item (%s): %.80s",
                                warn_prefix, exc, piece,
                            )
                        i = j + 1
                        break
            j += 1
        else:
            break
    return items


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    text = clean_markdown_fences(text.strip())
    results: List[Dict[str, Any]] = []
    i = 0
    while i < len(text):
        start = text.find("{", i)
        if start == -1:
            break
        depth = 0
        j = start
        in_string = False
        escape_next = False
        while j < len(text):
            c = text[j]
            if escape_next:
                escape_next = False
            elif c == "\\" and in_string:
                escape_next = True
            elif c == '"':
                in_string = not in_string
            elif not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = _sanitise_for_json(text[start : j + 1])
                        try:
                            obj = _loads_lenient(chunk)
                            if isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError as exc:
                            logger.warning("extract_json_objects decode error (%s): %.80s", exc, chunk)
                        i = j + 1
                        break
            j += 1
        else:
            break
    return results


_CLEAN_TEXT_RE = re.compile(
    r'"clean_text"\s*:\s*"((?:[^"\\]|\\.)*)"',
    flags=re.DOTALL,
)
_CLEAN_TEXT_TRUNC_RE = re.compile(
    r'"clean_text"\s*:\s*"((?:[^"\\]|\\.)*)$',
    flags=re.DOTALL,
)


def _unescape_json_string(s: str) -> str:
    try:
        return json.loads(f'"{s}"')
    except Exception:
        return (
            s.replace("\\n", "\n")
             .replace("\\r", "\r")
             .replace("\\t", "\t")
             .replace('\\"', '"')
             .replace("\\\\", "\\")
        )


def extract_clean_text(raw: str, *, default: str = "") -> str:
    obj = parse_json_object(raw)
    ct = obj.get("clean_text") if isinstance(obj, dict) else None
    if isinstance(ct, str) and ct.strip():
        return ct.strip()

    text = clean_markdown_fences(raw)

    m = _CLEAN_TEXT_RE.search(text)
    if m:
        return _unescape_json_string(m.group(1)).strip()

    m = _CLEAN_TEXT_TRUNC_RE.search(text)
    if m:
        return _unescape_json_string(m.group(1)).strip()

    return default


def parse_timeline_events(text: str) -> List[Dict[str, Any]]:
    text_stripped = clean_markdown_fences(text.strip())
    start = text_stripped.find("[")
    end = text_stripped.rfind("]")
    if start != -1 and end != -1:
        chunk = _sanitise_for_json(text_stripped[start : end + 1])
        try:
            result = _loads_lenient(chunk)
            if isinstance(result, list):
                return [x for x in result if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    return extract_json_objects(text_stripped)
