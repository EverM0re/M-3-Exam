from __future__ import annotations

import json
from typing import Any


def normalize_summary_to_string(parsed: Any, raw_fallback: str) -> str:
    if parsed is None:
        return (raw_fallback or "").strip()

    cur: Any = parsed
    seen = 0
    while isinstance(cur, dict) and "summary" in cur and seen < 4:
        nxt = cur.get("summary")
        if isinstance(nxt, str):
            return nxt.strip()
        cur = nxt
        seen += 1

    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    return str(parsed).strip()
