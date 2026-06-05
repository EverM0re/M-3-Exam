from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from m3exam.baselines._runtime.multimodel_common.fm_retrieve_helpers import (
    corpus_text_overlap_round_ids,
    fm_overlap_round_ids,
    rounds_from_lightrag_context,
)


def _snippet_from_item(item: Dict[str, Any], max_len: int = 320) -> str:
    parts: List[str] = []
    for k in ("details", "summary", "caption", "name", "title", "description", "value"):
        v = str(item.get(k) or "").strip()
        if v and v not in parts:
            parts.append(v)
    text = " ".join(parts).strip() or str(item)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def extract_round_candidates(
    raw: Dict[str, Any],
    *,
    id_to_round: Dict[str, str],
    corpus_meta: List[Dict[str, Any]],
    item_round_id_fn: Any,
    extra_items: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[str, str, str]]:
    seen: set[str] = set()
    out: List[Tuple[str, str, str]] = []

    def _add_item(item: Dict[str, Any], source: str) -> None:
        if not isinstance(item, dict):
            return
        rid = item_round_id_fn(
            item, id_to_round=id_to_round, corpus_meta=corpus_meta
        )
        if not rid or rid in seen:
            return
        seen.add(rid)
        out.append((rid, _snippet_from_item(item), source))

    if isinstance(raw, dict):
        mem = raw.get("memories") or {}
        for mt, data in mem.items():
            if not isinstance(data, dict):
                continue
            items = list(data.get("items") or [])
            if mt == "episodic" and not items:
                for bucket in ("relevant", "recent"):
                    items.extend(data.get(bucket) or [])
            for item in items:
                _add_item(item, f"retrieve:{mt}")
        for item in raw.get("results") or []:
            if isinstance(item, dict):
                _add_item(item, "retrieve:results")

    for item in extra_items or []:
        _add_item(item, "search")

    return out


def merge_search_hits_into_items(
    episodic_hits: List[Dict[str, Any]],
    semantic_hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(episodic_hits or []) + list(semantic_hits or []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "") or json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def llm_rerank_round_ids(
    llm_client: Any,
    model: str,
    question: str,
    candidates: Sequence[Tuple[str, str, str]],
    *,
    top_k: int,
    max_candidates: int = 24,
) -> List[str]:
    if not llm_client or not model or not candidates:
        return [c[0] for c in candidates[:top_k]]

    pool = list(candidates)[:max_candidates]
    lines: List[str] = []
    for i, (rid, snip, src) in enumerate(pool, start=1):
        lines.append(f"{i}. [{rid}] ({src}) {snip}")

    prompt = (
        "You rank conversation rounds by relevance to a question.\n"
        f"Question: {question}\n\n"
        "Candidates:\n"
        + "\n".join(lines)
        + "\n\n"
        f"Reply with ONLY a JSON array of up to "
        f"{max(1, top_k)} candidate numbers in best-first order, e.g. [3,1,7]. "
        "Use [] if none are relevant."
    )
    try:
        from llm_chat_kwargs import chat_completion_kwargs

        resp = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **chat_completion_kwargs(model, max_output_tokens=64, temperature=0.0),
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\[[\d,\s]*\]", text)
        if not m:
            return [c[0] for c in pool[:top_k]]
        nums = json.loads(m.group(0))
        if not isinstance(nums, list):
            return [c[0] for c in pool[:top_k]]
        picked: List[str] = []
        for n in nums:
            try:
                idx = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(pool):
                rid = pool[idx][0]
                if rid not in picked:
                    picked.append(rid)
            if len(picked) >= top_k:
                break
        if picked:
            return picked
    except Exception:
        pass
    return [c[0] for c in pool[:top_k]]


def offline_round_fallback(
    question: str,
    corpus_meta: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    images_dir: Any,
    *,
    budget: int,
    multimodal: bool = False,
) -> List[str]:
    cap = max(budget, 12)
    text_hits = corpus_text_overlap_round_ids(
        question, corpus_meta, budget_rounds=cap
    )
    if not multimodal:
        return text_hits
    img_hits = fm_overlap_round_ids(
        question,
        corpus_meta,
        sessions,
        images_dir,
        budget_rounds=cap,
    )
    return order_rounds_primary_first(text_hits, img_hits, cap=cap)


def order_rounds_primary_first(
    primary: List[str],
    secondary: List[str],
    *,
    cap: int,
) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for r in list(primary) + list(secondary):
        rr = str(r).replace(" ", "")
        if not rr or rr in seen:
            continue
        seen.add(rr)
        out.append(rr)
        if cap > 0 and len(out) >= cap:
            break
    return out


def expand_ordered_rounds(
    reranked: Sequence[str],
    candidates: Sequence[str],
    offline: Sequence[str],
    *,
    cap: int,
) -> List[str]:
    pool = order_rounds_primary_first(
        list(reranked),
        list(candidates),
        cap=0,
    )
    return order_rounds_primary_first(pool, list(offline), cap=cap)


def rounds_from_ctx_and_raw(
    ctx: str,
    raw: Dict[str, Any],
    item_round_id_fn: Any,
    *,
    id_to_round: Dict[str, str],
    corpus_meta: List[Dict[str, Any]],
    extra_items: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    cands = extract_round_candidates(
        raw,
        id_to_round=id_to_round,
        corpus_meta=corpus_meta,
        item_round_id_fn=item_round_id_fn,
        extra_items=extra_items,
    )
    from_ctx = rounds_from_lightrag_context(ctx)
    return order_rounds_primary_first(
        from_ctx + [c[0] for c in cands],
        [],
        cap=max(len(from_ctx) + len(cands), 32),
    )
