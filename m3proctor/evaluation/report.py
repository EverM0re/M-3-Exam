from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

TYPE_ORDER: Sequence[str] = ("ss", "ms", "tr", "mr", "fm", "th", "ii", "fj")
EM_ONLY_TYPES = {"fm", "fj"}
SOFT_AVG_EXCLUDE = {"fm", "fj"}

METRIC_LABELS = [
    ("em_score", "EM"),
    ("f1_score", "F1"),
    ("bleu1_score", "BLEU-1"),
    ("llm_score", "LLM-J"),
]


def _fmt(v: Optional[float], dash_if_none: bool = True) -> str:
    if v is None or not isinstance(v, (int, float)):
        return "-" if dash_if_none else ""
    return f"{v:.4f}"


def _per_type_value(rubric: Dict[str, Any], q_type: str, metric_key: str) -> Optional[float]:
    pt = (rubric or {}).get("per_type", {}) or {}
    block = pt.get(q_type)
    if not isinstance(block, dict):
        return None
    if block.get("count", 0) == 0:
        return None
    if q_type in EM_ONLY_TYPES and metric_key in ("f1_score", "bleu1_score", "llm_score"):
        return None
    v = block.get(metric_key)
    return float(v) if isinstance(v, (int, float)) else None


def _avg_value(rubric: Dict[str, Any], metric_key: str) -> Optional[float]:
    pt = (rubric or {}).get("per_type", {}) or {}
    total_n = 0
    total_sum = 0.0
    for t, blk in pt.items():
        if not isinstance(blk, dict):
            continue
        n = blk.get("count") or 0
        if not n:
            continue
        if metric_key in ("f1_score", "bleu1_score", "llm_score") and t in SOFT_AVG_EXCLUDE:
            continue
        v = blk.get(metric_key)
        if not isinstance(v, (int, float)):
            continue
        total_sum += float(v) * n
        total_n += n
    if total_n == 0:
        return None
    return total_sum / total_n


def render_per_type_table(
    title: str,
    summaries: Dict[str, Dict[str, Any]],
    model_order: Sequence[str],
) -> str:
    cols = ["Model", "Metric"] + [t.upper() for t in TYPE_ORDER] + ["Avg."]
    widths = [max(14, len(cols[0])), max(7, len(cols[1]))]
    widths += [8] * len(TYPE_ORDER) + [8]

    body_rows: List[List[str]] = []
    for m in model_order:
        s = summaries.get(m)
        if not isinstance(s, dict):
            for label_idx, (_, lab) in enumerate(METRIC_LABELS):
                body_rows.append(
                    [m if label_idx == 0 else "", lab] + ["-"] * (len(TYPE_ORDER) + 1)
                )
            continue
        if "error" in s or "_error" in s:
            for label_idx, (_, lab) in enumerate(METRIC_LABELS):
                body_rows.append(
                    [m if label_idx == 0 else "", lab] + ["ERROR"] * (len(TYPE_ORDER) + 1)
                )
            continue
        rubric = s.get("rubric") or {}
        for label_idx, (mkey, lab) in enumerate(METRIC_LABELS):
            row = [m if label_idx == 0 else "", lab]
            for t in TYPE_ORDER:
                row.append(_fmt(_per_type_value(rubric, t, mkey)))
            row.append(_fmt(_avg_value(rubric, mkey)))
            body_rows.append(row)

    for r in body_rows:
        for i, cell in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def _line(cells: Sequence[str]) -> str:
        parts = []
        for i, c in enumerate(cells):
            parts.append(f"{c:<{widths[i]}}" if i < 2 else f"{c:>{widths[i]}}")
        return " | ".join(parts)

    sep_width = sum(widths) + len(" | ") * (len(widths) - 1)
    out: List[str] = []
    if title:
        out.append("=" * sep_width)
        out.append(title.center(sep_width))
        out.append("=" * sep_width)
    out.append(_line(cols))
    out.append("-" * sep_width)
    last_model_row = ""
    for r in body_rows:
        if r[0] and last_model_row and r[0] != last_model_row:
            out.append("." * sep_width)
        if r[0]:
            last_model_row = r[0]
        out.append(_line(r))
    out.append("=" * sep_width)
    return "\n".join(out)


def render_cascade_table(
    title: str,
    summaries: Dict[str, Dict[str, Any]],
    model_order: Sequence[str],
) -> str:
    cols: List[str] = ["Model", "overall"]
    cols += [t.upper() for t in TYPE_ORDER]

    body_rows: List[List[str]] = []
    any_cascade = False
    for m in model_order:
        s = summaries.get(m)
        if not isinstance(s, dict) or "error" in s or "_error" in s:
            body_rows.append([m] + ["-"] * (len(cols) - 1))
            continue
        cs = s.get("cascade_summary")
        if not isinstance(cs, dict):
            body_rows.append([m, "-"] + ["-"] * len(TYPE_ORDER))
            continue
        any_cascade = True
        per_type = cs.get("per_type", {}) or {}
        row = [m, _fmt(cs.get("overall_cascade_rate"), False) or "-"]
        for t in TYPE_ORDER:
            blk = per_type.get(t)
            if isinstance(blk, dict) and blk.get("count"):
                row.append(_fmt(blk.get("cascade_rate"), False) or "-")
            else:
                row.append("-")
        body_rows.append(row)

    if not any_cascade:
        return ""

    widths = [max(14, len(cols[0]))] + [max(8, len(c)) for c in cols[1:]]
    for r in body_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def _line(cells: Sequence[str]) -> str:
        parts = []
        for i, c in enumerate(cells):
            parts.append(f"{c:<{widths[i]}}" if i == 0 else f"{c:>{widths[i]}}")
        return " | ".join(parts)

    sep_width = sum(widths) + len(" | ") * (len(widths) - 1)
    out: List[str] = []
    if title:
        out.append("=" * sep_width)
        out.append(title.center(sep_width))
        out.append("=" * sep_width)
    out.append(_line(cols))
    out.append("-" * sep_width)
    for r in body_rows:
        out.append(_line(r))
    out.append("=" * sep_width)
    out.append(
        "rate = fraction of (per-type) questions escalated to multimodal Stage 2."
    )
    return "\n".join(out)


def render_dataset_report(
    dataset: str,
    summaries: Dict[str, Dict[str, Any]],
    model_order: Sequence[str],
    *,
    llm_name: str = "",
) -> str:
    title1 = f"{llm_name + ' | ' if llm_name else ''}{dataset}  -  per-type metrics"
    title3 = f"{llm_name + ' | ' if llm_name else ''}{dataset}  -  cascade summary"
    parts = [render_per_type_table(title1, summaries, model_order)]
    cascade_tbl = render_cascade_table(title3, summaries, model_order)
    if cascade_tbl:
        parts.append(cascade_tbl)
    return "\n\n".join(parts)
