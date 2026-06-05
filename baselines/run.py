#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from m3exam.baselines._runtime.extended_common.dataset_io import (  # noqa: E402
    load_questions,
    load_sessions,
)
from m3exam.baselines._runtime.extended_common.llm_client import SharedLLM  # noqa: E402

from m3exam.m3proctor.evaluation.metrics import (  # noqa: E402
    aggregate,
    bleu1_text,
    em_text,
    f1_text,
    fm_em_image,
)
from m3exam.m3proctor.evaluation.report import render_dataset_report  # noqa: E402


EXTENDED_BASELINES = {"a_mem", "memoryos", "mem0_text", "mem0_visual", "nano_graphrag"}
MULTIMODEL_BASELINES = {"mirix", "memverse", "ngm", "raganything", "universalrag"}

EXTENDED_REGISTRY: Dict[str, Tuple[str, str, Dict[str, Any]]] = {
    "a_mem":         ("m3exam.baselines.amem.evaluator",         "AMemEvaluator",         {}),
    "memoryos":      ("m3exam.baselines.memoryos.evaluator",     "MemoryOSEvaluator",     {}),
    "mem0_text":     ("m3exam.baselines.mem0.evaluator",         "Mem0Evaluator",         {"variant": "mem0_text"}),
    "mem0_visual":   ("m3exam.baselines.mem0.evaluator",         "Mem0Evaluator",         {"variant": "mem0_visual"}),
    "nano_graphrag": ("m3exam.baselines.nano_graphrag.evaluator","NanoGraphRAGEvaluator", {}),
}

SUBFOLDER: Dict[str, str] = {
    "a_mem":         "amem",
    "memoryos":      "memoryos",
    "mem0_text":     "mem0",
    "mem0_visual":   "mem0",
    "nano_graphrag": "nano_graphrag",
    "mirix":         "mirix",
    "memverse":      "memverse",
    "ngm":           "ngm",
    "raganything":   "raganything",
    "universalrag":  "universalrag",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline", required=True,
        choices=sorted(EXTENDED_BASELINES | MULTIMODEL_BASELINES),
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--datasets-root", default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--no-judge", action="store_true")
    return p.parse_args()


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg(baseline: str) -> Dict[str, Any]:
    top = yaml.safe_load((_THIS_DIR / "config.yaml").read_text(encoding="utf-8")) or {}
    sub_path = _THIS_DIR / SUBFOLDER[baseline] / "config.yaml"
    if sub_path.is_file():
        sub = yaml.safe_load(sub_path.read_text(encoding="utf-8")) or {}
    else:
        sub = {}
    return _deep_merge(top, sub)


def _cfg_get(cfg: Dict[str, Any], *keys: Any, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _cfg_int(cfg: Dict[str, Any], *keys: Any, default: int = 0) -> int:
    try:
        return int(_cfg_get(cfg, *keys, default=default))
    except Exception:
        return default


def _cfg_float(cfg: Dict[str, Any], *keys: Any, default: float = 0.0) -> float:
    try:
        return float(_cfg_get(cfg, *keys, default=default))
    except Exception:
        return default


def _cfg_str(cfg: Dict[str, Any], *keys: Any, default: str = "") -> str:
    v = _cfg_get(cfg, *keys, default=default)
    return str(v) if v is not None else default


def _cfg_bool(cfg: Dict[str, Any], *keys: Any, default: bool = False) -> bool:
    v = _cfg_get(cfg, *keys, default=default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v) if v is not None else default


def _empty_to_none_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _pdf_limits(cfg: Dict[str, Any], model_name: str) -> Tuple[int, int, int]:
    return (
        _cfg_int(cfg, "models", model_name, "embed_pdf_max_chars", default=8000),
        _cfg_int(cfg, "models", model_name, "context_pdf_max_chars", default=12000),
        _cfg_int(cfg, "models", model_name, "min_native_pdf_chars", default=400),
    )


def _lightrag_index_llm_kwargs(cfg: Dict[str, Any]) -> Dict[str, str]:
    sub = _cfg_get(cfg, "lightrag", "index_llm", default=None)
    if not isinstance(sub, dict):
        return {}
    key = _empty_to_none_str(sub.get("api_key"))
    base = _empty_to_none_str(sub.get("base_url"))
    model = _empty_to_none_str(sub.get("model"))
    if not key or not base or not model:
        return {}
    return {
        "lightrag_index_api_key": key,
        "lightrag_index_base_url": base,
        "lightrag_index_model": model,
    }


def _lightrag_retrieve_mode(cfg: Dict[str, Any]) -> str:
    raw = _cfg_get(cfg, "lightrag", "retrieve_mode", default="mix")
    s = str(raw).strip().lower() if raw else "mix"
    return s or "mix"


def _lightrag_embedding_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "embed_backend": _cfg_str(cfg, "embedding", "backend", default="local").lower().strip(),
        "embed_model": _cfg_str(cfg, "embedding", "model", default="text-embedding-3-small"),
        "embed_dim": _cfg_int(cfg, "embedding", "dim", default=768),
        "embed_local_model": _cfg_str(cfg, "embedding", "local_model", default="thenlper/gte-base"),
        "embed_api_base_url": _empty_to_none_str(_cfg_get(cfg, "embedding", "base_url", default="")),
        "embed_api_key": _empty_to_none_str(_cfg_get(cfg, "embedding", "api_key", default="")),
    }


def _memory_agent_model_from_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    llm_b = cfg.get("llm") or {}
    mv_b = (cfg.get("models") or {}).get("memverse") or {}
    a = str(llm_b.get("memory_agent_model") or "").strip()
    b = str(mv_b.get("memory_agent_model") or "").strip()
    s = a or b
    return s or None


def _build_extended(baseline: str, cfg: Dict[str, Any], llm: SharedLLM, run_id: str):
    mod_path, cls_name, kwargs = EXTENDED_REGISTRY[baseline]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls(cfg, llm, run_id, **kwargs)


def _build_multimodel(baseline: str, cfg: Dict[str, Any], llm: SharedLLM, run_output_root: Path):
    from m3exam.baselines._runtime.multimodel_common.pdf_session_text import resolve_pdf_policy

    use_gpu = _cfg_bool(cfg, "models", "use_gpu", default=True)
    llm_client = getattr(llm, "_main", None)
    llm_client_raw = getattr(llm_client, "client", None) if llm_client else None
    llm_model = llm.model

    if baseline == "ngm":
        from m3exam.baselines.ngm.evaluator import NGMEvaluator
        pol = resolve_pdf_policy("ngm", cfg, llm_model=llm_model, probe_native_pdf=None)
        emb_c, ctx_c, min_n = _pdf_limits(cfg, "ngm")
        return NGMEvaluator(
            semantic_threshold=_cfg_float(cfg, "models", "ngm", "semantic_threshold", default=0.5),
            use_gpu=use_gpu,
            llm_client=llm_client_raw,
            llm_model=llm_model,
            pdf_policy=pol,
            embed_pdf_max_chars=emb_c,
            context_pdf_max_chars=ctx_c,
            min_native_pdf_chars=min_n,
            clip_model=_cfg_str(cfg, "models", "ngm", "clip_model", default="openai/clip-vit-base-patch32"),
            retrieve_dialogue_mode=_cfg_str(cfg, "models", "ngm", "retrieve_dialogue_mode", default="query_native"),
            retrieve_context_max_rounds=_cfg_int(cfg, "models", "ngm", "retrieve_context_max_rounds", default=0),
            append_graph_ranked_round_markers=_cfg_bool(cfg, "models", "ngm", "append_graph_ranked_round_markers", default=True),
            related_neighbor_limit=_cfg_int(cfg, "models", "ngm", "related_neighbor_limit", default=10),
        )

    if baseline == "universalrag":
        from m3exam.baselines.universalrag.evaluator import UniversalRAGEvaluator
        pol = resolve_pdf_policy("universalrag", cfg, llm_model=llm_model, probe_native_pdf=None)
        emb_c, ctx_c, min_n = _pdf_limits(cfg, "universalrag")
        return UniversalRAGEvaluator(
            text_model=_cfg_str(cfg, "models", "universalrag", "text_model", default="BAAI/bge-large-en-v1.5"),
            clip_model=_cfg_str(cfg, "models", "universalrag", "clip_model", default="openai/clip-vit-base-patch32"),
            use_gpu=use_gpu,
            alpha=_cfg_float(cfg, "models", "universalrag", "alpha", default=0.2),
            llm_client=llm_client_raw,
            llm_model=llm_model,
            pdf_policy=pol,
            embed_pdf_max_chars=emb_c,
            context_pdf_max_chars=ctx_c,
            min_native_pdf_chars=min_n,
            use_gpt_router=_cfg_bool(cfg, "models", "universalrag", "use_gpt_router", default=True),
        )

    if baseline == "raganything":
        from m3exam.baselines.raganything.evaluator import RAGAnythingEvaluator
        pol = resolve_pdf_policy("raganything", cfg, llm_model=llm_model, probe_native_pdf=None)
        emb_c, ctx_c, min_n = _pdf_limits(cfg, "raganything")
        return RAGAnythingEvaluator(
            text_model=_cfg_str(cfg, "models", "raganything", "text_model", default="sentence-transformers/all-mpnet-base-v2"),
            clip_model=_cfg_str(cfg, "models", "raganything", "clip_model", default="openai/clip-vit-large-patch14"),
            use_gpu=use_gpu,
            alpha=_cfg_float(cfg, "models", "raganything", "alpha", default=0.35),
            llm_client=llm_client_raw,
            llm_model=llm_model,
            pdf_policy=pol,
            embed_pdf_max_chars=emb_c,
            context_pdf_max_chars=ctx_c,
            min_native_pdf_chars=min_n,
            chunk_words=_cfg_int(cfg, "models", "raganything", "chunk_words", default=80),
            chunk_overlap=_cfg_int(cfg, "models", "raganything", "chunk_overlap", default=12),
            lightrag_mode=_lightrag_retrieve_mode(cfg),
            **_lightrag_embedding_kwargs(cfg),
            **_lightrag_index_llm_kwargs(cfg),
            lightrag_working_dir=run_output_root / ".eval_upstream" / "raganything_lightrag",
        )

    if baseline == "memverse":
        from m3exam.baselines.memverse.evaluator import MemVerseEvaluator
        pol = resolve_pdf_policy("memverse", cfg, llm_model=llm_model, probe_native_pdf=None)
        emb_c, ctx_c, min_n = _pdf_limits(cfg, "memverse")
        return MemVerseEvaluator(
            text_model=_cfg_str(cfg, "models", "memverse", "text_model", default="thenlper/gte-base"),
            clip_model=_cfg_str(cfg, "models", "memverse", "clip_model", default="openai/clip-vit-base-patch32"),
            use_gpu=use_gpu,
            alpha=_cfg_float(cfg, "models", "memverse", "alpha", default=0.4),
            llm_client=llm_client_raw,
            llm_model=llm_model,
            memory_agent_model=_memory_agent_model_from_cfg(cfg),
            pdf_policy=pol,
            embed_pdf_max_chars=emb_c,
            context_pdf_max_chars=ctx_c,
            min_native_pdf_chars=min_n,
            recency_gamma=_cfg_float(cfg, "models", "memverse", "recency_gamma", default=0.22),
            use_official_mmkg=_cfg_bool(cfg, "models", "memverse", "use_official_mmkg", default=True),
            lightrag_mode=_lightrag_retrieve_mode(cfg),
            **_lightrag_embedding_kwargs(cfg),
            **_lightrag_index_llm_kwargs(cfg),
            lightrag_working_dir=run_output_root / ".eval_upstream" / "memverse_lightrag",
        )

    if baseline == "mirix":
        from m3exam.baselines.mirix.evaluator import MirixEvaluator
        emb_c, ctx_c, min_n = _pdf_limits(cfg, "mirix")
        return MirixEvaluator(
            cfg=cfg,
            llm_client=llm_client_raw,
            llm_model=llm_model,
            probe_native_pdf=None,
            mode=_cfg_str(cfg, "models", "mirix", "mode", default="local"),
            local_sqlite=_cfg_bool(cfg, "models", "mirix", "local_sqlite", default=True),
            mirix_data_dir=_cfg_str(cfg, "models", "mirix", "mirix_data_dir", default=""),
            mirix_data_dir_under_run_output=_cfg_bool(cfg, "models", "mirix", "mirix_data_dir_under_run_output", default=False),
            run_output_root=run_output_root,
            api_url=_cfg_str(cfg, "models", "mirix", "api_url", default=""),
            api_key=_cfg_str(cfg, "models", "mirix", "api_key", default=""),
            client_id=_cfg_str(cfg, "models", "mirix", "client_id", default=""),
            org_id=_cfg_str(cfg, "models", "mirix", "org_id", default=""),
            embed_pdf_max_chars=emb_c,
            context_pdf_max_chars=ctx_c,
            min_native_pdf_chars=min_n,
            task_agent_model=_cfg_str(cfg, "models", "mirix", "task_agent_model", default=""),
            task_agent_api_key=_cfg_str(cfg, "models", "mirix", "task_agent_api_key", default=""),
            task_agent_base_url=_cfg_str(cfg, "models", "mirix", "task_agent_base_url", default=""),
            task_agent_max_tool_rounds=_cfg_int(cfg, "models", "mirix", "task_agent_max_tool_rounds", default=5),
            task_agent_search_limit=_cfg_int(cfg, "models", "mirix", "task_agent_search_limit", default=15),
            ingest_dialogue_images=_cfg_bool(cfg, "models", "mirix", "ingest_dialogue_images", default=True),
            ingest_max_dialogue_images=_cfg_int(cfg, "models", "mirix", "ingest_max_dialogue_images", default=10),
            index_pdf_via_ocr=_cfg_bool(cfg, "models", "mirix", "index_pdf_via_ocr", default=True),
        )

    raise ValueError(f"Unknown multimodel baseline: {baseline}")


def score_record(
    rec: Dict[str, Any], pred: str, *, judge_enabled: bool, llm: SharedLLM
) -> Dict[str, Any]:
    q_type = rec.get("type", "")
    answers = rec.get("answer", []) or []
    label = rec.get("label") or (answers[0] if answers else "")
    rec["model_answer"] = pred

    if q_type == "fj":
        rec["em_score"] = em_text(pred, answers)
        rec["f1_score"] = 0.0
        rec["bleu1_score"] = 0.0
        rec["llm_score"] = 0.0
        rec["llm_judge_raw"] = ""
        return rec
    if q_type == "fm":
        em, _ = fm_em_image(pred, answers)
        rec["em_score"] = em
        rec["f1_score"] = 0.0
        rec["bleu1_score"] = 0.0
        rec["llm_score"] = 0.0
        rec["llm_judge_raw"] = ""
        return rec

    rec["em_score"] = em_text(pred, answers)
    rec["f1_score"] = round(f1_text(pred, label), 4)
    rec["bleu1_score"] = round(bleu1_text(pred, label), 4)
    if judge_enabled and (pred or "").strip():
        s, raw, usage = llm.judge(rec.get("question", ""), label, pred)
        rec["llm_score"] = s
        rec["llm_judge_raw"] = raw
        rec["llm_judge_usage"] = usage
    else:
        rec["llm_score"] = 0.0
        rec["llm_judge_raw"] = ""
    return rec


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.baseline)
    if args.top_k is not None:
        cfg.setdefault("eval", {})["top_k"] = args.top_k
    top_k = int(cfg.get("eval", {}).get("top_k", 5))
    judge_enabled = (
        bool(cfg.get("eval", {}).get("judge", {}).get("enabled", True))
        and not args.no_judge
    )

    datasets_root_raw = (
        args.datasets_root
        or cfg.get("eval", {}).get("datasets_root")
        or ""
    )
    if not datasets_root_raw:
        print("[error] datasets_root not configured.", file=sys.stderr)
        return 2
    datasets_root = Path(datasets_root_raw)
    if not datasets_root.is_absolute():
        datasets_root = (_REPO_ROOT / datasets_root).resolve()
    data_dir = datasets_root / args.dataset
    if not data_dir.is_dir():
        print(f"[error] dataset dir not found: {data_dir}", file=sys.stderr)
        return 2

    sessions, sessions_path = load_sessions(data_dir)
    images_dir = sessions_path.parent / "images"
    questions = load_questions(data_dir)
    if args.max_questions > 0:
        questions = questions[: args.max_questions]
    print(
        f"baseline={args.baseline}  dataset={data_dir}  "
        f"sessions={len(sessions)}  questions={len(questions)}"
    )

    llm = SharedLLM(cfg.get("llm", {}))

    out_root = (
        _THIS_DIR / cfg.get("eval", {}).get("output_root", "outputs")
        / args.dataset / args.baseline
    )
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{data_dir.name}__{args.baseline}"

    if args.baseline in EXTENDED_BASELINES:
        evalr = _build_extended(args.baseline, cfg, llm, run_id)
    else:
        # Forward env vars so upstream OpenAI / vLLM clients can find creds.
        if cfg.get("llm", {}).get("api_key"):
            os.environ.setdefault("OPENAI_API_KEY", cfg["llm"]["api_key"])
        if cfg.get("llm", {}).get("base_url"):
            os.environ.setdefault("OPENAI_BASE_URL", cfg["llm"]["base_url"])
        evalr = _build_multimodel(args.baseline, cfg, llm, out_root)

    print(f"[{args.baseline}] build_index on {len(sessions)} sessions ...")
    t0 = time.time()
    # Extended baselines take (sessions, images_dir, data_dir);
    # multimodel baselines take (sessions, images_dir).
    if args.baseline in EXTENDED_BASELINES:
        evalr.build_index(sessions, images_dir, data_dir)
    else:
        evalr.build_index(sessions, images_dir)
    build_seconds = time.time() - t0
    print(f"[{args.baseline}] build_index done in {build_seconds:.1f}s")

    records: List[Dict[str, Any]] = []
    max_img = int(cfg.get("eval", {}).get("max_images_per_chat", 10))

    for i, q in enumerate(questions, start=1):
        question = q.get("question", "")
        rec: Dict[str, Any] = {
            "question": question,
            "answer": q.get("answer", []),
            "label": q.get("label") or (q.get("answer", [""]) or [""])[0],
            "supporting_facts": q.get("supporting_facts", ""),
            "type": q.get("type", ""),
            "model": args.baseline,
        }
        try:
            ret = evalr.retrieve(question, top_k=top_k)
            # Both extended and multimodel evaluators return either
            # (context, rounds, image_paths) or just (context, rounds).
            if isinstance(ret, tuple) and len(ret) == 3:
                context, _rounds, image_paths = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                context, _rounds = ret
                image_paths = []
            else:
                context, image_paths = "", []
        except Exception as e:
            print(f"[{args.baseline}] retrieve failed (q={question[:60]!r}): {e}")
            context, image_paths = "", []

        pred = ""
        usage: Dict[str, Any] = {}
        try:
            ans = evalr.answer(question, context, image_paths, max_images=max_img)
            if isinstance(ans, tuple) and len(ans) >= 2:
                pred, usage = ans[0], ans[1]
            else:
                pred = str(ans)
            rec["answer_usage"] = usage
            rec["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0) if isinstance(usage, dict) else 0
        except Exception as e:
            print(f"[{args.baseline}] answer failed (q={question[:60]!r}): {e}")
            rec["prompt_tokens"] = 0

        rec = score_record(rec, pred, judge_enabled=judge_enabled, llm=llm)
        records.append(rec)

        if i % 10 == 0 or i == len(questions):
            print(
                f"  [{args.baseline}] [{i:3d}/{len(questions)}] type={rec['type']:<3} "
                f"em={rec.get('em_score',0):.2f} f1={rec.get('f1_score',0):.2f} "
                f"llm={rec.get('llm_score',0):.2f}  -> {(pred or '')[:60]!r}"
            )

    rubric = aggregate(records)

    def _avg(xs):
        xs = [float(x) for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    prompt_tokens_list = [r.get("prompt_tokens", 0) for r in records]

    summary = {
        "model": args.baseline,
        "num_questions": len(records),
        "em_score": rubric["total"]["em_score"],
        "f1_score": rubric["total"]["f1_score"],
        "bleu1_score": rubric["total"]["bleu1_score"],
        "llm_score": rubric["total"]["llm_score"],
        "judge_disabled": not judge_enabled,
        "llm_model": llm.model,
        "judge_model": llm.judge_model,
        "build_index_seconds": round(build_seconds, 4),
        "avg_prompt_tokens": _avg(prompt_tokens_list),
        "rubric": rubric,
    }

    (out_root / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = render_dataset_report(
        dataset=args.dataset,
        summaries={args.baseline: summary},
        model_order=[args.baseline],
        llm_name=llm.model,
    )
    (out_root.parent / f"report_{args.baseline}.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\nResults: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
