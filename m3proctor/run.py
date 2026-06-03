#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from m3exam.m3proctor.infra.dataset import (  # noqa: E402
    load_questions,
    load_sessions,
)
from m3exam.m3proctor.infra.llm_client import SharedLLM  # noqa: E402
from m3exam.m3proctor.evaluation.metrics import (  # noqa: E402
    aggregate,
    bleu1_text,
    em_text,
    f1_text,
    fm_em_image,
)
from m3exam.m3proctor.evaluation.report import render_dataset_report  # noqa: E402

from m3exam.m3proctor.core.pipeline import M3ProctorEvaluator  # noqa: E402
from m3exam.m3proctor.infra.logging_setup import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets-root",
        default=None,
        help="Datasets root directory; overrides eval.datasets_root in config.yaml.",
    )
    p.add_argument("--dataset", required=True, help="Dataset sub-directory name.")
    p.add_argument("--config", default=str(_THIS_DIR / "config.yaml"))
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument(
        "--no-ablation-no-cascade",
        action="store_true",
        help="Skip the no-cascade ablation pass.",
    )
    p.add_argument(
        "--export-cascade-case-study",
        action="store_true",
        help="Export cascade case-study bundle (MR + Stage1 judge below threshold + Stage2 judge above threshold).",
    )
    return p.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
        em, _img_metrics = fm_em_image(pred, answers)
        rec["em_score"] = em
        rec["f1_score"] = 0.0
        rec["bleu1_score"] = 0.0
        rec["llm_score"] = 0.0
        rec["llm_judge_raw"] = ""
        return rec

    rec["em_score"] = em_text(pred, answers)
    rec["f1_score"] = round(f1_text(pred, label), 4)
    rec["bleu1_score"] = round(bleu1_text(pred, label), 4)
    if judge_enabled:
        if not (pred or "").strip():
            rec["llm_score"] = 0.0
            rec["llm_judge_raw"] = ""
            rec["llm_judge_usage"] = {}
        else:
            s, raw, usage = llm.judge(rec.get("question", ""), label, pred)
            rec["llm_score"] = s
            rec["llm_judge_raw"] = raw
            rec["llm_judge_usage"] = usage
    else:
        rec["llm_score"] = 0.0
        rec["llm_judge_raw"] = ""
    return rec


def _slug_dir(s: str, max_len: int = 56) -> str:
    slug = re.sub(r"[^\w\-]+", "_", (s or "").strip(), flags=re.UNICODE).strip("_")
    return slug[:max_len] if slug else "q"


def _cascade_case_study_should_run(cfg: Dict[str, Any], args: argparse.Namespace) -> bool:
    om = cfg.get("m3proctor", {}) or {}
    cs = om.get("cascade_case_study_export", {}) or {}
    return bool(cs.get("enabled")) or bool(args.export_cascade_case_study)


def export_cascade_case_study_bundle(
    *,
    outputs_dataset_root: Path,
    study_relative_name: str,
    run_idx: int,
    serial: int,
    dataset_name: str,
    question: str,
    gold_label: str,
    answers: Any,
    supporting_facts: str,
    rec: Dict[str, Any],
    retrieved_context_text: str,
    ans_dbg: Dict[str, Any],
    judge_stage1_score: Optional[float],
    judge_stage1_raw: str,
    judge_stage1_usage: Dict[str, Any],
    final_pred_text: str,
) -> Dict[str, Any]:
    case_root = outputs_dataset_root / study_relative_name
    case_root.mkdir(parents=True, exist_ok=True)
    qs = _slug_dir(question.replace("\n", " "), 48)
    sub = case_root / f"case_{serial:04d}_run{run_idx:03d}_{qs}"
    try:
        sub.mkdir(parents=True, exist_ok=False)
        att = sub / "attachments"
        att.mkdir(parents=True, exist_ok=True)
    except OSError:
        sub_alt = case_root / f"case_{serial:04d}_run{run_idx:03d}"
        sub_alt.mkdir(parents=True, exist_ok=True)
        sub = sub_alt
        att = sub / "attachments"
        att.mkdir(parents=True, exist_ok=True)

    imgs = ans_dbg.get("stage2_image_paths") or []
    pdfs_rendered = ans_dbg.get("stage2_pdf_render_paths") or []
    copied_images: List[Dict[str, str]] = []
    copied_pdfs: List[Dict[str, str]] = []
    for idx, ip in enumerate(imgs):
        src = Path(str(ip))
        if not src.is_file():
            copied_images.append(
                {"source": str(ip), "dest": "", "skipped": "missing_or_not_file"}
            )
            continue
        name = src.name or f"image_{idx}"
        dst = att / name
        if dst.exists():
            stem, suf = Path(name).stem, Path(name).suffix
            dst = att / f"{stem}__r{serial}{idx}{suf}"
        shutil.copy2(src, dst)
        copied_images.append({"source": str(src), "dest": dst.name})

    for idx, pp in enumerate(pdfs_rendered):
        src = Path(str(pp))
        if not src.is_file():
            copied_pdfs.append(
                {"source": str(pp), "dest": "", "skipped": "missing_or_not_file"}
            )
            continue
        name = src.name or f"pdf_render_{idx}.png"
        dst = att / name
        if dst.exists():
            stem, suf = Path(name).stem, Path(name).suffix
            dst = att / f"{stem}__r{serial}{idx}{suf}"
        shutil.copy2(src, dst)
        copied_pdfs.append({"source": str(src), "dest": dst.name})

    (sub / "retrieved_context.txt").write_text(
        retrieved_context_text or "", encoding="utf-8"
    )
    (sub / "stage1_answer_before_cascade.txt").write_text(
        (ans_dbg.get("stage1_answer") or "").strip(), encoding="utf-8"
    )
    (sub / "stage2_final_model_answer.txt").write_text(
        (final_pred_text or "").strip(), encoding="utf-8"
    )

    meta: Dict[str, Any] = {
        "dataset": dataset_name,
        "question_run_index_1_based": run_idx,
        "type": rec.get("type"),
        "question": question,
        "gold_label": gold_label,
        "gold_answer_list": answers,
        "supporting_facts": supporting_facts,
        "cascade_answerer_debug": {
            "upgrade_reason_before_stage2": ans_dbg.get("upgrade_reason"),
            "upgrade_to_stage2_heuristic": ans_dbg.get("upgrade_to_stage2"),
            "multi_img_compose": ans_dbg.get("multi_img_compose"),
            "visual_score": ans_dbg.get("visual_score"),
            "pdf_score": ans_dbg.get("pdf_score"),
            "chart_score": ans_dbg.get("chart_score"),
            "filename_intent": ans_dbg.get("filename_intent"),
            "n_images_stage2": len(imgs),
            "n_pdf_renders_stage2": len(pdfs_rendered),
            "candidate_filenames_stage2": ans_dbg.get("candidate_filenames"),
        },
        "stage1_via_llm_judge": {
            "score": judge_stage1_score,
            "raw_response": judge_stage1_raw,
            "usage_estimate": judge_stage1_usage or {},
        },
        "stage2_final_via_llm_judge": {
            "score": rec.get("llm_score"),
            "raw_response": rec.get("llm_judge_raw"),
            "usage_estimate": rec.get("llm_judge_usage") or {},
        },
        "attachments": {"images": copied_images, "pdf_page_renders": copied_pdfs},
        "other_metrics_on_final_answer": {
            "em_score": rec.get("em_score"),
            "f1_score": rec.get("f1_score"),
            "bleu1_score": rec.get("bleu1_score"),
        },
    }
    (sub / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        rel = str(sub.relative_to(outputs_dataset_root))
    except ValueError:
        rel = str(sub)
    return {"case_relative_path": rel, "case_slug": sub.name, "meta": meta}


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.top_k is not None:
        cfg.setdefault("eval", {})["top_k"] = args.top_k
    top_k = int(cfg.get("eval", {}).get("top_k", 5))
    judge_enabled = (
        bool(cfg.get("eval", {}).get("judge", {}).get("enabled", True))
        and not args.no_judge
    )
    run_ablation = bool(
        cfg.get("m3proctor", {}).get("run_no_cascade_ablation", True)
    ) and not args.no_ablation_no_cascade

    datasets_root_raw = (
        args.datasets_root
        or cfg.get("eval", {}).get("datasets_root")
        or ""
    )
    if not datasets_root_raw:
        print(
            "[error] datasets_root not configured (set eval.datasets_root in config.yaml "
            "or pass --datasets-root)",
            file=sys.stderr,
        )
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
    print(f"dataset: {data_dir}  sessions={len(sessions)}  questions={len(questions)}")

    llm = SharedLLM(cfg.get("llm", {}))

    output_root = (
        _THIS_DIR / cfg.get("eval", {}).get("output_root", "outputs") / args.dataset
    )
    out_dir = output_root / "m3proctor"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = setup_logger(output_root, dataset_name=args.dataset)
    logger.info("config: %s", args.config)
    logger.info(
        "dataset: %s  sessions=%d  questions=%d  top_k=%d  judge=%s  llm=%s  judge_llm=%s  no_cascade_ablation=%s",
        data_dir, len(sessions), len(questions), top_k, judge_enabled,
        llm.model, llm.judge_model, run_ablation,
    )

    run_id = f"{data_dir.name}__m3proctor"
    evalr = M3ProctorEvaluator(cfg, llm, run_id)

    print(f"[m3proctor] build_index on {len(sessions)} sessions ...")
    t0 = time.time()
    evalr.build_index(sessions, images_dir, data_dir)
    build_seconds = time.time() - t0
    print(f"[m3proctor] build_index done in {build_seconds:.1f}s")

    cascade_cs_active = _cascade_case_study_should_run(cfg, args)
    cs_cfg_top = cfg.get("m3proctor", {}).get("cascade_case_study_export", {}) or {}
    cs_study_dir_name = str(
        cs_cfg_top.get("output_dir_name") or "cascade_case_study"
    )
    cs_manifest_entries: List[Dict[str, Any]] = []

    logger.info(
        "cascade_case_study_export=%s  study_subdir=%s",
        cascade_cs_active, cs_study_dir_name,
    )
    if cascade_cs_active and not judge_enabled:
        logger.warning(
            "cascade_case_study_export enabled but LLM-as-judge is disabled; "
            "no cases will be exported this run."
        )

    records: List[Dict[str, Any]] = []
    records_no_cascade: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        question = q.get("question", "")
        sf = q.get("supporting_facts", "")
        q_type = q.get("type", "")
        rec: Dict[str, Any] = {
            "question": question,
            "answer": q.get("answer", []),
            "label": q.get("label") or (q.get("answer", [""]) or [""])[0],
            "supporting_facts": sf,
            "type": q_type,
            "model": "m3proctor",
        }
        try:
            context, _rounds, image_paths = evalr.retrieve(question, top_k=top_k)
        except Exception as e:
            logger.exception(
                "[%d/%d] retrieve failed (q=%r): %s",
                i, len(questions), question[:80], e,
            )
            context, image_paths = "", []

        max_img = int(cfg.get("eval", {}).get("max_images_per_chat", 10))

        if run_ablation:
            try:
                pred_flat, usage_flat = evalr.answer(
                    question, context, image_paths,
                    max_images=max_img,
                    enable_two_stage=False,
                )
            except Exception as e:
                logger.exception(
                    "[%d/%d] answer (no-cascade) failed (q=%r): %s",
                    i, len(questions), question[:80], e,
                )
                pred_flat, usage_flat = "", {}
            rec_nc = {**rec}
            rec_nc["model"] = "m3proctor_no_cascade"
            rec_nc["prompt_tokens"] = int(usage_flat.get("prompt_tokens", 0) or 0)
            rec_nc["answer_usage"] = usage_flat
            ans_dbg_nc = getattr(evalr, "_last_answer_debug", {}) or {}
            rec_nc["cascade_stage"] = ans_dbg_nc.get("stage")
            rec_nc["upgrade_reason"] = ans_dbg_nc.get("upgrade_reason", "")
            rec_nc["cascaded"] = False
            rec_nc["n_images_attached"] = ans_dbg_nc.get("n_images_attached", 0)
            rec_nc["n_pdf_pages_attached"] = ans_dbg_nc.get("n_pdf_pages_attached", 0)
            records_no_cascade.append(
                score_record(rec_nc, pred_flat, judge_enabled=judge_enabled, llm=llm)
            )

        ans_dbg: Dict[str, Any] = {}
        pred = ""
        try:
            pred, usage = evalr.answer(
                question, context, image_paths, max_images=max_img,
            )
            rec["answer_usage"] = usage
            rec["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
            ans_dbg = getattr(evalr, "_last_answer_debug", {}) or {}
            if ans_dbg:
                rec["cascade_stage"] = ans_dbg.get("stage")
                rec["upgrade_reason"] = ans_dbg.get("upgrade_reason")
                rec["visual_score"] = ans_dbg.get("visual_score")
                rec["pdf_score"] = ans_dbg.get("pdf_score")
                rec["n_images_attached"] = ans_dbg.get("n_images_attached", 0)
                rec["n_pdf_pages_attached"] = ans_dbg.get("n_pdf_pages_attached", 0)
                rec["cascaded"] = ans_dbg.get("stage") == 2
        except Exception as e:
            logger.exception(
                "[%d/%d] answer failed (q=%r): %s",
                i, len(questions), question[:80], e,
            )
            pred = ""
            rec["prompt_tokens"] = 0
            ans_dbg = {}

        ans_list = list(rec.get("answer") or [])
        label_for_judge = (
            rec.get("label") or (ans_list[0] if ans_list else "") or ""
        )
        qh_types_cs = {
            str(x).strip().lower()
            for x in (cs_cfg_top.get("question_types") or ["mr"])
            if x is not None
        }
        need_judge_s1_export = (
            cascade_cs_active
            and judge_enabled
            and (q_type or "").strip().lower() in qh_types_cs
            and int(ans_dbg.get("stage", 0) or 0) == 2
        )
        judge_s1_llm_score: Optional[float] = None
        judge_s1_raw = ""
        judge_s1_usage: Dict[str, Any] = {}
        if need_judge_s1_export:
            stage1_plain = (ans_dbg.get("stage1_answer") or "").strip()
            if stage1_plain:
                try:
                    judge_s1_llm_score, judge_s1_raw, judge_s1_usage = llm.judge(
                        question, str(label_for_judge), stage1_plain
                    )
                except Exception as e_j1:
                    logger.exception(
                        "[%d/%d] cascade case study: Stage1 judge failed: %s",
                        i, len(questions), e_j1,
                    )
                    judge_s1_llm_score = None
            else:
                judge_s1_llm_score = 0.0

        rec = score_record(rec, pred, judge_enabled=judge_enabled, llm=llm)
        records.append(rec)

        if need_judge_s1_export and judge_s1_llm_score is not None:
            nb = float(cs_cfg_top.get("stage1_negative_if_below", 1.0))
            pg = float(cs_cfg_top.get("stage2_positive_if_at_least", 1.0))
            fin_sc = float(rec.get("llm_score", -9.99))
            if judge_s1_llm_score + 1e-9 < nb and fin_sc + 1e-9 >= pg:
                try:
                    bd = export_cascade_case_study_bundle(
                        outputs_dataset_root=output_root,
                        study_relative_name=cs_study_dir_name,
                        run_idx=i,
                        serial=len(cs_manifest_entries) + 1,
                        dataset_name=args.dataset,
                        question=question,
                        gold_label=str(label_for_judge),
                        answers=rec.get("answer"),
                        supporting_facts=sf,
                        rec=rec,
                        retrieved_context_text=context,
                        ans_dbg=ans_dbg,
                        judge_stage1_score=judge_s1_llm_score,
                        judge_stage1_raw=judge_s1_raw,
                        judge_stage1_usage=judge_s1_usage,
                        final_pred_text=pred,
                    )
                    if bd:
                        slim = {
                            "case_slug": bd.get("case_slug"),
                            "case_relative_path": bd.get("case_relative_path"),
                            "question_preview": question[:120],
                            "stage1_judge_llm_score": judge_s1_llm_score,
                            "stage2_judge_llm_score": fin_sc,
                        }
                        cs_manifest_entries.append(slim)
                        logger.info(
                            "cascade case study exported -> %s",
                            bd.get("case_relative_path"),
                        )
                except Exception as e_ex:
                    logger.exception(
                        "[%d/%d] cascade case study export failed: %s",
                        i, len(questions), e_ex,
                    )

        if i % 10 == 0 or i == len(questions):
            print(
                f"  [m3proctor] [{i:3d}/{len(questions)}] type={rec['type']:<3} "
                f"em={rec.get('em_score',0):.2f} f1={rec.get('f1_score',0):.2f} "
                f"llm={rec.get('llm_score',0):.2f}  -> {(pred or '')[:60]!r}"
            )

    if getattr(evalr, "_classifier", None):
        evalr._classifier.flush()

    if cascade_cs_active and judge_enabled:
        study_out = output_root / cs_study_dir_name
        study_out.mkdir(parents=True, exist_ok=True)
        readme_txt = (
            "Cascade case study bundle (exported when filters match).\n\n"
            "Filter (see config.yaml m3proctor.cascade_case_study_export):\n"
            "  - type in question_types (default: mr only)\n"
            "  - Stage 2 actually occurred (upgrade_to_stage2=True, stage==2)\n"
            "  - Stage 1 LLM-as-judge score < stage1_negative_if_below (default 1.0)\n"
            "  - Stage 2 (final) LLM-as-judge score >= stage2_positive_if_at_least (default 1.0)\n\n"
            "Each case_* directory:\n"
            "  - meta.json                structured metadata, cascade debug, Stage1/2 judge scores\n"
            "  - retrieved_context.txt   retrieval context sent to the answerer\n"
            "  - stage1_answer_before_cascade.txt ; stage2_final_model_answer.txt\n"
            "  - attachments/            images and rendered PDF pages used in Stage 2\n"
        )
        (study_out / "README.txt").write_text(readme_txt, encoding="utf-8")
        (study_out / "exported_manifest.json").write_text(
            json.dumps(cs_manifest_entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "cascade case study: %d bundles under %s",
            len(cs_manifest_entries), study_out,
        )

    summary = aggregate(records)

    def _avg(xs):
        xs = [float(x) for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    prompt_tokens_list = [r.get("prompt_tokens", 0) for r in records]

    cascade_by_type: Dict[str, Dict[str, Any]] = {}
    for r in records:
        t = r.get("type", "") or "?"
        blk = cascade_by_type.setdefault(t, {"count": 0, "cascaded": 0})
        blk["count"] += 1
        if r.get("cascaded"):
            blk["cascaded"] += 1
    for t, blk in cascade_by_type.items():
        blk["cascade_rate"] = (
            round(blk["cascaded"] / blk["count"], 4) if blk["count"] else 0.0
        )

    n_cascaded_total = sum(1 for r in records if r.get("cascaded"))
    cascade_summary: Dict[str, Any] = {
        "overall_cascade_rate": (
            round(n_cascaded_total / len(records), 4) if records else 0.0
        ),
        "n_cascaded": n_cascaded_total,
        "n_total": len(records),
        "per_type": cascade_by_type,
    }

    stage1_tok = [r.get("prompt_tokens", 0) for r in records if not r.get("cascaded")]
    cascaded_tok = [r.get("prompt_tokens", 0) for r in records if r.get("cascaded")]

    summary_nc: Optional[Dict[str, Any]] = None
    prompt_tokens_nc: List[int] = []
    if run_ablation and records_no_cascade:
        summary_nc = aggregate(records_no_cascade)
        prompt_tokens_nc = [
            int(r.get("prompt_tokens", 0) or 0) for r in records_no_cascade
        ]

    experiment_nc: Dict[str, Any]
    if run_ablation and summary_nc is not None:
        experiment_nc = {
            "enabled": True,
            "num_questions": len(records_no_cascade),
            "avg_prompt_tokens": _avg(prompt_tokens_nc),
            "rubric": summary_nc,
        }
    else:
        experiment_nc = {"enabled": False}

    summary_full = {
        "model": "m3proctor",
        "num_questions": len(records),
        "em_score": summary["total"]["em_score"],
        "f1_score": summary["total"]["f1_score"],
        "bleu1_score": summary["total"]["bleu1_score"],
        "llm_score": summary["total"]["llm_score"],
        "judge_disabled": not judge_enabled,
        "llm_model": llm.model,
        "judge_model": llm.judge_model,
        "build_index_seconds": round(build_seconds, 4),
        "avg_prompt_tokens": _avg(prompt_tokens_list),
        "avg_prompt_tokens_stage1_only": _avg(stage1_tok),
        "avg_prompt_tokens_cascaded": _avg(cascaded_tok),
        "cascade_summary": cascade_summary,
        "rubric": summary,
        "experiment_no_cascade": experiment_nc,
    }
    (out_dir / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if records_no_cascade:
        (out_dir / "results_ablation_no_cascade.json").write_text(
            json.dumps(records_no_cascade, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary_full, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = render_dataset_report(
        dataset=args.dataset,
        summaries={"m3proctor": summary_full},
        model_order=["m3proctor"],
        llm_name=llm.model,
    )
    (output_root / "report.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    logger.info("DONE -> %s", out_dir)
    logger.info("log file -> %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
