from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from m3exam.m3proctor.interfaces.base_evaluator import BaseEvaluator

from m3exam.m3proctor.core.answerer import answer_question
from m3exam.m3proctor.core.indexer import (
    IndexedChunk,
    build_pdf_page_chunks,
    build_round_chunk,
    build_session_summary_chunk,
    load_caption_cache,
    load_pdf_digest_cache,
    save_caption_cache,
    save_pdf_digest_cache,
)
from m3exam.m3proctor.core.modality_classifier import ModalityClassifier
from m3exam.m3proctor.core.retriever import ModalityAwareRetriever, RetrievedItem


# Round-id literals carried through to context strings:
#   [D12:4]            (bracketed)
#   [ROUND_TAG] D12:4  (round tag line)
#   [SOURCE_ROUND] D12:4 (pdf-page chunk back-pointer)
_RE_ROUND_IN_BRACKETS = re.compile(r"\[([A-Z]\d+:\d+)\]")
_RE_ROUND_TAG_LINE = re.compile(r"\[ROUND_TAG\]\s*([A-Z]\d+:\d+)", re.I)
_RE_SOURCE_ROUND_LINE = re.compile(r"\[SOURCE_ROUND\]\s*([A-Z]\d+:\d+)", re.I)


def merge_round_ids_from_context(ctx_text: str, rounds: List[str]) -> List[str]:
    seen = set(rounds)
    out = list(rounds)
    for pat in (_RE_ROUND_IN_BRACKETS, _RE_ROUND_TAG_LINE, _RE_SOURCE_ROUND_LINE):
        for m in pat.finditer(ctx_text or ""):
            rid = m.group(1)
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out


class _SyncEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        import numpy as np
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension() or 768))
        return self.model.encode(list(texts), normalize_embeddings=False, show_progress_bar=False)


class M3ProctorEvaluator(BaseEvaluator):
    name = "m3proctor"
    needs_images_in_chat = True

    def __init__(self, cfg: Dict[str, Any], llm, run_id: str):
        super().__init__(cfg, llm, run_id)
        self.cfg_method = cfg.get("m3proctor", {}) or {}
        self.cfg_emb = cfg.get("embedding", {}) or {}
        self.cfg_eval = cfg.get("eval", {}) or {}
        self.cfg_pdf = cfg.get("pdf", {}) or {}

        self._chunks: Dict[str, IndexedChunk] = {}
        self._round_chunk_index: Dict[str, str] = {}
        self._summary_chunk_index: Dict[str, str] = {}
        self._session_to_round_ids: Dict[str, List[str]] = {}
        self._img_filename_index: Dict[str, str] = {}
        self._img_to_chunk_id: Dict[str, str] = {}
        self._retriever: Optional[ModalityAwareRetriever] = None
        self._classifier: Optional[ModalityClassifier] = None

        storage_root_cfg = (self.cfg_method.get("storage_root") or "").strip()
        if storage_root_cfg:
            self._storage_root = Path(storage_root_cfg) / self.run_id
        else:
            self._storage_root = (
                Path(__file__).resolve().parent / "_storage" / self.run_id
            )
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._caption_cache_path = self._storage_root / "img_captions.json"
        self._chart_caption_cache_path = self._storage_root / "chart_captions.json"
        self._classifier_cache_path = self._storage_root / "modality_cls.json"
        self._pdf_digest_cache_path = self._storage_root / "pdf_digests.json"
        self._pdf_render_dir: Path = self._storage_root / "pdf_renders"

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        data_dir: Path,
    ) -> None:
        if images_dir.is_dir():
            for p in images_dir.iterdir():
                if not p.is_file():
                    continue
                name = p.name
                stem = p.stem
                low_name = name.lower()
                low_stem = stem.lower()
                self._img_filename_index[low_name] = str(p)
                self._img_filename_index[low_stem] = str(p)
                self._img_filename_index[low_stem.replace("-", "_")] = str(p)

        caption_cache = load_caption_cache(self._caption_cache_path)
        chart_caption_cache = load_caption_cache(self._chart_caption_cache_path)
        pdf_digest_cache = load_pdf_digest_cache(self._pdf_digest_cache_path)
        pdf_max_chars = int(self.cfg_pdf.get("text_max_chars", 4000))
        do_summary = bool(self.cfg_method.get("session_summary", True))
        do_vlm_caption = bool(self.cfg_method.get("vlm_caption_missing", True))
        do_rich_chart = bool(self.cfg_method.get("rich_chart_caption", True))
        do_pdf_page_chunks = bool(self.cfg_method.get("pdf_page_chunks", True))
        pdf_page_text_max = int(self.cfg_method.get("pdf_page_text_max_chars", 1800))
        pdf_seen: set[str] = set()

        for sess in sessions:
            for dlg in sess.get("dialogues", []):
                ch = build_round_chunk(
                    sess,
                    dlg,
                    images_dir,
                    data_dir,
                    self.llm,
                    caption_missing_img=do_vlm_caption,
                    pdf_max_chars=pdf_max_chars,
                    img_caption_cache=caption_cache,
                    rich_chart_caption=do_rich_chart,
                    chart_caption_cache=chart_caption_cache,
                )
                if ch.chunk_id in self._chunks:
                    ch.chunk_id = f"{ch.chunk_id}_dup"
                self._chunks[ch.chunk_id] = ch
                self._round_chunk_index[ch.round_id] = ch.chunk_id
                self._session_to_round_ids.setdefault(ch.session_id, []).append(ch.round_id)
                for img_fn in ch.img_filenames:
                    self._img_to_chunk_id.setdefault(img_fn.lower(), ch.chunk_id)
                    self._img_to_chunk_id.setdefault(Path(img_fn).stem.lower(), ch.chunk_id)
                    self._img_to_chunk_id.setdefault(
                        Path(img_fn).stem.lower().replace("-", "_"), ch.chunk_id
                    )

                if do_pdf_page_chunks:
                    for pdf_path in ch.pdf_paths:
                        pdf_filename = Path(pdf_path).name
                        key = f"{pdf_path}::{ch.round_id}"
                        if key in pdf_seen:
                            continue
                        pdf_seen.add(key)
                        try:
                            page_chunks = build_pdf_page_chunks(
                                pdf_path,
                                pdf_filename=pdf_filename,
                                session_id=ch.session_id,
                                round_id=ch.round_id,
                                llm=self.llm,
                                digest_cache=pdf_digest_cache,
                                page_text_max_chars=pdf_page_text_max,
                            )
                        except Exception as e:
                            print(f"[indexer] pdf-page-chunks failed for {pdf_filename}: {e}")
                            page_chunks = []
                        for pc in page_chunks:
                            if pc.chunk_id in self._chunks:
                                pc.chunk_id = f"{pc.chunk_id}_dup"
                            self._chunks[pc.chunk_id] = pc

            if do_summary:
                sch = build_session_summary_chunk(sess, self.llm)
                if sch.chunk_id in self._chunks:
                    sch.chunk_id = f"{sch.chunk_id}_dup"
                self._chunks[sch.chunk_id] = sch
                self._summary_chunk_index[sch.session_id] = sch.chunk_id

        save_caption_cache(self._caption_cache_path, caption_cache)
        save_caption_cache(self._chart_caption_cache_path, chart_caption_cache)
        save_pdf_digest_cache(self._pdf_digest_cache_path, pdf_digest_cache)

        emb_model = self.cfg_emb.get("local_model", "thenlper/gte-base")
        embedder = _SyncEmbedder(emb_model)
        self._retriever = ModalityAwareRetriever(
            chunks=self._chunks,
            embedder=embedder,
            alpha_image=float(self.cfg_method.get("alpha_image", 0.20)),
            alpha_pdf=float(self.cfg_method.get("alpha_pdf", 0.25)),
            alpha_chart=float(self.cfg_method.get("alpha_chart", 0.15)),
            over_fetch=int(self.cfg_method.get("over_fetch", 4)),
            summary_boost=float(self.cfg_method.get("summary_boost", 0.10)),
            guarantee_summary=bool(self.cfg_method.get("guarantee_summary", True)),
            guarantee_min_sessions=int(self.cfg_method.get("guarantee_min_sessions", 2)),
        )
        asyncio.get_event_loop().run_until_complete(self._retriever.build_async())

        self._classifier = ModalityClassifier(
            llm=self.llm,
            cache_path=self._classifier_cache_path,
            dataset_name=data_dir.name,
        )

    def retrieve(
        self, question: str, top_k: int = 5
    ) -> Tuple[str, List[str], List[str]]:
        if not self._retriever or not self._classifier:
            return "", [], []
        flags = self._classifier.classify(question)
        items = asyncio.get_event_loop().run_until_complete(
            self._retriever.retrieve_async(question, flags, top_k=top_k)
        )

        # Force-inject round chunks for any `img_<id>` literal mentioned in the
        # question, so cross-session questions whose retrieval missed the right
        # round still see the right image's surrounding text.
        seen_chunk_ids = {it.chunk.chunk_id for it in items}
        for m in re.finditer(r"\b(img[_-]?\d+)\b", question, flags=re.IGNORECASE):
            key = m.group(1).lower().replace("-", "_")
            cid = self._img_to_chunk_id.get(key)
            if cid and cid not in seen_chunk_ids and cid in self._chunks:
                items.append(RetrievedItem(
                    chunk=self._chunks[cid],
                    base_score=0.0, boost=0.0, final_score=0.0,
                ))
                seen_chunk_ids.add(cid)

        self._last_flags = flags
        self._last_items = items

        summary_parts: List[str] = []
        round_parts: List[str] = []
        pdf_page_parts: List[str] = []
        for it in items:
            if it.chunk.kind == "summary":
                summary_parts.append(
                    f"[GLOBAL VIEW of session {it.chunk.session_id}]\n{it.chunk.content}"
                )
            elif it.chunk.kind == "pdf_page":
                pdf_page_parts.append(it.chunk.content)
            else:
                round_parts.append(it.chunk.content)

        ctx_text = ""
        if summary_parts:
            ctx_text += "=== SESSION CONTEXT ===\n" + "\n\n".join(summary_parts) + "\n\n"
        if round_parts:
            ctx_text += "=== ROUND CHUNKS ===\n" + "\n\n".join(round_parts) + "\n\n"
        if pdf_page_parts:
            ctx_text += "=== PDF PAGE EVIDENCE ===\n" + "\n\n".join(pdf_page_parts)

        rounds: List[str] = []
        for it in items:
            if it.chunk.kind == "round" and it.chunk.round_id:
                if it.chunk.round_id not in rounds:
                    rounds.append(it.chunk.round_id)
            elif it.chunk.kind == "pdf_page" and it.chunk.round_id:
                if it.chunk.round_id not in rounds:
                    rounds.append(it.chunk.round_id)

        if bool(self.cfg_method.get("relax_round_recall_from_context", False)):
            rounds = merge_round_ids_from_context(ctx_text, rounds)

        image_paths: List[str] = []
        seen: set = set()
        forced_imgs: List[str] = []
        for m in re.finditer(r"\b(img[_-]?\d+)\b", question, flags=re.IGNORECASE):
            key = m.group(1).lower().replace("-", "_")
            ip = self._img_filename_index.get(key)
            if ip and ip not in seen:
                seen.add(ip)
                forced_imgs.append(ip)
                image_paths.append(ip)
        for it in items:
            for ip in it.chunk.image_paths:
                if ip not in seen:
                    seen.add(ip)
                    image_paths.append(ip)

        self._last_forced_images = forced_imgs

        return ctx_text, rounds, image_paths

    def answer(
        self,
        question: str,
        context: str,
        image_paths: Optional[List[str]] = None,
        *,
        max_images: int = 10,
        enable_two_stage: Optional[bool] = None,
    ) -> Tuple[str, Dict[str, int]]:
        items = getattr(self, "_last_items", []) or []
        flags = getattr(self, "_last_flags", {}) or {}
        cfg_eval = self.cfg_eval
        pdf_total_cap = int(self.cfg_method.get("pdf_total_cap", 6))
        pages_per_pdf = int(self.cfg_method.get("pdf_vision_max_pages", 3))
        dpi = int(self.cfg_pdf.get("render_dpi", 144))
        v_th = float(self.cfg_method.get("visual_score_threshold", 0.40))
        p_th = float(self.cfg_method.get("pdf_score_threshold", 0.40))
        two_stage = bool(self.cfg_method.get("enable_two_stage", True))
        if enable_two_stage is not None:
            two_stage = bool(enable_two_stage)
        c_chart = float(self.cfg_method.get("cascade_chart_numeric_min", 0.55))
        c_hallu = float(self.cfg_method.get("cascade_hallucination_modal_min", 0.72))

        forced_imgs = getattr(self, "_last_forced_images", []) or []
        try:
            text, usage, _dbg = answer_question(
                question,
                "",
                items,
                flags,
                self.llm,
                max_images_per_chat=int(cfg_eval.get("max_images_per_chat", max_images)),
                pdf_vision_max_pages=pages_per_pdf,
                pdf_render_dpi=dpi,
                pdf_cache_dir=self._pdf_render_dir,
                pdf_total_cap=pdf_total_cap,
                visual_score_threshold=v_th,
                pdf_score_threshold=p_th,
                enable_two_stage=two_stage,
                cascade_chart_numeric_min=c_chart,
                cascade_hallucination_modal_min=c_hallu,
                forced_image_paths=forced_imgs,
            )
            self._last_answer_debug = _dbg
        except Exception as e:
            print(f"[m3proctor] answer failed: {e}")
            return "", {}
        return text, usage
