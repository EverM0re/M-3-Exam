from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_MM_DIR = Path(__file__).resolve().parent
if str(_MM_DIR) not in sys.path:
    sys.path.insert(0, str(_MM_DIR))
_MM_COMMON_DIR = '/Users/evermore_01/Documents/Code/Multimodaltalk_git/m3exam/baselines/_runtime/multimodel_common'
if _MM_COMMON_DIR not in sys.path:
    sys.path.insert(0, _MM_COMMON_DIR)


_RAGA_ROOT = Path(__file__).resolve().parent / "upstream"
if str(_RAGA_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGA_ROOT))

from m3exam.baselines._runtime.multimodel_common.base_evaluator import BaseEvaluator  # noqa: E402
from m3exam.baselines._runtime.multimodel_common.pdf_session_text import (  # noqa: E402
    PdfPolicy,
    build_session_pdf_context_blocks,
    build_session_pdf_snippets,
    dialogue_image_description,
)
from m3exam.baselines._runtime.multimodel_common.eval_metrics import (  # noqa: E402
    openai_usage_to_dict,
    prepend_eval_round_trace,
    stable_lightrag_doc_id_from_file_path,
)
from m3exam.baselines._runtime.multimodel_common.multimodal_llm import build_answer_messages  # noqa: E402

from m3exam.baselines._runtime.multimodel_common.load_upstream_bridge import load_module  # noqa: E402
from m3exam.baselines._runtime.multimodel_common.lightrag_index_reuse import (  # noqa: E402
    force_rebuild_lightrag_index_requested,
    lightrag_vdb_chunks_nonempty,
    reuse_lightrag_index_requested,
)
from m3exam.baselines._runtime.multimodel_common.fm_retrieve_helpers import (  # noqa: E402
    dialogue_image_rel_paths,
    dialogue_image_trace_basenames,
    fm_resolve_image_paths,
    merge_supporting_facts_images_first,
    rounds_from_lightrag_context,
)

_ra_mod = load_module(
    "_raganything_lightrag_bridge",
    _RAGA_ROOT / "multimodal_bridge" / "raganything_lightrag_runner.py",
)
RAGAnythingLightRAGSession = _ra_mod.RAGAnythingLightRAGSession


class RAGAnythingEvaluator(BaseEvaluator):
    upstream_repo = str(_RAGA_ROOT.resolve())
    upstream_meta = {
        "rag_factory": "raganything.RAGAnything + raganything_rag_core.create_lightrag_instance",
        "batch_insert": "raganything_rag_core.batch_insert_text_chunks",
        "bridge": "multimodal_bridge/raganything_lightrag_runner.RAGAnythingLightRAGSession",
        "insert": "raganything.utils.insert_text_content",
        "shell": "raganything._ensure_lightrag_initialized (pre-injected LightRAG)",
    }

    def __init__(
        self,
        text_model: str = "sentence-transformers/all-mpnet-base-v2",
        clip_model: str = "openai/clip-vit-large-patch14",
        use_gpu: bool = True,
        alpha: float = 0.35,
        llm_client=None,
        llm_model: str = "",
        pdf_policy: PdfPolicy = "native_only",
        embed_pdf_max_chars: int = 8000,
        context_pdf_max_chars: int = 12000,
        min_native_pdf_chars: int = 400,
        chunk_words: int = 80,
        chunk_overlap: int = 12,
        embed_backend: str = "local",
        embed_model: str = "text-embedding-3-small",
        embed_dim: int = 768,
        embed_local_model: str = "thenlper/gte-base",
        embed_api_base_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        lightrag_mode: str = "mix",
        lightrag_index_api_key: Optional[str] = None,
        lightrag_index_base_url: Optional[str] = None,
        lightrag_index_model: Optional[str] = None,
        lightrag_working_dir: Optional[Path] = None,
    ):
        super().__init__("RAG-Anything")
        self.text_model_name = text_model
        self.clip_model_name = clip_model
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.pdf_policy: PdfPolicy = pdf_policy
        self.embed_pdf_max_chars = embed_pdf_max_chars
        self.context_pdf_max_chars = context_pdf_max_chars
        self.min_native_pdf_chars = min_native_pdf_chars
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
        self.embed_backend = embed_backend
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.embed_local_model = embed_local_model
        self.embed_api_base_url = embed_api_base_url
        self.embed_api_key = embed_api_key
        self.lightrag_mode = lightrag_mode
        self.lightrag_index_api_key = lightrag_index_api_key
        self.lightrag_index_base_url = lightrag_index_base_url
        self.lightrag_index_model = lightrag_index_model
        self.lightrag_working_dir: Optional[Path] = (
            Path(lightrag_working_dir).resolve()
            if lightrag_working_dir
            else None
        )

        self._session: Optional[RAGAnythingLightRAGSession] = None
        self._workdir: Optional[Path] = None
        self._pdf_llm_block: Dict[str, str] = {}
        self._sessions_pdf_injected: set = set()
        self._corpus_meta: List[Dict[str, Any]] = []

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
    ) -> None:
        idx_key = (self.lightrag_index_api_key or "").strip()
        idx_base = (self.lightrag_index_base_url or "").strip()
        idx_model = (self.lightrag_index_model or "").strip()
        if idx_key and idx_base and idx_model:
            api_key = idx_key
            base_url = idx_base
            session_llm = idx_model
            print(
                f"  [LightRAG 建index LLM] {session_llm} @ {base_url} (answering仍useprimaryconfig llm) ",
                flush=True,
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get(
                "OPENAI_BASE_URL"
            )
            session_llm = self.llm_model or os.environ.get(
                "LLM_MODEL", "gpt-5-nano-2025-08-07"
            )
        if self.lightrag_working_dir is not None:
            workdir = self.lightrag_working_dir
            print(
                f"  [RAG-Anything→LightRAG] indexdirectory (eval.lightrag_cache_location=output → current run root/.eval_upstream) : {workdir}",
                flush=True,
            )
        else:
            workdir = (
                images_dir.parent / ".eval_upstream" / "raganything_lightrag"
            ).resolve()
        self._workdir = workdir
        self._lightrag_index_reused = False
        _reuse = reuse_lightrag_index_requested()
        _force = force_rebuild_lightrag_index_requested()

        def _cache_valid() -> bool:
            return workdir.exists() and lightrag_vdb_chunks_nonempty(workdir)

        if workdir.exists():
            if _force or not _reuse:
                shutil.rmtree(workdir, ignore_errors=True)
            elif not _cache_valid():
                shutil.rmtree(workdir, ignore_errors=True)

        self._session = RAGAnythingLightRAGSession(
            working_dir=workdir,
            llm_model=session_llm,
            embed_backend=self.embed_backend,
            embed_model=self.embed_model,
            embed_dim=self.embed_dim,
            embed_local_model=self.embed_local_model,
            embed_api_base_url=self.embed_api_base_url,
            embed_api_key=self.embed_api_key,
            api_key=api_key or None,
            base_url=base_url or None,
        )

        pdfs_dir = images_dir.parent / "pdfs"
        pdf_snip = {}
        self._pdf_llm_block = {}
        if pdfs_dir.is_dir():
            pdf_snip = build_session_pdf_snippets(
                sessions,
                pdfs_dir,
                policy=self.pdf_policy,
                embed_max_chars=self.embed_pdf_max_chars,
                min_native_chars=self.min_native_pdf_chars,
            )
            self._pdf_llm_block = build_session_pdf_context_blocks(
                sessions,
                pdfs_dir,
                policy=self.pdf_policy,
                context_max_chars=self.context_pdf_max_chars,
                min_native_chars=self.min_native_pdf_chars,
            )

        chunks: List[str] = []
        fpaths: List[str] = []
        self._corpus_meta = []

        for sess in sessions:
            sid = str(sess.get("session_id", ""))
            date = sess.get("date", "")
            snip = pdf_snip.get(sid, "")
            for dlg in sess.get("dialogues") or []:
                rnd = dlg.get("round", "")
                user = dlg.get("user", "")
                asst = dlg.get("assistant", "")
                vis = dialogue_image_description(dlg)
                blob = f"User: {user}\nAssistant: {asst}\n"
                if vis:
                    blob += f"[Visual]: {vis}\n"
                if snip:
                    blob += f"\n[PDF excerpt]\n{snip}\n"
                fp_one = f"raganything_{sid}_{rnd.replace(':', '_')}.txt"
                body = f"=== Session {sid} | Round {rnd} | {date} ===\n{blob.strip()}"
                chunks.append(
                    prepend_eval_round_trace(
                        body,
                        rnd,
                        chunk_id=stable_lightrag_doc_id_from_file_path(fp_one),
                        image_basenames=dialogue_image_trace_basenames(dlg),
                    )
                )
                fpaths.append(fp_one)
                files = dialogue_image_rel_paths(dlg)
                self._corpus_meta.append(
                    {
                        "session_id": sid,
                        "date": date,
                        "round": rnd,
                        "user": user,
                        "assistant": asst,
                        "img_files": files,
                    }
                )

        if _reuse and not _force and _cache_valid():
            self._lightrag_index_reused = True
            print(
                f"  [RAG-Anything→LightRAG] reusealready haveindex {self._workdir}, skip {len(chunks)}  entryingest"
            )
        else:
            print(
                f"  [RAG-Anything→insert_text_content] write {len(chunks)}  entry (仓库: {self.upstream_repo}) ..."
            )
            session = self._session
            assert session is not None
            session.build_sync(chunks, fpaths)

        self._maybe_lightrag_storage_health()

    def _maybe_lightrag_storage_health(self) -> None:
        try:
            from lightrag_storage_health import sync_run_storage_health_and_backfill
        except ImportError:
            return
        if self._session is None or self._workdir is None:
            return

        async def _ensure() -> Dict[str, Any]:
            rag = await self._session._ensure_rag()
            return {"raganything_lightrag": rag}

        try:
            from lightrag_storage_health import LightRAGStorageNotReadyError

            rags = self._session._run_async(_ensure())
            sync_run_storage_health_and_backfill(
                self._workdir,
                rags,
                mmkg_triple=False,
                run_async=self._session._run_async,
            )
        except LightRAGStorageNotReadyError:
            raise
        except Exception as exc:
            print(f"  [LightRAG health] 补向量skip: {exc}", flush=True)

    def retrieve(
        self,
        question: str,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        supporting_facts: str = "",
        top_k: int = 5,
        question_type: str = "",
    ) -> Tuple[str, List[str]]:
        del question_type
        if self._session is None:
            return "", []
        ctx = self._session.retrieve_context_sync(
            question, mode=self.lightrag_mode, top_k=top_k
        )
        rounds = rounds_from_lightrag_context(ctx)
        img_paths = fm_resolve_image_paths(
            question,
            ctx,
            self._corpus_meta,
            sessions,
            images_dir,
            top_k=top_k,
        )
        img_paths = merge_supporting_facts_images_first(
            supporting_facts, sessions, images_dir, img_paths
        )

        lines: List[str] = []
        last_sid = None
        self._sessions_pdf_injected = set()
        want = set(rounds)
        for meta in self._corpus_meta:
            if meta["round"] not in want:
                continue
            sid = meta["session_id"]
            if sid != last_sid:
                h = f"=== Session {sid}"
                if meta.get("date"):
                    h += f" ({meta['date']})"
                h += " ==="
                lines.append(h)
                last_sid = sid
                pdf_ctx = self._pdf_llm_block.get(sid)
                if pdf_ctx and sid not in self._sessions_pdf_injected:
                    lines.append("[Attached PDF excerpt for this session]\n" + pdf_ctx)
                    self._sessions_pdf_injected.add(sid)
            lines.append(f"\n[{meta['round']}]")
            lines.append(f"User      : {meta['user']}")
            lines.append(f"Assistant : {meta['assistant']}")
            for fn in meta.get("img_files") or []:
                rel = str(fn).strip().replace("\\", "/")
                if not rel:
                    continue
                p = str((images_dir / rel).resolve())
                if os.path.isfile(p) and p not in img_paths:
                    img_paths.append(p)

        body = "\n".join(lines) if lines else ""
        if ctx:
            body = (
                "[LightRAG aquery_data chunks — RAGAnything 包装  LightRAG + insert_text_content]\n"
                + ctx
                + ("\n\n---\n" + body if body else "")
            )
        return body, img_paths

    def answer(
        self,
        question: str,
        context: str,
        image_paths: Optional[List[str]] = None,
        *,
        vision_pdf_page_paths: Optional[List[str]] = None,
        vision_pdf_page_labels: Optional[List[str]] = None,
        pdf_answer_mode: str = "text_only",
    ) -> str:
        self.last_answer_usage = {}
        if self.llm_client is None:
            lines = [
                l.replace("Assistant : ", "").strip()
                for l in context.splitlines()
                if l.startswith("Assistant")
            ]
            return lines[0] if lines else ""
        try:
            from llm_chat_kwargs import chat_completion_kwargs

            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=build_answer_messages(
                    question,
                    context,
                    image_paths,
                    vision_pdf_page_paths=vision_pdf_page_paths,
                    vision_pdf_page_labels=vision_pdf_page_labels,
                    pdf_answer_mode=pdf_answer_mode,
                ),
                **chat_completion_kwargs(
                    self.llm_model, max_output_tokens=150, temperature=0.0
                ),
            )
            self.last_answer_usage = openai_usage_to_dict(getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            self.last_answer_usage = {}
            print(f"    [RAG-Anything] LLM failed: {exc}")
            return ""
