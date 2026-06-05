from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_MM_DIR = Path(__file__).resolve().parent
if str(_MM_DIR) not in sys.path:
    sys.path.insert(0, str(_MM_DIR))
_MM_COMMON_DIR = '/Users/evermore_01/Documents/Code/Multimodaltalk_git/m3exam/baselines/_runtime/multimodel_common'
if _MM_COMMON_DIR not in sys.path:
    sys.path.insert(0, _MM_COMMON_DIR)


_MEMVERSE_ROOT = Path(__file__).resolve().parent / "upstream"

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
    lightrag_mmkg_all_vdb_nonempty,
    lightrag_vdb_chunks_nonempty,
    reuse_lightrag_index_requested,
    wipe_mmkg_if_triplet_inconsistent,
)
from m3exam.baselines._runtime.multimodel_common.fm_retrieve_helpers import (  # noqa: E402
    dialogue_image_rel_paths,
    dialogue_image_trace_basenames,
    fm_resolve_image_paths,
    merge_supporting_facts_images_first,
    resolve_dialogue_image_paths,
    rounds_from_lightrag_context,
)

_mv_mod = load_module(
    "_memverse_lightrag_bridge",
    _MEMVERSE_ROOT / "multimodal_bridge" / "lightrag_runner.py",
)
MemVerseLightRAGSession = _mv_mod.MemVerseLightRAGSession

_mmkg_mod = load_module(
    "_memverse_mmkg_bridge",
    _MEMVERSE_ROOT / "multimodal_bridge" / "memverse_mmkg_runner.py",
)
MemVerseOfficialMMKGSession = _mmkg_mod.MemVerseOfficialMMKGSession


class MemVerseEvaluator(BaseEvaluator):
    """MemVerse-main: 官方 MMKG threeimageor单库 LightRAG (由 use_official_mmkg 切换) . """

    upstream_repo = str(_MEMVERSE_ROOT.resolve())
    upstream_meta = {
        "official_mmkg": "MemoryKB/build_memory agent x3 + MMKG/{core,episodic,semantic} LightRAG",
        "rag_factory": "MemVerse-main/memverse_rag_core.create_lightrag_instance",
        "batch_insert_legacy": "memverse_rag_core.batch_ainsert_text_chunks",
        "mmkg_bridge": "multimodal_bridge/memverse_mmkg_runner.MemVerseOfficialMMKGSession",
        "lightrag_bridge": "multimodal_bridge/lightrag_runner.MemVerseLightRAGSession",
        "orchestrator_ref": "MemVerse-main/orchestrator.py (threeimage + insert_chunks_from_json)",
    }

    def __init__(
        self,
        text_model: str = "thenlper/gte-base",
        clip_model: str = "openai/clip-vit-base-patch32",
        use_gpu: bool = True,
        alpha: float = 0.4,
        llm_client=None,
        llm_model: str = "",
        memory_agent_model: Optional[str] = None,
        pdf_policy: PdfPolicy = "native_then_ocr",
        embed_pdf_max_chars: int = 8000,
        context_pdf_max_chars: int = 12000,
        min_native_pdf_chars: int = 400,
        recency_gamma: float = 0.22,
        embed_backend: str = "local",
        embed_model: str = "text-embedding-3-small",
        embed_dim: int = 768,
        embed_local_model: str = "thenlper/gte-base",
        embed_api_base_url: Optional[str] = None,
        embed_api_key: Optional[str] = None,
        lightrag_mode: str = "hybrid",
        lightrag_index_api_key: Optional[str] = None,
        lightrag_index_base_url: Optional[str] = None,
        lightrag_index_model: Optional[str] = None,
        use_official_mmkg: bool = True,
        lightrag_working_dir: Optional[Path] = None,
    ):
        super().__init__("MemVerse")
        self.text_model_name = text_model
        self.clip_model_name = clip_model
        self.use_gpu = use_gpu
        self.alpha = float(alpha)
        self.llm_client = llm_client
        self.llm_model = llm_model
        mam = (memory_agent_model or "").strip()
        self.memory_agent_model: Optional[str] = mam if mam else None
        self.pdf_policy: PdfPolicy = pdf_policy
        self.embed_pdf_max_chars = embed_pdf_max_chars
        self.context_pdf_max_chars = context_pdf_max_chars
        self.min_native_pdf_chars = min_native_pdf_chars
        self.recency_gamma = float(recency_gamma)
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
        self.use_official_mmkg = bool(use_official_mmkg)
        self.lightrag_working_dir: Optional[Path] = (
            Path(lightrag_working_dir).resolve()
            if lightrag_working_dir
            else None
        )

        self._session: Optional[
            Union[MemVerseLightRAGSession, MemVerseOfficialMMKGSession]
        ] = None
        self._workdir: Optional[Path] = None
        self._pdf_llm_block: Dict[str, str] = {}
        self._sessions_pdf_injected: set = set()
        self._sessions: List[Dict[str, Any]] = []
        self._images_dir: Optional[Path] = None
        self._corpus_meta: List[Dict[str, Any]] = []

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
    ) -> None:
        self._sessions = sessions
        self._images_dir = images_dir
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
        _agent_model = (
            self.memory_agent_model
            or self.llm_model
            or session_llm
            or os.environ.get("LLM_MODEL", "")
        ).strip() or session_llm
        if self.memory_agent_model and self.memory_agent_model != (self.llm_model or "").strip():
            print(
                f"  [MemVerse MMKG] memory agents model={self.memory_agent_model!r} "
                f" (answering / judge 仍use llm.model={self.llm_model!r}) ",
                flush=True,
            )
        if self.lightrag_working_dir is not None:
            workdir = self.lightrag_working_dir
            print(
                f"  [MemVerse→LightRAG] indexdirectory (eval.lightrag_cache_location=output → current run root/.eval_upstream) : {workdir}",
                flush=True,
            )
        else:
            workdir = (images_dir.parent / ".eval_upstream" / "memverse_lightrag").resolve()
        self._workdir = workdir
        self._lightrag_index_reused = False
        _reuse = reuse_lightrag_index_requested()
        _force = force_rebuild_lightrag_index_requested()

        try:
            from eval_stage_checkpoint import (
                load_mmkg_build_progress,
                sanitize_stale_mmkg_lightrag_checkpoint,
                stage_checkpoint_enabled,
            )
        except ImportError:
            load_mmkg_build_progress = None  # type: ignore[misc, assignment]
            sanitize_stale_mmkg_lightrag_checkpoint = None  # type: ignore[misc, assignment]
            stage_checkpoint_enabled = lambda: False  # type: ignore[misc, assignment]

        def _mmkg_partial_in_progress() -> bool:
            if not self.use_official_mmkg or not stage_checkpoint_enabled():
                return False
            if load_mmkg_build_progress is None:
                return False
            prog = load_mmkg_build_progress(workdir)
            return bool(prog) and not prog.get("index_complete")

        if self.use_official_mmkg and workdir.exists():
            if sanitize_stale_mmkg_lightrag_checkpoint is not None:
                sanitize_stale_mmkg_lightrag_checkpoint(workdir)
            if _mmkg_partial_in_progress():
                print(
                    "  [checkpoint] 检测to未完as MMKG 建index (mmkg_build_progress.json) , "
                    "keep memory_chunks/ andalready have MMKG, will续建",
                    flush=True,
                )
            elif wipe_mmkg_if_triplet_inconsistent(workdir):
                print(
                    "  [MemVerse MMKG] three库构image进度not一致, alreadydelete MMKG/ (keep memory_chunks) , "
                    "willre-执linestage 3",
                    flush=True,
                )

        def _cache_valid() -> bool:
            if not workdir.exists():
                return False
            if self.use_official_mmkg:
                return lightrag_mmkg_all_vdb_nonempty(workdir)
            return lightrag_vdb_chunks_nonempty(workdir)

        if workdir.exists():
            if _force or not _reuse:
                shutil.rmtree(workdir, ignore_errors=True)
            elif _mmkg_partial_in_progress():
                pass
            elif not _cache_valid():
                shutil.rmtree(workdir, ignore_errors=True)

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
        global_i = 0
        total = sum(len(s.get("dialogues") or []) for s in sessions)

        for sess in sessions:
            sid = str(sess.get("session_id", ""))
            date = sess.get("date", "")
            snip = pdf_snip.get(sid, "")
            for dlg in sess.get("dialogues") or []:
                rnd = dlg.get("round", "")
                user = dlg.get("user", "")
                asst = dlg.get("assistant", "")
                vis = dialogue_image_description(dlg)
                pdf_part = f"\n[PDF excerpt]\n{snip}" if snip else ""
                fp_one = f"{sid}_{rnd.replace(':', '_')}.txt"
                # to齐 MemoryKB/build_memory.py  多field拼connect风case; first部cantrackback标记and doc-text-* id 一致
                body = (
                    f"=== Session {sid} | Round {rnd} | {date} ===\n"
                    f"Query: {user}\n"
                    f"Assistant: {asst}\n"
                )
                mem = prepend_eval_round_trace(
                    body,
                    rnd,
                    chunk_id=stable_lightrag_doc_id_from_file_path(fp_one),
                    image_basenames=dialogue_image_trace_basenames(dlg),
                )
                if vis:
                    mem += f"Image: {vis}\n"
                mem += pdf_part
                chunks.append(mem.strip())
                fpaths.append(fp_one)
                w = 1.0 + self.recency_gamma * (global_i / max(1, total - 1))
                self._corpus_meta.append(
                    {
                        "session_id": sid,
                        "date": date,
                        "round": rnd,
                        "user": user,
                        "assistant": asst,
                        "img_files": dialogue_image_rel_paths(dlg),
                        "recency_w": float(w),
                        "dialogue_vis": vis,
                    }
                )
                global_i += 1

        if self.use_official_mmkg:
            if self.llm_client is None:
                raise RuntimeError(
                    "官方 MemVerse MMKG 需要config llm.api_key/base_url (llm_client) , "
                    "use于 core/episodic/semantic three套 memory agent. "
                )
            # 断point续建must优先于「vdb non-空i.e.reuse」: otherwise semantic etc.子库 ainsert 未完aswhen, 
            # will因three库already have部分向量而误走「skipingest」, only靠 health 补向量unable to补all未构image  chunk. 
            if _reuse and not _force and _mmkg_partial_in_progress():
                print(
                    f"  [checkpoint] MemVerse MMKG 续建 ({len(self._corpus_meta)}  entrymemory) …",
                    flush=True,
                )
                self._session = MemVerseOfficialMMKGSession(
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
                    agent_client=self.llm_client,
                    agent_model=_agent_model,
                )
                entries_resume: List[Dict[str, Any]] = []
                for meta in self._corpus_meta:
                    dlg_query = meta["user"]
                    sid = meta["session_id"]
                    snip = pdf_snip.get(sid, "")
                    if snip:
                        dlg_query = f"{dlg_query}\n\n[PDF excerpt]\n{snip}"
                    imagecaption = (meta.get("dialogue_vis") or "").strip()
                    image_paths = resolve_dialogue_image_paths(
                        {"img_files": meta.get("img_files") or []},
                        images_dir,
                        max_images=10,
                    )
                    entries_resume.append(
                        {
                            "id": str(meta["round"]),
                            "query": dlg_query,
                            "videocaption": "",
                            "audiocaption": "",
                            "imagecaption": imagecaption,
                            "image_paths": image_paths,
                        }
                    )
                mmkg_sess = self._session
                assert mmkg_sess is not None
                mmkg_sess.build_sync(entries_resume)
            elif _reuse and not _force and _cache_valid():
                self._lightrag_index_reused = True
                print(
                    f"  [MemVerse MMKG] reusethreeimageindex {workdir}/MMKG, skip agent+ingest"
                )
                self._session = MemVerseOfficialMMKGSession(
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
                    agent_client=self.llm_client,
                    agent_model=_agent_model,
                )
            else:
                print(
                    f"  [MemVerse MMKG] 官方threeimage + agent ({len(self._corpus_meta)}  entrymemory) …"
                )
                entries: List[Dict[str, Any]] = []
                for meta in self._corpus_meta:
                    dlg_query = meta["user"]
                    sid = meta["session_id"]
                    snip = pdf_snip.get(sid, "")
                    if snip:
                        dlg_query = f"{dlg_query}\n\n[PDF excerpt]\n{snip}"
                    # and orchestrator   imagecaption 语义一致 (描述text, non-file名list) 
                    imagecaption = (meta.get("dialogue_vis") or "").strip()
                    image_paths = resolve_dialogue_image_paths(
                        {"img_files": meta.get("img_files") or []},
                        images_dir,
                        max_images=10,
                    )
                    entries.append(
                        {
                            "id": str(meta["round"]),
                            "query": dlg_query,
                            "videocaption": "",
                            "audiocaption": "",
                            "imagecaption": imagecaption,
                            "image_paths": image_paths,
                        }
                    )
                self._session = MemVerseOfficialMMKGSession(
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
                    agent_client=self.llm_client,
                    agent_model=_agent_model,
                )
                mmkg_sess = self._session
                assert mmkg_sess is not None
                mmkg_sess.build_sync(entries)
        else:
            self._session = MemVerseLightRAGSession(
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
            if _reuse and not _force and _cache_valid():
                self._lightrag_index_reused = True
                print(
                    f"  [MemVerse→LightRAG] reusealready haveindex {workdir}, skip {len(chunks)}  entryingest"
                )
            else:
                print(
                    f"  [MemVerse→LightRAG] 单库write {len(chunks)}  entry (仓库: {self.upstream_repo}) ..."
                )
                lr_sess = self._session
                assert lr_sess is not None
                lr_sess.build_sync(chunks, fpaths)

        self._maybe_lightrag_storage_health()

    def _maybe_lightrag_storage_health(self) -> None:
        try:
            from lightrag_storage_health import (
                run_storage_health_and_backfill,
                storage_backfill_all_from_env,
                storage_backfill_batch_size,
                storage_backfill_max_chunks,
                storage_health_enabled_from_env,
                storage_health_rebuild_threshold,
            )
        except ImportError:
            return
        sess = self._session
        if sess is None:
            return
        wd = self._workdir
        if wd is None:
            return
        if not storage_health_enabled_from_env():
            return

        async def _health_run() -> int:
            if self.use_official_mmkg:
                await sess._ensure_rags()
                rags = {
                    "MMKG/core": sess._rag_core,
                    "MMKG/episodic": sess._rag_epi,
                    "MMKG/semantic": sess._rag_sem,
                }
            else:
                lr_sess = sess
                rag = getattr(lr_sess, "_rag", None)
                if rag is None and hasattr(lr_sess, "_ensure_rag"):
                    rag = await lr_sess._ensure_rag()
                rags = {"lightrag": rag} if rag is not None else {}
            return await run_storage_health_and_backfill(
                wd,
                {k: v for k, v in rags.items() if v is not None},
                mmkg_triple=self.use_official_mmkg,
                threshold_rebuild=storage_health_rebuild_threshold(),
                backfill_all=storage_backfill_all_from_env(),
                batch_size=storage_backfill_batch_size(),
                max_per_store=storage_backfill_max_chunks(),
            )

        try:
            from lightrag_storage_health import LightRAGStorageNotReadyError

            done = sess._run_async(_health_run())
            if done:
                print(f"  [LightRAG health] 本轮共补向量 {done}  entry", flush=True)
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
        del question_type  # only mirix use于 FM/MR imagestrategy
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
        for meta in self._corpus_meta:
            if meta["round"] not in rounds:
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

        body = "\n".join(lines) if lines else ""
        if ctx:
            body = (
                "[LightRAG retrieved chunks — MemVerse-main / lightrag]\n"
                + ctx
                + ("\n\n---\n[Structured dialogue rounds matching chunk references]\n" + body if body else "")
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
            print(
                f"    [MemVerse] chat.completions failed model={self.llm_model!r}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    body = (resp.text or "")[:2000]
                    print(
                        f"    [MemVerse] HTTP {getattr(resp, 'status_code', '?')} body:\n{body}",
                        flush=True,
                    )
                except Exception as read_exc:
                    print(f"    [MemVerse] unable toread error body: {read_exc}", flush=True)
            traceback.print_exc()
            return ""
