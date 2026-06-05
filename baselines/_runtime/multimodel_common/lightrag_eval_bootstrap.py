# 在firsttimes import lightrag 之前call ensure_lightrag_eval_env(); 
# 在 import LightRAG (from而load operate) 之后依timescall: 
#   patch_lightrag_pick_by_vector_similarity, patch_lightrag_keyword_fallback. 

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

_PATCH_DONE = False
_KW_FALLBACK_PATCHED = False
_LIBSTDCXX_PRELOADED = False


def preload_conda_libstdcxx() -> None:
    global _LIBSTDCXX_PRELOADED
    if _LIBSTDCXX_PRELOADED:
        return
    _LIBSTDCXX_PRELOADED = True
    for candidate in (
        Path(sys.prefix) / "lib" / "libstdc++.so.6",
        Path(sys.prefix) / "lib" / "libstdc++.so",
    ):
        if candidate.exists():
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break


def ensure_lightrag_eval_env() -> None:
    preload_conda_libstdcxx()
    os.environ.setdefault("RERANK_BY_DEFAULT", "false")


def patch_lightrag_pick_by_vector_similarity() -> None:
    global _PATCH_DONE
    if _PATCH_DONE:
        return

    import lightrag.operate as op
    import lightrag.utils as u

    if getattr(u, "_multimodel_pick_vs_patched", False):
        _PATCH_DONE = True
        return

    cosine_similarity = u.cosine_similarity
    logger = u.logger

    async def pick_by_vector_similarity(
        query: str,
        text_chunks_storage: Any,
        chunks_vdb: Any,
        num_of_chunks: int,
        entity_info: list[dict[str, Any]],
        embedding_func: Callable,
        query_embedding=None,
    ) -> list[str]:
        if not entity_info:
            return []

        seen: set[str] = set()
        all_chunk_ids: list[str] = []
        for entity in entity_info:
            for cid in entity.get("sorted_chunks", []):
                sid = str(cid)
                if sid not in seen:
                    seen.add(sid)
                    all_chunk_ids.append(sid)

        if not all_chunk_ids:
            logger.warning(
                "Vector similarity chunk selection:  no chunk IDs found in entity_info"
            )
            return []

        if num_of_chunks <= 0:
            num_of_chunks = max(1, min(len(all_chunk_ids), 32))

        try:
            if query_embedding is None:
                query_embedding = await embedding_func([query], context="query")
                query_embedding = query_embedding[0]

            chunk_vectors = await chunks_vdb.get_vectors_by_ids(all_chunk_ids)
            chunk_vectors = {str(k): v for k, v in chunk_vectors.items()}

            present_ids = [cid for cid in all_chunk_ids if cid in chunk_vectors]
            if not present_ids:
                logger.debug(
                    "Vector similarity chunk selection: no vectors in chunks_vdb for %d candidate chunk IDs",
                    len(all_chunk_ids),
                )
                return []

            if len(present_ids) < len(all_chunk_ids):
                logger.debug(
                    "Vector similarity chunk selection: using %d/%d chunk IDs with stored vectors",
                    len(present_ids),
                    len(all_chunk_ids),
                )

            similarities: list[tuple[str, float]] = []
            for chunk_id in present_ids:
                chunk_embedding = chunk_vectors[chunk_id]
                try:
                    sim = cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append((chunk_id, sim))
                except Exception as e:
                    logger.warning(
                        f"Vector similarity chunk selection: failed to calculate similarity for chunk {chunk_id}: {e}"
                    )

            similarities.sort(key=lambda x: x[1], reverse=True)
            selected = [cid for cid, _ in similarities[:num_of_chunks]]

            logger.debug(
                "Vector similarity chunk selection: %d chunks from %d candidates with vectors",
                len(selected),
                len(present_ids),
            )
            return selected

        except Exception as e:
            logger.error(f"[VECTOR_SIMILARITY] Error in vector similarity sorting: {e}")
            import traceback

            logger.error(
                f"[VECTOR_SIMILARITY] Traceback: {traceback.format_exc()}"
            )
            logger.debug("[VECTOR_SIMILARITY] Falling back to simple truncation")
            return all_chunk_ids[:num_of_chunks]

    setattr(u, "_multimodel_pick_vs_patched", True)
    u.pick_by_vector_similarity = pick_by_vector_similarity
    op.pick_by_vector_similarity = pick_by_vector_similarity
    _PATCH_DONE = True


def patch_lightrag_keyword_fallback() -> None:
    global _KW_FALLBACK_PATCHED
    if _KW_FALLBACK_PATCHED:
        return

    import lightrag.operate as op

    if getattr(op, "_multimodel_kw_fallback_patched", False):
        _KW_FALLBACK_PATCHED = True
        return

    _orig = op.get_keywords_from_query

    async def get_keywords_from_query(
        query: str,
        query_param,
        global_config: dict,
        hashing_kv=None,
    ):
        hl_keywords, ll_keywords = await _orig(
            query, query_param, global_config, hashing_kv
        )
        mode = query_param.mode
        q = (query or "").strip()
        if not q:
            return hl_keywords, ll_keywords
        if not ll_keywords and mode in ("local", "hybrid", "mix"):
            ll_keywords = [q]
        if not hl_keywords and mode in ("global", "hybrid", "mix"):
            hl_keywords = [q]
        return hl_keywords, ll_keywords

    op.get_keywords_from_query = get_keywords_from_query
    setattr(op, "_multimodel_kw_fallback_patched", True)
    _KW_FALLBACK_PATCHED = True


# LightRAG 在 kg.shared_storage inprocess级cache asyncio.Lock; lock绑定「firsttimesusewhen 」event循环. 
# if MemVerse / RAG-Anything / MMKG eachfrom asyncio.new_event_loop(), 同process连run多modelwilltrigger
# «Lock ... is bound to a different event loop». evaluation脚本统一reuse下column循环. 
_LIGHT_EVAL_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LIGHT_EVAL_LOOP_LOCK = threading.Lock()


def get_lightrag_eval_event_loop() -> asyncio.AbstractEventLoop:
    global _LIGHT_EVAL_LOOP
    if _LIGHT_EVAL_LOOP is None or _LIGHT_EVAL_LOOP.is_closed():
        _LIGHT_EVAL_LOOP = asyncio.new_event_loop()
    return _LIGHT_EVAL_LOOP


def run_on_lightrag_eval_loop(coro: Any) -> Any:
    loop = get_lightrag_eval_event_loop()
    with _LIGHT_EVAL_LOOP_LOCK:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
