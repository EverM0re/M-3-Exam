from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from m3exam.baselines._runtime.extended_common.dataset_io import session_to_text
from m3exam.baselines._base.evaluator import BaseEvaluator


_ROUND_RE = re.compile(r"\[(D\d+:\d+)\]")


def _make_openai_embedding_fn(model: str, dim: int, base_url: str, api_key: str):
    from openai import AsyncOpenAI
    from nano_graphrag._utils import wrap_embedding_func_with_attrs  # type: ignore

    client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)

    @wrap_embedding_func_with_attrs(embedding_dim=dim, max_token_size=8192)
    async def _embed(texts):
        import numpy as np
        rsp = await client.embeddings.create(model=model, input=texts, encoding_format="float")
        return np.array([d.embedding for d in rsp.data])

    return _embed


def _make_local_embedding_fn(model_name: str, dim: int):
    from nano_graphrag._utils import wrap_embedding_func_with_attrs  # type: ignore
    import numpy as np

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "Local embedding requires sentence-transformers."
        ) from e

    st_model = SentenceTransformer(model_name)
    actual_dim = st_model.get_sentence_embedding_dimension() or dim

    @wrap_embedding_func_with_attrs(embedding_dim=actual_dim, max_token_size=8192)
    async def _embed(texts):
        loop = asyncio.get_event_loop()
        emb = await loop.run_in_executor(
            None, lambda: st_model.encode(texts, normalize_embeddings=True)
        )
        return np.array(emb)

    return _embed


class NanoGraphRAGEvaluator(BaseEvaluator):
    name = "nano_graphrag"
    needs_images_in_chat = False

    def __init__(self, cfg: Dict[str, Any], llm, run_id: str):
        super().__init__(cfg, llm, run_id)
        self.rag = None

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        data_dir: Path,
    ) -> None:
        llm_cfg = self.cfg.get("llm", {})
        emb_cfg = self.cfg.get("embedding", {})
        ng_cfg = self.cfg.get("models", {}).get("nano_graphrag", {})

        os.environ["OPENAI_API_KEY"] = llm_cfg.get("api_key", "") or "EMPTY"
        if llm_cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = llm_cfg["base_url"]

        working_dir = (
            Path(__file__).resolve().parent / "_storage" / self.run_id
        )
        working_dir.mkdir(parents=True, exist_ok=True)

        try:
            from nano_graphrag import GraphRAG  # type: ignore
            from nano_graphrag._llm import openai_complete_if_cache  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "nano_graphrag baseline requires `pip install nano-graphrag>=0.0.6`."
            ) from e

        model_name = llm_cfg.get("model", "gpt-4o-mini")

        async def model_func(prompt, system_prompt=None, history_messages=None, **kw):
            # Some vLLM backends don't accept response_format
            kw.pop("response_format", None)
            return await openai_complete_if_cache(
                model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                **kw,
            )

        if emb_cfg.get("backend", "local") == "local":
            embed_fn = _make_local_embedding_fn(
                emb_cfg.get("local_model", "thenlper/gte-base"),
                int(emb_cfg.get("dim", 768)),
            )
        else:
            embed_fn = _make_openai_embedding_fn(
                emb_cfg.get("openai_model", "text-embedding-3-small"),
                int(emb_cfg.get("openai_dim", 1536)),
                emb_cfg.get("openai_base_url") or "",
                llm_cfg.get("api_key", ""),
            )

        self.rag = GraphRAG(
            working_dir=str(working_dir),
            enable_local=bool(ng_cfg.get("enable_local", False)),
            enable_naive_rag=bool(ng_cfg.get("enable_naive_rag", True)),
            chunk_token_size=int(ng_cfg.get("chunk_token_size", 1200)),
            chunk_overlap_token_size=int(ng_cfg.get("chunk_overlap_token_size", 100)),
            best_model_func=model_func,
            cheap_model_func=model_func,
            embedding_func=embed_fn,
        )

        # Naive RAG doesn't need entity graphs. The default ainsert() runs
        # entity_extraction + community_report (one LLM call per chunk),
        # which is wasted work here. Monkey-patch ainsert to do only chunking
        # + write chunks_vdb / full_docs / text_chunks.
        from nano_graphrag._utils import compute_mdhash_id  # type: ignore
        from nano_graphrag._op import get_chunks  # type: ignore

        rag = self.rag

        async def _naive_only_ainsert(string_or_strings):
            await rag._insert_start()
            try:
                if isinstance(string_or_strings, str):
                    string_or_strings = [string_or_strings]
                new_docs = {
                    compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                    for c in string_or_strings
                }
                _add_doc_keys = await rag.full_docs.filter_keys(list(new_docs.keys()))
                new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
                if not new_docs:
                    return
                chunks = get_chunks(
                    new_docs=new_docs,
                    chunk_func=rag.chunk_func,
                    overlap_token_size=rag.chunk_overlap_token_size,
                    max_token_size=rag.chunk_token_size,
                    tokenizer_wrapper=rag.tokenizer_wrapper,
                )
                _add_chunk_keys = await rag.text_chunks.filter_keys(list(chunks.keys()))
                chunks = {k: v for k, v in chunks.items() if k in _add_chunk_keys}
                if not chunks:
                    return
                if rag.enable_naive_rag and rag.chunks_vdb is not None:
                    await rag.chunks_vdb.upsert(chunks)
                await rag.full_docs.upsert(new_docs)
                await rag.text_chunks.upsert(chunks)
            finally:
                await rag._insert_done()

        self.rag.ainsert = _naive_only_ainsert  # type: ignore[assignment]

        docs = [session_to_text(s) for s in sessions]
        try:
            self.rag.insert(docs)
        except Exception as e:
            print(f"[nano_graphrag] insert failed: {e}")
            raise

    def retrieve(
        self, question: str, top_k: int = 5
    ) -> Tuple[str, List[str], List[str]]:
        if not self.rag:
            return "", [], []
        from nano_graphrag import QueryParam  # type: ignore

        try:
            ctx = self.rag.query(
                question,
                QueryParam(mode="naive", only_need_context=True, top_k=max(top_k, 5)),
            )
        except Exception as e:
            print(f"[nano_graphrag] query failed: {e}")
            return "", [], []
        ctx = ctx or ""
        rounds: List[str] = []
        for r in _ROUND_RE.findall(ctx):
            if r not in rounds:
                rounds.append(r)
        return ctx, rounds, []
