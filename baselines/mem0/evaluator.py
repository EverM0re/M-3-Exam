from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from m3exam.baselines._runtime.extended_common.dataset_io import (
    collect_dialogue_images,
    collect_dialogue_pdfs,
    dialogue_to_text,
    iter_dialogues,
)
from m3exam.baselines._runtime.extended_common.pdf_render import render_pdf_pages
from m3exam.baselines._base.evaluator import BaseEvaluator


_ROUND_TAG_RE = re.compile(r"\[ROUND_TAG:([A-Z]\d+:\d+)\]")


def _image_url_obj(path: str) -> Optional[Dict[str, Any]]:
    import base64
    import mimetypes
    from pathlib import Path as _P

    p = _P(path)
    if not p.is_file():
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    try:
        data = base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        return None
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


class Mem0Evaluator(BaseEvaluator):
    needs_images_in_chat = False

    def __init__(
        self,
        cfg: Dict[str, Any],
        llm,
        run_id: str,
        *,
        variant: str = "mem0_text",
    ):
        super().__init__(cfg, llm, run_id)
        self.variant = variant
        self.name = variant
        self.mem = None
        self._user_id = f"u_{run_id}"
        self._round2imgs: Dict[str, List[str]] = {}
        self.needs_images_in_chat = variant == "mem0_visual"

    def _mem0_config(self) -> Dict[str, Any]:
        llm_cfg = self.cfg.get("llm", {})
        emb_cfg = self.cfg.get("embedding", {})
        var_cfg = self.cfg.get("models", {}).get(self.variant, {})

        coll = f"{var_cfg.get('collection_prefix', self.variant)}__{self.run_id}"
        storage_root = (
            Path(__file__).resolve().parent / "_storage" / self.variant / self.run_id
        )
        storage_root.mkdir(parents=True, exist_ok=True)

        llm_inner: Dict[str, Any] = {
            "provider": "openai",
            "config": {
                "model": llm_cfg.get("model", "gpt-4o-mini"),
                "api_key": llm_cfg.get("api_key", ""),
                "openai_base_url": llm_cfg.get("base_url") or None,
                "temperature": 0.0,
                "max_tokens": 1024,
            },
        }
        if var_cfg.get("enable_vision"):
            llm_inner["config"]["enable_vision"] = True
            llm_inner["config"]["vision_details"] = var_cfg.get("vision_details", "auto")

        if emb_cfg.get("backend", "local") == "local":
            emb_inner = {
                "provider": "huggingface",
                "config": {
                    "model": emb_cfg.get("local_model", "all-MiniLM-L6-v2"),
                },
            }
        else:
            emb_inner = {
                "provider": "openai",
                "config": {
                    "model": emb_cfg.get("openai_model", "text-embedding-3-small"),
                    "api_key": llm_cfg.get("api_key", ""),
                    "openai_base_url": emb_cfg.get("openai_base_url") or None,
                },
            }

        # FAISS + cosine: mem0's default chroma config inverts the score
        # direction (L2 distance vs. score_and_rank's "higher=better"), which
        # collapses retrieval quality. FAISS with cosine returns inner-product
        # scores in the right direction.
        vector_inner = {
            "provider": "faiss",
            "config": {
                "collection_name": coll,
                "path": str(storage_root / "faiss"),
                "distance_strategy": "cosine",
                "embedding_model_dims": int(self.cfg.get("embedding", {}).get("dim", 768)),
                "normalize_L2": False,
            },
        }

        return {
            "llm": llm_inner,
            "embedder": emb_inner,
            "vector_store": vector_inner,
            "version": "v1.1",
        }

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        data_dir: Path,
    ) -> None:
        try:
            from mem0 import Memory  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mem0 baseline requires `pip install mem0ai`."
            ) from e

        os.environ.setdefault("OPENAI_API_KEY", self.cfg.get("llm", {}).get("api_key", ""))
        if self.cfg.get("llm", {}).get("base_url"):
            os.environ.setdefault("OPENAI_BASE_URL", self.cfg["llm"]["base_url"])

        cfg = self._mem0_config()
        self.mem = Memory.from_config(cfg)

        max_imgs = int(self.cfg.get("eval", {}).get("max_images_per_chat", 10))
        pdf_cfg = self.cfg.get("pdf", {})
        pdf_max_pages = int(pdf_cfg.get("max_pages_per_pdf", 0))
        pdf_dpi = int(pdf_cfg.get("render_dpi", 144))
        pdf_cache = data_dir / pdf_cfg.get("cache_dir", ".pdf_renders")

        for sid, rnd, dlg in iter_dialogues(sessions):
            text_block = dialogue_to_text(dlg, sid)
            text_tagged = f"{text_block}\n[ROUND_TAG:{rnd}]"

            if self.variant == "mem0_text":
                msgs = [{"role": "user", "content": text_tagged}]
                self._safe_add(msgs, dlg, sid, rnd)
                continue

            imgs = collect_dialogue_images(dlg, images_dir)
            for pdf in collect_dialogue_pdfs(dlg, data_dir):
                imgs += render_pdf_pages(
                    pdf, pdf_cache, dpi=pdf_dpi, max_pages=pdf_max_pages
                )
            if imgs:
                self._round2imgs.setdefault(rnd, []).extend(imgs)
            if not imgs:
                msgs = [{"role": "user", "content": text_tagged}]
                self._safe_add(msgs, dlg, sid, rnd)
                continue

            n_batches = math.ceil(len(imgs) / max_imgs)
            for bi in range(n_batches):
                batch = imgs[bi * max_imgs : (bi + 1) * max_imgs]
                content: List[Dict[str, Any]] = [{"type": "text", "text": text_tagged}]
                for ip in batch:
                    obj = _image_url_obj(ip)
                    if obj:
                        content.append(obj)
                msgs = [{"role": "user", "content": content}]
                self._safe_add(msgs, dlg, sid, rnd, batch_idx=bi, n_batches=n_batches)

    def _safe_add(
        self,
        messages: List[Dict[str, Any]],
        dlg: Dict[str, Any],
        sid: str,
        rnd: str,
        *,
        batch_idx: int = 0,
        n_batches: int = 1,
    ) -> None:
        meta = {
            "session_id": sid,
            "round_id": rnd,
            "batch_idx": batch_idx,
            "n_batches": n_batches,
            "date": str(dlg.get("date") or ""),
        }
        try:
            self.mem.add(
                messages=messages,
                user_id=self._user_id,
                metadata=meta,
                infer=False,
            )
        except Exception as e:
            print(f"[{self.variant}] add failed for {rnd} (batch {batch_idx}): {e}")

    def retrieve(
        self, question: str, top_k: int = 5
    ) -> Tuple[str, List[str], List[str]]:
        if not self.mem:
            return "", [], []
        try:
            res = self.mem.search(
                query=question, top_k=max(top_k * 2, top_k),
                filters={"user_id": self._user_id},
            )
        except Exception as e:
            print(f"[{self.variant}] search failed: {e}")
            return "", [], []
        items = res.get("results", []) if isinstance(res, dict) else (res or [])
        ctx_parts: List[str] = []
        rounds: List[str] = []
        retrieved_round_ids: List[str] = []
        seen_rounds: set = set()
        retrieved_imgs: List[str] = []
        for it in items:
            md = it.get("metadata") or {}
            rnd = md.get("round_id") or ""
            if rnd and rnd in seen_rounds:
                continue
            seen_rounds.add(rnd)
            text = it.get("memory") or ""
            ctx_parts.append(text)
            if rnd:
                retrieved_round_ids.append(rnd)
            for r in _ROUND_TAG_RE.findall(text):
                if r not in rounds:
                    rounds.append(r)
            if len(ctx_parts) >= top_k:
                break
        if self.variant == "mem0_visual" and retrieved_round_ids:
            retrieved_imgs = self._lookup_images_for_rounds(retrieved_round_ids)
        return "\n\n".join(ctx_parts), rounds, retrieved_imgs

    def _lookup_images_for_rounds(self, rounds: List[str]) -> List[str]:
        out: List[str] = []
        rmap = getattr(self, "_round2imgs", {})
        for r in rounds:
            for p in rmap.get(r, []):
                if p not in out:
                    out.append(p)
        return out[: int(self.cfg.get("eval", {}).get("max_images_per_chat", 10))]
