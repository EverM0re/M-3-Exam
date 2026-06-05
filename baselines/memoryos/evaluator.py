from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from m3exam.baselines._runtime.extended_common.dataset_io import (
    dialogue_to_text,
    iter_dialogues,
)
from m3exam.baselines._base.evaluator import BaseEvaluator

_UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream"
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))


_ROUND_TAG_RE = re.compile(r"\[ROUND_TAG:([A-Z]\d+:\d+)\]")


class MemoryOSEvaluator(BaseEvaluator):
    name = "memoryos"
    needs_images_in_chat = False

    def __init__(self, cfg: Dict[str, Any], llm, run_id: str):
        super().__init__(cfg, llm, run_id)
        self.mos = None
        self.user_id = "m3exam_user"

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        data_dir: Path,
    ) -> None:
        from memoryos import Memoryos  # type: ignore

        llm_cfg = self.cfg.get("llm", {})
        mos_cfg = self.cfg.get("models", {}).get("memoryos", {})
        emb_model = self.cfg.get("embedding", {}).get("local_model", "all-MiniLM-L6-v2")

        storage_root = (
            Path(__file__).resolve().parent / "_storage" / self.run_id
        )
        storage_root.mkdir(parents=True, exist_ok=True)

        self.mos = Memoryos(
            user_id=self.user_id,
            openai_api_key=llm_cfg.get("api_key", ""),
            data_storage_path=str(storage_root),
            openai_base_url=llm_cfg.get("base_url") or None,
            short_term_capacity=int(mos_cfg.get("short_term_capacity", 10)),
            mid_term_capacity=int(mos_cfg.get("mid_term_capacity", 2000)),
            long_term_knowledge_capacity=int(mos_cfg.get("long_term_knowledge_capacity", 100)),
            retrieval_queue_capacity=int(mos_cfg.get("retrieval_queue_capacity", 7)),
            mid_term_similarity_threshold=float(mos_cfg.get("mid_term_similarity_threshold", 0.6)),
            llm_model=llm_cfg.get("model", "gpt-4o-mini"),
            embedding_model_name=emb_model,
        )

        for sid, rnd, dlg in iter_dialogues(sessions):
            user_in = dlg.get("user", "") or ""
            asst_in = dlg.get("assistant", "") or ""
            text_block = dialogue_to_text(dlg, sid)
            extras: List[str] = []
            for line in text_block.split("\n"):
                if line.startswith("[Images attached:") or line.startswith("[PDF attached:"):
                    extras.append(line)
            if extras:
                user_in = user_in + "\n" + "\n".join(extras)
            asst_tagged = f"{asst_in}\n[ROUND_TAG:{rnd}]"
            try:
                self.mos.add_memory(
                    user_input=user_in,
                    agent_response=asst_tagged,
                    timestamp=str(dlg.get("date") or ""),
                )
            except Exception as e:
                print(f"[memoryos] add_memory failed for {rnd}: {e}")

    def retrieve(
        self, question: str, top_k: int = 5
    ) -> Tuple[str, List[str], List[str]]:
        if not self.mos:
            return "", [], []
        try:
            res = self.mos.retriever.retrieve_context(
                user_query=question, user_id=self.user_id
            )
        except Exception as e:
            print(f"[memoryos] retrieve failed: {e}")
            return "", [], []
        pages = res.get("retrieved_pages", [])[: top_k]
        ctx_parts: List[str] = []
        rounds: List[str] = []
        for p in pages:
            u = p.get("user_input", "")
            a = p.get("agent_response", "")
            ts = p.get("timestamp", "")
            ctx_parts.append(f"[Memory @ {ts}]\nUser: {u}\nAssistant: {a}")
            for rnd in _ROUND_TAG_RE.findall(a):
                if rnd not in rounds:
                    rounds.append(rnd)
        for kn in res.get("retrieved_user_knowledge", [])[:3]:
            ctx_parts.append(f"[User Knowledge] {kn.get('knowledge','')}")
        for kn in res.get("retrieved_assistant_knowledge", [])[:3]:
            ctx_parts.append(f"[Assistant Knowledge] {kn.get('knowledge','')}")
        return "\n\n".join(ctx_parts), rounds, []
