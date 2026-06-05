from __future__ import annotations

import os
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


class AMemEvaluator(BaseEvaluator):
    name = "a_mem"
    needs_images_in_chat = False

    def __init__(self, cfg: Dict[str, Any], llm, run_id: str):
        super().__init__(cfg, llm, run_id)
        self.mem = None
        self._round_index: List[str] = []

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        data_dir: Path,
    ) -> None:
        llm_cfg = self.cfg.get("llm", {})
        os.environ["OPENAI_API_KEY"] = llm_cfg.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        if llm_cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = llm_cfg["base_url"]
        # Upstream OpenAIController hard-codes temperature=0.7; some endpoints
        # reject that. A_MEM_TEMPERATURE env var lets us override.
        amem_temp = self.cfg.get("models", {}).get("a_mem", {}).get("temperature")
        os.environ["A_MEM_TEMPERATURE"] = str(amem_temp) if amem_temp is not None else "1.0"

        from memory_layer import AgenticMemorySystem  # type: ignore

        emb_model = self.cfg.get("embedding", {}).get("local_model", "all-MiniLM-L6-v2")
        amem_cfg = self.cfg.get("models", {}).get("a_mem", {})
        self.mem = AgenticMemorySystem(
            model_name=emb_model,
            llm_backend="openai",
            llm_model=llm_cfg.get("model", "gpt-4o-mini"),
            evo_threshold=int(amem_cfg.get("evo_threshold", 100)),
            api_key=llm_cfg.get("api_key", ""),
        )
        self._round_index.clear()
        dlgs = iter_dialogues(sessions)
        for sid, rnd, dlg in dlgs:
            text = dialogue_to_text(dlg, sid)
            text_tagged = f"{text}\n[ROUND_TAG:{rnd}]"
            try:
                self.mem.add_note(content=text_tagged, time=str(dlg.get("date") or ""))
                self._round_index.append(rnd)
            except Exception as e:
                print(f"[a_mem] add_note failed for {rnd}: {e}")

    def retrieve(
        self, question: str, top_k: int = 5
    ) -> Tuple[str, List[str], List[str]]:
        if not self.mem or not self.mem.memories:
            return "", [], []
        try:
            indices = self.mem.retriever.search(question, top_k)
        except Exception as e:
            print(f"[a_mem] retrieve failed: {e}")
            return "", [], []
        all_memories = list(self.mem.memories.values())
        ctx_parts: List[str] = []
        rounds: List[str] = []
        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            content = m.content or ""
            ctx_parts.append(content)
            for rnd in _ROUND_TAG_RE.findall(content):
                if rnd not in rounds:
                    rounds.append(rnd)
        return "\n\n".join(ctx_parts), rounds, []
