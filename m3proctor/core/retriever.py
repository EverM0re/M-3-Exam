from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from m3exam.m3proctor.core.indexer import IndexedChunk


@dataclass
class RetrievedItem:
    chunk: IndexedChunk
    base_score: float
    boost: float
    final_score: float


class ModalityAwareRetriever:
    def __init__(
        self,
        chunks: Dict[str, IndexedChunk],
        embedder,
        *,
        alpha_image: float = 0.20,
        alpha_pdf: float = 0.25,
        alpha_chart: float = 0.15,
        over_fetch: int = 4,
        summary_boost: float = 0.10,
        guarantee_summary: bool = True,
        guarantee_min_sessions: int = 2,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.alpha_image = float(alpha_image)
        self.alpha_pdf = float(alpha_pdf)
        self.alpha_chart = float(alpha_chart)
        self.over_fetch = max(2, int(over_fetch))
        self.summary_boost = float(summary_boost)
        self.guarantee_summary = bool(guarantee_summary)
        self.guarantee_min_sessions = max(1, int(guarantee_min_sessions))
        self._chunk_ids: List[str] = list(chunks.keys())
        self._matrix = None

    async def build_async(self) -> None:
        import numpy as np

        texts = [self.chunks[cid].content for cid in self._chunk_ids]
        try:
            emb = await self.embedder(texts)
        except TypeError:
            emb = self.embedder(texts)
        emb = np.asarray(emb)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = emb / norms

    async def retrieve_async(
        self,
        query: str,
        modality_flags: Dict[str, bool],
        top_k: int = 5,
    ) -> List[RetrievedItem]:
        import numpy as np

        try:
            qemb = await self.embedder([query])
        except TypeError:
            qemb = self.embedder([query])
        qemb = np.asarray(qemb)[0]
        qnorm = np.linalg.norm(qemb) or 1.0
        qemb = qemb / qnorm

        sims = self._matrix @ qemb

        pool = min(len(self._chunk_ids), max(top_k * self.over_fetch, top_k + 2))
        top_idx = np.argpartition(-sims, pool - 1)[:pool]
        cand = sorted(top_idx.tolist(), key=lambda i: -sims[i])

        items: List[RetrievedItem] = []
        need_img = bool(modality_flags.get("needs_image"))
        need_pdf = bool(modality_flags.get("needs_pdf"))
        need_chart = bool(modality_flags.get("needs_chart"))
        for i in cand:
            cid = self._chunk_ids[i]
            ch = self.chunks[cid]
            base = float(sims[i])
            boost = 0.0
            if need_img and ch.has_image:
                boost += self.alpha_image
            if need_pdf and ch.has_pdf:
                boost += self.alpha_pdf
            if need_chart and ch.has_chart:
                boost += self.alpha_chart
            if ch.kind == "summary":
                boost += self.summary_boost
            items.append(
                RetrievedItem(chunk=ch, base_score=base, boost=boost, final_score=base + boost)
            )

        items.sort(key=lambda x: x.final_score, reverse=True)
        picked = items[:top_k]

        if self.guarantee_summary and not any(it.chunk.kind == "summary" for it in picked):
            for it in items[top_k:]:
                if it.chunk.kind == "summary":
                    non_sum = [j for j in range(len(picked)) if picked[j].chunk.kind != "summary"]
                    if non_sum:
                        picked[non_sum[-1]] = it
                    break

        sessions_now = {it.chunk.session_id for it in picked}
        if len(sessions_now) < self.guarantee_min_sessions:
            for it in items[top_k:]:
                if it.chunk.session_id in sessions_now:
                    continue
                from collections import Counter
                c = Counter(p.chunk.session_id for p in picked)
                over_sid = c.most_common(1)[0][0]
                drop_indices = [
                    j for j in range(len(picked)) if picked[j].chunk.session_id == over_sid
                ]
                if drop_indices:
                    drop_idx = min(drop_indices, key=lambda j: picked[j].final_score)
                    picked[drop_idx] = it
                    sessions_now.add(it.chunk.session_id)
                if len(sessions_now) >= self.guarantee_min_sessions:
                    break

        # For every summary chunk in top-k, ensure at least one round chunk
        # from the same session is also present (otherwise the LLM only sees
        # the condensed timeline, losing access to round-level details).
        summary_sessions = {it.chunk.session_id for it in picked if it.chunk.kind == "summary"}
        for sid in summary_sessions:
            already_has_round = any(
                it.chunk.kind == "round" and it.chunk.session_id == sid for it in picked
            )
            if already_has_round:
                continue
            candidate = None
            for it in items:
                if it.chunk.kind == "round" and it.chunk.session_id == sid:
                    candidate = it
                    break
            if candidate is None:
                continue
            replaceable = [
                j for j in range(len(picked))
                if picked[j].chunk.kind != "summary"
                and picked[j].chunk.session_id != sid
            ]
            if replaceable:
                drop = min(replaceable, key=lambda j: picked[j].final_score)
                picked[drop] = candidate

        picked.sort(key=lambda x: x.final_score, reverse=True)
        return picked
