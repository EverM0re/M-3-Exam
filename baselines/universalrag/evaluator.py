from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
_MM_COMMON_DIR = '/Users/evermore_01/Documents/Code/Multimodaltalk_git/m3exam/baselines/_runtime/multimodel_common'
if _MM_COMMON_DIR not in sys.path:
    sys.path.insert(0, _MM_COMMON_DIR)


_UNI_ROOT = Path(__file__).resolve().parent / "upstream"
if str(_UNI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UNI_ROOT))

from m3exam.baselines._runtime.multimodel_common.base_evaluator import BaseEvaluator  # noqa: E402
from m3exam.baselines._runtime.multimodel_common.pdf_session_text import (  # noqa: E402
    PdfPolicy,
    build_session_pdf_context_blocks,
    build_session_pdf_snippets,
    dialogue_image_description,
)
from m3exam.baselines._runtime.multimodel_common.eval_metrics import openai_usage_to_dict  # noqa: E402
from m3exam.baselines._runtime.multimodel_common.multimodal_llm import build_answer_messages, is_image_filename_question  # noqa: E402

from m3exam.baselines._runtime.multimodel_common.load_upstream_bridge import load_module  # noqa: E402

_ur_mod = load_module(
    "_universalrag_rag_core",
    _UNI_ROOT / "universalrag_rag_core.py",
)
UniversalRAGOnlineSession = _ur_mod.UniversalRAGOnlineSession


class UniversalRAGEvaluator(BaseEvaluator):
    upstream_repo = str(_UNI_ROOT.resolve())
    upstream_meta = {
        "entry": "UniversalRAG-main/universalrag_rag_core.py",
        "bridge": "UniversalRAG-main/multimodal_bridge/online_runner.py",
        "router_prompt": "UniversalRAG-main/route/gpt/prompt.py:ROUTER_PROMPT",
    }

    def __init__(
        self,
        text_model: str = "BAAI/bge-large-en-v1.5",
        clip_model: str = "openai/clip-vit-base-patch32",
        use_gpu: bool = True,
        alpha: float = 0.2,
        llm_client=None,
        llm_model: str = "",
        pdf_policy: PdfPolicy = "native_then_ocr",
        embed_pdf_max_chars: int = 8000,
        context_pdf_max_chars: int = 12000,
        min_native_pdf_chars: int = 400,
        use_gpt_router: bool = True,
    ):
        super().__init__("UniversalRAG")
        self.text_model_name = text_model
        self.clip_model_name = clip_model
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.pdf_policy = pdf_policy
        self.embed_pdf_max_chars = embed_pdf_max_chars
        self.context_pdf_max_chars = context_pdf_max_chars
        self.min_native_pdf_chars = min_native_pdf_chars
        self.use_gpt_router = use_gpt_router

        self._ur: Optional[UniversalRAGOnlineSession] = None
        self._corpus: List[Dict[str, Any]] = []
        self._text_embeddings: Optional[np.ndarray] = None
        self._img_corpus: List[Dict[str, Any]] = []
        self._img_embeddings: Optional[np.ndarray] = None
        self._pdf_llm_block: Dict[str, str] = {}
        self._sessions_pdf_injected: set = set()
        self._last_route: str = ""

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
    ) -> None:
        self._ur = UniversalRAGOnlineSession(
            text_model=self.text_model_name,
            clip_model=self.clip_model_name,
            use_gpu=self.use_gpu,
            alpha=self.alpha,
        )
        self._corpus = []
        self._img_corpus = []
        texts: List[str] = []

        pdfs_dir = images_dir.parent / "pdfs"
        pdf_snip: Dict[str, str] = {}
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

        for sess in sessions:
            sid = str(sess.get("session_id", ""))
            date = sess.get("date", "")
            snip = pdf_snip.get(sid, "")
            for dlg in sess.get("dialogues") or []:
                rnd = dlg.get("round", "")
                user = dlg.get("user", "")
                asst = dlg.get("assistant", "")
                files = dlg.get("img_file") or dlg.get("image_file") or []
                if isinstance(files, str):
                    files = [files]
                vis = dialogue_image_description(dlg)
                blob = f"{user} {asst} {vis}"
                if snip:
                    blob = f"{blob}\n[PDF excerpt]\n{snip}"
                entry = {
                    "session_id": sid,
                    "date": date,
                    "round": rnd,
                    "user": user,
                    "assistant": asst,
                    "img_files": files,
                    "images_dir": str(images_dir),
                    "vis_txt": vis,
                }
                self._corpus.append(entry)
                texts.append(blob)
                valid = [
                    str(images_dir / str(fn).strip())
                    for fn in files
                    if (images_dir / str(fn).strip()).is_file()
                ]
                if valid:
                    self._img_corpus.append({**entry, "valid_img_paths": valid})

        print(f"  [UniversalRAG] BGE 编码 (仓库: {self.upstream_repo}) ...")
        self._text_embeddings = self._ur.embed_documents(texts)
        if self._img_corpus:
            vecs = []
            kept = []
            for e in self._img_corpus:
                v = self._ur.blend_image_caption(
                    e["valid_img_paths"][0], str(e.get("vis_txt", ""))
                )
                if v is not None:
                    vecs.append(v)
                    kept.append(e)
            self._img_embeddings = np.stack(vecs) if vecs else None
            self._img_corpus = kept
        else:
            self._img_embeddings = None

    def _route(self, question: str) -> str:
        if self.use_gpt_router and self.llm_client and self.llm_model:
            try:
                return self._ur.route_with_llm(question, self.llm_client, self.llm_model)
            except Exception:
                pass
        return self._ur.route_heuristic(question)

    def _retrieve_text(self, question: str, top_k: int) -> List[int]:
        if self._text_embeddings is None or not self._corpus:
            return []
        qe = self._ur.embed_query(question)
        sims = cosine_similarity(qe, self._text_embeddings).flatten()
        return list(np.argsort(sims)[::-1][:top_k])

    def _retrieve_image(self, question: str, top_k: int) -> Tuple[List[int], List[str]]:
        if self._img_embeddings is None or not self._img_corpus:
            return [], []
        qe = self._ur.encode_query_clip(question).reshape(1, -1)
        sims = cosine_similarity(qe, self._img_embeddings).flatten()
        idx = list(np.argsort(sims)[::-1][:top_k])
        paths: List[str] = []
        for i in idx:
            for p in self._img_corpus[i].get("valid_img_paths", []):
                if os.path.isfile(p) and p not in paths:
                    paths.append(p)
        return idx, paths

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
        cat = self._route(question)
        self._last_route = cat
        use_image_branch = cat in ("image", "clip", "video") or is_image_filename_question(
            question
        )

        text_idx = self._retrieve_text(question, top_k)
        img_paths: List[str] = []
        img_idx: List[int] = []
        if use_image_branch:
            img_idx, img_paths = self._retrieve_image(question, top_k)

        seen: set = set()
        ordered_idx: List[int] = []
        for i in text_idx:
            if i not in seen:
                seen.add(i)
                ordered_idx.append(i)
        for ii in img_idx:
            if not (0 <= ii < len(self._img_corpus)):
                continue
            er = self._img_corpus[ii]["round"]
            for j, c in enumerate(self._corpus):
                if c["round"] == er and j not in seen:
                    seen.add(j)
                    ordered_idx.append(j)
                    break

        if not ordered_idx:
            ordered_idx = self._retrieve_text(question, top_k)

        round_order = {self._corpus[i]["round"]: i for i in range(len(self._corpus))}
        entries = [self._corpus[i] for i in ordered_idx if 0 <= i < len(self._corpus)]
        entries.sort(key=lambda e: round_order.get(e["round"], 9999))

        lines: List[str] = []
        last_sid = None
        self._sessions_pdf_injected = set()
        for entry in entries:
            sid = entry["session_id"]
            if sid != last_sid:
                h = f"=== Session {sid}"
                if entry.get("date"):
                    h += f" ({entry['date']})"
                h += " ==="
                lines.append(h)
                last_sid = sid
                pdf_ctx = self._pdf_llm_block.get(sid)
                if pdf_ctx and sid not in self._sessions_pdf_injected:
                    lines.append("[Attached PDF excerpt for this session]\n" + pdf_ctx)
                    self._sessions_pdf_injected.add(sid)
            lines.append(f"\n[{entry['round']}]")
            lines.append(f"User      : {entry['user']}")
            lines.append(f"Assistant : {entry['assistant']}")
            for fn in entry.get("img_files", []):
                p = str(Path(entry.get("images_dir", str(images_dir))) / str(fn).strip())
                if os.path.isfile(p) and p not in img_paths:
                    img_paths.append(p)

        hdr = f"[UniversalRAG route={cat} | repo online_runner + BGE/CLIP]\n"
        return hdr + "\n".join(lines), img_paths

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
            print(f"    [UniversalRAG] LLM callfailed: {exc}")
            return ""
