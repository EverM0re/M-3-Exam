from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
_MM_COMMON_DIR = '/Users/evermore_01/Documents/Code/Multimodaltalk_git/m3exam/baselines/_runtime/multimodel_common'
if _MM_COMMON_DIR not in sys.path:
    sys.path.insert(0, _MM_COMMON_DIR)


_NGM_ROOT = Path(__file__).resolve().parent / "upstream"

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

_ngm_mod = load_module(
    "_ngm_graph_bridge",
    _NGM_ROOT / "multimodal_bridge" / "ngm_runner.py",
)
NGMGraphSession = _ngm_mod.NGMGraphSession


def _ngm_round_label_from_node_id(node_id: str) -> str:
    if not isinstance(node_id, str) or not node_id.startswith("round_"):
        return ""
    body = node_id[len("round_") :]
    if "_" not in body:
        return body
    return body.replace("_", ":", 1)


class NGMEvaluator(BaseEvaluator):
    pdf_policy: PdfPolicy

    upstream_repo = str(_NGM_ROOT.resolve())
    upstream_meta = {
        "bridge": "Neural-Graph-Memory-NGM-main/multimodal_bridge/ngm_runner.py",
        "core": "SRC/memory.py:NeuralGraphMemory",
    }

    def __init__(
        self,
        semantic_threshold: float = 0.5,
        use_gpu: bool = True,
        llm_client=None,
        llm_model: str = "",
        pdf_policy: PdfPolicy = "native_then_ocr",
        embed_pdf_max_chars: int = 8000,
        context_pdf_max_chars: int = 12000,
        min_native_pdf_chars: int = 400,
        clip_model: str = "openai/clip-vit-base-patch32",
        retrieve_dialogue_mode: str = "query_native",
        retrieve_context_max_rounds: int = 0,
        append_graph_ranked_round_markers: bool = True,
        related_neighbor_limit: int = 10,
    ):
        super().__init__("NGM")
        self.semantic_threshold = semantic_threshold
        self.use_gpu = use_gpu
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.pdf_policy = pdf_policy
        self.embed_pdf_max_chars = embed_pdf_max_chars
        self.context_pdf_max_chars = context_pdf_max_chars
        self.min_native_pdf_chars = min_native_pdf_chars
        self.clip_model_name = clip_model
        _rdm = (retrieve_dialogue_mode or "query_native").strip().lower()
        # top_k_graph: 历史match名, 现and上游一致 (only query return串) 
        if _rdm == "top_k_graph":
            _rdm = "query_native"
        if _rdm not in ("query_native", "structured_rounds", "full_corpus"):
            _rdm = "query_native"
        self.retrieve_dialogue_mode = _rdm
        self.retrieve_context_max_rounds = retrieve_context_max_rounds
        self.append_graph_ranked_round_markers = bool(append_graph_ranked_round_markers)
        self.related_neighbor_limit = max(0, min(int(related_neighbor_limit), 48))

        self._session: Optional[NGMGraphSession] = None
        self._pdf_llm_block: Dict[str, str] = {}
        self._sessions_pdf_injected: set = set()
        self._corpus_order: List[Dict[str, Any]] = []
        self._round_to_entry: Dict[str, Dict[str, Any]] = {}
        self._images_dir: Optional[Path] = None

    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
    ) -> None:
        self._images_dir = images_dir
        graph_sess = NGMGraphSession(
            semantic_threshold=self.semantic_threshold,
            related_neighbor_limit=self.related_neighbor_limit,
        )
        self._session = graph_sess
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

        chunks: List[Dict[str, Any]] = []
        self._corpus_order = []
        self._round_to_entry = {}
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
                content = f"[{rnd}] User: {user} | Assistant: {asst}"
                if vis:
                    content = f"{content} | ImageDesc: {vis}"
                if snip:
                    content = f"{content}\n[PDF excerpt]\n{snip}"
                chunks.append(
                    {
                        "session_id": sid,
                        "date": date,
                        "round": rnd,
                        "user": user,
                        "assistant": asst,
                        "img_files": files,
                        "text_for_node": content,
                    }
                )
                entry = {
                    "session_id": sid,
                    "date": date,
                    "round": rnd,
                    "user": user,
                    "assistant": asst,
                    "img_files": files,
                }
                self._corpus_order.append(entry)
                self._round_to_entry[str(rnd)] = entry

        print(f"  [NGM] NeuralGraphMemory 构image (仓库: {self.upstream_repo}) ...")
        graph_sess.build_from_dialogue_chunks(chunks, images_dir)

    def _ordered_round_labels_from_graph(
        self, question: str, max_rounds: int
    ) -> List[str]:
        if self._session is None:
            return []
        ranked_ids = self._session.ranked_text_round_node_ids(
            question, max(1, max_rounds), use_traversal=True
        )
        out: List[str] = []
        seen: set[str] = set()
        for nid in ranked_ids:
            rk = _ngm_round_label_from_node_id(nid)
            if rk and rk not in seen:
                seen.add(rk)
                out.append(rk)
        return out

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
        mem, img_paths = self._session.image_paths_for_question(question, top_k=top_k)
        lines: List[str] = []
        last_sid = None
        self._sessions_pdf_injected = set()

        def _append_entries_in_order(entries_in_order: List[Dict[str, Any]]) -> None:
            nonlocal last_sid
            for entry in entries_in_order:
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
                        lines.append(
                            "[Attached PDF excerpt for this session]\n" + pdf_ctx
                        )
                        self._sessions_pdf_injected.add(sid)
                lines.append(f"\n[{entry['round']}]")
                lines.append(f"User      : {entry['user']}")
                lines.append(f"Assistant : {entry['assistant']}")

        if self.retrieve_dialogue_mode == "full_corpus":
            _append_entries_in_order(list(self._corpus_order))
            dlg_header = "[Full dialogue index for VLM context]"
            ctx = "\n".join(lines)
            if mem:
                ctx = (
                    "[NeuralGraphMemory.query graph_traversal]\n"
                    + str(mem)
                    + "\n\n---\n"
                    + dlg_header
                    + "\n"
                    + ctx
                )
        elif self.retrieve_dialogue_mode == "structured_rounds":
            max_r = (
                self.retrieve_context_max_rounds
                if self.retrieve_context_max_rounds > 0
                else top_k
            )
            max_r = max(1, max_r)
            ranked_ids = self._session.ranked_text_round_node_ids(
                question, max_r, use_traversal=True
            )
            picked: List[Dict[str, Any]] = []
            seen_r: set = set()
            for nid in ranked_ids:
                rk = _ngm_round_label_from_node_id(nid)
                if not rk or rk in seen_r:
                    continue
                ent = self._round_to_entry.get(rk)
                if ent:
                    seen_r.add(rk)
                    picked.append(ent)
            _append_entries_in_order(picked)
            dlg_header = f"[Top-{max_r} dialogue rounds from graph retrieval]"
            ctx = "\n".join(lines)
            if mem:
                ctx = (
                    "[NeuralGraphMemory.query graph_traversal]\n"
                    + str(mem)
                    + "\n\n---\n"
                    + dlg_header
                    + "\n"
                    + ctx
                )
            elif lines:
                ctx = dlg_header + "\n" + ctx
            else:
                ctx = ""
        else:
            # query_native: and SRC/memory.NeuralGraphMemory.query / _graph_traversal_retrieval 一致, 
            # primarynodeall文 + if干 Related (related_neighbor_limit, 按and question相似度排序) . 
            # optionaltrack加image排序round  ``[Dx:y]`` 单line标记: 几乎not占 token, but能让 recall_at_k and「imageretrieve排序」to齐; 
            # 金标if含non-常规format (such as D17:source) 仍mayunable tomatch eval_metrics   则. 
            ctx = "[NeuralGraphMemory.query graph_traversal]\n" + str(mem or "").strip()
            if self.append_graph_ranked_round_markers:
                n_mark = (
                    self.retrieve_context_max_rounds
                    if self.retrieve_context_max_rounds > 0
                    else top_k
                )
                labels = self._ordered_round_labels_from_graph(question, max(1, n_mark))
                if labels:
                    ctx += "\n\n---\n[Graph-ranked round ids]\n" + "\n".join(
                        f"[{lb}]" for lb in labels
                    )
        if (is_image_filename_question(question) or "image" in (question or "").lower()) and (
            len(img_paths) < max(top_k, 5)
        ):
            _, more = self._session.image_paths_for_question(
                question, top_k=max(top_k * 2, 12)
            )
            for p in more:
                if p not in img_paths:
                    img_paths.append(p)
        return ctx, img_paths

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
            if self._session:
                return self._session.query_memory(question, top_k=5, use_traversal=True)
            return ""
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
            print(f"    [NGM] LLM callfailed: {exc}")
            return (
                self._session.query_memory(question, top_k=5, use_traversal=True)
                if self._session
                else ""
            )
