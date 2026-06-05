from __future__ import annotations

import json
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BaseEvaluator(ABC):
    """
    abstract base class for all model adapters. 

    subclassmust implement:
      - build_index(sessions, images_dir)   buildretrieval index
      - retrieve(question, sessions, images_dir, top_k) retrieve related context
      - answer(question, context, image_paths)  generate answer
    """

    def __init__(self, name: str):
        self.name = name

    # ──────────────────────────────────────────────────────────────────────
    # abstract method: subclassmust implement
    # ──────────────────────────────────────────────────────────────────────

    @abstractmethod
    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
    ) -> None:
        ...

    @abstractmethod
    def retrieve(
        self,
        question: str,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        supporting_facts: str = "",
        top_k: int = 5,
        question_type: str = "",
    ) -> Tuple[str, List[str]]:
        ...

    @abstractmethod
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
        ...

    # ──────────────────────────────────────────────────────────────────────
    # data loading helpers (shared) 
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def load_sessions(data_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
        # first try direct path, then try subdirectories
        direct = data_dir / "sessions.json"
        if direct.is_file():
            path = direct
        else:
            candidates = sorted(data_dir.glob("*/sessions.json"))
            if not candidates:
                raise FileNotFoundError(f"sessions.json not found in {data_dir}")
            path = candidates[0]

        with open(path, encoding="utf-8") as f:
            sessions = json.load(f)
        return sessions, path

    @staticmethod
    def load_questions(question_path: Path) -> List[Dict[str, Any]]:
        if question_path.is_dir():
            question_path = question_path / "question.json"
        if not question_path.is_file():
            raise FileNotFoundError(f"question.json not found: {question_path}")
        raw = question_path.read_text(encoding="utf-8")
        raw = re.sub(r",\s*([\]\}])", r"\1", raw)
        return json.loads(raw)

    @staticmethod
    def session_to_text(session: Dict[str, Any]) -> str:
        sid = session.get("session_id", "?")
        date = session.get("date", "")
        header = f"=== Session {sid}" + (f" ({date})" if date else "") + " ==="
        lines = [header]
        for dlg in session.get("dialogues", []):
            rnd = dlg.get("round", "")
            lines.append(f"\n[{rnd}]")
            lines.append(f"User      : {dlg.get('user', '')}")
            lines.append(f"Assistant : {dlg.get('assistant', '')}")
        return "\n".join(lines)

    @staticmethod
    def sessions_to_text(sessions: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            BaseEvaluator.session_to_text(s) for s in sessions
        )

    @staticmethod
    def collect_images(
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        max_images: int = 10,
    ) -> List[str]:
        seen: set = set()
        paths: List[str] = []
        for sess in sessions:
            for dlg in sess.get("dialogues", []):
                for fn in (dlg.get("img_file") or []) + (dlg.get("image_file") or []):
                    fn = str(fn).strip()
                    if fn and fn not in seen:
                        seen.add(fn)
                        full = images_dir / fn
                        if full.is_file():
                            paths.append(str(full))
                        if len(paths) >= max_images:
                            return paths
        return paths

    @staticmethod
    def parsing_session_ids(supporting_facts: str) -> List[str]:
        seen: set = set()
        ids: List[str] = []
        for sid in re.findall(r"([A-Z]\d+):\d+", supporting_facts or ""):
            if sid not in seen:
                seen.add(sid)
                ids.append(sid)
        return ids

    # ──────────────────────────────────────────────────────────────────────
    # generic evaluationmain loop (cansubclassoverride) 
    # ──────────────────────────────────────────────────────────────────────

    def run_evaluation(
        self,
        data_dir: Path,
        question_path: Path,
        top_k: int = 5,
        max_images: int = 10,
    ) -> List[Dict[str, Any]]:
        # loaddata
        sessions, sessions_path = self.load_sessions(data_dir)
        session_index = {str(s["session_id"]): s for s in sessions}
        images_dir = sessions_path.parent / "images"
        questions = self.load_questions(question_path)

        print(f"[{self.name}] building index ({len(sessions)} sessions) ...")
        self.build_index(sessions, images_dir)

        records: List[Dict[str, Any]] = []
        total = len(questions)

        for i, q in enumerate(questions, start=1):
            question   = q.get("question", "")
            answer_list = q.get("answer", [])
            label      = q.get("label", answer_list[0] if answer_list else "")
            sf         = q.get("supporting_facts", "")
            q_type     = q.get("type", "")

            print(f"  [{i:3d}/{total}] {question[:70]}")

            # retrieve
            try:
                context, img_paths = self.retrieve(
                    question=question,
                    sessions=sessions,
                    images_dir=images_dir,
                    top_k=top_k,
                )
                img_paths = img_paths[:max_images]
            except Exception as exc:
                print(f"    [WARN] retrieve failed: {exc}")
                context, img_paths = "", []

            # answer
            try:
                model_answer = self.answer(question, context, img_paths)
            except Exception as exc:
                print(f"    [WARN] answer failed: {exc}")
                model_answer = ""

            print(f"    → {model_answer[:80]!r}")

            records.append({
                "question":          question,
                "answer":            answer_list,
                "label":             label,
                "supporting_facts":  sf,
                "type":              q_type,
                "model_answer":      model_answer,
                "num_images_attached": len(img_paths),
                "model":             self.name,
            })

        return records
