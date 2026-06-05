from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BaseEvaluator(ABC):
    name: str = "base"
    needs_images_in_chat: bool = False

    def __init__(self, cfg: Dict[str, Any], llm, run_id: str):
        self.cfg = cfg
        self.llm = llm
        self.run_id = run_id

    @abstractmethod
    def build_index(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        data_dir: Path,
    ) -> None: ...

    @abstractmethod
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> Tuple[str, List[str], List[str]]:
        ...

    def answer(
        self,
        question: str,
        context: str,
        image_paths: Optional[List[str]] = None,
        *,
        max_images: int = 10,
    ) -> Tuple[str, Dict[str, int]]:
        imgs = image_paths if (self.needs_images_in_chat and image_paths) else []
        return self.llm.answer(question, context, imgs, max_images=max_images)
