from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError as e:
    raise RuntimeError("openai>=1.0 is required: pip install openai") from e


def _image_data_url(path: str) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    try:
        data = base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        return None
    return f"data:{mime};base64,{data}"


JUDGE_PROMPT_TEMPLATE = (
    "You are an impartial judge evaluating the memory capabilities of an AI assistant "
    "with the question-answering task. Your task is to compare the Assistant's Answer "
    "against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1.\n\n"
    "Question     : {question}\n"
    "Gold Answer  : {gold}\n"
    "Model Answer : {model_answer}\n\n"
    "Scoring Rubric\n"
    "Score 0 (Incorrect / Miss):\n"
    "- The answer contradicts the Ground Truth.\n"
    "- For Yes/No questions: The answer has the wrong polarity (e.g., says \"Yes\" when Ground Truth is \"No\").\n"
    "- For Open-ended questions: The answer provides factually wrong information or hallucinations.\n"
    "- The assistant fails to provide the required information.\n"
    "Score 0.25 (Poor / Tangential):\n"
    "- The answer touches on the topic but misses the core entity or key value required.\n"
    "- The answer contains a mix of minor correct details and significant hallucinations or wrong associations.\n"
    "- The answer is excessively vague to the point of being useless (e.g., answering \"a dog\" instead of \"a golden retriever\").\n"
    "Score 0.5 (Partial / Vague / Excessive):\n"
    "- The answer is technically correct, but lacks confidence or is incomplete.\n"
    "- The answer captures the main entity or concept correctly but misses a part of the required supporting details.\n"
    "- The answer includes the correct information, but is over informative or excessive.\n"
    "- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain (e.g., \"I think it might be Yes\").\n"
    "- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.\n"
    "Score 0.75 (Good / Minor Imperfection):\n"
    "- The answer is largely accurate and captures the core information confidently.\n"
    "- It misses only minor details (e.g., specific adjectives or secondary details) that do not alter the main truth.\n"
    "- The answer contains all the correct information but includes unnecessary \"fluff\" or slight conversational filler that reduces precision.\n"
    "Score 1 (Correct / Exact):\n"
    "- The answer is accurate, precise, and confident.\n"
    "- For Yes/No questions: The polarity matches the Ground Truth perfectly.\n"
    "- For Open-ended questions: The answer contains all the core information and necessary details required by the Ground Truth without hallucinations.\n\n"
    "Reply with ONLY the score digit. No other text."
)


_ALLOWED_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _parse_judge_score(text: str) -> float:
    import re as _re

    s = (text or "").strip()
    for k in ("0.75", "0.25", "0.5", "1.0", "1", "0.0", "0"):
        if s == k:
            v = float(k)
            if v in _ALLOWED_SCORES:
                return v
    m = _re.search(r"(?<![\d.])(0?\.\d+|1(?:\.0+)?|0)(?![\d])", s)
    if m:
        try:
            v = float(m.group(1))
        except ValueError:
            return 0.0
        return min(_ALLOWED_SCORES, key=lambda x: abs(x - v))
    return 0.0


class _SubClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout: float,
        max_retries: int,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        kw: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if self.base_url:
            kw["base_url"] = self.base_url
        self.client = OpenAI(**kw)

    def create(self, **kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "temperature" in kwargs and "temperature" in msg and "does not support" in msg:
                kwargs.pop("temperature", None)
                return self.client.chat.completions.create(**kwargs)
            raise


class SharedLLM:
    def __init__(self, cfg: Dict[str, Any]):
        self.api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
        self.model = cfg.get("model", "gpt-4o-mini")
        self.answer_max_tokens = int(cfg.get("answer_max_tokens", 256))
        self.judge_max_tokens = int(cfg.get("judge_max_tokens", 8))
        self.timeout = float(cfg.get("http_timeout_sec", 900))
        self.max_retries = int(cfg.get("http_max_retries", 3))

        raw_t = cfg.get("temperature", None)
        self.temperature_override: Optional[float] = (
            float(raw_t) if raw_t is not None else None
        )
        raw_jt = cfg.get("judge_temperature", None)
        self.judge_temperature: Optional[float] = (
            float(raw_jt) if raw_jt is not None else None
        )

        self._main = _SubClient(
            self.api_key, self.base_url, self.model,
            timeout=self.timeout, max_retries=self.max_retries,
        )

        # Optional dedicated judge endpoint (cfg["judge"] mirrors api_key/base_url/model).
        # When absent, judge uses the main client.
        judge_cfg = cfg.get("judge") or {}
        if judge_cfg.get("api_key") or judge_cfg.get("model"):
            self._judge = _SubClient(
                judge_cfg.get("api_key") or self.api_key,
                judge_cfg.get("base_url") or self.base_url,
                judge_cfg.get("model") or self.model,
                timeout=float(judge_cfg.get("http_timeout_sec", self.timeout)),
                max_retries=int(judge_cfg.get("http_max_retries", self.max_retries)),
            )
        else:
            self._judge = self._main
        self.judge_model = self._judge.model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        if self.temperature_override is not None:
            t = self.temperature_override
        else:
            t = temperature
        kwargs: Dict[str, Any] = {
            "model": self._main.model,
            "messages": messages,
            "max_tokens": max_tokens or self.answer_max_tokens,
        }
        if t is not None:
            kwargs["temperature"] = t
        rsp = self._main.create(**kwargs)
        text = (rsp.choices[0].message.content or "").strip()
        usage = {}
        if getattr(rsp, "usage", None):
            usage = {
                "prompt_tokens": getattr(rsp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(rsp.usage, "completion_tokens", 0),
                "total_tokens": getattr(rsp.usage, "total_tokens", 0),
            }
        return text, usage

    def answer(
        self,
        question: str,
        context: str,
        image_paths: Optional[List[str]] = None,
        *,
        max_images: int = 10,
    ) -> Tuple[str, Dict[str, int]]:
        prompt = (
            "You are an AI assistant. Answer the question based ONLY on the conversation "
            "history below (and any attached images).\n\n"
            "=== CONVERSATION HISTORY ===\n"
            f"{context}\n"
            "=== END ===\n\n"
            f"Question: {question}\n\n"
            "Give a short, direct answer (a few words or a brief phrase). "
            "Do not add explanations or repeat the question."
        )
        paths = [p for p in (image_paths or []) if p][:max_images]
        if not paths:
            messages = [{"role": "user", "content": prompt}]
        else:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for p in paths:
                url = _image_data_url(p)
                if url:
                    content.append({"type": "image_url", "image_url": {"url": url}})
            messages = [{"role": "user", "content": content}]
        return self.chat(messages, temperature=0.2)

    def judge(self, question: str, label: str, model_answer: str) -> Tuple[float, str, Dict[str, int]]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, gold=label, model_answer=model_answer
        )
        kwargs: Dict[str, Any] = {
            "model": self._judge.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(self.judge_max_tokens, 16),
        }
        if self.judge_temperature is not None:
            kwargs["temperature"] = self.judge_temperature
        rsp = self._judge.create(**kwargs)
        text = (rsp.choices[0].message.content or "").strip()
        usage: Dict[str, int] = {}
        if getattr(rsp, "usage", None):
            usage = {
                "prompt_tokens": getattr(rsp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(rsp.usage, "completion_tokens", 0),
                "total_tokens": getattr(rsp.usage, "total_tokens", 0),
            }
        return _parse_judge_score(text), text, usage
