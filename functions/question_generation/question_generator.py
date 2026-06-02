from __future__ import annotations
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from m3exam.config.config_loader import cfg
from m3exam.global_methods import run_chatgpt, set_openai_key
from m3exam.functions.common.vision_llm import call_vision_llm
from m3exam.functions.question_generation.question_utils import (
    MR_PDF_QUESTION_PROMPT,
    PROMPTS,
)


VALID_TYPES = ("SS", "MS", "TR", "TH", "II", "MR", "FM", "FJ")


class QuestionGenerator:

    def __init__(self) -> None:
        self.dialogue_route   = cfg("question_generation", "question_dialogue_route")
        self.question_number  = int(cfg("question_generation", "question_number",  default=50))
        self.question_per_turn= int(cfg("question_generation", "question_per_turn", default=5))
        self.question_type    = str(cfg("question_generation", "question_type",    default="SS")).upper().strip()
        self.rounds_per_sample= int(cfg("question_generation", "rounds_per_sample", default=4))
        self.output_dir       = cfg("question_generation", "output_dir")

        if not self.output_dir:
            self.output_dir = str(Path(self.dialogue_route).parent / "questions")

        self.user_persona = (
            cfg("thematic_subset",       "user_persona")
            or cfg("timeline_generation", "user_persona")
            or ""
        )
        self.core_event = cfg("thematic_subset", "core_event") or ""

        self.sessions: list[dict[str, Any]] = self._load_sessions()
        self.timeline: list[dict[str, Any]] = self._load_timeline()

    def _load_sessions(self) -> list[dict[str, Any]]:
        path = Path(self.dialogue_route) / "sessions.json"
        if not path.exists():
            raise FileNotFoundError(f"sessions.json not found at {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_timeline(self) -> list[dict[str, Any]]:
        route = Path(self.dialogue_route)
        candidates = [route / "timeline.json"]
        candidates.extend(sorted(route.glob("timeline_*.json")))
        for path in candidates:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        return []

    def _round_image_files(self, dlg: dict) -> list[str]:
        for key in ("img_file", "image_file"):
            v = dlg.get(key)
            if not v:
                continue
            if isinstance(v, str):
                return [v]
            if isinstance(v, list):
                return [str(x) for x in v if x]
        return []

    def _round_pdf_files(self, dlg: dict) -> list[str]:
        v = dlg.get("pdf_file")
        if not v:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return []

    def _fmt_rounds(
        self,
        rounds: list[dict],
        session_date: str = "",
        show_date_header: bool = False,
        include_image_files: bool = False,
    ) -> str:
        lines: list[str] = []
        if show_date_header and session_date:
            lines.append(f"[Session Date: {session_date}]")
        for r in rounds:
            lines.append(f"[{r['round']}] User: {r.get('user', '')}")
            lines.append(f"[{r['round']}] Assistant: {r.get('assistant', '')}")
            if include_image_files:
                imgs = self._round_image_files(r)
                if imgs:
                    lines.append(f"[{r['round']}] Images: {', '.join(imgs)}")
        return "\n".join(lines)

    def _timeline_summary(self) -> str:
        if not self.timeline:
            return "No timeline available."
        return "\n".join(
            f"- [{e.get('Event_Time', 'N/A')}] {e.get('Event_Description', '')}"
            for e in self.timeline
        )

    def _sample_ss(self) -> tuple[list[dict], str]:
        session = random.choice(self.sessions)
        dialogues = session["dialogues"]
        n = self.rounds_per_sample
        if len(dialogues) <= n:
            return dialogues, session.get("date", "")
        start = random.randint(0, len(dialogues) - n)
        return dialogues[start : start + n], session.get("date", "")

    def _sample_ms(self) -> str:
        n_sessions = min(3, len(self.sessions))
        n_sessions = max(n_sessions, min(2, len(self.sessions)))

        chosen = random.sample(self.sessions, n_sessions)
        parts: list[str] = []
        per_session = max(2, self.rounds_per_sample // n_sessions)

        for sess in chosen:
            dlg = sess["dialogues"]
            if len(dlg) <= per_session:
                rounds = dlg
            else:
                start = random.randint(0, len(dlg) - per_session)
                rounds = dlg[start : start + per_session]
            header = f"=== Session {sess['session_id']} (Date: {sess.get('date','unknown')}) ==="
            parts.append(header + "\n" + self._fmt_rounds(rounds, sess.get("date", ""), show_date_header=False))

        return "\n\n".join(parts)

    def _sample_tr(self) -> str:
        n_sessions = min(4, len(self.sessions))
        n_sessions = max(n_sessions, min(2, len(self.sessions)))

        chosen = random.sample(self.sessions, n_sessions)
        chosen.sort(key=lambda s: s.get("date", ""))
        parts: list[str] = []
        per_session = max(2, self.rounds_per_sample // 2)

        for sess in chosen:
            dlg = sess["dialogues"]
            if len(dlg) <= per_session:
                rounds = dlg
            else:
                start = random.randint(0, len(dlg) - per_session)
                rounds = dlg[start : start + per_session]
            header = f"=== Session {sess['session_id']} - Date: {sess.get('date','unknown')} ==="
            parts.append(header + "\n" + self._fmt_rounds(rounds))

        return "\n\n".join(parts)

    def _sample_th(self) -> tuple[list[dict], str]:
        session = random.choice(self.sessions)
        dialogues = session["dialogues"]
        target = max(self.rounds_per_sample + 2, 5)
        target = min(target, len(dialogues), 8)
        if len(dialogues) <= target:
            return dialogues, session.get("session_id", "")
        start = random.randint(0, len(dialogues) - target)
        return dialogues[start : start + target], session.get("session_id", "")

    def _sample_ii(self) -> str:
        n_sessions = min(5, len(self.sessions))
        n_sessions = max(n_sessions, min(2, len(self.sessions)))

        chosen = random.sample(self.sessions, n_sessions)
        parts: list[str] = []
        per_session = max(2, self.rounds_per_sample // 2)

        for sess in chosen:
            dlg = sess["dialogues"]
            if len(dlg) <= per_session:
                rounds = dlg
            else:
                start = random.randint(0, len(dlg) - per_session)
                rounds = dlg[start : start + per_session]
            header = f"=== Session {sess['session_id']} (Date: {sess.get('date','unknown')}) ==="
            parts.append(header + "\n" + self._fmt_rounds(rounds, sess.get("date", ""), show_date_header=False))

        return "\n\n".join(parts)

    def _rounds_with_images(self, dialogues: list[dict]) -> list[dict]:
        return [d for d in dialogues if self._round_image_files(d)]

    def _sample_mr(self) -> tuple[str, str]:
        eligible = [s for s in self.sessions if self._rounds_with_images(s["dialogues"])]
        if not eligible:
            eligible = self.sessions

        session = random.choice(eligible)
        dialogues = session["dialogues"]
        with_imgs = self._rounds_with_images(dialogues)
        if not with_imgs:
            return self._fmt_rounds(dialogues, session.get("date", ""), include_image_files=True), ""

        center = random.choice(with_imgs)
        center_idx = dialogues.index(center)
        n = max(self.rounds_per_sample, 4)
        half = n // 2
        start = max(0, center_idx - half)
        end = min(len(dialogues), start + n)
        rounds = dialogues[start:end]

        excerpt = (
            f"=== Session {session['session_id']} (Date: {session.get('date','unknown')}) ===\n"
            + self._fmt_rounds(rounds, session.get("date", ""), include_image_files=True)
        )

        inventory_lines: list[str] = []
        for r in rounds:
            imgs = self._round_image_files(r)
            if imgs:
                inventory_lines.append(f"{r['round']}: {', '.join(imgs)}")
        inventory = "\n".join(inventory_lines) or "(no images in this window)"
        return excerpt, inventory

    def _rounds_with_pdfs(self, dialogues: list[dict]) -> list[dict]:
        return [d for d in dialogues if self._round_pdf_files(d)]

    def _resolve_pdf_path(self, pdf_basename: str) -> Optional[Path]:
        candidates: list[Path] = []
        if self.dialogue_route:
            base = Path(self.dialogue_route)
            candidates.extend([base / "pdfs" / pdf_basename, base / pdf_basename])
        pdf_route = (cfg("thematic_subset", "pdf_route") or "").strip()
        if pdf_route:
            candidates.append(Path(pdf_route) / pdf_basename)
        for p in candidates:
            if p.is_file():
                return p
        return None

    def _render_pdf_pages(self, pdf_path: Path, max_pages: int = 30, dpi: int = 144) -> list[str]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("  [WARN] PyMuPDF (fitz) not installed; cannot render PDF pages - falling back to text MR.")
            return []
        import hashlib
        cache_root = Path(os.environ.get("M3EXAM_QGEN_PDF_CACHE", "/tmp/m3exam_qgen_pdf_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)
        try:
            mtime = int(pdf_path.stat().st_mtime)
        except OSError:
            mtime = 0
        digest = hashlib.sha1(f"{pdf_path.resolve()}:{mtime}".encode("utf-8")).hexdigest()[:16]
        cache_dir = cache_root / digest
        if cache_dir.is_dir():
            cached = sorted(p for p in cache_dir.glob("page_*.png"))
            if cached:
                return [str(p) for p in cached[:max_pages]]
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            print(f"  [WARN] Could not open PDF {pdf_path}: {exc}")
            return []
        out: list[str] = []
        try:
            n = min(doc.page_count, max(1, int(max_pages)))
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            for i in range(n):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                page_path = cache_dir / f"page_{i + 1:03d}.png"
                pix.save(str(page_path))
                out.append(str(page_path))
        finally:
            doc.close()
        return out

    def _sample_mr_pdf(self) -> Optional[Tuple[str, str, str, list[str]]]:
        candidates: list[tuple[dict, dict]] = []
        for s in self.sessions:
            for d in s.get("dialogues", []):
                if self._round_pdf_files(d):
                    candidates.append((s, d))
        if not candidates:
            return None

        session, center = random.choice(candidates)
        dialogues = session["dialogues"]
        center_idx = dialogues.index(center)

        pdfs = self._round_pdf_files(center)
        if not pdfs:
            return None
        pdf_basename = pdfs[0]
        pdf_path = self._resolve_pdf_path(pdf_basename)
        if pdf_path is None:
            print(f"  [WARN] PDF '{pdf_basename}' referenced in dialogue not found on disk - falling back to text MR.")
            return None

        page_paths = self._render_pdf_pages(pdf_path)
        if not page_paths:
            return None

        n = max(self.rounds_per_sample, 4)
        half = n // 2
        start = max(0, center_idx - half)
        end = min(len(dialogues), start + n)
        rounds = dialogues[start:end]
        excerpt = (
            f"=== Session {session.get('session_id', '?')} "
            f"(Date: {session.get('date','unknown')}) ===\n"
            + self._fmt_rounds(rounds, session.get("date", ""), include_image_files=False)
        )
        doc_title = os.path.splitext(pdf_basename)[0]
        return doc_title, pdf_basename, excerpt, page_paths

    def _sample_fm(self) -> tuple[str, str]:
        eligible = []
        for s in self.sessions:
            for d in s["dialogues"]:
                if len(self._round_image_files(d)) >= 2:
                    eligible.append((s, d))
        if not eligible:
            for s in self.sessions:
                for d in s["dialogues"]:
                    if self._round_image_files(d):
                        eligible.append((s, d))
        if not eligible:
            return self._sample_mr()

        session, target_round = random.choice(eligible)
        dialogues = session["dialogues"]
        idx = dialogues.index(target_round)
        start = max(0, idx - 1)
        end = min(len(dialogues), idx + 2)
        rounds = dialogues[start:end]

        excerpt = (
            f"=== Session {session['session_id']} (Date: {session.get('date','unknown')}) ===\n"
            + self._fmt_rounds(rounds, session.get("date", ""), include_image_files=True)
        )

        inventory_lines = [
            f"{target_round['round']}: {', '.join(self._round_image_files(target_round))}"
        ]
        inventory = "\n".join(inventory_lines)
        return excerpt, inventory

    def _sample_fj(self) -> str:
        return self._sample_ms()

    def _call_llm(self, prompt: str) -> str:
        return run_chatgpt(prompt, num_tokens_request=2500, temperature=0.8)

    def _parse_response(self, response: str, q_type: str) -> list[dict]:
        text = response.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()

        start = text.find("[")
        end   = text.rfind("]") + 1
        if start == -1 or end == 0:
            print(f"  [WARN] No JSON array found in LLM response. First 300 chars:\n  {response[:300]}")
            return []

        try:
            raw: list[dict] = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON parse failed ({exc}). Attempting partial recovery...")
            raw = self._recover_partial_json(text[start:end])

        valid: list[dict] = []
        type_lower = q_type.lower()
        for item in raw:
            if not isinstance(item, dict):
                continue
            if "question" not in item or "answer" not in item:
                continue
            if not isinstance(item["answer"], list) or len(item["answer"]) == 0:
                continue
            item.setdefault("type",   type_lower)
            item.setdefault("label",  item["answer"][0])
            item.setdefault("supporting_facts", "")

            if type_lower == "fj":
                item["answer"] = item["answer"][:1]
                item["label"] = item["answer"][0]
            else:
                item["answer"] = item["answer"][:4]

            valid.append(item)

        return valid

    @staticmethod
    def _recover_partial_json(text: str) -> list[dict]:
        results: list[dict] = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(text[start : i + 1])
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
        return results

    def _build_prompt(self) -> Union[str, Tuple[str, List[str]]]:
        t = self.question_type
        template = PROMPTS[t]
        common = {
            "n": self.question_per_turn,
            "persona": self.user_persona,
            "core_event": self.core_event,
        }

        if t == "SS":
            rounds, date = self._sample_ss()
            return template.format(
                **common,
                date=date,
                dialogue_excerpt=self._fmt_rounds(rounds),
            )

        if t == "MS":
            return template.format(
                **common,
                timeline_summary=self._timeline_summary(),
                dialogue_excerpt=self._sample_ms(),
            )

        if t == "TR":
            return template.format(
                **common,
                timeline_with_dates=self._timeline_summary(),
                dialogue_excerpt=self._sample_tr(),
            )

        if t == "TH":
            rounds, _sid = self._sample_th()
            return template.format(
                **common,
                dialogue_excerpt=self._fmt_rounds(rounds),
            )

        if t == "II":
            return template.format(
                **common,
                timeline_summary=self._timeline_summary(),
                dialogue_excerpt=self._sample_ii(),
            )

        if t == "MR":
            pdf_pick = self._sample_mr_pdf()
            if pdf_pick is not None:
                doc_title, pdf_basename, excerpt, page_paths = pdf_pick
                prompt = MR_PDF_QUESTION_PROMPT.format(
                    **common,
                    doc_title=doc_title,
                    pdf_file=pdf_basename,
                    dialogue_excerpt=excerpt,
                )
                return prompt, page_paths
            excerpt, inventory = self._sample_mr()
            return template.format(
                **common,
                dialogue_excerpt=excerpt,
                image_inventory=inventory,
            )

        if t == "FM":
            excerpt, inventory = self._sample_fm()
            return template.format(
                **common,
                dialogue_excerpt=excerpt,
                image_inventory=inventory,
            )

        if t == "FJ":
            return template.format(
                **common,
                dialogue_excerpt=self._sample_fj(),
            )

        raise ValueError(f"Unknown question_type: {t!r}")

    def generate(self) -> list[dict]:
        if self.question_type not in VALID_TYPES:
            raise ValueError(
                f"Unknown question_type '{self.question_type}'. "
                f"Valid values: {', '.join(VALID_TYPES)}."
            )

        set_openai_key()

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        question_file = output_path / "question.json"

        if question_file.exists():
            with open(question_file, encoding="utf-8") as f:
                all_questions: list[dict] = json.load(f)
            print(f"Resuming: loaded {len(all_questions)} existing questions from {question_file}")
        else:
            all_questions = []

        already_have = len(all_questions)
        still_needed = self.question_number - already_have
        if still_needed <= 0:
            print(f"Target of {self.question_number} questions already reached. Nothing to do.")
            return all_questions

        n_turns = math.ceil(still_needed / self.question_per_turn)
        print(
            f"\nGenerating {still_needed} more {self.question_type} questions "
            f"({n_turns} turn(s) x up to {self.question_per_turn} per turn)..."
        )
        print(f"Output -> {question_file}\n")

        generated_this_run = 0

        for turn in range(1, n_turns + 1):
            remaining = self.question_number - len(all_questions)
            if remaining <= 0:
                break

            original_per_turn = self.question_per_turn
            if remaining < self.question_per_turn:
                self.question_per_turn = remaining

            print(f"  Turn {turn}/{n_turns} - requesting {self.question_per_turn} questions...", end=" ", flush=True)

            try:
                built = self._build_prompt()
                if isinstance(built, tuple):
                    prompt, image_paths = built
                    response = call_vision_llm(prompt, image_paths, max_tokens=2500)
                else:
                    prompt = built
                    response = self._call_llm(prompt)
                new_qs = self._parse_response(response, self.question_type)
            except Exception as exc:
                print(f"\n  [ERROR] Turn {turn} failed: {exc}")
                self.question_per_turn = original_per_turn
                continue
            finally:
                self.question_per_turn = original_per_turn

            new_qs = new_qs[: self.question_number - len(all_questions)]
            all_questions.extend(new_qs)
            generated_this_run += len(new_qs)
            print(f"got {len(new_qs)}. Total so far: {len(all_questions)}")

            with open(question_file, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, indent=2, ensure_ascii=False)

        print(f"\nDone. Generated {generated_this_run} question(s) this run.")
        print(f"Total in file: {len(all_questions)} / {self.question_number} target.")
        print(f"Saved to: {question_file}")
        return all_questions
