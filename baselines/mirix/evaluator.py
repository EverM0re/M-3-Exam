from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from collections import defaultdict
import mimetypes
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
_MM_COMMON_DIR = '/Users/evermore_01/Documents/Code/Multimodaltalk_git/m3exam/baselines/_runtime/multimodel_common'
if _MM_COMMON_DIR not in sys.path:
    sys.path.insert(0, _MM_COMMON_DIR)


_MIRIX_ROOT = _THIS_DIR / "upstream"
sys.path.insert(0, str(_MIRIX_ROOT))

from m3exam.baselines._runtime.multimodel_common.base_evaluator import BaseEvaluator  # noqa: E402
from m3exam.baselines._runtime.multimodel_common.eval_metrics import (  # noqa: E402
    append_answer_round_citation,
    normalize_round_id_for_trace,
    openai_usage_to_dict,
    prepend_eval_round_trace,
    ROUND_CITATION_PROMPT,
    strip_answer_round_citation,
)
from m3exam.baselines._runtime.multimodel_common.fm_retrieve_helpers import (  # noqa: E402
    fm_resolve_image_paths_traced,
    image_paths_for_rounds,
    question_wants_image_gallery,
    rounds_from_lightrag_context,
)
from m3exam.baselines._runtime.multimodel_common.mirix_retrieve_helpers import (  # noqa: E402
    expand_ordered_rounds,
    extract_round_candidates,
    llm_rerank_round_ids,
    offline_round_fallback,
    order_rounds_primary_first,
)
from m3exam.baselines._runtime.multimodel_common.multimodal_llm import build_answer_messages, is_image_filename_question  # noqa: E402
from m3exam.baselines._runtime.multimodel_common.pdf_session_text import (  # noqa: E402
    build_session_pdf_context_blocks,
    build_session_pdf_snippets,
    collect_session_pdf_origins,
    dialogue_image_description,
    resolve_pdf_policy,
)


# remote HTTP ingest单image上限 (base64 into JSON, too large; triggers gateway/queue limit) 
_REMOTE_INGEST_IMAGE_MAX_BYTES = 12 * 1024 * 1024
# PDF 经 file_uri by serverread盘ingest (must share with the API containershare host path) 
_REMOTE_INGEST_PDF_MAX_BYTES = 32 * 1024 * 1024


class MirixIngestIncompleteError(RuntimeError):
    """asyncingestoverride ratebelow async_ingest_min_ratio, 且 fail_on_incomplete_ingest=true whenraise. """


INDEX_PROGRESS_VERSION = 2
PDF_INDEX_CACHE_VERSION = 1

INGEST_ANOMALY_CONTENT_FILTER = "content_filter"
INGEST_ANOMALY_ADD_FAILED = "add_failed"
INGEST_ANOMALY_MISSING_EPISODIC = "missing_episodic_after_add"
INGEST_ANOMALY_CONFIRM_TIMEOUT = "confirm_timeout"


class _IngestAnomalyLogHandler(logging.Handler):
    """capture MIRIX memory-agent log   Ark content_filter (usually notpropagate to Python) . """

    def __init__(
        self,
        anomaly_rounds: Dict[str, str],
        ctx: Dict[str, str],
    ) -> None:
        super().__init__(level=logging.WARNING)
        self.anomaly_rounds = anomaly_rounds
        self.ctx = ctx

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if "content_filter" not in msg.lower():
            return
        rid = str(self.ctx.get("current_round_id") or "").strip()
        if rid:
            self.anomaly_rounds[rid] = INGEST_ANOMALY_CONTENT_FILTER


def _record_ingest_anomaly(
    anomaly_rounds: Dict[str, str], round_id: str, reason: str
) -> None:
    rid = str(round_id or "").strip()
    if not rid:
        return
    prev = anomaly_rounds.get(rid)
    if prev == INGEST_ANOMALY_CONTENT_FILTER:
        return
    anomaly_rounds[rid] = reason


def load_index_progress_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"    [MIRIX] unable toread index progress {path}: {exc}", flush=True)
        return None
    return data if isinstance(data, dict) else None


def save_index_progress_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_pdf_index_cache_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"    [MIRIX] unable toread PDF indexcache {path}: {exc}", flush=True)
        return None
    return data if isinstance(data, dict) else None


def save_pdf_index_cache_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _mirix_ingest_progress_bar(total: int, *, remote_images: bool) -> Any:
    if total <= 0:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    if not sys.stderr.isatty():
        return None
    desc = "MIRIX ingest"
    if remote_images:
        desc += " (image base64)"
    return tqdm(
        total=total,
        desc=desc,
        unit="轮",
        leave=False,
        ncols=100,
        dynamic_ncols=False,
        file=sys.stderr,
    )


def _mirix_async_wait_progress_bar(total: int) -> Any:
    if total <= 0:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    if not sys.stderr.isatty():
        return None
    return tqdm(
        total=total,
        desc="MIRIX asyncingestpersist",
        unit="轮",
        leave=True,
        ncols=100,
        dynamic_ncols=False,
        file=sys.stderr,
    )


def _local_image_to_remote_image_url_part(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data or len(data) > _REMOTE_INGEST_IMAGE_MAX_BYTES:
        return None
    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    b64 = base64.standard_b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "auto"},
    }


def _pdf_file_uri_part(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > _REMOTE_INGEST_PDF_MAX_BYTES:
        return None
    return {"type": "file_uri", "file_uri": str(path.resolve())}


def _pdf_file_base64_part(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data or len(data) > _REMOTE_INGEST_PDF_MAX_BYTES:
        return None
    b64 = base64.standard_b64encode(data).decode("ascii")
    return {
        "type": "file_base64",
        "file_base64": {
            "filename": path.name,
            "mime_type": "application/pdf",
            "data": f"data:application/pdf;base64,{b64}",
        },
    }


def _pdf_ingest_content_part(path: Path, *, mode: str) -> Optional[Dict[str, Any]]:
    del mode
    return _pdf_file_base64_part(path)


def _cfg_str(cfg: Dict[str, Any], *keys: str, default: str = "") -> str:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    if cur is None:
        return default
    return str(cur).strip() if isinstance(cur, str) else str(cur)


def _cfg_bool(cfg: Dict[str, Any], *keys: str, default: bool = False) -> bool:
    v = _cfg_str(cfg, *keys, default="")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _cfg_int(cfg: Dict[str, Any], *keys: str, default: int = 0) -> int:
    try:
        cur: Any = cfg
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return int(cur)
    except (TypeError, ValueError):
        return default


def _cfg_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur.get(k)
    return cur


_PERSISTENT_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _persistent_async_loop() -> asyncio.AbstractEventLoop:
    global _PERSISTENT_ASYNC_LOOP
    if _PERSISTENT_ASYNC_LOOP is None or _PERSISTENT_ASYNC_LOOP.is_closed():
        _PERSISTENT_ASYNC_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_PERSISTENT_ASYNC_LOOP)
    return _PERSISTENT_ASYNC_LOOP


def _run_async(coro):
    loop = _persistent_async_loop()
    return loop.run_until_complete(coro)


def _default_meta_config_path(mode: str) -> Path:
    if mode == "local":
        local_eval = _THIS_DIR / "configs" / "mirix_local_eval.yaml"
        if local_eval.is_file():
            return local_eval
    remote_docker = _THIS_DIR / "configs" / "mirix_remote_docker.yaml"
    if mode == "remote" and remote_docker.is_file():
        return remote_docker
    return _MIRIX_ROOT / "mirix" / "configs" / "mirix.yaml"


_PG_ENV_KEYS = (
    "MIRIX_PG_USER",
    "MIRIX_PG_PASSWORD",
    "MIRIX_PG_DB",
    "MIRIX_PG_HOST",
    "MIRIX_PG_PORT",
    "MIRIX_PG_URI",
)


def _resolve_local_mirix_data_dir(
    *,
    explicit: str,
    under_run: bool,
    run_output_root: Optional[Path],
) -> Tuple[Optional[Path], Optional[str]]:
    exp = (explicit or "").strip()
    if exp:
        return Path(exp).expanduser().resolve(), None
    if under_run:
        if run_output_root is None:
            return None, (
                "mirix_data_dir_under_run_output=true but未passed incurrent run output root, "
                "already skipped (viamain entry run_dataset_suite batch run, or use mirix_data_dir explicit path) "
            )
        return Path(run_output_root).expanduser().resolve() / ".mirix_data", None
    return None, None


def _apply_local_runtime_env(
    cfg: Dict[str, Any],
    *,
    local_sqlite: bool,
    meta_agent_config_path: Optional[Path] = None,
) -> None:
    llm_key = _cfg_str(cfg, "llm", "api_key")
    llm_base = _cfg_str(cfg, "llm", "base_url")
    skip_openai_base_override = False
    if meta_agent_config_path and meta_agent_config_path.is_file():
        try:
            meta_raw = _load_meta_yaml(meta_agent_config_path)
            emb = meta_raw.get("embedding_config") or {}
            mem_llm = meta_raw.get("llm_config") or {}
            if isinstance(emb, dict) and str(emb.get("embedding_endpoint") or "").strip():
                skip_openai_base_override = True
            if isinstance(mem_llm, dict) and str(mem_llm.get("model_endpoint") or "").strip():
                skip_openai_base_override = True
        except (OSError, ValueError):
            pass
    mem_base = ""
    if meta_agent_config_path and meta_agent_config_path.is_file():
        try:
            mem_llm = (_load_meta_yaml(meta_agent_config_path).get("llm_config") or {})
            if isinstance(mem_llm, dict):
                mem_base = str(mem_llm.get("model_endpoint") or "").strip().rstrip("/")
        except (OSError, ValueError):
            pass
    share_vllm_with_eval = bool(
        llm_base and mem_base and _same_openai_base(llm_base, mem_base)
    )
    if llm_key and (not skip_openai_base_override or share_vllm_with_eval):
        os.environ["OPENAI_API_KEY"] = llm_key
    if llm_base and not skip_openai_base_override:
        os.environ["OPENAI_API_BASE"] = llm_base
        os.environ.setdefault("OPENAI_BASE_URL", llm_base)
    os.environ["MIRIX_REDIS_ENABLED"] = "false"
    # Ark quota is tight: fewer retries; other endpointskeep a higher limit (env-var override possible) 
    retry_default = "8"
    if meta_agent_config_path and meta_agent_config_path.is_file():
        try:
            if _meta_memory_llm_is_ark(meta_agent_config_path):
                retry_default = "2"
        except (OSError, ValueError):
            pass
    os.environ.setdefault("MIRIX_LLM_RETRY_LIMIT", retry_default)
    if local_sqlite:
        # must pre-set as "None" rather than pop/empty: mirix/settings.py on importwill run load_dotenv(), 
        # pop 后will rewrite MIRIX-main/.env rewrite; empty string又unable toparse as Optional[int] pg_port. 
        for key in _PG_ENV_KEYS:
            os.environ[key] = "None"


def _load_meta_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"MIRIX configmust be YAML mapping: {path}")
    return data


def _meta_yaml_build_embeddings_flag(meta_path: Path) -> Optional[bool]:
    try:
        raw = _load_meta_yaml(meta_path)
    except (OSError, ValueError):
        return None
    val = raw.get("build_embeddings_for_memory")
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def _set_build_embeddings_for_memory(enabled: bool) -> None:
    os.environ["BUILD_EMBEDDINGS_FOR_MEMORY"] = "true" if enabled else "false"
    try:
        import mirix.constants as mirix_constants

        mirix_constants.BUILD_EMBEDDINGS_FOR_MEMORY = enabled
    except ImportError:
        pass
    try:
        import mirix.services.episodic_memory_manager as episodic_memory_manager

        episodic_memory_manager.BUILD_EMBEDDINGS_FOR_MEMORY = enabled
    except ImportError:
        pass


def _truncate_for_embedding(text: str, max_chars: int = 12000) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def _mirix_meta_embedding_client(
    meta: Dict[str, Any], *, eval_cfg: Dict[str, Any]
) -> Tuple[Any, str]:
    emb = meta.get("embedding_config") or {}
    if not isinstance(emb, dict):
        raise ValueError("meta YAML missing embedding_config")
    model = str(emb.get("embedding_model") or "").strip()
    base = str(emb.get("embedding_endpoint") or "").strip().rstrip("/")
    if not model or not base:
        raise ValueError("embedding_config must contain embedding_model and embedding_endpoint")
    if "host.docker.internal" in base:
        base = base.replace("host.docker.internal", "127.0.0.1")
    api_key = str(emb.get("api_key") or "").strip()
    if not api_key:
        api_key = _cfg_str(eval_cfg, "llm", "api_key") or os.environ.get(
            "OPENAI_API_KEY", ""
        ).strip()
    if not api_key:
        raise ValueError("embedding_config.api_key is empty and eval llm.api_key is unusable")
    from openai import OpenAI

    timeout = min(120, _cfg_int(eval_cfg, "llm", "http_timeout_sec", default=120))
    client = OpenAI(api_key=api_key, base_url=base, timeout=timeout, max_retries=2)
    return client, model


def _openai_embeddings_batch(
    client: Any,
    model: str,
    texts: List[str],
    *,
    batch_size: int,
) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        data = sorted(resp.data, key=lambda d: d.index)
        out.extend([list(d.embedding) for d in data])
    return out


def _preflight_memory_llm(meta_path: Path, *, eval_cfg: Dict[str, Any]) -> None:
    raw = _load_meta_yaml(meta_path)
    llm = raw.get("llm_config")
    if not isinstance(llm, dict):
        return
    model = str(llm.get("model") or "").strip()
    base = str(llm.get("model_endpoint") or "").strip().rstrip("/")
    if not model or not base:
        return
    if "host.docker.internal" in base:
        base = base.replace("host.docker.internal", "127.0.0.1")
    api_key = _resolve_openai_api_key_for_endpoint(
        str(llm.get("api_key") or ""),
        base,
        eval_cfg=eval_cfg,
    )
    timeout = min(60, _cfg_int(eval_cfg, "llm", "http_timeout_sec", default=120))
    try:
        from openai import OpenAI
    except ImportError:
        return
    client = OpenAI(api_key=api_key, base_url=base, timeout=timeout)
    from llm_chat_kwargs import chat_completion_kwargs

    vllm_tool_hint = (
        "memory LLM must support OpenAI-style tool calling (tool_choice=required) ; "
        "local vLLM also requires --enable-auto-tool-choice --tool-call-parsingr hermes (Qwen2.5) . "
    )
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            **chat_completion_kwargs(
                model, max_output_tokens=4, temperature=0.0, eval_cfg=eval_cfg
            ),
        )
    except Exception as exc:
        err = str(exc)
        if "429" in err or "Quota" in err or "quota" in err.lower():
            raise RuntimeError(f"MIRIX memory LLM preflight failed (quota/rate-limited) : {exc}") from exc
        if "401" in err or "Unauthorized" in err or "Authentication" in err:
            raise RuntimeError(f"MIRIX memory LLM preflight failed (authentication) : {exc}") from exc
        raise RuntimeError(f"MIRIX memory LLM preflight failed: {exc}") from exc
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "no-op",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice="required",
            **chat_completion_kwargs(
                model, max_output_tokens=8, temperature=0.0, eval_cfg=eval_cfg
            ),
        )
    except Exception as exc:
        err = str(exc)
        if "tool-call-parsingr" in err or "tool_choice" in err:
            raise RuntimeError(
                f"MIRIX memory LLM does not support tool calling (agent unable toingest) : {exc}. {vllm_tool_hint}"
            ) from exc
        raise RuntimeError(f"MIRIX memory LLM tool-call preflight failed: {exc}. {vllm_tool_hint}") from exc
    print(f"    [MIRIX] memory LLM 预检via (含 tool_choice) : {model} @ {base}", flush=True)


def _normalize_llm_endpoint(endpoint: Any) -> str:
    ep = str(endpoint or "").strip().rstrip("/")
    if "host.docker.internal" in ep:
        ep = ep.replace("host.docker.internal", "127.0.0.1")
    return ep


def _llm_config_endpoint(llm_cfg: Any) -> str:
    if llm_cfg is None:
        return ""
    if isinstance(llm_cfg, dict):
        return _normalize_llm_endpoint(llm_cfg.get("model_endpoint"))
    return _normalize_llm_endpoint(getattr(llm_cfg, "model_endpoint", None))


def _llm_config_model(llm_cfg: Any) -> str:
    if llm_cfg is None:
        return ""
    if isinstance(llm_cfg, dict):
        return str(llm_cfg.get("model") or "").strip()
    return str(getattr(llm_cfg, "model", None) or "").strip()


def _meta_agent_matches_request(found: Any, request: Any) -> bool:
    req_llm = getattr(request, "llm_config", None)
    found_llm = getattr(found, "llm_config", None)
    req_ep = _llm_config_endpoint(req_llm)
    found_ep = _llm_config_endpoint(found_llm)
    if req_ep and found_ep and req_ep != found_ep:
        return False
    req_model = _llm_config_model(req_llm)
    found_model = _llm_config_model(found_llm)
    if req_model and found_model and req_model != found_model:
        return False
    return True


def _build_create_meta_agent(
    cfg: Dict[str, Any],
    *,
    eval_cfg: Optional[Dict[str, Any]] = None,
):
    from mirix.schemas.agent import CreateMetaAgent
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig

    if eval_cfg is not None:
        cfg = _inject_eval_api_keys_into_meta_cfg(cfg, eval_cfg=eval_cfg)
    meta_cfg = cfg.get("meta_agent_config") or {}
    kwargs: Dict[str, Any] = {
        "llm_config": LLMConfig(**cfg["llm_config"]),
        "embedding_config": EmbeddingConfig(**cfg["embedding_config"]),
    }
    for key in ("agents", "name", "system_prompts", "description", "memory_blocks"):
        if key in meta_cfg:
            kwargs[key] = meta_cfg[key]
    return CreateMetaAgent(**kwargs)


def _role_messages_to_mirix_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages or not isinstance(messages[0], dict) or "role" not in messages[0]:
        return messages
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        converted.append(
            {
                "type": "text",
                "text": "[USER]" if msg.get("role") == "user" else "[ASSISTANT]",
            }
        )
        content = msg.get("content")
        if isinstance(content, str):
            converted.append({"type": "text", "text": content})
        elif isinstance(content, list):
            converted.extend(content)
        else:
            raise ValueError(f"Invalid message content type: {type(content)!r}")
    return converted


class _LocalMirixBridge:
    """process内 MIRIX: reuse REST levelmemoryretrieve/ingest逻辑, not经 HTTP. """

    def __init__(
        self,
        *,
        client_id: str,
        org_id: str,
        retrieve_skip_topic_extraction: bool = True,
    ) -> None:
        self.client_id = client_id or "multimodal-eval-client"
        self.org_id = org_id or "demo-org"
        self.retrieve_skip_topic_extraction = bool(retrieve_skip_topic_extraction)
        self._local = None
        self._meta_agent = None
        self._ingest_lock: Optional[asyncio.Lock] = None
        self._embedding_insert_fallback_logged = False

    async def _require_actor(self):
        local = await self._get_local()
        actor = local.client
        if actor is None:
            raise RuntimeError("LocalClient.client is not initialized")
        if actor.write_scope is None:
            raise ValueError("Client has no write_scope")
        return local, actor

    async def _get_local(self):
        if self._local is None:
            from mirix import LocalClient

            self._local = await LocalClient.create(
                client_id=self.client_id,
                org_id=self.org_id,
            )
        return self._local

    async def _retrieval_agent_state(self, server: Any, actor: Any) -> Any:
        if self._meta_agent is not None:
            return self._meta_agent
        all_agents = await server.agent_manager.list_agents(actor=actor, limit=1000)
        if not all_agents:
            raise RuntimeError("No agents found")
        preferred = str(getattr(self._meta_agent, "id", "") or "") or None
        return _pick_retrieval_agent_state(all_agents, preferred_id=preferred)

    async def _find_loadable_meta_agent(self, actor: Any) -> Any:
        from mirix.orm.errors import NoResultFound
        from mirix.schemas.agent import AgentType

        local = await self._get_local()
        for agent in await local.list_agents():
            if getattr(agent, "agent_type", None) != AgentType.meta_memory_agent:
                continue
            try:
                await local.server.agent_manager.get_agent_by_id(
                    agent_id=agent.id, actor=actor
                )
                return agent
            except NoResultFound:
                continue
        return None

    async def initialize_meta_agent(
        self,
        *,
        config_path: str,
        update_agents: bool = True,
        eval_cfg: Optional[Dict[str, Any]] = None,
    ):
        from mirix.orm.errors import NoResultFound
        from mirix.schemas.agent import AgentType, UpdateMetaAgent

        local = await self._get_local()
        if self._meta_agent is not None and not update_agents:
            return self._meta_agent

        meta_raw = _load_meta_yaml(Path(config_path))
        request = _build_create_meta_agent(meta_raw, eval_cfg=eval_cfg)
        _, actor = await self._require_actor()

        if not update_agents:
            found = await self._find_loadable_meta_agent(actor)
            if found is not None:
                if _meta_agent_matches_request(found, request):
                    disk_ep = _llm_config_endpoint(getattr(found, "llm_config", None))
                    print(
                        f"    [MIRIX] reuse库内 meta agent (memory LLM @ {disk_ep or '?'}) ",
                        flush=True,
                    )
                    self._meta_agent = found
                    return found
                req_ep = _llm_config_endpoint(getattr(request, "llm_config", None))
                disk_ep = _llm_config_endpoint(getattr(found, "llm_config", None))
                print(
                    f"    [MIRIX] meta agent configalready变more ({disk_ep or '?'} → {req_ep or '?'}) , "
                    "will按when前 YAML update (avoid误连no tool-parsingr  local vLLM) ",
                    flush=True,
                )
                update_agents = True
            else:
                self._meta_agent = await local.create_meta_agent(request=request)
                return self._meta_agent

        existing = await local.server.agent_manager.list_agents(actor=actor, limit=1000)
        meta_candidates = [
            a
            for a in existing
            if getattr(a, "agent_type", None) == AgentType.meta_memory_agent
        ]
        valid_meta: List[Any] = []
        for agent in meta_candidates:
            try:
                await local.server.agent_manager.get_agent_by_id(
                    agent_id=agent.id, actor=actor
                )
                valid_meta.append(agent)
            except NoResultFound:
                continue

        if valid_meta:
            target = _pick_retrieval_agent_state(valid_meta)
            update = UpdateMetaAgent(
                name=request.name,
                llm_config=request.llm_config,
                embedding_config=request.embedding_config,
                agents=request.agents,
                system_prompts=request.system_prompts,
            )
            try:
                self._meta_agent = await local.server.agent_manager.update_meta_agent(
                    meta_agent_id=target.id,
                    meta_agent_update=update,
                    actor=actor,
                )
                return self._meta_agent
            except NoResultFound:
                print(
                    "    [MIRIX] meta agent updatefailed (库内 agent already失效) , willre-建 meta+子 agent…",
                    flush=True,
                )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, NoResultFound) or "not found" in err.lower():
                    print(
                        f"    [MIRIX] meta agent updatefailed ({err}) , willre-建…",
                        flush=True,
                    )
                else:
                    raise

        self._meta_agent = await local.create_meta_agent(request=request)
        return self._meta_agent

    async def _ensure_user(self, user_id: str):
        from mirix.schemas.user import User as PydanticUser
        from mirix.services.user_manager import UserManager

        local = await self._get_local()
        user_manager = UserManager()
        try:
            return await user_manager.get_user_by_id(user_id)
        except Exception:
            return await user_manager.create_user(
                pydantic_user=PydanticUser(
                    id=user_id,
                    name=user_id,
                    organization_id=local.org_id,
                    timezone=user_manager.DEFAULT_TIME_ZONE,
                    status="active",
                    is_deleted=False,
                    is_admin=False,
                )
            )

    def _get_ingest_lock(self) -> asyncio.Lock:
        if self._ingest_lock is None:
            self._ingest_lock = asyncio.Lock()
        return self._ingest_lock

    async def _episodic_agent_state(self, server: Any, actor: Any) -> Any:
        from mirix.schemas.agent import AgentType

        if self._meta_agent is None:
            raise ValueError("Meta agent not initialized")
        children = await server.agent_manager.list_agents(
            actor=actor, parent_id=self._meta_agent.id, limit=100
        )
        for agent in children:
            if getattr(agent, "agent_type", None) == AgentType.episodic_memory_agent:
                return agent
        raise RuntimeError(
            "未找to episodic_memory_agent (请确认 meta_agent_config 含 episodic_memory_agent) "
        )

    async def purge_user_episodic(self, user_id: str) -> int:
        local, _actor = await self._require_actor()
        return await local.server.episodic_memory_manager.delete_by_user_id(user_id)

    async def direct_insert_episodic_turn(
        self,
        *,
        user_id: str,
        meta_agent_id: str,
        episodic_agent_state: Any,
        summary: str,
        details: str,
        filter_tags: Dict[str, Any],
        occurred_at: Optional[datetime] = None,
        use_cache: bool = True,
        summary_fallback_max: int = 280,
    ) -> Any:
        local, actor = await self._require_actor()
        user = await self._ensure_user(user_id)
        tags = dict(filter_tags or {})
        if actor.write_scope:
            tags.setdefault("scope", actor.write_scope)
        ts = occurred_at or datetime.now()
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        smax = max(40, int(summary_fallback_max))
        org_id = str(actor.organization_id or self.org_id or local.org_id)
        mgr = local.server.episodic_memory_manager
        summary_text = summary.strip() or details[:smax]
        details_text = details.strip()

        async def _insert_event() -> Any:
            return await mgr.insert_event(
                actor=actor,
                agent_state=episodic_agent_state,
                agent_id=meta_agent_id,
                timestamp=ts,
                event_type="conversation_turn",
                event_actor="user",
                summary=summary_text,
                details=details_text,
                organization_id=org_id,
                filter_tags=tags,
                use_cache=use_cache,
                client_id=actor.id,
                user_id=user.id,
            )

        try:
            return await _insert_event()
        except Exception as exc:
            if not _is_embedding_api_not_found_error(exc):
                raise
            _set_build_embeddings_for_memory(False)
            if not self._embedding_insert_fallback_logged:
                self._embedding_insert_fallback_logged = True
                print(
                    "    [MIRIX] direct_episodic embedding API 404/NotFound, "
                    "alreadyclose BUILD_EMBEDDINGS_FOR_MEMORY, 后续roundonly BM25 ingest",
                    flush=True,
                )
            return await _insert_event()

    async def add(
        self,
        *,
        user_id: str,
        messages: List[Dict[str, Any]],
        chaining: bool = False,
        filter_tags: Optional[Dict[str, Any]] = None,
        occurred_at: Optional[str] = None,
        async_add: bool = False,
        verbose: bool = False,
        use_cache: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        del headers  # local patternnotuse HTTP 头
        if self._meta_agent is None:
            raise ValueError("Meta agent not initialized")
        async with self._get_ingest_lock():
            if async_add:
                from mirix.queue.queue_util import put_messages
                from mirix.utils import convert_message_to_mirix_message

                local, actor = await self._require_actor()
                filter_tags = dict(filter_tags or {})
                filter_tags["scope"] = actor.write_scope
                input_messages = await convert_message_to_mirix_message(
                    _role_messages_to_mirix_input(messages),
                    org_id=local.org_id,
                )
                await put_messages(
                    actor=actor,
                    agent_id=self._meta_agent.id,
                    input_messages=input_messages,
                    chaining=chaining,
                    user_id=user_id,
                    verbose=verbose,
                    filter_tags=filter_tags,
                    use_cache=use_cache,
                    occurred_at=occurred_at,
                )
                return {
                    "success": True,
                    "status": "queued",
                    "agent_id": self._meta_agent.id,
                    "message_count": len(input_messages),
                }

            from mirix.utils import convert_message_to_mirix_message

            local, actor = await self._require_actor()
            filter_tags = dict(filter_tags or {})
            filter_tags["scope"] = actor.write_scope
            user = await self._ensure_user(user_id)
            input_messages = await convert_message_to_mirix_message(
                _role_messages_to_mirix_input(messages),
                org_id=local.org_id,
            )
            await local.server.send_messages(
                actor=actor,
                agent_id=self._meta_agent.id,
                input_messages=input_messages,
                chaining=chaining,
                user=user,
                verbose=verbose,
                filter_tags=filter_tags,
                use_cache=use_cache,
                occurred_at=occurred_at,
            )
            return {
                "success": True,
                "status": "processed",
                "agent_id": self._meta_agent.id,
                "message_count": len(input_messages),
            }

    async def retrieve_with_conversation(
        self,
        *,
        user_id: str,
        messages: List[Dict[str, Any]],
        limit: int = 10,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        skip_memory_types: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        from mirix.server.rest_api import (
            extract_topics_and_temporal_info,
            retrieve_memories_by_keywords,
        )
        from mirix.temporal.temporal_parsingr import parsing_temporal_expression

        local, actor = await self._require_actor()
        server = local.server
        try:
            retrieval_agent = await self._retrieval_agent_state(server, actor)
        except RuntimeError:
            return {"success": False, "error": "No agents found", "memories": {}}

        has_content = False
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                for content_item in msg.get("content", []):
                    if isinstance(content_item, dict) and content_item.get("text", "").strip():
                        has_content = True
                        break
                if has_content:
                    break

        topics: Optional[str] = None
        temporal_expr: Optional[str] = None
        skip_topic_llm = bool(getattr(self, "retrieve_skip_topic_extraction", True))
        if has_content and not skip_topic_llm:
            topics, temporal_expr = await extract_topics_and_temporal_info(
                messages, retrieval_agent.llm_config
            )
        elif has_content:
            parts: List[str] = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                for content_item in msg.get("content", []):
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        t = str(content_item.get("text") or "").strip()
                        if t:
                            parts.append(t)
            topics = " ".join(parts)[:800] if parts else None
        key_words = topics or ""

        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        if temporal_expr:
            try:
                user = await server.user_manager.get_user_by_id(user_id)
                import pytz

                reference_time = datetime.now(pytz.timezone(user.timezone))
            except Exception:
                reference_time = datetime.now()
            temporal_range = parsing_temporal_expression(temporal_expr, reference_time)
            if temporal_range:
                start_date = temporal_range.start
                end_date = temporal_range.end
                if start_date and start_date.tzinfo:
                    start_date = start_date.replace(tzinfo=None)
                if end_date and end_date.tzinfo:
                    end_date = end_date.replace(tzinfo=None)

        memories = await retrieve_memories_by_keywords(
            server=server,
            client=actor,
            user_id=user_id,
            agent_state=retrieval_agent,
            key_words=key_words,
            limit=limit,
            filter_tags=dict(filter_tags or {}),
            use_cache=use_cache,
            start_date=start_date,
            end_date=end_date,
            skip_memory_types=skip_memory_types,
        )
        return {
            "success": True,
            "topics": topics,
            "temporal_expression": temporal_expr,
            "memories": memories,
        }

    async def episodic_round_map(self, user_id: str, *, limit: int = 500) -> Dict[str, str]:
        local, actor = await self._require_actor()
        server = local.server
        try:
            agent_state = await self._retrieval_agent_state(server, actor)
        except RuntimeError:
            return {}
        user = await self._ensure_user(user_id)
        items = await server.episodic_memory_manager.list_episodic_memory(
            agent_state=agent_state,
            user=user,
            limit=max(1, limit),
        )
        out: Dict[str, str] = {}
        for item in items:
            rid = str((item.filter_tags or {}).get("round") or "").strip()
            if rid and item.id:
                out[str(item.id)] = rid
        return out

    async def episodic_total_count(self, user_id: str) -> int:
        local, actor = await self._require_actor()
        user = await self._ensure_user(user_id)
        return int(
            await local.server.episodic_memory_manager.get_total_number_of_items(user=user)
        )

    async def search(
        self,
        *,
        user_id: str,
        query: str,
        memory_type: str = "all",
        search_field: str = "null",
        search_method: str = "embedding",
        limit: int = 10,
        filter_tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        local, actor = await self._require_actor()
        server = local.server
        user = await self._ensure_user(user_id)
        try:
            agent_state = await self._retrieval_agent_state(server, actor)
        except RuntimeError:
            return {"success": False, "error": "No agents", "results": [], "count": 0}
        scopes = actor.read_scopes
        ft = dict(filter_tags or {})
        ft["scope"] = actor.write_scope
        lim = max(1, int(limit))
        field = search_field if search_field not in ("null", "", None) else "details"
        results: List[Dict[str, Any]] = []

        async def _pull(manager, list_fn: str, *, mtype: str, default_field: str):
            nonlocal results
            if len(results) >= lim and memory_type != "all":
                return
            mgr = manager
            fn = getattr(mgr, list_fn)
            items = await fn(
                agent_state=agent_state,
                user=user,
                query=query or "",
                search_field=field if field != "null" else default_field,
                search_method=search_method,
                limit=lim,
                filter_tags=ft,
                scopes=scopes,
            )
            for item in items:
                row: Dict[str, Any] = {"memory_type": mtype}
                if mtype == "episodic":
                    row.update(
                        {
                            "summary": item.summary,
                            "details": item.details,
                            "occurred_at": (
                                item.occurred_at.isoformat() if item.occurred_at else None
                            ),
                            "filter_tags": item.filter_tags,
                        }
                    )
                elif mtype == "resource":
                    row.update(
                        {
                            "title": getattr(item, "title", None),
                            "summary": getattr(item, "summary", None),
                            "content": getattr(item, "content", None),
                            "filter_tags": getattr(item, "filter_tags", None),
                        }
                    )
                elif mtype == "semantic":
                    row.update(
                        {
                            "name": getattr(item, "name", None),
                            "summary": getattr(item, "summary", None),
                            "details": getattr(item, "details", None),
                        }
                    )
                results.append(row)

        if memory_type in ("all", "episodic"):
            await _pull(server.episodic_memory_manager, "list_episodic_memory", mtype="episodic", default_field="details")
        if memory_type in ("all", "resource"):
            try:
                await _pull(server.resource_memory_manager, "list_resources", mtype="resource", default_field="content")
            except Exception:
                pass
        if memory_type in ("all", "semantic"):
            try:
                await _pull(server.semantic_memory_manager, "list_semantic_items", mtype="semantic", default_field="details")
            except Exception:
                pass
        cap = lim if memory_type != "all" else lim * 3
        return {"success": True, "results": results[:cap], "count": min(len(results), cap)}


def _round_trace_header(round_id: str) -> str:
    rid = normalize_round_id_for_trace(round_id)
    if not rid:
        return ""
    return f"[{rid}] Round {rid}\n[TURN_ID] {rid}"


def _item_round_id(
    item: Dict[str, Any],
    *,
    id_to_round: Optional[Dict[str, str]] = None,
    corpus_meta: Optional[List[Dict[str, Any]]] = None,
) -> str:
    ft = item.get("filter_tags") or {}
    if isinstance(ft, dict):
        rid = str(ft.get("round") or "").strip()
        if rid:
            return rid.replace(" ", "")
    iid = str(item.get("id") or "").strip()
    if iid and id_to_round and iid in id_to_round:
        return str(id_to_round[iid]).replace(" ", "")
    blob = " ".join(
        str(item.get(k) or "") for k in ("summary", "details", "caption", "name", "title", "description", "value")
    ).lower()
    if corpus_meta and blob.strip():
        best_rid = ""
        best_score = 0.0
        blob_tokens = {w for w in re.split(r"\W+", blob) if len(w) > 2}
        if not blob_tokens:
            return ""
        for meta in corpus_meta:
            rnd = str(meta.get("round") or "").replace(" ", "")
            if not rnd:
                continue
            meta_blob = " ".join(
                str(meta.get(k) or "") for k in ("user", "assistant", "dialogue_vis")
            ).lower()
            mt = {w for w in re.split(r"\W+", meta_blob) if len(w) > 2}
            if not mt:
                continue
            score = len(blob_tokens & mt) / max(len(blob_tokens), 1)
            if score > best_score:
                best_score = score
                best_rid = rnd
        if best_score >= 0.12:
            return best_rid
    for pat in (
        re.compile(r"\[([Dd]\d+:\d+)\]"),
        re.compile(r"\b([Dd]\d+:\d+)\b"),
    ):
        m = pat.search(blob)
        if m:
            raw = m.group(1).replace(" ", "")
            return raw if raw[:1] != "d" else "D" + raw[1:]
    return ""


def _collect_episodic_items(memories: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = (memories.get("memories") or {}).get("episodic") or {}
    if not data:
        return []
    items: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for bucket in ("relevant", "recent", "items"):
        for item in data.get(bucket) or []:
            iid = str(item.get("id") or "")
            key = iid or json.dumps(item, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    return items


def _format_retrieved_memories(
    memories: Dict[str, Any],
    *,
    id_to_round: Optional[Dict[str, str]] = None,
    corpus_meta: Optional[List[Dict[str, Any]]] = None,
    max_items_per_type: int = 0,
) -> str:
    lines: List[str] = []
    if not memories.get("memories"):
        return ""

    lines.append("[MIRIX retrieve_with_conversation]")
    if not any(
        (data or {}).get("total_count", 0) > 0 or (data or {}).get("items")
        for data in memories["memories"].values()
    ):
        return ""

    for memory_type, data in memories["memories"].items():
        if not data or data.get("total_count", 0) == 0:
            continue
        items = list(data.get("items") or [])
        if memory_type == "episodic" and not items:
            items = _collect_episodic_items(memories)
        if not items and "recent" in data:
            items = list(data.get("recent") or [])
        if not items:
            continue
        lines.append(f"--- {memory_type} ---")
        cap = int(max_items_per_type) if max_items_per_type and max_items_per_type > 0 else 0
        shown = 0
        for item in items:
            if cap and shown >= cap:
                break
            if memory_type == "core":
                label = item.get("label", "")
                value = item.get("value", "")
                content = f"{label}: {value}".strip(": ").strip() or str(item)
            else:
                rid = _item_round_id(item, id_to_round=id_to_round, corpus_meta=corpus_meta)
                parts: List[str] = []
                if rid:
                    hdr = _round_trace_header(rid)
                    if hdr:
                        parts.append(hdr)
                detail = str(item.get("details") or "").strip()
                summary = str(
                    item.get("summary")
                    or item.get("caption")
                    or item.get("name")
                    or item.get("title")
                    or item.get("description")
                    or item.get("value", "")
                    or ""
                ).strip()
                if detail:
                    parts.append(detail)
                if summary and summary != detail:
                    parts.append(summary)
                if not parts:
                    parts.append(str(item))
                ts = item.get("timestamp") or item.get("occurred_at")
                content = "\n".join(parts)
                if ts:
                    content = f"[{ts}] {content}"
            lines.append(content)
            shown += 1
    return "\n".join(lines)


def _format_wrap_user_prompt_memories(
    memories: Dict[str, Any],
    *,
    id_to_round: Optional[Dict[str, str]] = None,
    corpus_meta: Optional[List[Dict[str, Any]]] = None,
    max_items: int = 12,
) -> str:
    lines: List[str] = ["<episodic_memory>"]
    found = False
    if not memories.get("memories"):
        lines.append("None")
        lines.append("</episodic_memory>")
        return "\n".join(lines)

    n = 0
    for memory_type, data in memories["memories"].items():
        if n >= max_items:
            break
        if not data or data.get("total_count", 0) == 0:
            continue
        items = list(data.get("items") or [])
        if memory_type == "episodic" and not items:
            items = _collect_episodic_items(memories)
        if not items and "recent" in data:
            items = list(data.get("recent") or [])
        for item in items:
            if n >= max_items:
                break
            if memory_type == "core":
                label = item.get("label", "")
                value = item.get("value", "")
                content = f"{label}: {value}".strip(": ").strip() or str(item)
            else:
                rid = _item_round_id(item, id_to_round=id_to_round, corpus_meta=corpus_meta)
                parts: List[str] = []
                if rid:
                    hdr = _round_trace_header(rid)
                    if hdr:
                        parts.append(hdr)
                content = str(
                    item.get("summary")
                    or item.get("caption")
                    or item.get("name")
                    or item.get("details")
                    or item.get("value", "")
                    or str(item)
                ).strip()
                if parts:
                    parts.append(content)
                    content = "\n".join(parts)
                ts = item.get("timestamp") or item.get("occurred_at")
                if ts:
                    content = f"[{ts}] {content}"
            if content.strip():
                lines.append(content)
                found = True
                n += 1
    if not found:
        lines.append("None")
    lines.append("</episodic_memory>")
    return "\n".join(lines)


def _truncate_context(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 24] + "\n...[context truncated]"


def _append_dialogue_blocks_for_rounds(
    ctx: str,
    rounds: List[str],
    corpus_meta: List[Dict[str, Any]],
    *,
    images_dir: Path,
    max_rounds: Optional[int] = None,
    max_chars: int = 12000,
) -> str:
    ordered: List[str] = []
    seen: Set[str] = set()
    for r in rounds:
        rr = str(r).replace(" ", "")
        if not rr or rr in seen:
            continue
        seen.add(rr)
        ordered.append(rr)
    if max_rounds is not None and max_rounds > 0:
        ordered = ordered[:max_rounds]
    want = set(ordered)
    if not want:
        return ctx

    meta_by_round: Dict[str, Dict[str, Any]] = {}
    for meta in corpus_meta:
        rnd = str(meta.get("round") or "").replace(" ", "")
        if rnd in want and rnd not in meta_by_round:
            meta_by_round[rnd] = meta

    lines: List[str] = []
    last_sid: Optional[str] = None
    for rnd in ordered:
        meta = meta_by_round.get(rnd)
        if not meta:
            continue
        sid = str(meta.get("session_id") or "")
        if sid != last_sid:
            h = f"=== Session {sid}"
            if meta.get("date"):
                h += f" ({meta['date']})"
            h += " ==="
            lines.append(h)
            last_sid = sid
        body = f"User:\n{meta.get('user', '')}\n\nAssistant:\n{meta.get('assistant', '')}"
        img_names = meta.get("img_files") or []
        if img_names:
            body += "\n\n[IMAGE_FILES] " + ", ".join(str(x) for x in img_names)
        body = prepend_eval_round_trace(
            body,
            round_id=rnd,
            image_basenames=img_names or None,
        )
        lines.append(body)
    if not lines:
        return ctx
    block = "\n\n---\n[MIRIX matched dialogue turns]\n" + "\n\n".join(lines)
    merged = ctx + block if ctx else block.lstrip()
    return _truncate_context(merged, max_chars)


def _model_likely_supports_openai_tools(model: str) -> bool:
    m = (model or "").lower()
    if not m:
        return False
    if "qwen" in m or "-vl" in m or "vllm" in m:
        return False
    return True


def _is_tool_choice_unsupported_error(exc: BaseException) -> bool:
    try:
        from mirix.llm_api.helpers import is_tool_choice_unsupported_error

        return is_tool_choice_unsupported_error(exc)
    except ImportError:
        msg = str(exc).lower()
        return "tool choice" in msg or "tool-call-parsingr" in msg or "enable-auto-tool-choice" in msg


def _is_embedding_api_not_found_error(exc: BaseException) -> bool:
    try:
        from mirix.llm_api.helpers import is_embedding_api_not_found_error

        if is_embedding_api_not_found_error(exc):
            return True
    except ImportError:
        pass
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "notfound" in name:
        return True
    status = getattr(exc, "status_code", None)
    if status == 404:
        return True
    return "404" in msg and ("not found" in msg or "embed" in msg)


def _meta_memory_llm_is_ark(meta_path: Path) -> bool:
    try:
        raw = _load_meta_yaml(meta_path)
        llm = raw.get("llm_config") or {}
        if not isinstance(llm, dict):
            return False
        ep = str(llm.get("model_endpoint") or "").lower()
        return "volces.com" in ep or "ark.cn-beijing" in ep
    except (OSError, ValueError):
        return False

def _is_content_filter_error(exc: BaseException) -> bool:
    return "content_filter" in str(exc).lower()


def _ingest_retry_text_only(err: str, exc: BaseException) -> bool:
    return (
        "org_id is required" in err
        or "file or image data" in err
        or _is_content_filter_error(exc)
    )


def _pick_retrieval_agent_state(
    all_agents: List[Any],
    *,
    preferred_id: Optional[str] = None,
) -> Any:
    if not all_agents:
        raise ValueError("no agents")

    def _created_key(agent: Any) -> str:
        ts = getattr(agent, "created_at", None)
        return ts.isoformat() if ts is not None else ""

    meta_agents: List[Any] = []
    try:
        from mirix.schemas.agent import AgentType

        meta_agents = [
            a for a in all_agents if getattr(a, "agent_type", None) == AgentType.meta_memory_agent
        ]
    except ImportError:
        pass
    if not meta_agents:
        for agent in all_agents:
            name = str(getattr(agent, "name", "") or "")
            if name == "meta_memory_agent" or name.endswith("_meta_memory_agent"):
                meta_agents.append(agent)
    if meta_agents:
        if preferred_id:
            for agent in meta_agents:
                if str(getattr(agent, "id", "") or "") == preferred_id:
                    return agent
        return max(meta_agents, key=_created_key)
    return all_agents[0]


def _read_openai_credentials_from_env_file(path: Path) -> Tuple[str, str]:
    if not path.is_file():
        return "", ""
    key, base = "", ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k == "OPENAI_API_KEY" and v:
            key = v
        elif k in ("OPENAI_API_BASE", "OPENAI_BASE_URL") and v:
            base = v
    return key, base


def _same_openai_base(a: str, b: str) -> bool:
    return (a or "").rstrip("/").lower() == (b or "").rstrip("/").lower()


_PLACEHOLDER_OPENAI_API_KEYS = frozenset(
    {"", "sk-local", "local", "your-api-key", "changeme", "none", "null"}
)


def _mirix_index_llm_base_url(eval_cfg: Dict[str, Any]) -> str:
    explicit = _cfg_str(eval_cfg, "models", "mirix", "index_llm_base_url").strip().rstrip("/")
    if explicit:
        return explicit
    meta_path = _cfg_str(eval_cfg, "models", "mirix", "meta_agent_config_path").strip()
    if not meta_path:
        return ""
    try:
        llm = (_load_meta_yaml(Path(meta_path).expanduser()).get("llm_config") or {})
        if isinstance(llm, dict):
            return str(llm.get("model_endpoint") or "").strip().rstrip("/")
    except (OSError, ValueError):
        pass
    return ""


def _resolve_openai_api_key_for_endpoint(
    api_key: str,
    endpoint: str,
    *,
    eval_cfg: Dict[str, Any],
) -> str:
    key = str(api_key or "").strip()
    base = str(endpoint or "").strip().rstrip("/")
    if "host.docker.internal" in base:
        base = base.replace("host.docker.internal", "127.0.0.1")
    eval_base = _cfg_str(eval_cfg, "llm", "base_url").strip().rstrip("/")
    eval_key = _cfg_str(eval_cfg, "llm", "api_key").strip()
    index_base = _mirix_index_llm_base_url(eval_cfg)
    for match_base in (eval_base, index_base):
        if eval_key and match_base and _same_openai_base(base, match_base):
            if key.lower() in _PLACEHOLDER_OPENAI_API_KEYS:
                return eval_key
    if key.lower() in _PLACEHOLDER_OPENAI_API_KEYS:
        return eval_key or os.environ.get("OPENAI_API_KEY", "sk-local")
    return key


def _inject_eval_api_keys_into_meta_cfg(
    cfg: Dict[str, Any], *, eval_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    import copy

    out = copy.deepcopy(cfg)
    eval_base = _cfg_str(eval_cfg, "llm", "base_url").strip().rstrip("/")
    index_base = _mirix_index_llm_base_url(eval_cfg)
    match_bases = [b for b in (eval_base, index_base) if b]
    if not match_bases:
        return out
    for section, ep_field in (
        ("llm_config", "model_endpoint"),
        ("topic_extraction_llm_config", "model_endpoint"),
        ("embedding_config", "embedding_endpoint"),
    ):
        sec = out.get(section)
        if not isinstance(sec, dict):
            continue
        ep = str(sec.get(ep_field) or "").strip().rstrip("/")
        if not ep or not any(_same_openai_base(ep, b) for b in match_bases):
            continue
        sec["api_key"] = _resolve_openai_api_key_for_endpoint(
            str(sec.get("api_key") or ""),
            ep,
            eval_cfg=eval_cfg,
        )
    return out


class _MirixEvalTaskAgent:
    """evaluationuse TaskAgent: 预retrieve + search_memory / check_raw_item 工具循环 (to齐官方 evals) . """

    _SYSTEM_PROMPT_BASE = (
        "You are the Chat Agent in a personal assistant with unified Mirix memory. "
        "A preliminary memory search is provided; use `search_memory` for episodic and "
        "semantic embedding searches. Use `check_raw_item` when raw_input_id is present "
        "and exact wording matters. "
        "Answer Format (CRITICAL): output the minimal direct answer phrase first—no full "
        "sentences, no preamble. For facts use 1–8 words when possible (e.g. "
        "\"Medial saphenous vein\" not \"The recommended vein is...\"). "
        "For lists use comma-separated items. Match memory wording when possible."
    )

    def __init__(
        self,
        *,
        llm_client: Any,
        model: str,
        mirix_client: Any,
        user_id: str,
        max_tool_rounds: int = 5,
        search_limit: int = 15,
        cite_source_rounds: bool = True,
        eval_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.mirix_client = mirix_client
        self.user_id = user_id
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.search_limit = max(5, int(search_limit))
        self.cite_source_rounds = bool(cite_source_rounds)
        self._eval_cfg = eval_cfg
        self.last_usage: Dict[str, Any] = {}
        self.use_openai_tools = _model_likely_supports_openai_tools(self.model)

    @property
    def _system_prompt(self) -> str:
        if not self.cite_source_rounds:
            return self._SYSTEM_PROMPT_BASE
        return self._SYSTEM_PROMPT_BASE + " " + ROUND_CITATION_PROMPT

    def _build_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search Mirix memories (try episodic and semantic separately).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "memory_type": {
                                "type": "string",
                                "enum": [
                                    "episodic",
                                    "resource",
                                    "procedural",
                                    "knowledge",
                                    "semantic",
                                    "all",
                                ],
                            },
                            "search_field": {"type": "string"},
                            "search_method": {"type": "string", "enum": ["bm25", "embedding"]},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_raw_item",
                    "description": "Fetch raw input payload by raw_input_id from search results.",
                    "parameters": {
                        "type": "object",
                        "properties": {"raw_input_id": {"type": "string"}},
                        "required": ["raw_input_id"],
                    },
                },
            },
        ]

    def _search_memory(self, params: Optional[Dict[str, Any]]) -> Any:
        if not params or not isinstance(params, dict):
            return {"success": False, "error": "Missing search parameters."}
        args = dict(params)
        if not args.get("search_method"):
            args["search_method"] = "embedding"
        if not args.get("limit") or int(args["limit"]) < 10:
            args["limit"] = self.search_limit
        mt = args.pop("memory_type", "all")
        if mt == "knowledge":
            mt = "knowledge_vault"
        raw = _run_async(
            self.mirix_client.search(
                user_id=self.user_id,
                query=str(args.pop("query", "")),
                memory_type=str(mt),
                **{k: v for k, v in args.items() if v is not None},
            )
        )
        if not isinstance(raw, dict) or not raw.get("success"):
            return raw
        out = raw.get("results") or []
        for row in out:
            if isinstance(row, dict):
                row.pop("id", None)
                row.pop("actor", None)
        return out

    def _check_raw_item(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rid = (params or {}).get("raw_input_id")
        if not rid:
            return {"success": False, "error": "raw_input_id is required."}
        fn = getattr(self.mirix_client, "check_raw_item", None)
        if not callable(fn):
            return {
                "success": False,
                "error": "check_raw_item not available on this MIRIX client build.",
            }
        try:
            return fn(rid)  # type: ignore[misc]
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    def answer_programmatic_search(self, question: str, preliminary_block: str) -> str:
        extra_blocks: List[str] = []
        for mt in ("episodic", "semantic"):
            hits = self._search_memory(
                {
                    "query": question,
                    "memory_type": mt,
                    "search_method": "embedding",
                    "limit": self.search_limit,
                }
            )
            if isinstance(hits, list) and hits:
                extra_blocks.append(
                    f"<{mt}_embedding_search>\n"
                    + json.dumps(hits[: self.search_limit], ensure_ascii=False, default=str)
                )
        memory_blob = preliminary_block
        if extra_blocks:
            memory_blob = memory_blob + "\n\n" + "\n\n".join(extra_blocks)
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using only the provided Mirix memory excerpts. "
                    "Be concise and output only the direct answer."
                    + (f" {ROUND_CITATION_PROMPT}" if self.cite_source_rounds else "")
                ),
            },
            {
                "role": "user",
                "content": f"Memories:\n{memory_blob}\n\nQuestion: {question}",
            },
        ]
        from llm_chat_kwargs import chat_completion_kwargs

        resp = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            **chat_completion_kwargs(
                self.model,
                max_output_tokens=128 if self.cite_source_rounds else 64,
                temperature=0.0,
                eval_cfg=self._eval_cfg,
            ),
        )
        self.last_usage = openai_usage_to_dict(getattr(resp, "usage", None))
        return (resp.choices[0].message.content or "").strip()

    def answer_with_preliminary(self, question: str, preliminary_block: str) -> str:
        if not self.use_openai_tools:
            return self.answer_programmatic_search(question, preliminary_block)
        try:
            return self._answer_with_tool_loop(question, preliminary_block)
        except Exception as exc:
            if _is_tool_choice_unsupported_error(exc):
                self.use_openai_tools = False
                return self.answer_programmatic_search(question, preliminary_block)
            raise

    def answer_with_preliminary_or_fallback(
        self,
        question: str,
        preliminary_block: str,
        *,
        fallback_client: Any,
        fallback_model: str,
    ) -> str:
        try:
            return self.answer_with_preliminary(question, preliminary_block)
        except Exception as exc:
            err_name = type(exc).__name__
            if err_name not in ("AuthenticationError", "PermissionDeniedError") and "401" not in str(
                exc
            ):
                raise
            fb = _MirixEvalTaskAgent(
                llm_client=fallback_client,
                model=fallback_model,
                mirix_client=self.mirix_client,
                user_id=self.user_id,
                max_tool_rounds=self.max_tool_rounds,
                search_limit=self.search_limit,
                cite_source_rounds=self.cite_source_rounds,
                eval_cfg=self._eval_cfg,
            )
            fb.use_openai_tools = False
            print(
                f"    [MIRIX] task_agent OhMyGPT authenticationfailed, time退本  LLM 程序化retrieveanswering"
                f" (model={fallback_model!r}) ",
                flush=True,
            )
            ans = fb.answer_programmatic_search(question, preliminary_block)
            self.last_usage = dict(fb.last_usage)
            return ans

    def _answer_with_tool_loop(self, question: str, preliminary_block: str) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "system",
                "content": (
                    "Preliminary memories retrieved for this query:\n" + preliminary_block
                ),
            },
            {"role": "user", "content": question},
        ]
        tools = self._build_tools()
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for round_num in range(self.max_tool_rounds + 1):
            is_last = round_num == self.max_tool_rounds
            if is_last:
                messages.append(
                    {
                        "role": "system",
                        "content": "Maximum searches reached. Provide your best concise answer now.",
                    }
                )
            from llm_chat_kwargs import chat_completion_kwargs

            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=None if is_last else tools,
                tool_choice=None if is_last else "auto",
                **chat_completion_kwargs(
                    self.model,
                    max_output_tokens=128 if self.cite_source_rounds else 64,
                    temperature=0.0,
                    eval_cfg=self._eval_cfg,
                ),
            )
            usage = openai_usage_to_dict(getattr(resp, "usage", None))
            for k in usage_total:
                usage_total[k] += int(usage.get(k, 0) or 0)
            message = resp.choices[0].message
            if not getattr(message, "tool_calls", None):
                self.last_answer_usage = {**usage_total, **usage}
                return (message.content or "").strip()
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                        for tc in message.tool_calls
                    ],
                }
            )
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_result: Any = {"success": False, "error": "Invalid tool arguments."}
                else:
                    if tool_call.function.name == "search_memory":
                        tool_result = self._search_memory(args)
                    elif tool_call.function.name == "check_raw_item":
                        tool_result = self._check_raw_item(args)
                    else:
                        tool_result = {"success": False, "error": "Unknown tool."}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )
        self.last_usage = usage_total
        return ""


def _pg_sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _pg_exec_sql(sql: str, *, timeout: int = 60) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "mirix_pgvector",
                "psql",
                "-U",
                "mirix",
                "-d",
                "mirix",
                "-t",
                "-A",
                "-c",
                sql,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def _load_episodic_round_map_pg(user_id: str) -> Dict[str, str]:
    if not user_id:
        return {}
    uid = _pg_sql_escape(user_id)
    sql = (
        "SELECT id, filter_tags->>'round' FROM episodic_memory "
        f"WHERE user_id='{uid}' "
        "AND is_deleted=false AND filter_tags->>'round' IS NOT NULL"
    )
    rc, out_text, _ = _pg_exec_sql(sql, timeout=30)
    if rc != 0:
        return {}
    out: Dict[str, str] = {}
    for line in out_text.splitlines():
        if "|" not in line:
            continue
        iid, rnd = line.split("|", 1)
        iid, rnd = iid.strip(), rnd.strip()
        if iid and rnd:
            out[iid] = rnd.replace(" ", "")
    return out


def _pg_episodic_ingested_rounds(user_id: str) -> Set[str]:
    if not user_id:
        return set()
    uid = _pg_sql_escape(user_id)
    sql = (
        "SELECT DISTINCT filter_tags->>'round' FROM episodic_memory "
        f"WHERE user_id='{uid}' AND is_deleted=false "
        "AND filter_tags->>'round' IS NOT NULL"
    )
    rc, out_text, _ = _pg_exec_sql(sql, timeout=30)
    if rc != 0:
        return set()
    return {
        ln.strip().replace(" ", "")
        for ln in out_text.splitlines()
        if ln.strip()
    }


def _purge_user_episodic_pg(user_id: str) -> int:
    if not user_id:
        return 0
    uid = _pg_sql_escape(user_id)
    sql = (
        "WITH d AS (DELETE FROM episodic_memory WHERE user_id='"
        + uid
        + "' RETURNING 1) SELECT COUNT(*) FROM d"
    )
    rc, out_text, _ = _pg_exec_sql(sql, timeout=120)
    if rc != 0:
        return 0
    try:
        return int((out_text or "").strip() or "0")
    except ValueError:
        return 0


class MirixEvaluator(BaseEvaluator):
    upstream_repo = str(_MIRIX_ROOT.resolve())
    upstream_meta = {
        "client": "mirix/local_client/local_client.py:LocalClient (mode=local) | mirix/client/remote_client.py:MirixClient (mode=remote)",
        "server": "in-process AsyncServer (local) | mirix/server/rest_api.py (remote)",
        "answer_mode": "rag_style | mirix_hybrid (VLM+PDFpage) | mirix_task_agent",
    }

    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        llm_client=None,
        llm_model: str = "",
        probe_native_pdf: Optional[bool] = None,
        mode: str = "local",
        api_url: str = "",
        api_key: str = "",
        client_id: str = "",
        org_id: str = "",
        local_sqlite: bool = True,
        mirix_data_dir: str = "",
        mirix_data_dir_under_run_output: bool = False,
        run_output_root: Optional[Path] = None,
        meta_agent_config_path: str = "",
        add_chaining: bool = False,
        add_use_async: bool = False,
        retrieve_limit_scale: int = 2,
        embed_pdf_max_chars: int = 8000,
        context_pdf_max_chars: int = 12000,
        min_native_pdf_chars: int = 400,
        answer_mode: str = "rag_style",
        task_agent_model: str = "",
        task_agent_api_key: str = "",
        task_agent_base_url: str = "",
        task_agent_max_tool_rounds: int = 5,
        task_agent_search_limit: int = 15,
        rag_append_max_rounds: int = 0,
        rag_context_max_chars: int = 12000,
        retrieve_compact_max_items: int = 12,
        ingest_dialogue_images: bool = True,
        ingest_max_dialogue_images: int = 10,
        ingest_defer_session_pdf_in_dialogue: bool = False,
        ingest_pdf_patch_after_index: bool = True,
        ingest_pdf_native: bool = False,
        ingest_pdf_episodic_text: bool = True,
        index_pdf_via_ocr: bool = True,
        reindex_pdf_episodic_only: bool = False,
        ingest_pdf_episodic_fallback: bool = True,
        async_ingest_min_ratio: float = 0.75,
        clear_user_memory_before_build: bool = True,
        ingest_confirm_per_round: bool = False,
        ingest_confirm_per_round_sec: int = 15,
        ingest_retry_missing_sync: bool = True,
        ingest_retry_anomaly_at_end: bool = True,
        ingest_sequential_wait: Optional[bool] = None,
        ingest_per_round_max_wait_sec: int = 300,
        async_ingest_max_wait_sec: int = 1200,
        fail_on_incomplete_ingest: bool = True,
        retrieve_rerank: bool = True,
        retrieve_rerank_candidates: int = 24,
        retrieve_mode: str = "eval_hybrid",
        ingest_to_memory: bool = True,
        no_ingest_multimodal: Optional[bool] = None,
        simulate_mirix_retrieve: bool = False,
        simulate_embed_batch_size: int = 24,
        search_supplement_limit: int = 15,
        retrieve_context_max_rounds: int = 36,
        search_memory_types: Optional[List[str]] = None,
        retrieve_skip_knowledge_vault: bool = True,
        cite_source_rounds: bool = True,
        fm_top_k_images: int = 10,
        memory_llm_preflight: Optional[bool] = None,
        ingest_text_only_on_ark: Optional[bool] = None,
        retrieve_skip_topic_extraction: bool = True,
        retrieve_min_limit: int = 8,
        use_gold_in_retrieve: bool = False,
        format_memory_max_items: int = 16,
        ingest_mode: str = "",
        direct_episodic_summary_max_chars: int = 280,
        direct_episodic_build_embeddings: bool = False,
        update_meta_agent_on_build: Optional[bool] = None,
    ):
        super().__init__("mirix")
        self._cfg = cfg
        self.llm_client = llm_client
        self.llm_model = llm_model or ""
        self.mode = (mode or "local").strip().lower()
        if self.mode not in ("local", "remote"):
            raise ValueError(f"models.mirix.mode must be local or remote, when前: {mode!r}")
        self.api_url = api_url.strip() or "http://127.0.0.1:8531"
        self.api_key = api_key.strip()
        self.client_id = client_id.strip() or "multimodal-eval-client"
        self.org_id = org_id.strip() or "demo-org"
        self.local_sqlite = local_sqlite
        self._run_output_root = (
            Path(run_output_root).expanduser().resolve() if run_output_root else None
        )
        iso_dir, iso_warn = _resolve_local_mirix_data_dir(
            explicit=mirix_data_dir,
            under_run=bool(mirix_data_dir_under_run_output),
            run_output_root=self._run_output_root,
        )
        if iso_warn:
            print(f"    [MIRIX] warning: {iso_warn}", flush=True)
        if iso_dir is not None:
            if self.mode != "local":
                print(
                    "    [MIRIX] already忽略 mirix_data_dir / mirix_data_dir_under_run_output (only mode=local when设  MIRIX_DIR) ",
                    flush=True,
                )
            else:
                iso_dir.mkdir(parents=True, exist_ok=True)
                os.environ["MIRIX_DIR"] = str(iso_dir)
                print(
                    f"    [MIRIX] already设  MIRIX_DIR={iso_dir} (localdatadirectory; evaluationendnotfrom动delete) ",
                    flush=True,
                )
        self.meta_agent_config_path = meta_agent_config_path.strip()
        self.add_chaining = add_chaining
        self.add_use_async = add_use_async
        self.retrieve_limit_scale = max(1, int(retrieve_limit_scale))
        self.embed_pdf_max_chars = embed_pdf_max_chars
        self.context_pdf_max_chars = context_pdf_max_chars
        self.min_native_pdf_chars = min_native_pdf_chars
        self._probe_native_pdf = probe_native_pdf
        self.answer_mode = (answer_mode or "rag_style").strip().lower()
        if self.answer_mode not in ("rag_style", "mirix_task_agent", "mirix_hybrid"):
            raise ValueError(
                f"models.mirix.answer_mode must be rag_style, mirix_task_agent or mirix_hybrid, "
                f"when前: {answer_mode!r}"
            )
        self.task_agent_model = (task_agent_model or llm_model or "").strip()
        self.task_agent_api_key = task_agent_api_key.strip()
        self.task_agent_base_url = task_agent_base_url.strip()
        self.task_agent_max_tool_rounds = max(1, int(task_agent_max_tool_rounds))
        self.task_agent_search_limit = max(5, int(task_agent_search_limit))
        self.rag_append_max_rounds = max(0, int(rag_append_max_rounds))
        self.rag_context_max_chars = max(2000, int(rag_context_max_chars))
        self.retrieve_compact_max_items = max(3, int(retrieve_compact_max_items))
        self.ingest_dialogue_images = bool(ingest_dialogue_images)
        self.ingest_max_dialogue_images = max(1, int(ingest_max_dialogue_images))
        self.ingest_defer_session_pdf_in_dialogue = bool(
            ingest_defer_session_pdf_in_dialogue
        )
        self.ingest_pdf_patch_after_index = bool(ingest_pdf_patch_after_index)
        self.ingest_pdf_native = bool(ingest_pdf_native)
        self.ingest_pdf_episodic_text = bool(ingest_pdf_episodic_text)
        self.index_pdf_via_ocr = bool(index_pdf_via_ocr)
        self.reindex_pdf_episodic_only = bool(reindex_pdf_episodic_only)
        self.ingest_pdf_episodic_fallback = bool(ingest_pdf_episodic_fallback)
        self._pdf_episodic_patched = False
        self.pdf_policy = resolve_pdf_policy(
            "mirix",
            cfg,
            llm_model=self.llm_model,
            probe_native_pdf=self._probe_native_pdf,
        )
        self.async_ingest_min_ratio = min(1.0, max(0.1, float(async_ingest_min_ratio)))
        self.clear_user_memory_before_build = bool(clear_user_memory_before_build)
        self.index_checkpoint_path: Optional[Path] = None
        self.resume_index: bool = True
        self.ingest_retry_missing_sync = bool(ingest_retry_missing_sync)
        self.ingest_retry_anomaly_at_end = bool(ingest_retry_anomaly_at_end)
        self._ingest_anomaly_rounds: Dict[str, str] = {}
        self.ingest_confirm_per_round = bool(ingest_confirm_per_round)
        self.ingest_confirm_per_round_sec = max(10, int(ingest_confirm_per_round_sec))
        self.ingest_per_round_max_wait_sec = max(30, int(ingest_per_round_max_wait_sec))
        if ingest_sequential_wait is None:
            self.ingest_sequential_wait = (
                self.mode == "remote"
                and self.add_use_async
                and self.ingest_to_memory
            )
        else:
            self.ingest_sequential_wait = bool(ingest_sequential_wait)
        self.async_ingest_max_wait_sec = max(0, int(async_ingest_max_wait_sec))
        self.fail_on_incomplete_ingest = bool(fail_on_incomplete_ingest)
        self.retrieve_rerank = bool(retrieve_rerank)
        self.retrieve_rerank_candidates = max(8, int(retrieve_rerank_candidates))
        _rm = (retrieve_mode or "eval_hybrid").strip().lower()
        if _rm not in ("pure_mirix", "eval_hybrid"):
            raise ValueError(
                "models.mirix.retrieve_mode must be pure_mirix or eval_hybrid, "
                f"when前: {retrieve_mode!r}"
            )
        self.retrieve_mode = _rm
        _ingest_mode_cfg = (ingest_mode or "").strip().lower()
        if _ingest_mode_cfg in ("offline", "direct_episodic", "agent_add"):
            self.ingest_mode = _ingest_mode_cfg
        elif ingest_to_memory:
            self.ingest_mode = "agent_add"
        else:
            self.ingest_mode = "offline"
        if self.ingest_mode == "direct_episodic":
            self.ingest_to_memory = True
            if self.mode != "local":
                raise ValueError("models.mirix.ingest_mode=direct_episodic only支持 mode=local")
        elif self.ingest_mode == "offline":
            self.ingest_to_memory = False
        else:
            self.ingest_to_memory = bool(ingest_to_memory)
        self.direct_episodic_summary_max_chars = max(40, int(direct_episodic_summary_max_chars))
        self.direct_episodic_build_embeddings = bool(direct_episodic_build_embeddings)
        self._direct_episodic_embeddings_effective = self.direct_episodic_build_embeddings
        self._update_meta_agent_on_build = update_meta_agent_on_build
        if no_ingest_multimodal is None:
            self.no_ingest_multimodal = self.ingest_mode in ("offline", "direct_episodic")
        else:
            self.no_ingest_multimodal = bool(no_ingest_multimodal)
        if (
            self.no_ingest_multimodal
            and self.ingest_mode == "direct_episodic"
            and self.ingest_to_memory
        ):
            print(
                "    [MIRIX] hint: no_ingest_multimodal=true 且 direct_episodic 仍willto话textwrite "
                "episodic; if完allnot要ingest请设 ingest_mode: offline",
                flush=True,
            )
        self.simulate_mirix_retrieve = bool(simulate_mirix_retrieve)
        self.simulate_embed_batch_size = max(4, int(simulate_embed_batch_size))
        if (
            not self.ingest_to_memory
            and self.retrieve_mode == "pure_mirix"
            and not self.simulate_mirix_retrieve
        ):
            print(
                "    [MIRIX] ingest_to_memory=false when pure_mirix nomemorycan检; "
                "alreadyfrom动改use retrieve_mode=eval_hybrid (本  corpus time退) ",
                flush=True,
            )
            self.retrieve_mode = "eval_hybrid"
        self.search_supplement_limit = max(0, int(search_supplement_limit))
        self.retrieve_context_max_rounds = max(8, int(retrieve_context_max_rounds))
        default_search_types = ("episodic", "semantic", "resource")
        if search_memory_types:
            self.search_memory_types = tuple(
                str(x).strip().lower()
                for x in search_memory_types
                if str(x).strip()
            )
        else:
            self.search_memory_types = default_search_types
        self.retrieve_skip_knowledge_vault = bool(retrieve_skip_knowledge_vault)
        self._retrieve_skip_memory_types: Optional[Set[str]] = (
            {"knowledge_vault"} if self.retrieve_skip_knowledge_vault else None
        )
        self.fm_top_k_images = max(10, int(fm_top_k_images))
        self.cite_source_rounds = bool(cite_source_rounds)

        if self.mode == "local":
            _apply_local_runtime_env(
                self._cfg,
                local_sqlite=self.local_sqlite,
                meta_agent_config_path=self._meta_config_path_optional(),
            )

        self._memory_llm_is_ark = False
        try:
            self._memory_llm_is_ark = _meta_memory_llm_is_ark(
                self._resolve_meta_config_path()
            )
        except FileNotFoundError:
            pass
        if ingest_text_only_on_ark is None:
            self.ingest_text_only_on_ark = True
        else:
            self.ingest_text_only_on_ark = bool(ingest_text_only_on_ark)
        if memory_llm_preflight is None:
            self.memory_llm_preflight = not self._memory_llm_is_ark
        else:
            self.memory_llm_preflight = bool(memory_llm_preflight)
        self.retrieve_skip_topic_extraction = bool(retrieve_skip_topic_extraction)
        self.retrieve_min_limit = max(3, int(retrieve_min_limit))
        self.use_gold_in_retrieve = bool(use_gold_in_retrieve)
        self.format_memory_max_items = max(4, int(format_memory_max_items))

        self._client: Optional[Union[_LocalMirixBridge, Any]] = None
        self._user_id = "multimodal_eval_user"
        self._corpus_meta: List[Dict[str, Any]] = []
        self._episodic_id_to_round: Dict[str, str] = {}
        self._expected_dialogue_turns = 0
        self._pdf_ctx_blocks: Dict[str, str] = {}
        self._pdf_snippets: Dict[str, str] = {}
        self._session_pdf_origins: Dict[str, List[Tuple[str, str]]] = {}
        self._native_pdfs_ingested: Set[str] = set()
        self._native_pdf_ingest_failures: List[str] = []
        self._sessions_pdf_text_fallback: Set[str] = set()
        self._task_agent_llm_client: Any = None
        self.last_answer_usage: Dict[str, Any] = {}
        self.last_fm_image_trace: List[Dict[str, Any]] = []
        self._last_retrieve_question_type: str = ""
        self._last_retrieve_rounds: List[str] = []
        self.last_retrieve_diag: Dict[str, Any] = {}
        self.last_index_integrity: Dict[str, Any] = {}
        self._shadow_chunks: List[Dict[str, Any]] = []
        self._shadow_vecs: Any = None
        self._shadow_by_mt: Dict[str, List[int]] = defaultdict(list)

    def _resolve_task_agent_llm(self) -> Tuple[Any, str]:
        model = self.task_agent_model or self.llm_model
        base_url = self.task_agent_base_url
        meta_key = ""
        if not self.task_agent_model or not base_url:
            try:
                meta = _load_meta_yaml(self._resolve_meta_config_path())
                llm_cfg = meta.get("llm_config") or {}
                if not self.task_agent_model:
                    model = str(llm_cfg.get("model") or model).strip()
                if not base_url:
                    base_url = str(llm_cfg.get("model_endpoint") or "").strip()
                meta_key = str(llm_cfg.get("api_key") or "").strip()
            except Exception:
                pass
        eval_base = _cfg_str(self._cfg, "llm", "base_url").strip()
        eval_key = _cfg_str(self._cfg, "llm", "api_key").strip()
        if not base_url:
            base_url = eval_base
        # Optional MIRIX .env override. Set MIRIX_ENV_FILE to point to one.
        mir_env_path = os.environ.get("MIRIX_ENV_FILE", "").strip()
        mir_env = Path(mir_env_path) if mir_env_path else (_THIS_DIR / "upstream" / ".env")
        mir_key, mir_base = _read_openai_credentials_from_env_file(mir_env)
        if not base_url and mir_base:
            base_url = mir_base

        remote_ohmy = "ohmygpt" in (base_url or "").lower()
        env_key = os.environ.get("OPENAI_API_KEY", "").strip()

        if self.task_agent_api_key:
            api_key = self.task_agent_api_key
        elif _same_openai_base(base_url, eval_base) and eval_key:
            api_key = eval_key
        elif remote_ohmy or (mir_base and _same_openai_base(base_url, mir_base)):
            # OhMyGPT: 优先 MIRIX Docker .env, avoid shell in OPENAI_API_KEY 指向本  vLLM
            api_key = mir_key or meta_key or env_key or eval_key
        else:
            api_key = env_key or mir_key or meta_key or eval_key

        if not api_key:
            raise ValueError(
                "mirix task_agent missing API key: 请设  models.mirix.task_agent_api_key "
                "or ~/MIRIX/.env   OPENAI_API_KEY"
            )

        if remote_ohmy and eval_key and api_key == eval_key and api_key != mir_key:
            print(
                "    [MIRIX] warning: task_agent willuse llm.api_key 访问 OhMyGPT, "
                "建议设  task_agent_api_key or ~/MIRIX/.env",
                flush=True,
            )

        cache_tag = f"{model}|{base_url}|{api_key[:8]}"
        if self._task_agent_llm_client is not None and getattr(
            self, "_task_agent_llm_cache_tag", ""
        ) == cache_tag:
            return self._task_agent_llm_client, model

        from openai import OpenAI

        timeout = _cfg_int(self._cfg, "llm", "http_timeout_sec", default=120)
        self._task_agent_llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )
        self._task_agent_llm_model_resolved = model
        self._task_agent_llm_cache_tag = cache_tag
        return self._task_agent_llm_client, model

    def _meta_config_path_optional(self) -> Optional[Path]:
        if self.meta_agent_config_path:
            p = Path(self.meta_agent_config_path).expanduser()
            return p.resolve() if p.is_file() else None
        default_yaml = _default_meta_config_path(self.mode)
        return default_yaml.resolve() if default_yaml.is_file() else None

    def _resolve_meta_config_path(self) -> Path:
        p = self._meta_config_path_optional()
        if p is not None:
            return p
        if self.meta_agent_config_path:
            raise FileNotFoundError(
                f"MIRIX meta_agent_config_path notishave效file: {self.meta_agent_config_path}"
            )
        raise FileNotFoundError(f"MIRIX defaultconfigmissing失: {_default_meta_config_path(self.mode)}")

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.mode == "local":
            _apply_local_runtime_env(
                self._cfg,
                local_sqlite=self.local_sqlite,
                meta_agent_config_path=self._meta_config_path_optional(),
            )
            try:
                from mirix import LocalClient  # noqa: F401 — validatecan安装
            except ImportError as exc:
                raise ImportError(
                    "unable to import mirix (local pattern) . 请在 MIRIX-main directory执line pip install -e ."
                ) from exc
            self._client = _LocalMirixBridge(
                client_id=self.client_id,
                org_id=self.org_id,
                retrieve_skip_topic_extraction=self.retrieve_skip_topic_extraction,
            )
            return

        try:
            from mirix import MirixClient  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "unable to import mirix (remote pattern) . 请确认 MIRIX-main already pip install -e ."
            ) from exc

        kwargs: Dict[str, Any] = {
            "base_url": self.api_url,
            "write_scope": "read_write",
            "read_scopes": ["read_write"],
            "timeout": _cfg_int(self._cfg, "llm", "http_timeout_sec", default=120),
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.client_id:
            kwargs["client_id"] = self.client_id
        if self.org_id:
            kwargs["org_id"] = self.org_id
        _persistent_async_loop()
        self._client = MirixClient(**kwargs)

    def _meta_agent_update_on_build(self) -> bool:
        if self._update_meta_agent_on_build is not None:
            return bool(self._update_meta_agent_on_build)
        explicit = _cfg_get(self._cfg, "models", "mirix", "update_meta_agent_on_build")
        if explicit is not None:
            return _cfg_bool(self._cfg, "models", "mirix", "update_meta_agent_on_build", default=False)
        return False

    def _resolve_direct_episodic_build_embeddings(self) -> bool:
        if not self.direct_episodic_build_embeddings:
            return False
        try:
            yaml_flag = _meta_yaml_build_embeddings_flag(self._resolve_meta_config_path())
        except FileNotFoundError:
            yaml_flag = None
        if yaml_flag is False:
            return False
        return True

    def _resolve_meta_build_embeddings_for_ingest(self) -> bool:
        try:
            yaml_flag = _meta_yaml_build_embeddings_flag(self._resolve_meta_config_path())
        except FileNotFoundError:
            yaml_flag = None
        if yaml_flag is False:
            return False
        if yaml_flag is True:
            return True
        if self.ingest_mode == "direct_episodic":
            return self._resolve_direct_episodic_build_embeddings()
        # agent_add: YAML 未writewhendefault关 (avoidto :8000 打 embeddings.create → 404) 
        return False

    def _ensure_meta_agent(self) -> None:
        self._ensure_client()
        assert self._client is not None
        if getattr(self._client, "_meta_agent", None) is not None:
            return
        config_path = str(self._resolve_meta_config_path())
        update_agents = self._meta_agent_update_on_build()
        _run_async(
            self._client.initialize_meta_agent(
                config_path=config_path,
                update_agents=update_agents,
                eval_cfg=self._cfg,
            )
        )

    def _refresh_episodic_round_index(self) -> None:
        self._episodic_id_to_round = {}
        if self._client is None:
            return
        try:
            if self.mode == "local" and isinstance(self._client, _LocalMirixBridge):
                self._episodic_id_to_round = _run_async(
                    self._client.episodic_round_map(self._user_id, limit=2000)
                )
            elif self.mode == "remote":
                self._episodic_id_to_round = _load_episodic_round_map_pg(self._user_id)
        except Exception as exc:
            print(f"    [MIRIX] episodic round index 刷新failed: {type(exc).__name__}: {exc}", flush=True)

    def _count_episodic_total(self) -> int:
        if self._client is None:
            return 0
        try:
            if self.mode == "remote":
                list_fn = getattr(self._client, "list_memory_components", None)
                if callable(list_fn):
                    data = _run_async(
                        list_fn(
                            user_id=self._user_id,
                            memory_type="episodic",
                            limit=1,
                        )
                    )
                    return int(
                        ((data or {}).get("memories") or {}).get("episodic", {}).get("total_count", 0) or 0
                    )
            if self.mode == "local" and isinstance(self._client, _LocalMirixBridge):
                return _run_async(self._client.episodic_total_count(self._user_id))
        except Exception:
            pass
        return 0

    def _ingested_round_ids(self) -> Set[str]:
        if self.mode == "remote":
            return _pg_episodic_ingested_rounds(self._user_id)
        self._refresh_episodic_round_index()
        return {
            str(v).replace(" ", "")
            for v in self._episodic_id_to_round.values()
            if v
        }

    def _wait_round_ingested_pg(
        self, round_id: str, *, timeout_sec: Optional[int] = None
    ) -> bool:
        if self.mode != "remote":
            return True
        rid = str(round_id or "").replace(" ", "")
        if not rid:
            return False
        if timeout_sec is None:
            timeout_sec = (
                self.ingest_per_round_max_wait_sec
                if self.ingest_sequential_wait
                else self.ingest_confirm_per_round_sec
            )
        deadline = time.time() + float(max(10, int(timeout_sec)))
        poll = 1.5
        last_log = 0.0
        while time.time() < deadline:
            if rid in _pg_episodic_ingested_rounds(self._user_id):
                return True
            now = time.time()
            if now - last_log >= 30.0:
                elapsed = int(now - (deadline - float(timeout_sec)))
                print(
                    f"    [MIRIX] etc.pendinground {rid} persist… already {elapsed}s / {timeout_sec}s",
                    flush=True,
                )
                last_log = now
            time.sleep(poll)
        return False

    def _confirm_dialogue_ingest(
        self, round_id: str, ingest_log: Any, *, after_error_retry: bool = False
    ) -> bool:
        if self.mode != "remote":
            return True
        if not self.ingest_sequential_wait and not self.ingest_confirm_per_round:
            return True
        timeout = (
            self.ingest_per_round_max_wait_sec
            if self.ingest_sequential_wait
            else self.ingest_confirm_per_round_sec
        )
        if self._wait_round_ingested_pg(round_id, timeout_sec=timeout):
            return True
        suffix = " (剥imageretry后仍timeout) " if after_error_retry else ""
        label = "顺序ingest" if self.ingest_sequential_wait else "async ingest PG 确认"
        ingest_log(
            f"    [MIRIX] round {round_id} {label}timeout"
            f" (>{timeout}s) {suffix}"
        )
        return False

    def _ingest_needs_async_wait(self) -> bool:
        if self._expected_dialogue_turns <= 0:
            return False
        if self.add_use_async:
            return True
        return self.mode == "remote"

    def _wait_async_ingest(self, *, timeout_override: Optional[int] = None) -> None:
        if not self._ingest_needs_async_wait():
            return
        min_distinct = max(
            1,
            min(
                self._expected_dialogue_turns,
                max(8, int(self._expected_dialogue_turns * self.async_ingest_min_ratio)),
            ),
        )
        if timeout_override is not None:
            timeout = max(0, int(timeout_override))
        else:
            budget = _cfg_int(self._cfg, "models", "mirix", "async_ingest_wait_sec", default=300)
            max_cap = _cfg_int(
                self._cfg,
                "models",
                "mirix",
                "async_ingest_max_wait_sec",
                default=self.async_ingest_max_wait_sec,
            )
            timeout = max(budget, max_cap) if max_cap > 0 else budget
        if timeout <= 0:
            print(
                "    [MIRIX] already skippedasyncingestendetc.pending (async_ingest_wait_sec≤0 且 max≤0) ; "
                "完整性检查only反映when前 PG 快照",
                flush=True,
            )
            return
        poll = 3.0
        deadline = time.time() + timeout
        stable_need = 2
        stable = 0
        last_n = -1
        wait_pbar = _mirix_async_wait_progress_bar(min_distinct)
        if wait_pbar is None:
            print(
                f"    [MIRIX] etc.pendingasyncingestpersist (target去re-round≥{min_distinct}/"
                f"{self._expected_dialogue_turns}, 最长 {timeout}s) …",
                flush=True,
            )
        try:
            while time.time() < deadline:
                n = len(self._ingested_round_ids())
                if wait_pbar is not None:
                    wait_pbar.n = min(n, min_distinct)
                    wait_pbar.set_postfix_str(
                        f"{n}/{self._expected_dialogue_turns}",
                        refresh=False,
                    )
                    wait_pbar.refresh()
                if n >= min_distinct and n == last_n:
                    stable += 1
                else:
                    stable = 0
                last_n = n
                if n >= min_distinct and stable >= stable_need:
                    if wait_pbar is not None:
                        wait_pbar.n = min_distinct
                        wait_pbar.refresh()
                        wait_pbar.close()
                        wait_pbar = None
                    total = self._count_episodic_total()
                    print(
                        f"    [MIRIX] asyncingest 绪: distinct_rounds={n} episodic_rows={total}",
                        flush=True,
                    )
                    return
                time.sleep(poll)
        finally:
            if wait_pbar is not None:
                wait_pbar.close()
        n = len(self._ingested_round_ids())
        print(
            f"    [MIRIX] asyncingestetc.pendingend (distinct_rounds={n}/"
            f"{self._expected_dialogue_turns} episodic_rows="
            f"{self._count_episodic_total()}; 未达 {min_distinct} willtry补库) ",
            flush=True,
        )

    def _wait_async_ingest_after_build(self) -> None:
        if not self._ingest_needs_async_wait():
            return
        if self.ingest_sequential_wait:
            tail = _cfg_int(
                self._cfg, "models", "mirix", "async_ingest_tail_wait_sec", default=90
            )
            if tail > 0:
                print(
                    f"    [MIRIX] 顺序ingestalready完as逐轮etc.pending; receive尾稳定检查 {tail}s…",
                    flush=True,
                )
                self._wait_async_ingest(timeout_override=tail)
            return
        self._wait_async_ingest()

    def _retrieve_limit_for_question(self, top_k: int) -> int:
        floor = max(3, int(self.retrieve_min_limit))
        return max(top_k * self.retrieve_limit_scale, top_k, floor)

    def _search_supplement_hits(
        self, question: str, *, extra_queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if self._client is None or self.search_supplement_limit <= 0:
            return []
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        queries: List[str] = []
        supported_mt = frozenset({"episodic", "semantic", "resource", "all"})
        for q in [question, *(extra_queries or [])]:
            qq = str(q or "").strip()
            if qq and qq not in queries:
                queries.append(qq)
        for query in queries:
            for mt in self.search_memory_types:
                if mt not in supported_mt:
                    continue
                raw: Optional[Dict[str, Any]] = None
                try:
                    raw = _run_async(
                        self._client.search(
                            user_id=self._user_id,
                            query=query,
                            memory_type=mt,
                            search_method="embedding",
                            limit=self.search_supplement_limit,
                        )
                    )
                except Exception as exc:
                    if _is_embedding_api_not_found_error(exc):
                        print(
                            f"    [MIRIX] search 补充 ({mt}) embedding 404, time退 string_match"
                            f" (请确认 agent embedding_endpoint non-evaluation vLLM) : {exc}",
                            flush=True,
                        )
                        try:
                            raw = _run_async(
                                self._client.search(
                                    user_id=self._user_id,
                                    query=query,
                                    memory_type=mt,
                                    search_method="string_match",
                                    limit=self.search_supplement_limit,
                                )
                            )
                        except Exception as fallback_exc:
                            print(
                                f"    [MIRIX] search 补充 ({mt}) string_match alsofailed:"
                                f" {type(fallback_exc).__name__}: {fallback_exc}",
                                flush=True,
                            )
                            continue
                    else:
                        print(
                            f"    [MIRIX] search 补充 ({mt}) failed: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                if not isinstance(raw, dict) or not raw.get("success"):
                    continue
                for item in raw.get("results") or []:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("id") or "") or json.dumps(
                        item, sort_keys=True, default=str
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(item)
        return merged

    def _verify_index_integrity(self) -> Dict[str, Any]:
        self._refresh_episodic_round_index()
        expected = int(self._expected_dialogue_turns or 0)
        corpus_rounds = len(self._corpus_meta)
        episodic_n = self._count_episodic_total()
        ingested = self._ingested_round_ids()
        distinct_rounds = len(ingested)
        mapped = len(self._episodic_id_to_round)
        ratio = (distinct_rounds / expected) if expected > 0 else 0.0
        ok = expected <= 0 or ratio >= self.async_ingest_min_ratio
        expected_ids = {
            str(m.get("round") or "").replace(" ", "")
            for m in self._corpus_meta
            if str(m.get("round") or "").strip()
        }
        missing = sorted(expected_ids - ingested)
        report = {
            "ingest_mode": getattr(self, "ingest_mode", ""),
            "user_id": self._user_id,
            "expected_dialogue_turns": expected,
            "corpus_meta_rounds": corpus_rounds,
            "episodic_total_count": episodic_n,
            "episodic_distinct_rounds": distinct_rounds,
            "episodic_id_mapped": mapped,
            "missing_rounds": missing[:32],
            "missing_round_count": len(missing),
            "coverage_ratio": round(ratio, 4),
            "min_ratio_required": self.async_ingest_min_ratio,
            "ok": ok,
        }
        self.last_index_integrity = report
        miss_hint = ""
        if missing:
            sample = ", ".join(missing[:8])
            if len(missing) > 8:
                sample += f", ... (+{len(missing) - 8})"
            miss_hint = f" missing=[{sample}]"
        print(
            f"    [MIRIX] index完整性: user_id={self._user_id} "
            f"distinct_rounds={distinct_rounds}/{expected} (ratio={ratio:.2f}) "
            f"episodic_rows={episodic_n} round_map={mapped} corpus={corpus_rounds} "
            f"ok={ok}{miss_hint}",
            flush=True,
        )
        if not ok:
            print(
                "    [MIRIX] warning: 去re-roundoverride ratebelow async_ingest_min_ratio, "
                "retrieve/recall may偏low; can增大 async_ingest_wait_sec, "
                "设 add_use_async:false (走 add_sync) , or开启 ingest_retry_missing_sync. ",
                flush=True,
            )
        return report

    def _enforce_index_integrity_or_fail(self, report: Dict[str, Any]) -> None:
        if not self.ingest_to_memory or not self.fail_on_incomplete_ingest:
            return
        if report.get("ok"):
            return
        distinct = int(report.get("episodic_distinct_rounds") or 0)
        expected = int(report.get("expected_dialogue_turns") or 0)
        ratio = float(report.get("coverage_ratio") or 0.0)
        min_ratio = float(report.get("min_ratio_required") or self.async_ingest_min_ratio)
        missing_n = int(report.get("missing_round_count") or 0)
        sample = report.get("missing_rounds") or []
        sample_s = ", ".join(str(x) for x in sample[:8])
        if missing_n > len(sample):
            sample_s += f", ... (+{missing_n - len(sample)})"
        raise MirixIngestIncompleteError(
            "MIRIX asyncingest未达override rate门槛, already 止evaluation (未进入 [2/2] answering) . "
            f" distinct_rounds={distinct}/{expected} (ratio={ratio:.2f}, "
            f"需要≥{min_ratio:.2f}). "
            f" missing失round {missing_n}  , example: [{sample_s}]. "
            " can增大 async_ingest_max_wait_sec, docker restart mirix_api 清queue, "
            "减小 --max-sessions, or设 fail_on_incomplete_ingest: false onlydebug. "
        )

    def _retrieve_raw(self, question: str, *, limit: int) -> Dict[str, Any]:
        assert self._client is not None
        skip_types = self._retrieve_skip_memory_types
        retrieve_kw: Dict[str, Any] = {
            "user_id": self._user_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                }
            ],
            "limit": limit,
        }
        if skip_types and isinstance(self._client, _LocalMirixBridge):
            retrieve_kw["skip_memory_types"] = skip_types
        return _run_async(self._client.retrieve_with_conversation(**retrieve_kw))

    def _rounds_from_retrieve_raw(self, raw: Dict[str, Any], ctx: str) -> List[str]:
        rounds = rounds_from_lightrag_context(ctx)
        if not rounds and isinstance(raw, dict):
            for item in _collect_episodic_items(raw):
                rid = _item_round_id(
                    item,
                    id_to_round=self._episodic_id_to_round,
                    corpus_meta=self._corpus_meta,
                )
                if rid:
                    rounds.append(rid)
        return rounds

    def _ingest_ark_text_only(self) -> bool:
        return bool(self._memory_llm_is_ark and self.ingest_text_only_on_ark)

    def _answer_multimodal_like_mirix(self) -> bool:
        return bool(self.no_ingest_multimodal) and self.answer_mode in (
            "mirix_hybrid",
            "rag_style",
        )

    def _retrieve_attach_dialogue_images(self) -> bool:
        if self._answer_multimodal_like_mirix():
            return True
        return self.answer_mode in ("mirix_hybrid", "rag_style")

    def _effective_ingest_dialogue_images(self) -> bool:
        if self._ingest_ark_text_only():
            return False
        return bool(self.ingest_dialogue_images)

    def _effective_ingest_pdf_native(self) -> bool:
        if self._ingest_ark_text_only():
            return False
        return bool(self.ingest_pdf_native)

    def _use_pdf_text_in_episodic(self) -> bool:
        return bool(
            self.ingest_pdf_episodic_text
            or self.index_pdf_via_ocr
            or self._ingest_ark_text_only()
            or self.ingest_mode == "direct_episodic"
        )

    def _should_skip_local_pdf_excerpts(self) -> bool:
        if self.no_ingest_multimodal:
            return False
        if self._use_pdf_text_in_episodic():
            return False
        return (
            self.retrieve_mode == "pure_mirix"
            and self._effective_ingest_pdf_native()
        )

    def _index_pdf_policy(self) -> str:
        pol = str(self.pdf_policy or "native_then_ocr")
        if self.index_pdf_via_ocr and pol == "native_only":
            return "native_then_ocr"
        return pol

    @staticmethod
    def _occurred_at_from_session_date(date: str) -> Optional[datetime]:
        ds = str(date or "").strip()[:19]
        if not ds:
            return None
        if "T" not in ds:
            ds = ds + "T12:00:00"
        if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ds):
            return None
        try:
            return datetime.fromisoformat(ds)
        except ValueError:
            return None

    def _direct_episodic_summary(self, user_t: str, asst_t: str, details: str) -> str:
        cap = self.direct_episodic_summary_max_chars
        for candidate in (asst_t.strip(), user_t.strip(), details.strip()):
            if candidate:
                return candidate[:cap]
        return details[:cap]

    def _sessions_fingerprint(self, sessions: List[Dict[str, Any]]) -> str:
        key_src = json.dumps(
            [s.get("session_id") for s in sessions], ensure_ascii=False
        )
        return hashlib.sha256(key_src.encode("utf-8")).hexdigest()

    def _load_completed_rounds_from_checkpoint(
        self, sessions: List[Dict[str, Any]]
    ) -> Tuple[Set[str], bool]:
        path = self.index_checkpoint_path
        if path is None or not self.resume_index:
            return set(), False
        ck = load_index_progress_file(Path(path))
        if not ck:
            return set(), False
        fp = self._sessions_fingerprint(sessions)
        if ck.get("sessions_fingerprint") != fp:
            print(
                "    [MIRIX] index progressandwhen前 sessions listnot一致, willfrom头建index",
                flush=True,
            )
            return set(), False
        if ck.get("ingest_mode") and ck.get("ingest_mode") != self.ingest_mode:
            print(
                f"    [MIRIX] index progress ingest_mode={ck.get('ingest_mode')!r} "
                f"andwhen前 {self.ingest_mode!r} not一致, willfrom头建index",
                flush=True,
            )
            return set(), False
        ver = int(ck.get("version") or 0)
        completed = {str(x) for x in (ck.get("completed_round_ids") or []) if x}
        if not completed and ver < 2 and ck.get("completed_session_ids"):
            print(
                "    [MIRIX] 检测tolegacy session 级 checkpoint, unable to按 round 续run, willfrom头建index",
                flush=True,
            )
            return set(), False
        index_complete = bool(ck.get("index_complete"))
        total_rounds = int(ck.get("total_rounds") or 0)
        if completed and not index_complete:
            print(
                f"    [MIRIX] 续建index: already完as {len(completed)}"
                f"{f'/{total_rounds}' if total_rounds else ''}   round",
                flush=True,
            )
        elif index_complete:
            print(
                f"    [MIRIX] indexalreadyall完as ({len(completed)}   round) , skipingest",
                flush=True,
            )
        if ck.get("pdf_episodic_patched"):
            self._pdf_episodic_patched = True
        return completed, index_complete

    def _load_ingest_anomalies_from_checkpoint(
        self, sessions: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        path = self.index_checkpoint_path
        if path is None or not self.resume_index:
            return {}
        ck = load_index_progress_file(Path(path))
        if not ck:
            return {}
        if ck.get("sessions_fingerprint") != self._sessions_fingerprint(sessions):
            return {}
        if ck.get("ingest_mode") and ck.get("ingest_mode") != self.ingest_mode:
            return {}
        raw = ck.get("anomaly_rounds") or {}
        if not isinstance(raw, dict):
            return {}
        out = {
            str(k).strip(): str(v)
            for k, v in raw.items()
            if str(k).strip() and str(v).strip()
        }
        if out:
            print(
                f"    [MIRIX] 续run: 载入 {len(out)}  pendingretryexception round"
                f" (example: {', '.join(sorted(out)[:6])}"
                f"{'...' if len(out) > 6 else ''}) ",
                flush=True,
            )
        return out

    def _save_ingest_anomalies_sidecar(self) -> None:
        path = self.index_checkpoint_path
        if path is None:
            return
        anomalies = self._ingest_anomaly_rounds or {}
        sidecar = Path(path).parent / "ingest_anomalies.json"
        if not anomalies:
            if sidecar.is_file():
                try:
                    sidecar.unlink()
                except OSError:
                    pass
            return
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "anomaly_rounds": dict(sorted(anomalies.items())),
            "count": len(anomalies),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_index_round_checkpoint(
        self,
        sessions: List[Dict[str, Any]],
        completed_round_ids: Set[str],
        *,
        total_rounds: int,
        index_complete: bool = False,
    ) -> None:
        path = self.index_checkpoint_path
        if path is None:
            return
        anomalies = dict(self._ingest_anomaly_rounds or {})
        prev = load_index_progress_file(Path(path)) or {}
        payload: Dict[str, Any] = {
            "version": INDEX_PROGRESS_VERSION,
            "sessions_fingerprint": self._sessions_fingerprint(sessions),
            "user_id": self._user_id,
            "ingest_mode": self.ingest_mode,
            "completed_round_ids": sorted(completed_round_ids),
            "completed_count": len(completed_round_ids),
            "total_rounds": total_rounds,
            "total_sessions": len(sessions),
            "index_complete": index_complete,
            "pdf_episodic_patched": bool(
                self._pdf_episodic_patched or prev.get("pdf_episodic_patched")
            ),
            "anomaly_rounds": anomalies,
            "anomaly_count": len(anomalies),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_index_progress_file(Path(path), payload)
        self._save_ingest_anomalies_sidecar()

    def _finish_round_ingest_checkpoint(
        self,
        sessions: List[Dict[str, Any]],
        completed_rounds: Set[str],
        round_id: str,
        *,
        total_rounds: int,
    ) -> None:
        rid = str(round_id or "").strip()
        if not rid:
            return
        completed_rounds.add(rid)
        self._save_index_round_checkpoint(
            sessions,
            completed_rounds,
            total_rounds=total_rounds,
            index_complete=False,
        )
        print(
            f"    [MIRIX] round {rid} index progressalreadysave "
            f"({len(completed_rounds)}/{total_rounds}) → {self.index_checkpoint_path}",
            flush=True,
        )

    def _ensure_corpus_meta_for_dialogue(
        self, sess: Dict[str, Any], dlg: Dict[str, Any]
    ) -> None:
        sid = str(sess.get("session_id", ""))
        rid = str(dlg.get("round", "") or "")
        for meta in self._corpus_meta:
            if meta.get("session_id") == sid and str(meta.get("round") or "") == rid:
                return
        self._append_corpus_meta_for_dialogue(sess, dlg)

    def _append_corpus_meta_for_dialogue(
        self, sess: Dict[str, Any], dlg: Dict[str, Any], *, ingest_pbar: Any = None
    ) -> None:
        sid = str(sess.get("session_id", ""))
        date = str(sess.get("date", "") or "")
        rid = dlg.get("round", "")
        user_t = str(dlg.get("user", "") or "")
        asst_t = str(dlg.get("assistant", "") or "")
        img_names = [Path(str(x)).name for x in (dlg.get("img_file") or [])]
        self._expected_dialogue_turns += 1
        self._corpus_meta.append(
            {
                "session_id": sid,
                "date": date,
                "round": str(rid),
                "user": user_t,
                "assistant": asst_t,
                "img_files": img_names,
                "dialogue_vis": dialogue_image_description(dlg),
            }
        )
        if ingest_pbar is not None:
            ingest_pbar.update(1)

    def _replay_session_corpus_only(
        self, sess: Dict[str, Any], *, ingest_pbar: Any = None
    ) -> None:
        for dlg in sess.get("dialogues") or []:
            self._append_corpus_meta_for_dialogue(sess, dlg, ingest_pbar=ingest_pbar)

    def _hydrate_corpus_meta_all_sessions(
        self, sessions: List[Dict[str, Any]], *, ingest_pbar: Any = None
    ) -> None:
        for sess in sessions:
            self._replay_session_corpus_only(sess, ingest_pbar=ingest_pbar)

    def _build_dialogue_ingest_body(
        self,
        *,
        sid: str,
        dlg: Dict[str, Any],
        user_t: str,
        asst_t: str,
        img_names: List[str],
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
        seen_session: set[str],
        sessions_pdf_fallback: set[str],
        sess: Dict[str, Any],
        pdfs_dir: Path,
        pdf_policy: str,
    ) -> str:
        img_line = ", ".join(img_names) if img_names else ""
        desc = dialogue_image_description(dlg)
        body = f"User:\n{user_t}\n\nAssistant:\n{asst_t}"
        if desc:
            body += f"\n\n[Image description]\n{desc}"
        if img_line:
            body += f"\n\n[IMAGE_FILES] {img_line}"
        if sid not in seen_session:
            seen_session.add(sid)
            use_pdf_episodic = (
                not self.ingest_defer_session_pdf_in_dialogue
                and (
                    self._use_pdf_text_in_episodic()
                    or (
                        self.ingest_pdf_episodic_fallback
                        and sid in sessions_pdf_fallback
                    )
                )
            )
            if (
                use_pdf_episodic
                and sid in sessions_pdf_fallback
                and sid not in pdf_snip
                and pdfs_dir.is_dir()
            ):
                self._lazy_pdf_excerpt_for_session(
                    sess, pdfs_dir, pdf_policy, pdf_snip, pdf_ctx_blocks
                )
            if use_pdf_episodic:
                ps = pdf_snip.get(sid, "")
                if ps:
                    body = f"[Session PDF snippet]\n{ps}\n\n---\n" + body
                pb = pdf_ctx_blocks.get(sid, "")
                if pb:
                    body = (
                        f"[Session PDF context]\n"
                        f"{pb[: self.context_pdf_max_chars]}\n\n---\n"
                        + body
                    )
        return body

    def _dialogue_user_content(
        self,
        body: str,
        dlg: Dict[str, Any],
        images_dir: Path,
        pdfs_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        del pdfs_dir  # PDF 在will话级 _ingest_native_pdf_file ingest, notduplicatelicatehunground
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": body}]
        if self._effective_ingest_dialogue_images():
            img_files = [str(x) for x in (dlg.get("img_file") or []) if str(x).strip()]
            cap = self.ingest_max_dialogue_images
            if len(img_files) > cap:
                body = (
                    body
                    + f"\n\n[MIRIX ingest] Dialogue images capped at {cap}/{len(img_files)} "
                    f"(vLLM limit); omitted: {', '.join(img_files[cap:])}"
                )
                user_content[0] = {"type": "text", "text": body}
                img_files = img_files[:cap]
            for rel in img_files:
                p = (images_dir / rel).resolve()
                # local/remote alluse base64 image_url, avoid file_uri 在 convert when走
                # _save_image_from_file_uri → SQLite (易 greenlet_spawn / database is locked) 
                part = _local_image_to_remote_image_url_part(p)
                if part:
                    user_content.append(part)
        return user_content

    def _ingest_native_pdf_file(
        self,
        pdf_path: Path,
        *,
        session_id: str,
        round_ref: str,
        date: str,
        ingest_log,
    ) -> None:
        assert self._client is not None
        key = str(pdf_path.resolve())
        if key in self._native_pdfs_ingested:
            return
        part = _pdf_ingest_content_part(pdf_path, mode=self.mode)
        if not part:
            ingest_log(
                f"    [MIRIX] skip PDF ingest (missing失ortoo large) : {pdf_path.name}"
            )
            return
        body = (
            f"[MIRIX PDF resource ingest] session={session_id} "
            f"round_ref={round_ref} file={pdf_path.name}"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": body}, part],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "(pdf stored for resource memory)"}],
            },
        ]
        add_kw: Dict[str, Any] = {
            "user_id": self._user_id,
            "messages": messages,
            "chaining": self.add_chaining,
            "filter_tags": {
                "session_id": session_id,
                "round": round_ref,
                "pdf_file": pdf_path.name,
                "ingest_kind": "pdf_resource",
            },
            "async_add": self.add_use_async,
        }
        add_headers: Optional[Dict[str, str]] = None
        if self.mode == "remote" and self.org_id:
            add_headers = {"X-Org-ID": self.org_id}
        if date:
            ds = str(date).strip()[:19]
            if "T" not in ds:
                ds = ds + "T12:00:00"
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ds):
                add_kw["occurred_at"] = ds
        try:
            if add_headers:
                _run_async(self._client.add(**add_kw, headers=add_headers))
            else:
                _run_async(self._client.add(**add_kw))
            self._native_pdfs_ingested.add(key)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if _ingest_retry_text_only(err, exc):
                ingest_log(
                    f"    [MIRIX] PDF {pdf_path.name} 多模态ingestfailed, time退纯text resource…"
                )
                try:
                    add_kw["messages"] = self._strip_binary_content(messages)
                    if add_headers:
                        _run_async(self._client.add(**add_kw, headers=add_headers))
                    else:
                        _run_async(self._client.add(**add_kw))
                    self._native_pdfs_ingested.add(key)
                    return
                except Exception:
                    pass
            self._native_pdf_ingest_failures.append(pdf_path.name)
            self._sessions_pdf_text_fallback.add(session_id)
            ingest_log(
                f"    [MIRIX] PDF resource add failed {pdf_path.name}: {err}"
            )
            traceback.print_exc()

    @staticmethod
    def _strip_binary_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                text_only = [c for c in content if isinstance(c, dict) and c.get("type") == "text"]
                out.append({**msg, "content": text_only or content})
            else:
                out.append(msg)
        return out

    def _retry_missing_episodic_sync(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        pdfs_dir: Path,
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
    ) -> None:
        if not self.ingest_retry_missing_sync or self.mode != "remote":
            return
        assert self._client is not None
        ingested = self._ingested_round_ids()
        expected_ids = {
            str(m.get("round") or "").replace(" ", "")
            for m in self._corpus_meta
            if str(m.get("round") or "").strip()
        }
        missing = sorted(expected_ids - ingested)
        if not missing:
            return
        print(
            f"    [MIRIX] to {len(missing)}  missing失round顺序补ingest"
            f" (每轮etc.pending PG≤{self.ingest_per_round_max_wait_sec}s; "
            f"example: {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}) ",
            flush=True,
        )
        missing_set = set(missing)
        seen_session: set[str] = set()
        ok = 0
        for sess in sessions:
            sid = str(sess.get("session_id", ""))
            date = str(sess.get("date", "") or "")
            for dlg in sess.get("dialogues") or []:
                rid = str(dlg.get("round", "")).replace(" ", "")
                if not rid or rid not in missing_set:
                    continue
                if rid in self._ingested_round_ids():
                    continue
                user_t = str(dlg.get("user", "") or "")
                asst_t = str(dlg.get("assistant", "") or "")
                img_names = [Path(str(x)).name for x in (dlg.get("img_file") or [])]
                desc = dialogue_image_description(dlg)
                body = f"User:\n{user_t}\n\nAssistant:\n{asst_t}"
                if desc:
                    body += f"\n\n[Image description]\n{desc}"
                if img_names:
                    body += "\n\n[IMAGE_FILES] " + ", ".join(img_names)
                if sid not in seen_session:
                    seen_session.add(sid)
                    use_pdf_episodic = self._use_pdf_text_in_episodic() or (
                        self.ingest_pdf_episodic_fallback
                        and sid in self._sessions_pdf_text_fallback
                    )
                    if use_pdf_episodic:
                        ps = pdf_snip.get(sid, "")
                        if ps:
                            body = f"[Session PDF snippet]\n{ps}\n\n---\n" + body
                        pb = pdf_ctx_blocks.get(sid, "")
                        if pb:
                            body = (
                                f"[Session PDF context]\n"
                                f"{pb[: self.context_pdf_max_chars]}\n\n---\n"
                                + body
                            )
                body = prepend_eval_round_trace(
                    body,
                    round_id=rid,
                    chunk_id=None,
                    image_basenames=img_names or None,
                )
                user_content = self._dialogue_user_content(
                    body, dlg, images_dir, pdfs_dir if pdfs_dir.is_dir() else None
                )
                messages = [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "(stored turn)"}],
                    },
                ]
                add_kw: Dict[str, Any] = {
                    "user_id": self._user_id,
                    "messages": messages,
                    "chaining": self.add_chaining,
                    "filter_tags": {"session_id": sid, "round": rid},
                    "async_add": True,
                }
                if date:
                    ds = str(date).strip()[:19]
                    if "T" not in ds:
                        ds = ds + "T12:00:00"
                    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ds):
                        add_kw["occurred_at"] = ds
                try:
                    _run_async(self._client.add(**add_kw))
                    if self._confirm_dialogue_ingest(rid, print):
                        ok += 1
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    if _ingest_retry_text_only(err, exc):
                        print(
                            f"    [MIRIX] round={rid} 多模态补ingestfailed, time退纯text…",
                            flush=True,
                        )
                        try:
                            add_kw["messages"] = self._strip_binary_content(messages)
                            _run_async(self._client.add(**add_kw))
                            ok += 1
                            continue
                        except Exception:
                            pass
                    print(
                        f"    [MIRIX] async 补ingestfailed round={rid}: {err}",
                        flush=True,
                    )
        if ok:
            print(f"    [MIRIX] 顺序补ingestpersistsuccess: {ok}/{len(missing)} 轮", flush=True)

    def _round_in_episodic_index(self, round_id: str) -> bool:
        rid = str(round_id or "").replace(" ", "")
        if not rid:
            return False
        return rid in self._ingested_round_ids()

    def _ingest_dialogue_round_add(
        self,
        *,
        sess: Dict[str, Any],
        dlg: Dict[str, Any],
        images_dir: Path,
        pdfs_dir: Path,
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
        pdf_policy: str,
        seen_session: set[str],
        sessions_pdf_fallback: set[str],
        ingest_log: Any,
        text_only: bool,
    ) -> Tuple[bool, Optional[str]]:
        assert self._client is not None
        sid = str(sess.get("session_id", ""))
        date = str(sess.get("date", "") or "")
        rid = str(dlg.get("round", "") or "").strip()
        user_t = str(dlg.get("user", "") or "")
        asst_t = str(dlg.get("assistant", "") or "")
        img_names = [Path(str(x)).name for x in (dlg.get("img_file") or [])]
        body = self._build_dialogue_ingest_body(
            sid=sid,
            dlg=dlg,
            user_t=user_t,
            asst_t=asst_t,
            img_names=img_names,
            pdf_snip=pdf_snip,
            pdf_ctx_blocks=pdf_ctx_blocks,
            seen_session=seen_session,
            sessions_pdf_fallback=sessions_pdf_fallback,
            sess=sess,
            pdfs_dir=pdfs_dir,
            pdf_policy=pdf_policy,
        )
        body = prepend_eval_round_trace(
            body,
            round_id=rid,
            chunk_id=None,
            image_basenames=img_names or None,
        )
        user_content = self._dialogue_user_content(
            body, dlg, images_dir, pdfs_dir if pdfs_dir.is_dir() else None
        )
        messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "(stored turn)"}],
            },
        ]
        if text_only:
            messages = self._strip_binary_content(messages)
        filter_tags = {"session_id": sid, "round": rid}
        add_kw: Dict[str, Any] = {
            "user_id": self._user_id,
            "messages": messages,
            "chaining": self.add_chaining,
            "filter_tags": filter_tags,
            "async_add": self.add_use_async,
        }
        if date:
            ds = str(date).strip()[:19]
            if "T" not in ds:
                ds = ds + "T12:00:00"
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ds):
                add_kw["occurred_at"] = ds
        try:
            _run_async(self._client.add(**add_kw))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            reason = (
                INGEST_ANOMALY_CONTENT_FILTER
                if _is_content_filter_error(exc)
                else INGEST_ANOMALY_ADD_FAILED
            )
            ingest_log(
                f"    [MIRIX] round={rid} add failed"
                f"{' (纯text) ' if text_only else ''}: {err}"
            )
            return False, reason
        if not self._confirm_dialogue_ingest(rid, ingest_log):
            return False, INGEST_ANOMALY_CONFIRM_TIMEOUT
        if not self._round_in_episodic_index(rid):
            return False, INGEST_ANOMALY_MISSING_EPISODIC
        return True, None

    def _retry_anomaly_rounds_at_end(
        self,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        pdfs_dir: Path,
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
        *,
        ingest_log: Any,
    ) -> None:
        if not self.ingest_retry_anomaly_at_end:
            return
        if self.ingest_mode != "agent_add" or not self.ingest_to_memory:
            return
        pending = {
            str(k).strip(): str(v)
            for k, v in (self._ingest_anomaly_rounds or {}).items()
            if str(k).strip()
        }
        if not pending:
            return
        assert self._client is not None
        pdf_policy = resolve_pdf_policy(
            "mirix",
            self._cfg,
            llm_model=self.llm_model,
            probe_native_pdf=self._probe_native_pdf,
        )
        self._refresh_episodic_round_index()
        sample = ", ".join(sorted(pending)[:8])
        print(
            f"    [MIRIX] exception round 统一retry: {len(pending)}  "
            f" ({sample}{'...' if len(pending) > 8 else ''}) ",
            flush=True,
        )
        ingest_log_ctx: Dict[str, str] = {"current_round_id": ""}
        log_handler = _IngestAnomalyLogHandler(self._ingest_anomaly_rounds, ingest_log_ctx)
        mirix_logger = logging.getLogger("Mirix")
        mirix_logger.addHandler(log_handler)
        seen_session: set[str] = set()
        ok_rounds: List[str] = []
        try:
            for sess in sessions:
                for dlg in sess.get("dialogues") or []:
                    rid = str(dlg.get("round", "") or "").strip()
                    if not rid or rid not in pending:
                        continue
                    reason = pending[rid]
                    ingest_log_ctx["current_round_id"] = rid
                    ingest_log(
                        f"    [MIRIX] retry round={rid} (原因={reason}) …"
                    )
                    if self._round_in_episodic_index(rid) and reason not in (
                        INGEST_ANOMALY_CONTENT_FILTER,
                        INGEST_ANOMALY_CONFIRM_TIMEOUT,
                    ):
                        self._ingest_anomaly_rounds.pop(rid, None)
                        ok_rounds.append(rid)
                        ingest_log(
                            f"    [MIRIX] round={rid} already在 episodic  , skipexceptionretry"
                        )
                        continue
                    ok_text, fail_text = self._ingest_dialogue_round_add(
                        sess=sess,
                        dlg=dlg,
                        images_dir=images_dir,
                        pdfs_dir=pdfs_dir,
                        pdf_snip=pdf_snip,
                        pdf_ctx_blocks=pdf_ctx_blocks,
                        pdf_policy=pdf_policy,
                        seen_session=seen_session,
                        sessions_pdf_fallback=self._sessions_pdf_text_fallback,
                        ingest_log=ingest_log,
                        text_only=True,
                    )
                    if ok_text and rid not in self._ingest_anomaly_rounds:
                        self._ingest_anomaly_rounds.pop(rid, None)
                        ok_rounds.append(rid)
                        ingest_log(f"    [MIRIX] round={rid} 纯textretrysuccess")
                        continue
                    if ok_text and self._round_in_episodic_index(rid):
                        if reason == INGEST_ANOMALY_CONTENT_FILTER:
                            ingest_log(
                                f"    [MIRIX] round={rid} 纯textalreadypersist"
                                f" (仍记录 content_filter, skip多模态二times add) "
                            )
                            self._ingest_anomaly_rounds.pop(rid, None)
                            ok_rounds.append(rid)
                            continue
                    ok_full, fail_full = self._ingest_dialogue_round_add(
                        sess=sess,
                        dlg=dlg,
                        images_dir=images_dir,
                        pdfs_dir=pdfs_dir,
                        pdf_snip=pdf_snip,
                        pdf_ctx_blocks=pdf_ctx_blocks,
                        pdf_policy=pdf_policy,
                        seen_session=seen_session,
                        sessions_pdf_fallback=self._sessions_pdf_text_fallback,
                        ingest_log=ingest_log,
                        text_only=False,
                    )
                    if ok_full:
                        self._ingest_anomaly_rounds.pop(rid, None)
                        ok_rounds.append(rid)
                        ingest_log(f"    [MIRIX] round={rid} 多模态retrysuccess")
                    else:
                        new_reason = fail_full or fail_text or reason
                        _record_ingest_anomaly(
                            self._ingest_anomaly_rounds, rid, new_reason
                        )
                        ingest_log(
                            f"    [MIRIX] round={rid} retry仍failed ({new_reason}) "
                        )
        finally:
            ingest_log_ctx["current_round_id"] = ""
            mirix_logger.removeHandler(log_handler)
        still = len(self._ingest_anomaly_rounds or {})
        print(
            f"    [MIRIX] exception round retryend: success {len(ok_rounds)}, "
            f"仍pendingprocess {still}",
            flush=True,
        )

    def _lazy_pdf_excerpt_for_session(
        self,
        sess: Dict[str, Any],
        pdfs_dir: Path,
        pdf_policy: str,
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
    ) -> None:
        sid = str(sess.get("session_id", ""))
        if not sid or pdf_snip.get(sid):
            return
        one_snip = build_session_pdf_snippets(
            [sess],
            pdfs_dir,
            policy=pdf_policy,  # type: ignore[arg-type]
            embed_max_chars=self.embed_pdf_max_chars,
            min_native_chars=self.min_native_pdf_chars,
        )
        one_ctx = build_session_pdf_context_blocks(
            [sess],
            pdfs_dir,
            policy=pdf_policy,  # type: ignore[arg-type]
            context_max_chars=self.context_pdf_max_chars,
            min_native_chars=self.min_native_pdf_chars,
        )
        pdf_snip.update(one_snip)
        pdf_ctx_blocks.update(one_ctx)

    def _pdf_index_cache_path(self) -> Optional[Path]:
        if self.index_checkpoint_path is None:
            return None
        return Path(self.index_checkpoint_path).parent / "pdf_index_cache.json"

    def _load_pdf_index_cache(
        self, sessions: List[Dict[str, Any]], pdf_policy: str
    ) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
        path = self._pdf_index_cache_path()
        if path is None:
            return None
        ck = load_pdf_index_cache_file(path)
        if not ck:
            return None
        if int(ck.get("version") or 0) != PDF_INDEX_CACHE_VERSION:
            return None
        if ck.get("sessions_fingerprint") != self._sessions_fingerprint(sessions):
            return None
        if str(ck.get("pdf_policy") or "") != str(pdf_policy):
            return None
        snip = ck.get("pdf_snippets")
        ctx = ck.get("pdf_ctx_blocks")
        if not isinstance(snip, dict) or not isinstance(ctx, dict):
            return None
        print(
            f"    [MIRIX] reuse PDF indexcache {path.name}"
            f" ({len(snip)}   session, skip OCR) ",
            flush=True,
        )
        return (
            {str(k): str(v) for k, v in snip.items()},
            {str(k): str(v) for k, v in ctx.items()},
        )

    def _save_pdf_index_cache(
        self,
        sessions: List[Dict[str, Any]],
        pdf_policy: str,
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
    ) -> None:
        path = self._pdf_index_cache_path()
        if path is None or not pdf_snip:
            return
        save_pdf_index_cache_file(
            path,
            {
                "version": PDF_INDEX_CACHE_VERSION,
                "sessions_fingerprint": self._sessions_fingerprint(sessions),
                "pdf_policy": pdf_policy,
                "pdf_snippets": pdf_snip,
                "pdf_ctx_blocks": pdf_ctx_blocks,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def _patch_pdf_episodic_index(
        self,
        sessions: List[Dict[str, Any]],
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
        *,
        ingest_log: Any = None,
    ) -> int:
        if not self.ingest_to_memory or not self._use_pdf_text_in_episodic():
            return 0
        if self.ingest_mode != "agent_add":
            print(
                "    [MIRIX] reindex_pdf_episodic_only only支持 ingest_mode=agent_add",
                flush=True,
            )
            return 0
        assert self._client is not None

        def _log(msg: str) -> None:
            if ingest_log is not None:
                ingest_log(msg)
            else:
                print(msg, flush=True)

        patched = 0
        for sess in sessions:
            sid = str(sess.get("session_id", "") or "").strip()
            if not sid:
                continue
            ps = (pdf_snip.get(sid) or "").strip()
            pb = (pdf_ctx_blocks.get(sid) or "").strip()
            if not ps and not pb:
                continue
            chunks: List[str] = []
            if ps:
                chunks.append(f"[Session PDF snippet]\n{ps}")
            if pb:
                chunks.append(
                    f"[Session PDF context]\n{pb[: self.context_pdf_max_chars]}"
                )
            body = "\n\n---\n".join(chunks)
            round_ref = f"{sid}:pdf_index"
            body = prepend_eval_round_trace(
                body,
                round_id=round_ref,
                chunk_id=None,
                image_basenames=None,
            )
            messages = [
                {"role": "user", "content": [{"type": "text", "text": body}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "(pdf ocr index patch)"}],
                },
            ]
            add_kw: Dict[str, Any] = {
                "user_id": self._user_id,
                "messages": messages,
                "chaining": False,
                "filter_tags": {"session_id": sid, "round": round_ref},
                "async_add": self.add_use_async,
            }
            date = str(sess.get("date", "") or "")
            occurred_dt = self._occurred_at_from_session_date(date)
            if occurred_dt is not None:
                add_kw["occurred_at"] = occurred_dt
            try:
                _run_async(self._client.add(**add_kw))
                patched += 1
                _log(f"    [MIRIX] PDF index补丁 session={sid} round={round_ref}")
            except Exception as exc:
                _log(
                    f"    [MIRIX] PDF index补丁failed session={sid}: "
                    f"{type(exc).__name__}: {exc}"
                )
        if patched:
            print(
                f"    [MIRIX] PDF episodic 补丁完as: {patched}/{len(sessions)}   session"
                " (未re-runto话 round) ",
                flush=True,
            )
        return patched

    def _needs_pdf_episodic_patch(
        self,
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
        *,
        checkpoint_pdf_patched: Optional[bool] = None,
    ) -> bool:
        if checkpoint_pdf_patched is True or self._pdf_episodic_patched:
            return False
        if not self.ingest_to_memory or not self._use_pdf_text_in_episodic():
            return False
        if not pdf_snip and not pdf_ctx_blocks:
            return False
        if self.reindex_pdf_episodic_only:
            return True
        if (
            self.ingest_defer_session_pdf_in_dialogue
            and self.ingest_pdf_patch_after_index
        ):
            return True
        return False

    def _maybe_patch_pdf_episodic_after_index(
        self,
        sessions: List[Dict[str, Any]],
        completed_rounds: Set[str],
        total_turns: int,
        pdf_snip: Dict[str, str],
        pdf_ctx_blocks: Dict[str, str],
        *,
        ingest_log: Any = None,
        checkpoint_pdf_patched: Optional[bool] = None,
    ) -> None:
        if not self._needs_pdf_episodic_patch(
            pdf_snip, pdf_ctx_blocks, checkpoint_pdf_patched=checkpoint_pdf_patched
        ):
            return
        n_sess = sum(
            1
            for s in sessions
            if (pdf_snip.get(str(s.get("session_id", "") or "")) or "").strip()
            or (pdf_ctx_blocks.get(str(s.get("session_id", "") or "")) or "").strip()
        )
        print(
            "    [MIRIX] to话 round alreadyallingest; begin PDF → episodic 补丁"
            f" (约 {n_sess}   session, non- {total_turns} 轮; "
            "answering仍canuse vision_pages look atimage) ",
            flush=True,
        )
        n_patch = self._patch_pdf_episodic_index(
            sessions, pdf_snip, pdf_ctx_blocks, ingest_log=ingest_log
        )
        if n_patch > 0:
            self._pdf_episodic_patched = True
            self._save_index_round_checkpoint(
                sessions,
                completed_rounds,
                total_rounds=total_turns,
                index_complete=True,
            )

    def build_index(self, sessions: List[Dict[str, Any]], images_dir: Path) -> None:
        if self.ingest_to_memory:
            emb_effective = self._resolve_meta_build_embeddings_for_ingest()
            if self.ingest_mode == "direct_episodic":
                self._direct_episodic_embeddings_effective = emb_effective
            _set_build_embeddings_for_memory(emb_effective)
            if not emb_effective:
                print(
                    "    [MIRIX] BUILD_EMBEDDINGS_FOR_MEMORY=false"
                    " (memoryingestnot算向量 embedding; retrieveuse BM25/already have episodic) ",
                    flush=True,
                )
            if self.memory_llm_preflight and self.ingest_mode == "agent_add":
                _preflight_memory_llm(self._resolve_meta_config_path(), eval_cfg=self._cfg)
            self._ensure_meta_agent()
            assert self._client is not None
            if self.ingest_mode == "direct_episodic":
                emb_on = self._direct_episodic_embeddings_effective
                emb_note = ""
                if self.direct_episodic_build_embeddings and not emb_on:
                    emb_note = " (meta YAML build_embeddings_for_memory=false alreadyoverride) "
                print(
                    "    [MIRIX] 档  B direct_episodic: skip add()/memory agent, "
                    f"直connect insert_event → SQLite (embedding={'on' if emb_on else 'off'}{emb_note}) ; "
                    "retrieve走官方 retrieve_with_conversation",
                    flush=True,
                )
            elif self.ingest_mode == "agent_add":
                print(
                    "    [MIRIX] ModalityTalk 真·适配: agent_add() → Meta/all套memory agent"
                    f" (image={'on' if self._effective_ingest_dialogue_images() else 'off'}, "
                    f"每轮最多 {self.ingest_max_dialogue_images} 张, "
                    f"PDF resource={'on' if self._effective_ingest_pdf_native() else 'off'}, "
                    f"chaining={'on' if self.add_chaining else 'off'}) ; "
                    "retrieve/answeringto齐 MIRIX-main/evals",
                    flush=True,
                )
                if self.ingest_defer_session_pdf_in_dialogue:
                    patch_note = (
                        "indexend后from动补丁"
                        if self.ingest_pdf_patch_after_index
                        else "indexend后not补丁 (需手动 reindex_pdf) "
                    )
                    print(
                        "    [MIRIX] 平衡加速: to话ingestnotwith Session PDF text ("
                        f"{patch_note}) ; answering仍走 mirix_hybrid + vision_pages",
                        flush=True,
                    )
            elif self._ingest_ark_text_only():
                print(
                    "    [MIRIX] Ark 省quota: memoryingestonlyuse纯text (image/PDF not进 add; "
                    "answering仍can由 VLM readimage) ",
                    flush=True,
                )

        # eachdatasetonce build: usewill话指纹区分 Mirix user, avoid跨dataset污染
        key_src = json.dumps([s.get("session_id") for s in sessions], ensure_ascii=False)
        self._user_id = "meval_" + hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:24]

        completed_rounds, index_already_complete = (
            self._load_completed_rounds_from_checkpoint(sessions)
        )
        self._ingest_anomaly_rounds = self._load_ingest_anomalies_from_checkpoint(
            sessions
        )
        clear_memory_this_build = (
            self.clear_user_memory_before_build and not completed_rounds
        )

        if (
            clear_memory_this_build
            and self.ingest_mode == "direct_episodic"
            and isinstance(self._client, _LocalMirixBridge)
        ):
            deleted = _run_async(self._client.purge_user_episodic(self._user_id))
            if deleted:
                print(
                    f"    [MIRIX] already清空该dataset旧 episodic: {deleted} line "
                    f"(user_id={self._user_id})",
                    flush=True,
                )

        if clear_memory_this_build and self.mode == "remote" and self.ingest_to_memory:
            deleted = _purge_user_episodic_pg(self._user_id)
            if deleted:
                print(
                    f"    [MIRIX] already清空该dataset旧 episodic: {deleted} line "
                    f"(user_id={self._user_id})",
                    flush=True,
                )

        pdf_policy = self._index_pdf_policy()
        pdfs_dir = images_dir.parent / "pdfs"
        pdf_snip: Dict[str, str] = {}
        pdf_ctx_blocks: Dict[str, str] = {}
        eff_pdf_native = self._effective_ingest_pdf_native()
        skip_local_pdf_excerpts = self._should_skip_local_pdf_excerpts()
        cached_pdf = (
            self._load_pdf_index_cache(sessions, pdf_policy)
            if pdfs_dir.is_dir() and not skip_local_pdf_excerpts
            else None
        )
        if cached_pdf is not None:
            pdf_snip, pdf_ctx_blocks = cached_pdf
        elif pdfs_dir.is_dir() and not skip_local_pdf_excerpts:
            if self.index_pdf_via_ocr or self._use_pdf_text_in_episodic():
                print(
                    f"    [MIRIX] index PDF: render+OCR/text layer (policy={pdf_policy}) → episodic 摘录",
                    flush=True,
                )
            pdf_snip = build_session_pdf_snippets(
                sessions,
                pdfs_dir,
                policy=pdf_policy,  # type: ignore[arg-type]
                embed_max_chars=self.embed_pdf_max_chars,
                min_native_chars=self.min_native_pdf_chars,
            )
            pdf_ctx_blocks = build_session_pdf_context_blocks(
                sessions,
                pdfs_dir,
                policy=pdf_policy,  # type: ignore[arg-type]
                context_max_chars=self.context_pdf_max_chars,
                min_native_chars=self.min_native_pdf_chars,
            )
            self._save_pdf_index_cache(sessions, pdf_policy, pdf_snip, pdf_ctx_blocks)
        elif skip_local_pdf_excerpts:
            print(
                "    [MIRIX] pure_mirix + ingest_pdf_native: skip本  PDF OCR/摘录"
                " (PDF bynativefileingest resource memory) ",
                flush=True,
            )
        self._pdf_ctx_blocks = dict(pdf_ctx_blocks)
        self._pdf_snippets = dict(pdf_snip)
        self._session_pdf_origins = self._build_session_pdf_origin_index(
            sessions, pdfs_dir if pdfs_dir.is_dir() else Path(images_dir.parent / "pdfs")
        )
        self._native_pdfs_ingested = set()
        self._native_pdf_ingest_failures = []
        self._sessions_pdf_text_fallback = set()

        _direct_episodic_agent: Any = None
        _direct_meta_id = ""
        if self.ingest_mode == "direct_episodic" and self._client is not None:
            bridge = self._client
            assert isinstance(bridge, _LocalMirixBridge)
            meta = bridge._meta_agent
            assert meta is not None

            async def _load_ep_agent(local_bridge: _LocalMirixBridge):
                local, actor = await local_bridge._require_actor()
                return await local_bridge._episodic_agent_state(local.server, actor)

            _direct_episodic_agent = _run_async(_load_ep_agent(bridge))
            _direct_meta_id = str(meta.id)

        seen_session: set[str] = set()
        self._corpus_meta = []
        total_turns = sum(len(sess.get("dialogues") or []) for sess in sessions)
        self._expected_dialogue_turns = 0
        remote_img_ingest = bool(
            self._effective_ingest_dialogue_images() and self.mode == "remote"
        )
        ingest_pbar = _mirix_ingest_progress_bar(total_turns, remote_images=remote_img_ingest)
        if ingest_pbar is not None and completed_rounds:
            ingest_pbar.update(len(completed_rounds))
        if remote_img_ingest and ingest_pbar is None:
            print(
                "    [MIRIX] remote pattern: to话imageby base64 image_url 上passingest"
                " (hostreadimage; 单image≤12MB) ",
                flush=True,
            )
        if self.ingest_to_memory and eff_pdf_native and pdfs_dir.is_dir():
            print(
                "    [MIRIX] PDF ingest: file_base64 → resource memory (local/remote) "
                " (andindex OCR 并line; on failurecantime退 episodic 摘录) ",
                flush=True,
            )
        elif (
            self.ingest_to_memory
            and pdfs_dir.is_dir()
            and self._use_pdf_text_in_episodic()
            and not eff_pdf_native
        ):
            print(
                "    [MIRIX] index PDF only OCR text进 episodic; answering侧 vision_pages rendered page走 VLM",
                flush=True,
            )
        elif self.ingest_to_memory and self._ingest_ark_text_only() and pdfs_dir.is_dir():
            print(
                "    [MIRIX] Ark 纯textingest: PDF by本 text摘录write episodic (skip resource native PDF) ",
                flush=True,
            )
        if not self.ingest_to_memory:
            print(
                "    [MIRIX] ingest_to_memory=false: notcall add(), onlybuild本  corpus/PDF 摘录",
                flush=True,
            )
            if self._answer_multimodal_like_mirix():
                print(
                    "    [MIRIX] notingest多模态: indexonly corpus + PDF path标记; answeringwhenmatched round"
                    " file_uri image + vision_pages native PDF (etc.同 MIRIX add 形态, notwrite episodic) ",
                    flush=True,
                )
        elif self.mode == "remote" and self.ingest_sequential_wait:
            print(
                f"    [MIRIX] remote 顺序ingest: 每轮 add 后etc.pending PG persist"
                f" (≤{self.ingest_per_round_max_wait_sec}s/轮, avoidqueue洪水积压) ",
                flush=True,
            )
        elif self.mode == "remote" and self.ingest_confirm_per_round:
            print(
                "    [MIRIX] remote ingest: async /memory/add + 逐轮 PG 确认"
                f" (≤{self.ingest_confirm_per_round_sec}s/轮; 8531 镜像no add_sync) ",
                flush=True,
            )
        elif self.mode == "remote" and not self.add_use_async:
            print(
                "    [MIRIX] remote ingest: request add_sync; if 8531 nothis端point则 fallback async, "
                "建index后willetc.pending PG persist",
                flush=True,
            )

        def _ingest_log(msg: str) -> None:
            if ingest_pbar is not None:
                ingest_pbar.write(msg)
            else:
                print(msg, flush=True)

        mirix_client = self._client
        if self.ingest_to_memory:
            assert mirix_client is not None

        if index_already_complete:
            ck_pdf = (
                load_index_progress_file(Path(self.index_checkpoint_path))
                if self.index_checkpoint_path
                else {}
            ) or {}
            if self._needs_pdf_episodic_patch(
                pdf_snip,
                pdf_ctx_blocks,
                checkpoint_pdf_patched=bool(ck_pdf.get("pdf_episodic_patched")),
            ):
                self._maybe_patch_pdf_episodic_after_index(
                    sessions,
                    completed_rounds,
                    total_turns,
                    pdf_snip,
                    pdf_ctx_blocks,
                    ingest_log=_ingest_log,
                    checkpoint_pdf_patched=bool(ck_pdf.get("pdf_episodic_patched")),
                )
            for sess in sessions:
                for dlg in sess.get("dialogues") or []:
                    self._ensure_corpus_meta_for_dialogue(sess, dlg)
            if ingest_pbar is not None:
                ingest_pbar.update(total_turns)
            if ingest_pbar is not None:
                ingest_pbar.close()
            if not self.ingest_to_memory:
                mode_note = (
                    "本 vocabulary overlapretrieve + VLM answering (zeromemory LLM) "
                    if self.retrieve_mode == "eval_hybrid"
                    and not self.simulate_mirix_retrieve
                    else f"retrievepattern={self.retrieve_mode}"
                )
                print(
                    f"    [MIRIX] skip MIRIX add/persist (ingest_to_memory=false) ; "
                    f"corpus={len(self._corpus_meta)} 轮; {mode_note}",
                    flush=True,
                )
                return
            self._refresh_episodic_round_index()
            if self._ingest_anomaly_rounds and self.ingest_retry_anomaly_at_end:
                pdfs_dir_resume = images_dir.parent / "pdfs"
                self._retry_anomaly_rounds_at_end(
                    sessions,
                    images_dir,
                    pdfs_dir_resume,
                    self._pdf_snippets or {},
                    self._pdf_ctx_blocks or {},
                    ingest_log=_ingest_log,
                )
                self._save_index_round_checkpoint(
                    sessions,
                    completed_rounds,
                    total_rounds=total_turns,
                    index_complete=False,
                )
            integrity = self._verify_index_integrity()
            self._enforce_index_integrity_or_fail(integrity)
            return

        ingest_log_ctx: Dict[str, str] = {"current_round_id": ""}
        anomaly_log_handler = _IngestAnomalyLogHandler(
            self._ingest_anomaly_rounds, ingest_log_ctx
        )
        mirix_ingest_logger = logging.getLogger("Mirix")
        mirix_ingest_logger.addHandler(anomaly_log_handler)

        for sess in sessions:
            sid = str(sess.get("session_id", ""))
            date = str(sess.get("date", "") or "")
            session_has_pending = any(
                str(d.get("round", "") or "") not in completed_rounds
                for d in (sess.get("dialogues") or [])
            )
            try:
                if (
                    session_has_pending
                    and self.ingest_to_memory
                    and eff_pdf_native
                    and pdfs_dir.is_dir()
                ):
                    for pdf_path, round_ref in collect_session_pdf_origins(sess, pdfs_dir):
                        self._ingest_native_pdf_file(
                            pdf_path,
                            session_id=sid,
                            round_ref=round_ref,
                            date=date,
                            ingest_log=_ingest_log,
                        )
                for dlg in sess.get("dialogues") or []:
                    rid = str(dlg.get("round", "") or "")
                    if rid in completed_rounds:
                        self._ensure_corpus_meta_for_dialogue(sess, dlg)
                        if ingest_pbar is not None:
                            ingest_pbar.update(1)
                        continue
                    self._expected_dialogue_turns += 1
                    user_t = str(dlg.get("user", "") or "")
                    asst_t = str(dlg.get("assistant", "") or "")
                    img_names = []
                    for x in dlg.get("img_file") or []:
                        img_names.append(Path(str(x)).name)
                    self._corpus_meta.append(
                        {
                            "session_id": sid,
                            "date": date,
                            "round": str(rid),
                            "user": user_t,
                            "assistant": asst_t,
                            "img_files": img_names,
                            "dialogue_vis": dialogue_image_description(dlg),
                        }
                    )
                    if not self.ingest_to_memory:
                        if ingest_pbar is not None:
                            ingest_pbar.update(1)
                        self._finish_round_ingest_checkpoint(
                            sessions, completed_rounds, rid, total_rounds=total_turns
                        )
                        continue
                    assert mirix_client is not None
                    body = self._build_dialogue_ingest_body(
                        sid=sid,
                        dlg=dlg,
                        user_t=user_t,
                        asst_t=asst_t,
                        img_names=img_names,
                        pdf_snip=pdf_snip,
                        pdf_ctx_blocks=pdf_ctx_blocks,
                        seen_session=seen_session,
                        sessions_pdf_fallback=self._sessions_pdf_text_fallback,
                        sess=sess,
                        pdfs_dir=pdfs_dir,
                        pdf_policy=pdf_policy,
                    )
                    body = prepend_eval_round_trace(
                        body,
                        round_id=rid,
                        chunk_id=None,
                        image_basenames=img_names or None,
                    )
                    filter_tags = {"session_id": sid, "round": str(rid)}
                    occurred_dt = self._occurred_at_from_session_date(date)

                    if self.ingest_mode == "direct_episodic":
                        assert isinstance(mirix_client, _LocalMirixBridge)
                        try:
                            _run_async(
                                mirix_client.direct_insert_episodic_turn(
                                    user_id=self._user_id,
                                    meta_agent_id=_direct_meta_id,
                                    episodic_agent_state=_direct_episodic_agent,
                                    summary=self._direct_episodic_summary(
                                        user_t, asst_t, body
                                    ),
                                    details=body,
                                    filter_tags=filter_tags,
                                    occurred_at=occurred_dt,
                                    summary_fallback_max=self.direct_episodic_summary_max_chars,
                                )
                            )
                        except Exception as exc:
                            _ingest_log(
                                f"    [MIRIX] direct_episodic failed round={rid}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            traceback.print_exc()
                        finally:
                            if ingest_pbar is not None:
                                ingest_pbar.update(1)
                        self._finish_round_ingest_checkpoint(
                            sessions, completed_rounds, rid, total_rounds=total_turns
                        )
                        continue

                    user_content = self._dialogue_user_content(
                        body, dlg, images_dir, pdfs_dir if pdfs_dir.is_dir() else None
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": user_content,
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "(stored turn)"}],
                        },
                    ]
                    add_kw: Dict[str, Any] = {
                        "user_id": self._user_id,
                        "messages": messages,
                        "chaining": self.add_chaining,
                        "filter_tags": filter_tags,
                        "async_add": self.add_use_async,
                    }
                    if date:
                        ds = str(date).strip()[:19]
                        if "T" not in ds:
                            ds = ds + "T12:00:00"
                        if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ds):
                            add_kw["occurred_at"] = ds
                    ingest_log_ctx["current_round_id"] = rid
                    round_add_ok = False
                    try:
                        _run_async(mirix_client.add(**add_kw))
                        if not self._confirm_dialogue_ingest(str(rid), _ingest_log):
                            _record_ingest_anomaly(
                                self._ingest_anomaly_rounds,
                                rid,
                                INGEST_ANOMALY_CONFIRM_TIMEOUT,
                            )
                        else:
                            round_add_ok = True
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {exc}"
                        if _ingest_retry_text_only(err, exc):
                            _ingest_log(
                                f"    [MIRIX] round={rid} 多模态ingestfailed, time退纯text…"
                            )
                            if _is_content_filter_error(exc):
                                _record_ingest_anomaly(
                                    self._ingest_anomaly_rounds,
                                    rid,
                                    INGEST_ANOMALY_CONTENT_FILTER,
                                )
                            try:
                                add_kw["messages"] = self._strip_binary_content(messages)
                                _run_async(mirix_client.add(**add_kw))
                                if self._confirm_dialogue_ingest(
                                    str(rid), _ingest_log, after_error_retry=True
                                ):
                                    round_add_ok = True
                                else:
                                    _record_ingest_anomaly(
                                        self._ingest_anomaly_rounds,
                                        rid,
                                        INGEST_ANOMALY_CONFIRM_TIMEOUT,
                                    )
                            except Exception as exc2:
                                reason = (
                                    INGEST_ANOMALY_CONTENT_FILTER
                                    if _is_content_filter_error(exc2)
                                    else INGEST_ANOMALY_ADD_FAILED
                                )
                                _record_ingest_anomaly(
                                    self._ingest_anomaly_rounds, rid, reason
                                )
                        else:
                            reason = (
                                INGEST_ANOMALY_CONTENT_FILTER
                                if _is_content_filter_error(exc)
                                else INGEST_ANOMALY_ADD_FAILED
                            )
                            _record_ingest_anomaly(
                                self._ingest_anomaly_rounds, rid, reason
                            )
                            _ingest_log(
                                f"    [MIRIX] add{' (async)' if self.add_use_async else '_sync'} "
                                f"failed round={rid}: {err}"
                            )
                            traceback.print_exc()
                    finally:
                        ingest_log_ctx["current_round_id"] = ""
                        if round_add_ok and self.ingest_mode == "agent_add":
                            cf_marked = (
                                rid in self._ingest_anomaly_rounds
                                and self._ingest_anomaly_rounds.get(rid)
                                == INGEST_ANOMALY_CONTENT_FILTER
                            )
                            if not cf_marked and not self._round_in_episodic_index(rid):
                                _record_ingest_anomaly(
                                    self._ingest_anomaly_rounds,
                                    rid,
                                    INGEST_ANOMALY_MISSING_EPISODIC,
                                )
                        if ingest_pbar is not None:
                            ingest_pbar.update(1)
                        self._finish_round_ingest_checkpoint(
                            sessions, completed_rounds, rid, total_rounds=total_turns
                        )

            except Exception:
                self._save_index_round_checkpoint(
                    sessions,
                    completed_rounds,
                    total_rounds=total_turns,
                    index_complete=False,
                )
                raise

        mirix_ingest_logger.removeHandler(anomaly_log_handler)

        if ingest_pbar is not None:
            ingest_pbar.close()

        if self._ingest_anomaly_rounds:
            reasons: Dict[str, int] = defaultdict(int)
            for v in self._ingest_anomaly_rounds.values():
                reasons[str(v)] += 1
            summary = ", ".join(f"{k}={n}" for k, n in sorted(reasons.items()))
            print(
                f"    [MIRIX] main loop标记 {len(self._ingest_anomaly_rounds)}  exception round"
                f" ({summary}) , will在receive尾统一retry",
                flush=True,
            )

        if not self.ingest_to_memory:
            mode_note = (
                "本 vocabulary overlapretrieve + VLM answering (zeromemory LLM) "
                if self.retrieve_mode == "eval_hybrid" and not self.simulate_mirix_retrieve
                else (
                    "影子向量retrieve (simulate_mirix_retrieve, 仍耗 embedding API) "
                    if self.simulate_mirix_retrieve
                    else f"retrievepattern={self.retrieve_mode}"
                )
            )
            print(
                f"    [MIRIX] skip MIRIX add/persist (ingest_to_memory=false) ; "
                f"corpus={len(self._corpus_meta)} 轮; {mode_note}",
                flush=True,
            )
            note = "未write MIRIX PG; eval_hybrid 依赖本  corpus/PDF OCR and MIRIX API 空结果"
            if self.simulate_mirix_retrieve:
                note += "; simulate_mirix_retrieve willuse meta embedding 建影子向量index"
                self._build_shadow_embed_index()
                if self._shadow_vecs is not None:
                    note = (
                        "未write MIRIX PG; simulate_mirix_retrieve already建影子向量index, "
                        "retrievestage模拟 retrieve_with_conversation + embedding search (pure context) "
                    )
            self.last_index_integrity = {
                "ingest_to_memory": False,
                "corpus_meta_rounds": len(self._corpus_meta),
                "simulate_mirix_retrieve": bool(self.simulate_mirix_retrieve),
                "shadow_chunks": len(self._shadow_chunks),
                "note": note,
            }
            return

        if eff_pdf_native and pdfs_dir.is_dir():
            fail_names = sorted(set(self._native_pdf_ingest_failures))
            print(
                f"    [MIRIX] PDF resource ingest: success {len(self._native_pdfs_ingested)}  file"
                f"{', failed ' + str(len(fail_names)) + ' (' + ', '.join(fail_names[:5]) + ('...' if len(fail_names) > 5 else '') + ')' if fail_names else ''}",
                flush=True,
            )
            if fail_names and self.ingest_pdf_episodic_fallback:
                print(
                    "    [MIRIX] alreadytofailed PDF 所属 session time退 episodic text摘录",
                    flush=True,
                )

        if self.ingest_mode != "direct_episodic":
            self._wait_async_ingest_after_build()
        elif self.ingest_mode == "direct_episodic":
            self._refresh_episodic_round_index()
        if self.ingest_retry_missing_sync and self.ingest_mode == "agent_add":
            self._retry_missing_episodic_sync(
                sessions,
                images_dir,
                pdfs_dir,
                pdf_snip,
                pdf_ctx_blocks,
            )
            if self._ingest_needs_async_wait():
                if self.ingest_sequential_wait:
                    tail = _cfg_int(
                        self._cfg,
                        "models",
                        "mirix",
                        "async_ingest_tail_wait_sec",
                        default=90,
                    )
                    if tail > 0:
                        print(
                            f"    [MIRIX] 补库后receive尾稳定检查 {tail}s…",
                            flush=True,
                        )
                        self._wait_async_ingest(timeout_override=tail)
                else:
                    retry_wait = _cfg_int(
                        self._cfg, "models", "mirix", "async_ingest_retry_wait_sec", default=300
                    )
                    if retry_wait > 0:
                        print(
                            f"    [MIRIX] 补库后againetc.pending {retry_wait}s…",
                            flush=True,
                        )
                        self._wait_async_ingest(timeout_override=retry_wait)
        if self._ingest_anomaly_rounds:
            self._retry_anomaly_rounds_at_end(
                sessions,
                images_dir,
                pdfs_dir,
                pdf_snip,
                pdf_ctx_blocks,
                ingest_log=_ingest_log,
            )
            self._save_index_round_checkpoint(
                sessions,
                completed_rounds,
                total_rounds=total_turns,
                index_complete=False,
            )
        self._refresh_episodic_round_index()
        self._maybe_patch_pdf_episodic_after_index(
            sessions,
            completed_rounds,
            total_turns,
            pdf_snip,
            pdf_ctx_blocks,
            ingest_log=_ingest_log,
        )
        integrity = self._verify_index_integrity()
        self._enforce_index_integrity_or_fail(integrity)
        self._save_index_round_checkpoint(
            sessions,
            completed_rounds,
            total_rounds=total_turns,
            index_complete=True,
        )
        print(
            f"    [MIRIX] all round index完as, 进度already标记 index_complete "
            f"({len(completed_rounds)}/{total_turns})",
            flush=True,
        )

    def _resolve_fm_images_pure_mirix(
        self,
        question: str,
        *,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        top_k: int,
        hint_rounds: List[str],
    ) -> List[str]:
        from fm_retrieve_helpers import collect_paths_for_rounds_ordered, image_paths_for_rounds

        trace: List[Dict[str, Any]] = []
        paths: List[str] = []
        seen: set[str] = set()
        for rnd in hint_rounds:
            rr = str(rnd).replace(" ", "")
            if not rr:
                continue
            for p in image_paths_for_rounds([rr], sessions, images_dir):
                if p in seen:
                    continue
                seen.add(p)
                paths.append(p)
                trace.append(
                    {
                        "image_file": Path(p).name,
                        "image_path": p,
                        "round_id": rr,
                        "resolve_source": "mirix_retrieve_round",
                    }
                )
        extra = self._mirix_resource_image_paths(question, sessions, images_dir, top_k=top_k)
        for p in extra:
            if p not in seen:
                seen.add(p)
                paths.append(p)
                trace.append(
                    {
                        "image_file": Path(p).name,
                        "image_path": p,
                        "round_id": "",
                        "resolve_source": "mirix_resource_search",
                    }
                )
        self.last_fm_image_trace = trace
        return paths[: max(top_k * 3, 12)]

    def _resolve_fm_images(
        self,
        question: str,
        ctx: str,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        *,
        top_k: int,
        hint_rounds: Optional[List[str]] = None,
        question_type: str = "",
    ) -> List[str]:
        paths, trace = fm_resolve_image_paths_traced(
            question,
            ctx,
            self._corpus_meta,
            sessions,
            images_dir,
            top_k=top_k,
            hint_rounds=hint_rounds,
            question_type=question_type,
            fm_min_images=self.fm_top_k_images,
        )
        extra = self._mirix_resource_image_paths(question, sessions, images_dir, top_k=top_k)
        seen = {str(p) for p in paths}
        for p in extra:
            if p not in seen:
                seen.add(p)
                paths.append(p)
                trace.append(
                    {
                        "image_file": Path(p).name,
                        "image_path": p,
                        "round_id": "",
                        "resolve_source": "mirix_resource_search",
                    }
                )
        self.last_fm_image_trace = trace
        return paths

    def _mirix_resource_image_paths(
        self,
        question: str,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        *,
        top_k: int,
    ) -> List[str]:
        if self._client is None:
            return []
        try:
            raw = _run_async(
                self._client.search(
                    user_id=self._user_id,
                    query=question,
                    memory_type="resource",
                    search_method="embedding",
                    limit=max(top_k * 3, 10),
                )
            )
        except Exception:
            return []
        if not isinstance(raw, dict) or not raw.get("success"):
            return []
        rounds: List[str] = []
        blob = json.dumps(raw.get("results") or [], ensure_ascii=False, default=str)
        rounds.extend(rounds_from_lightrag_context(blob))
        if not rounds:
            return []
        from fm_retrieve_helpers import collect_paths_for_rounds_ordered

        return collect_paths_for_rounds_ordered(
            rounds, sessions, images_dir, max_paths=max(top_k * 15, 36)
        )

    def _build_retrieve_context(
        self,
        raw_dict: Dict[str, Any],
        *,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        top_k: int,
        ordered_rounds: Optional[List[str]] = None,
    ) -> str:
        ctx = _format_retrieved_memories(
            raw_dict,
            id_to_round=self._episodic_id_to_round,
            corpus_meta=self._corpus_meta,
            max_items_per_type=self.format_memory_max_items,
        )
        if self.retrieve_mode == "pure_mirix":
            ctx = _truncate_context(ctx, self.rag_context_max_chars)
            rounds = list(ordered_rounds or [])
            if not rounds:
                rounds = self._rounds_from_retrieve_raw(raw_dict, ctx)
            if rounds:
                ctx = self._append_pdf_context_for_rounds(ctx, rounds)
            return ctx

        rounds = list(ordered_rounds or [])
        if not rounds:
            rounds = self._rounds_from_retrieve_raw(raw_dict, ctx)
        append_n = self.retrieve_context_max_rounds
        if self.rag_append_max_rounds > 0:
            append_n = max(append_n, self.rag_append_max_rounds)
        append_n = max(append_n, top_k)
        extra_from_raw = self._rounds_from_retrieve_raw(raw_dict, ctx)
        rounds = order_rounds_primary_first(
            rounds,
            extra_from_raw,
            cap=append_n,
        )
        dialogue_chars = self.rag_context_max_chars
        ctx = _append_dialogue_blocks_for_rounds(
            ctx,
            rounds,
            self._corpus_meta,
            images_dir=images_dir,
            max_rounds=append_n,
            max_chars=dialogue_chars,
        )
        return self._append_pdf_context_for_rounds(ctx, rounds)

    def _append_pdf_context_for_rounds(self, ctx: str, rounds: List[str]) -> str:
        if not rounds:
            return ctx
        pdf_blocks = getattr(self, "_pdf_ctx_blocks", None) or {}
        want_sessions: Set[str] = set()
        round_norm = {str(r).replace(" ", "") for r in rounds if str(r).strip()}
        for meta in self._corpus_meta:
            rnd = str(meta.get("round") or "").replace(" ", "")
            if rnd in round_norm:
                sid = str(meta.get("session_id") or "")
                if sid:
                    want_sessions.add(sid)
        if not want_sessions:
            return ctx
        blocks: List[str] = []
        for sid in sorted(want_sessions):
            pb = pdf_blocks.get(sid, "") if pdf_blocks else ""
            if not pb:
                origins = getattr(self, "_session_pdf_origins", {}).get(sid) or []
                if origins:
                    stub_lines = []
                    for pdf_name, round_ref in origins[:3]:
                        stub_lines.append(
                            f"[PDF_PATH] {pdf_name}\n[ROUND_REF] {round_ref}\n[PDF_PAGE] 1"
                        )
                    pb = "\n\n".join(stub_lines)
            if pb:
                blocks.append(f"=== Session {sid} PDF ===\n{pb}")
        if not blocks:
            return ctx
        merged = ctx + "\n\n---\n[MIRIX session PDF excerpts]\n" + "\n\n".join(blocks)
        return _truncate_context(merged, self.rag_context_max_chars * 3)

    def _task_agent_preliminary(self, question: str, context: str) -> str:
        prelim = (context or "").strip()
        if prelim:
            return prelim
        lim = max(10, self.retrieve_compact_max_items)
        raw = self._retrieve_raw(question, limit=lim)
        return _format_retrieved_memories(
            raw if isinstance(raw, dict) else {},
            id_to_round=self._episodic_id_to_round,
            corpus_meta=self._corpus_meta,
        )

    def _answer_with_vlm(
        self,
        question: str,
        context: str,
        image_paths: Optional[List[str]],
        *,
        vision_pdf_page_paths: Optional[List[str]] = None,
        vision_pdf_page_labels: Optional[List[str]] = None,
        pdf_answer_mode: str = "text_only",
    ) -> str:
        if self.llm_client is None:
            return ""
        llm_client = self.llm_client
        image_labels: Optional[List[str]] = None
        if image_paths and self.last_fm_image_trace:
            by_path = {t["image_path"]: t for t in self.last_fm_image_trace}
            image_labels = []
            for p in image_paths:
                t = by_path.get(str(p), {})
                rid = str(t.get("round_id") or "")
                src = str(t.get("resolve_source") or "")
                parts = [f"exact file name: {Path(p).name}"]
                if rid:
                    parts.append(f"[ROUND] {rid}")
                if src:
                    parts.append(f"[SOURCE] {src}")
                image_labels.append(" ".join(parts))

        from answer_postprocess import (
            is_degenerate_vlm_answer,
            max_completion_tokens_for_qtype,
            truncate_degenerate_answer,
        )

        qtype = getattr(self, "_last_retrieve_question_type", "") or ""
        max_tok = max_completion_tokens_for_qtype(qtype)

        from llm_chat_kwargs import chat_completion_kwargs

        def _call_vlm(msgs: List[Dict[str, Any]], *, tokens: int) -> str:
            r = llm_client.chat.completions.create(
                model=self.llm_model,
                messages=msgs,
                **chat_completion_kwargs(
                    self.llm_model,
                    max_output_tokens=tokens,
                    temperature=0.0,
                    eval_cfg=self._cfg,
                ),
            )
            u = openai_usage_to_dict(getattr(r, "usage", None))
            if u:
                for k, v in u.items():
                    self.last_answer_usage[k] = int(self.last_answer_usage.get(k, 0) or 0) + int(
                        v or 0
                    )
            return (r.choices[0].message.content or "").strip()

        msgs = build_answer_messages(
            question,
            context,
            image_paths,
            vision_pdf_page_paths=vision_pdf_page_paths,
            vision_pdf_page_labels=vision_pdf_page_labels,
            pdf_answer_mode=pdf_answer_mode,
            image_attach_labels=image_labels,
            cite_source_rounds=self.cite_source_rounds,
            question_type=qtype,
        )
        self.last_answer_usage = {}
        raw = _call_vlm(msgs, tokens=max_tok)
        if is_degenerate_vlm_answer(raw):
            raw = truncate_degenerate_answer(raw)
        if is_degenerate_vlm_answer(raw):
            anti = (
                "\n\nCRITICAL: Reply with one short factual answer only. "
                "Do NOT repeat the same word or token."
            )
            rc = msgs[0].get("content")
            if isinstance(rc, list):
                for part in rc:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = str(part.get("text") or "") + anti
                        break
            else:
                msgs[0]["content"] = str(rc or "") + anti
            raw = _call_vlm(msgs, tokens=min(max_tok, 96))
            if is_degenerate_vlm_answer(raw):
                raw = truncate_degenerate_answer(raw) or ""
        body, _ = strip_answer_round_citation(raw)
        if not body.strip():
            retry_msgs = build_answer_messages(
                question,
                context,
                image_paths,
                vision_pdf_page_paths=vision_pdf_page_paths,
                vision_pdf_page_labels=vision_pdf_page_labels,
                pdf_answer_mode=pdf_answer_mode,
                image_attach_labels=image_labels,
                cite_source_rounds=False,
                question_type=qtype,
            )
            extra = (
                "\n\nYou must output a short factual answer phrase first. "
                "Do not reply with only [rounds: ...] and do not leave the answer empty."
            )
            rc = retry_msgs[0].get("content")
            if isinstance(rc, list):
                for part in rc:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = str(part.get("text") or "") + extra
                        break
            else:
                retry_msgs[0]["content"] = str(rc or "") + extra
            raw = _call_vlm(retry_msgs, tokens=min(max_completion_tokens_for_qtype(qtype), 96))
            if is_degenerate_vlm_answer(raw):
                raw = truncate_degenerate_answer(raw) or ""
        return self._finalize_answer_with_round_citation(raw)

    def _finalize_answer_with_round_citation(self, answer: str) -> str:
        if not self.cite_source_rounds:
            return (answer or "").strip()
        return append_answer_round_citation(
            answer, getattr(self, "_last_retrieve_rounds", []) or []
        )

    def _build_session_pdf_origin_index(
        self,
        sessions: List[Dict[str, Any]],
        pdfs_dir: Path,
    ) -> Dict[str, List[Tuple[str, str]]]:
        out: Dict[str, List[Tuple[str, str]]] = {}
        if not pdfs_dir.is_dir():
            return out
        for sess in sessions:
            sid = str(sess.get("session_id", ""))
            origins = collect_session_pdf_origins(sess, pdfs_dir)
            if origins:
                out[sid] = [(p.name, str(rr)) for p, rr in origins]
        return out

    def _resolve_mirix_style_dialogue_images(
        self,
        ordered_rounds: List[str],
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        *,
        top_k: int,
        question: str = "",
        question_type: str = "",
    ) -> List[str]:
        wants_gallery = question_wants_image_gallery(question, question_type)
        img_cap_rounds = max(top_k, 6) if self.no_ingest_multimodal else max(
            top_k * 3, self.retrieve_context_max_rounds, 12
        )
        head = [str(r).replace(" ", "") for r in ordered_rounds[:img_cap_rounds] if str(r).strip()]
        cap = self.fm_top_k_images if wants_gallery else min(self.fm_top_k_images, max(6, top_k))
        paths = image_paths_for_rounds(head, sessions, images_dir)
        trace: List[Dict[str, Any]] = []
        for p in paths[:cap]:
            rnd = ""
            for meta in self._corpus_meta:
                for rel in meta.get("img_files") or []:
                    if Path(str(rel)).name == Path(p).name:
                        rnd = str(meta.get("round") or "")
                        break
                if rnd:
                    break
            trace.append(
                {
                    "image_file": Path(p).name,
                    "image_path": p,
                    "round_id": rnd.replace(" ", ""),
                    "resolve_source": "mirix_no_ingest_file_uri",
                }
            )
        self.last_fm_image_trace = trace
        return paths[:cap]

    def render_pdf_vision_for_retrieve(
        self,
        sessions: List[Dict[str, Any]],
        pdfs_dir: Path,
        round_ids: List[str],
        *,
        max_pages: int = 3,
        out_dir: Path,
        tag: str = "0",
    ) -> Tuple[List[str], int, List[Dict[str, Any]]]:
        from pdf_session_text import render_pdf_pages_for_rounds

        return render_pdf_pages_for_rounds(
            sessions,
            pdfs_dir,
            list(round_ids),
            self._corpus_meta,
            max_pages=max_pages,
            out_dir=out_dir,
            tag=tag,
        )

    def _complete_retrieve_from_raw_dict(
        self,
        raw_dict: Dict[str, Any],
        question: str,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        top_k: int,
        question_type: str,
        *,
        search_items: List[Dict[str, Any]],
        pure_override: Optional[bool] = None,
        diag_apis: Optional[List[str]] = None,
        gold_rounds: Optional[Set[str]] = None,
    ) -> Tuple[str, List[str]]:
        raw_dict = raw_dict if isinstance(raw_dict, dict) else {}
        candidates = extract_round_candidates(
            raw_dict,
            id_to_round=self._episodic_id_to_round,
            corpus_meta=self._corpus_meta,
            item_round_id_fn=_item_round_id,
            extra_items=search_items,
        )
        all_candidate_rids = [c[0] for c in candidates]
        rerank_head_k = min(
            self.retrieve_rerank_candidates,
            max(top_k, 12),
        )
        pure = (
            bool(pure_override)
            if pure_override is not None
            else (self.retrieve_mode == "pure_mirix")
        )
        used_llm_rerank = False
        used_offline_fallback = False
        if (
            not pure
            and self.retrieve_rerank
            and self.llm_client
            and candidates
        ):
            reranked_head = llm_rerank_round_ids(
                self.llm_client,
                self.llm_model,
                question,
                candidates,
                top_k=rerank_head_k,
                max_candidates=self.retrieve_rerank_candidates,
            )
            used_llm_rerank = True
        else:
            reranked_head = all_candidate_rids[:rerank_head_k]

        gold_list = sorted(gold_rounds or [])
        if pure:
            ordered_rounds = order_rounds_primary_first(
                gold_list + list(reranked_head),
                list(all_candidate_rids),
                cap=max(top_k, self.retrieve_context_max_rounds),
            )
        else:
            offline = offline_round_fallback(
                question,
                self._corpus_meta,
                sessions,
                images_dir,
                budget=max(self.retrieve_context_max_rounds * 2, top_k * 4, 24),
                multimodal=bool(self.no_ingest_multimodal),
            )
            used_offline_fallback = bool(offline)
            ordered_rounds = expand_ordered_rounds(
                gold_list + list(reranked_head),
                all_candidate_rids,
                offline,
                cap=self.retrieve_context_max_rounds,
            )
        self._last_retrieve_rounds = list(ordered_rounds)
        apis = diag_apis or ["retrieve_with_conversation", "search"]
        self.last_retrieve_diag = {
            "retrieve_mode": self.retrieve_mode,
            "apis": apis,
            "search_memory_types": list(self.search_memory_types),
            "candidate_rounds": len(all_candidate_rids),
            "ordered_rounds": len(ordered_rounds),
            "used_llm_rerank": used_llm_rerank,
            "used_offline_fallback": used_offline_fallback,
            "appended_corpus_dialogue": not pure,
            "appended_local_pdf_ocr": not pure,
        }

        ctx = self._build_retrieve_context(
            raw_dict,
            sessions=sessions,
            images_dir=images_dir,
            top_k=top_k,
            ordered_rounds=ordered_rounds,
        )
        wants_gallery = question_wants_image_gallery(question, question_type)
        if self._retrieve_attach_dialogue_images():
            img_paths = self._resolve_mirix_style_dialogue_images(
                ordered_rounds,
                sessions,
                images_dir,
                top_k=top_k,
                question=question,
                question_type=question_type,
            )
            if wants_gallery:
                if pure:
                    extra = self._resolve_fm_images_pure_mirix(
                        question,
                        sessions=sessions,
                        images_dir=images_dir,
                        top_k=max(top_k, self.fm_top_k_images // 2),
                        hint_rounds=ordered_rounds,
                    )
                else:
                    extra = self._resolve_fm_images(
                        question,
                        ctx,
                        sessions,
                        images_dir,
                        top_k=max(top_k, self.fm_top_k_images // 2),
                        hint_rounds=ordered_rounds,
                        question_type=question_type,
                    )
                seen = set(img_paths)
                for p in extra:
                    if p not in seen:
                        seen.add(p)
                        img_paths.append(p)
                img_paths = img_paths[: self.fm_top_k_images]
        elif wants_gallery:
            if pure:
                img_paths = self._resolve_fm_images_pure_mirix(
                    question,
                    sessions=sessions,
                    images_dir=images_dir,
                    top_k=max(top_k, self.fm_top_k_images // 2),
                    hint_rounds=ordered_rounds,
                )
            else:
                img_paths = self._resolve_fm_images(
                    question,
                    ctx,
                    sessions,
                    images_dir,
                    top_k=max(top_k, self.fm_top_k_images // 2),
                    hint_rounds=ordered_rounds,
                    question_type=question_type,
                )
            if not img_paths and not pure and self._corpus_meta:
                print(
                    "    [MIRIX] FM/MR image: retrieve/rerank 未parsingtoimage, already走 overlap/corpus time退",
                    flush=True,
                )
        else:
            self.last_fm_image_trace = []
            img_paths = []
        return ctx, img_paths

    def _build_shadow_embed_index(self) -> None:
        self._shadow_chunks = []
        self._shadow_vecs = None
        self._shadow_by_mt = defaultdict(list)
        if not self.simulate_mirix_retrieve or not self.meta_agent_config_path:
            return
        try:
            meta = _load_meta_yaml(Path(self.meta_agent_config_path))
            client, model = _mirix_meta_embedding_client(meta, eval_cfg=self._cfg)
        except Exception as exc:
            print(
                f"    [MIRIX] simulate_mirix_retrieve: unable toload embedding config, skip影子index: {exc}",
                flush=True,
            )
            return
        for meta in self._corpus_meta:
            rnd = str(meta.get("round") or "").replace(" ", "")
            sid = str(meta.get("session_id") or "")
            body = (
                f"User:\n{meta.get('user', '')}\n\nAssistant:\n{meta.get('assistant', '')}"
            )
            dv = str(meta.get("dialogue_vis") or "").strip()
            if dv:
                body += f"\n\n[Image description]\n{dv}"
            img_names = meta.get("img_files") or []
            if img_names:
                body += "\n\n[IMAGE_FILES] " + ", ".join(str(x) for x in img_names)
            body = prepend_eval_round_trace(
                body.strip(),
                round_id=rnd,
                chunk_id=None,
                image_basenames=[Path(str(x)).name for x in img_names] if img_names else None,
            )
            txt = _truncate_for_embedding(body)
            if not txt.strip():
                continue
            idx = len(self._shadow_chunks)
            self._shadow_chunks.append(
                {
                    "text": txt,
                    "memory_type": "episodic",
                    "round": rnd,
                    "session_id": sid,
                }
            )
            self._shadow_by_mt["episodic"].append(idx)
        for sid, snip in (self._pdf_snippets or {}).items():
            s = str(snip or "").strip()
            if not s:
                continue
            txt = _truncate_for_embedding(f"[Session {sid} PDF text]\n{s}")
            idx = len(self._shadow_chunks)
            self._shadow_chunks.append(
                {
                    "text": txt,
                    "memory_type": "semantic",
                    "round": "",
                    "session_id": str(sid),
                }
            )
            self._shadow_by_mt["semantic"].append(idx)
        if not self._shadow_chunks:
            print("    [MIRIX] simulate_mirix_retrieve: nocanusetextblock, skip影子index", flush=True)
            return
        texts = [c["text"] for c in self._shadow_chunks]
        try:
            embs = _openai_embeddings_batch(
                client,
                model,
                texts,
                batch_size=self.simulate_embed_batch_size,
            )
        except Exception as exc:
            print(
                f"    [MIRIX] simulate_mirix_retrieve: embedding API failed, skip影子index: {exc}",
                flush=True,
            )
            self._shadow_chunks = []
            self._shadow_by_mt = defaultdict(list)
            return
        try:
            import numpy as np

            mat = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            self._shadow_vecs = mat / norms
        except Exception:
            self._shadow_chunks = []
            self._shadow_by_mt = defaultdict(list)
            self._shadow_vecs = None
            print(
                "    [MIRIX] simulate_mirix_retrieve: numpy is unusableor向量归一化failed, skip影子index",
                flush=True,
            )
            return
        print(
            f"    [MIRIX] simulate_mirix_retrieve: 影子向量index 绪 "
            f"chunks={len(self._shadow_chunks)} dim={len(embs[0]) if embs else 0}",
            flush=True,
        )

    def _shadow_item_dict(self, chunk_i: int) -> Dict[str, Any]:
        c = self._shadow_chunks[chunk_i]
        mt = str(c.get("memory_type") or "episodic")
        return {
            "id": f"shadow-{chunk_i}",
            "details": c["text"],
            "summary": "",
            "memory_type": mt,
            "filter_tags": {
                "round": c.get("round") or "",
                "session_id": c.get("session_id") or "",
            },
        }

    def _shadow_embed_question(self, question: str) -> Any:
        import numpy as np

        meta = _load_meta_yaml(Path(self.meta_agent_config_path))
        client, model = _mirix_meta_embedding_client(meta, eval_cfg=self._cfg)
        vec = _openai_embeddings_batch(
            client, model, [_truncate_for_embedding(question)], batch_size=1
        )[0]
        v = np.array(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        return v

    def _shadow_top_chunk_indices(self, qvec: Any, indices: List[int], top_k: int) -> List[int]:
        import numpy as np

        if not indices or self._shadow_vecs is None or top_k <= 0:
            return []
        rows = self._shadow_vecs[np.array(indices, dtype=np.int64)]
        sims = rows @ qvec
        order = np.argsort(-sims)[:top_k]
        return [int(indices[int(j)]) for j in order]

    def _shadow_retrieve_raw_dict(self, qvec: Any, limit: int) -> Dict[str, Any]:
        epi_idx = self._shadow_by_mt.get("episodic") or []
        top_i = self._shadow_top_chunk_indices(qvec, epi_idx, limit)
        items = [self._shadow_item_dict(i) for i in top_i]
        return {
            "success": True,
            "topics": "",
            "memories": {
                "episodic": {"items": items, "total_count": len(items)},
            },
        }

    def _shadow_search_supplement_items(
        self, qvec: Any, question: str, *, extra_queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        queries: List[str] = []
        for q in [question, *(extra_queries or [])]:
            qq = str(q or "").strip()
            if qq and qq not in queries:
                queries.append(qq)
        qvecs = [qvec]
        if len(queries) > 1:
            try:
                for qq in queries[1:]:
                    qvecs.append(self._shadow_embed_question(qq))
            except Exception:
                pass
        for qv in qvecs:
            for mt in self.search_memory_types:
                idxs = self._shadow_by_mt.get(str(mt).strip().lower()) or []
                if not idxs:
                    continue
                top_i = self._shadow_top_chunk_indices(
                    qv, idxs, self.search_supplement_limit
                )
                for i in top_i:
                    item = dict(self._shadow_item_dict(i))
                    item["id"] = f"shadow-search-{mt}-{i}-{len(merged)}"
                    item["memory_type"] = str(mt).strip().lower()
                    key = item["id"]
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(item)
        return merged

    def _retrieve_shadow_simulated_mirix(
        self,
        question: str,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        *,
        top_k: int,
        question_type: str,
    ) -> Tuple[str, List[str]]:
        lim = self._retrieve_limit_for_question(top_k)
        try:
            qvec = self._shadow_embed_question(question)
        except Exception as exc:
            print(
                f"    [MIRIX] shadow retrieve: 问 questions embedding failed: {exc}, 退time eval_hybrid 本 ",
                flush=True,
            )
            return self._retrieve_local_corpus_only(
                question,
                sessions,
                images_dir,
                top_k=top_k,
                question_type=question_type,
            )
        raw_dict = self._shadow_retrieve_raw_dict(qvec, lim)
        topics = str(raw_dict.get("topics") or "").strip()
        extra_queries = [topics] if topics and topics != question.strip() else []
        search_items = self._shadow_search_supplement_items(
            qvec, question, extra_queries=extra_queries
        )
        ctx, imgs = self._complete_retrieve_from_raw_dict(
            raw_dict,
            question,
            sessions,
            images_dir,
            top_k,
            question_type,
            search_items=search_items,
            pure_override=True,
            diag_apis=["shadow_embedding_retrieve", "shadow_embedding_search"],
        )
        self.last_retrieve_diag["simulate_mirix_retrieve"] = True
        self.last_retrieve_diag["ingest_to_memory"] = False
        return ctx, imgs

    def _retrieve_local_corpus_only(
        self,
        question: str,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        *,
        top_k: int,
        question_type: str,
    ) -> Tuple[str, List[str]]:
        budget = max(self.retrieve_context_max_rounds * 2, top_k * 4, 24)
        offline = offline_round_fallback(
            question,
            self._corpus_meta,
            sessions,
            images_dir,
            budget=budget,
            multimodal=bool(self.no_ingest_multimodal),
        )
        ordered_rounds = expand_ordered_rounds(
            [],
            [],
            offline,
            cap=self.retrieve_context_max_rounds,
        )
        if self.no_ingest_multimodal:
            mm_cap = min(
                self.retrieve_context_max_rounds,
                max(top_k * 2, top_k + 4, 8),
            )
            ordered_rounds = ordered_rounds[:mm_cap]
        self._last_retrieve_rounds = list(ordered_rounds)
        apis = ["local_corpus_offline"]
        if self.no_ingest_multimodal:
            apis.append("mirix_no_ingest_multimodal")
        self.last_retrieve_diag = {
            "retrieve_mode": self.retrieve_mode,
            "apis": apis,
            "ingest_to_memory": False,
            "no_ingest_multimodal": bool(self.no_ingest_multimodal),
            "candidate_rounds": len(offline),
            "ordered_rounds": len(ordered_rounds),
            "used_llm_rerank": False,
            "used_offline_fallback": bool(offline),
            "appended_corpus_dialogue": True,
            "appended_local_pdf_ocr": True,
        }
        ctx = self._build_retrieve_context(
            {},
            sessions=sessions,
            images_dir=images_dir,
            top_k=top_k,
            ordered_rounds=ordered_rounds,
        )
        wants_gallery = question_wants_image_gallery(question, question_type)
        if self._retrieve_attach_dialogue_images():
            img_paths = self._resolve_mirix_style_dialogue_images(
                ordered_rounds,
                sessions,
                images_dir,
                top_k=top_k,
                question=question,
                question_type=question_type,
            )
        elif wants_gallery:
            img_paths = self._resolve_fm_images(
                question,
                ctx,
                sessions,
                images_dir,
                top_k=max(top_k, self.fm_top_k_images // 2),
                hint_rounds=ordered_rounds,
                question_type=question_type,
            )
            if not img_paths and self._corpus_meta:
                print(
                    "    [MIRIX] FM/MR image: 本  corpus time退未parsingtoimage, already走 overlap/corpus",
                    flush=True,
                )
        else:
            self.last_fm_image_trace = []
            img_paths = []
        return ctx, img_paths

    def retrieve(
        self,
        question: str,
        sessions: List[Dict[str, Any]],
        images_dir: Path,
        supporting_facts: str = "",
        top_k: int = 5,
        question_type: str = "",
    ) -> Tuple[str, List[str]]:
        gold_rounds: Set[str] = set()
        if self.use_gold_in_retrieve and supporting_facts:
            from eval_metrics import parsing_gold_rounds

            gold_rounds = parsing_gold_rounds(supporting_facts)
        self._last_retrieve_question_type = str(question_type or "").strip().lower()
        if (
            not self.ingest_to_memory
            and self.simulate_mirix_retrieve
            and self._shadow_vecs is not None
        ):
            return self._retrieve_shadow_simulated_mirix(
                question,
                sessions,
                images_dir,
                top_k=top_k,
                question_type=question_type,
            )
        if not self.ingest_to_memory and self.retrieve_mode == "eval_hybrid":
            return self._retrieve_local_corpus_only(
                question,
                sessions,
                images_dir,
                top_k=top_k,
                question_type=question_type,
            )
        self._ensure_meta_agent()
        assert self._client is not None
        if not self._episodic_id_to_round:
            self._refresh_episodic_round_index()
        lim = self._retrieve_limit_for_question(top_k)
        try:
            raw = self._retrieve_raw(question, limit=lim)
        except Exception as exc:
            print(
                f"    [MIRIX] retrieve_with_conversation failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            self.last_fm_image_trace = []
            self.last_retrieve_diag = {
                "retrieve_mode": self.retrieve_mode,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return "", []
        raw_dict = raw if isinstance(raw, dict) else {}
        topics = str(raw_dict.get("topics") or "").strip()
        extra_queries = [topics] if topics and topics != question.strip() else []
        search_items = self._search_supplement_hits(question, extra_queries=extra_queries)
        return self._complete_retrieve_from_raw_dict(
            raw_dict,
            question,
            sessions,
            images_dir,
            top_k,
            question_type,
            search_items=search_items,
            pure_override=None,
            diag_apis=None,
            gold_rounds=gold_rounds,
        )

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
            return ""

        fm_visual = bool(image_paths) and is_image_filename_question(question)
        mr_visual = (
            bool(image_paths)
            and (getattr(self, "_last_retrieve_question_type", "") or "") == "mr"
        )
        has_pdf_vision = bool(vision_pdf_page_paths)
        use_task_agent = (
            self.answer_mode == "mirix_task_agent"
            and not fm_visual
            and not mr_visual
            and not has_pdf_vision
        )

        if use_task_agent:
            self._ensure_meta_agent()
            assert self._client is not None
            prelim = self._task_agent_preliminary(question, context)
            try:
                ta_client, ta_model = self._resolve_task_agent_llm()
                agent = _MirixEvalTaskAgent(
                    llm_client=ta_client,
                    model=ta_model,
                    mirix_client=self._client,
                    user_id=self._user_id,
                    max_tool_rounds=self.task_agent_max_tool_rounds,
                    search_limit=self.task_agent_search_limit,
                    cite_source_rounds=self.cite_source_rounds,
                    eval_cfg=self._cfg,
                )
                ans = agent.answer_with_preliminary_or_fallback(
                    question,
                    prelim,
                    fallback_client=self.llm_client,
                    fallback_model=self.llm_model,
                )
                ans = self._finalize_answer_with_round_citation(ans)
                self.last_answer_usage = dict(agent.last_usage)
                return ans
            except Exception as exc:
                if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError") or "401" in str(
                    exc
                ):
                    print(
                        f"    [MIRIX] task_agent authentication仍failed, tryfinallyonce本 程序化answering: {exc}",
                        flush=True,
                    )
                    fb = _MirixEvalTaskAgent(
                        llm_client=self.llm_client,
                        model=self.llm_model,
                        mirix_client=self._client,
                        user_id=self._user_id,
                        max_tool_rounds=self.task_agent_max_tool_rounds,
                        search_limit=self.task_agent_search_limit,
                        cite_source_rounds=self.cite_source_rounds,
                        eval_cfg=self._cfg,
                    )
                    fb.use_openai_tools = False
                    ans = fb.answer_programmatic_search(question, prelim)
                    self.last_answer_usage = dict(fb.last_usage)
                    return self._finalize_answer_with_round_citation(ans)
                self.last_answer_usage = {}
                print(
                    f"    [MIRIX] task_agent answeringfailed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                return ""

        try:
            return self._finalize_answer_with_round_citation(
                self._answer_with_vlm(
                    question,
                    context,
                    image_paths,
                    vision_pdf_page_paths=vision_pdf_page_paths,
                    vision_pdf_page_labels=vision_pdf_page_labels,
                    pdf_answer_mode=pdf_answer_mode,
            )
            )
        except Exception as exc:
            self.last_answer_usage = {}
            print(
                f"    [MIRIX] chat.completions failed model={self.llm_model!r}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            return ""
