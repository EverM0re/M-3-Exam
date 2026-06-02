import os
import time
import openai

from m3exam.config.config_loader import cfg

_openai_client = None


def _text_llm_base_url() -> str:
    return cfg("api", "text_llm", "base_url") or os.environ.get("OPENAI_BASE_URL", "")


def _text_llm_api_key() -> str:
    return cfg("api", "text_llm", "api_key") or os.environ.get("OPENAI_API_KEY", "")


def _text_llm_model() -> str:
    return cfg("api", "text_llm", "model") or os.environ.get("OPENAI_MODEL", "")


def set_openai_key() -> None:
    global _openai_client
    api_key  = _text_llm_api_key()
    base_url = _text_llm_base_url()

    if not api_key:
        raise ValueError(
            "text_llm api_key is not configured. "
            "Set api.text_llm.api_key in config/config.yaml "
            "or the OPENAI_API_KEY environment variable."
        )

    _openai_client = (
        openai.OpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else openai.OpenAI(api_key=api_key)
    )


def _get_actual_model(model_alias: str, use_16k: bool = False) -> str:
    if model_alias == "chatgpt":
        return _text_llm_model() or "gpt-3.5-turbo"
    return model_alias


def run_chatgpt(
    query: str,
    num_gen: int = 1,
    num_tokens_request: int = 1000,
    model: str = "chatgpt",
    use_16k: bool = False,
    temperature: float = 1.0,
    wait_time: int = 1,
) -> str:
    if _openai_client is None:
        raise ValueError("OpenAI client not initialised. Call set_openai_key() first.")

    actual_model = _get_actual_model(model, use_16k)

    max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "10"))
    max_backoff = int(os.environ.get("OPENAI_MAX_BACKOFF_SECONDS", "120"))
    backoff = max(1, min(wait_time, max_backoff))

    completion = None
    last_err = None
    for attempt in range(max_retries):
        try:
            role = "user"
            completion = _openai_client.chat.completions.create(
                model=actual_model,
                temperature=temperature,
                max_tokens=num_tokens_request,
                n=num_gen,
                messages=[{"role": role, "content": query}],
            )
            break
        except openai.APIError as e:
            last_err = e
            print(f"API error: {e}  - retry {attempt + 1}/{max_retries}, sleeping {backoff}s")
        except openai.APIConnectionError as e:
            last_err = e
            print(f"Connection error: {e}  - retry {attempt + 1}/{max_retries}, sleeping {backoff}s")
        except openai.RateLimitError as e:
            last_err = e
            print(f"Rate limit exceeded: {e}  - retry {attempt + 1}/{max_retries}, sleeping {backoff}s")
        except Exception as e:
            last_err = e
            print(f"Unexpected error: {e}  - retry {attempt + 1}/{max_retries}, sleeping {backoff}s")
        if attempt < max_retries - 1:
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    if completion is None:
        raise RuntimeError(
            f"run_chatgpt: failed after {max_retries} attempts"
        ) from last_err

    if num_gen > 1:
        return [choice.message.content for choice in completion.choices]
    return completion.choices[0].message.content
