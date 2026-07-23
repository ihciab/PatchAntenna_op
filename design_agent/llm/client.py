"""Provider-agnostic LLM client interfaces and simple implementations."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


class LLMClient:
    """Abstract LLM client used by design skills.

    Future implementations can wrap OpenAI, local models, MCP tools, LangGraph
    nodes, or any other backend while keeping skill interfaces stable.
    """

    def __init__(self, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the client with optional model and provider configuration."""

        self.model_name = model_name
        self.config = config or {}

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a text response from an LLM backend."""

        raise NotImplementedError("LLMClient.generate must be implemented by a concrete backend.")


class OpenAICompatibleLLMClient(LLMClient):
    """Minimal client for OpenAI-compatible chat completions APIs.

    This client works with providers that expose a ``/chat/completions`` endpoint
    compatible with the OpenAI request shape. It is intentionally small so the
    framework can test LLM connectivity before the full design algorithms are
    implemented.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
        api_url: Base URL or full chat-completions endpoint URL.
            api_key: API key used in the ``Authorization`` header.
            model_name: Model identifier accepted by the provider.
            config: Optional provider-specific parameters.
        """

        super().__init__(model_name=model_name, config=config)
        self.api_url = normalize_chat_completions_url(api_url)
        self.api_key = api_key

    @classmethod
    def from_config_file(cls, config_path: str = "config.json") -> "OpenAICompatibleLLMClient":
        """Create a client from ``config.json`` and environment overrides.

        Expected JSON section:

        .. code-block:: json

            {
              "agent_api": {
                "API_URL": ".../chat/completions",
                "MODEL_NAME": "model-name",
                "API_KEY": "..."
              }
            }

        Environment variables take priority over the JSON values:
        ``AGENT_API_URL``, ``AGENT_MODEL_NAME``, and ``AGENT_API_KEY``.
        """

        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        api_config = data.get("agent_api", {})

        api_url = os.getenv("AGENT_API_URL") or api_config.get("API_URL")
        api_key = (
            os.getenv("AGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or api_config.get("API_KEY")
        )
        model_name = os.getenv("AGENT_MODEL_NAME") or api_config.get("MODEL_NAME")

        if not api_url:
            raise ValueError("Missing API URL. Set agent_api.API_URL or AGENT_API_URL.")
        if not api_key:
            raise ValueError("Missing API key. Set agent_api.API_KEY or AGENT_API_KEY.")
        if not model_name:
            raise ValueError("Missing model name. Set agent_api.MODEL_NAME or AGENT_MODEL_NAME.")

        return cls(api_url=api_url, api_key=api_key, model_name=model_name, config=api_config)

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a text response using a chat-completions request."""

        context = context or {}
        label = str(context.get("log_label") or "llm.generate")
        timeout = float(context.get("timeout", self.config.get("timeout", 60)))
        started_at = time.perf_counter()
        print(
            "[LLM] START {0} model={1} prompt_chars={2} timeout={3:.0f}s".format(
                label,
                self.model_name,
                len(prompt),
                timeout,
            ),
            flush=True,
        )
        messages: List[Dict[str, str]] = []
        system_prompt = context.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": context.get("temperature", self.config.get("temperature", 0.2)),
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": "Bearer {0}".format(self.api_key),
                "Content-Type": "application/json",
                "HTTP-Referer": str(self.config.get("HTTP_REFERER", "https://local-design-agent")),
                "X-OpenRouter-Title": str(self.config.get("APP_TITLE", "Auto-py2cst Design Agent")),
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            elapsed = time.perf_counter() - started_at
            error_body = exc.read().decode("utf-8", errors="replace")
            print(
                "[LLM] FAIL {0} elapsed={1:.1f}s status={2} body={3}".format(
                    label,
                    elapsed,
                    exc.code,
                    error_body,
                ),
                flush=True,
            )
            raise RuntimeError(
                "LLM request failed with HTTP {0}: {1}".format(exc.code, error_body)
            ) from exc
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            print(
                "[LLM] FAIL {0} elapsed={1:.1f}s error={2}: {3}".format(
                    label,
                    elapsed,
                    exc.__class__.__name__,
                    exc,
                ),
                flush=True,
            )
            raise

        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("LLM response did not contain choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise ValueError("LLM response did not contain message content.")
        elapsed = time.perf_counter() - started_at
        print(
            "[LLM] DONE {0} elapsed={1:.1f}s response_chars={2}".format(
                label,
                elapsed,
                len(str(content)),
            ),
            flush=True,
        )
        return str(content)


def normalize_chat_completions_url(api_url: str) -> str:
    """Return a chat-completions endpoint from a base URL or endpoint URL."""

    clean = str(api_url).strip().rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return clean + "/chat/completions"
