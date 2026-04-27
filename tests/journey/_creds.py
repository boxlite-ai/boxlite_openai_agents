"""Load custom OpenAI-compatible LLM credentials from
``integration_design/KEY/custom_openai_llm.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

from openai import AsyncOpenAI


_KEY_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "integration_design"
    / "KEY"
    / "custom_openai_llm.md"
)


def load_creds() -> dict[str, str]:
    text = _KEY_FILE.read_text()
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def normalize_base_url(url: str) -> str:
    for suffix in ("/chat/completions",):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def build_openai_client() -> tuple[str, AsyncOpenAI]:
    """Return ``(model_name, AsyncOpenAI client)`` ready for
    ``OpenAIChatCompletionsModel``.
    """
    creds = load_creds()
    return creds["model"], AsyncOpenAI(
        base_url=normalize_base_url(creds["base_url"]),
        api_key=creds["llm_key"],
    )
