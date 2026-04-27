"""Custom OpenAI-compatible LLM provider — drive BoxLite without OpenAI keys.

Maps to the *Models and providers* tutorial
https://developers.openai.com/api/docs/guides/agents/models

Any OpenAI-Chat-Completions compatible gateway works — DeepSeek, Qwen via
SiliconFlow, OpenRouter, self-hosted vLLM, Ollama via LiteLLM, etc. This
example reads credentials from
``integration_design/KEY/custom_openai_llm.md`` (the same file the journey
test uses), so you can flip providers without touching code.

Two non-obvious details that bite first-time users:

1. **`capabilities=[Shell()]`** — the default ``Capabilities.default()`` adds
   a hosted ``apply_patch`` tool that the Chat Completions API does not
   support. Non-OpenAI providers will reject the request outright unless
   you trim the capability list.
2. **Imperative prompts** — reasoning models (Qwen3.6, DeepSeek-R1, QwQ)
   tend to "think" their way to an answer instead of calling the tool.
   Spell out *MUST use the tool* when you actually want a sandbox exec.

Run::

    python examples/05_custom_llm.py
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from agents import OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from openai import AsyncOpenAI

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


_KEY_FILE = (
    Path(__file__).resolve().parent.parent
    / "integration_design"
    / "KEY"
    / "custom_openai_llm.md"
)


def load_creds() -> dict[str, str]:
    creds: dict[str, str] = {}
    for line in _KEY_FILE.read_text().splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*(.+?)\s*$", line)
        if m:
            creds[m.group(1)] = m.group(2)
    return creds


def normalize_base_url(url: str) -> str:
    # Some gateways publish the URL with the chat-completions path baked in.
    # AsyncOpenAI wants only the API root.
    for suffix in ("/chat/completions",):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


async def main() -> None:
    creds = load_creds()
    custom_client = AsyncOpenAI(
        base_url=normalize_base_url(creds["base_url"]),
        api_key=creds["llm_key"],
    )

    agent = SandboxAgent(
        name="Custom-LLM Coder",
        instructions=(
            "You MUST use the exec_command tool. Do not compute anything yourself. "
            "When asked, run the shell command exactly as requested and reply with "
            "only the stdout."
        ),
        capabilities=[Shell()],     # required for non-OpenAI providers
        model=OpenAIChatCompletionsModel(
            model=creds["model"],
            openai_client=custom_client,
        ),
    )

    result = await Runner.run(
        agent,
        "Run `python -c 'print(7*6)'` in the sandbox and report the stdout.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(
                client=BoxLiteSandboxClient(),
                options=BoxLiteSandboxClientOptions(image="python:3.12-slim"),
            ),
            workflow_name="boxlite-custom-llm",
        ),
        max_turns=6,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
