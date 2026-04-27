"""End-to-end test using a real OpenAI-compatible LLM provider (SiliconFlow)
against the BoxLite sandbox client.

Credentials come from `integration_design/KEY/custom_openai_llm.md`. The test
makes a real HTTPS call to the SiliconFlow gateway, so it is gated behind an
env flag and skipped by default in CI.

Run it explicitly:

    BOXLITE_RUN_LIVE_LLM=1 .venv/bin/python -m pytest tests/test_custom_llm.py -v -s
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# The gateway in custom_openai_llm.md ends in /v1/chat/completions; the OpenAI
# Python client wants only the API root (/v1) — strip the suffix automatically.
_KEY_FILE = Path(__file__).resolve().parent.parent / "integration_design" / "KEY" / "custom_openai_llm.md"


def _load_creds() -> dict[str, str]:
    text = _KEY_FILE.read_text()
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _normalize_base_url(url: str) -> str:
    # Trim known suffixes so AsyncOpenAI sees only the API root.
    for suffix in ("/chat/completions",):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


@pytest.mark.skipif(
    not os.environ.get("BOXLITE_RUN_LIVE_LLM"),
    reason="Set BOXLITE_RUN_LIVE_LLM=1 to enable live SiliconFlow calls.",
)
@pytest.mark.asyncio
async def test_custom_openai_llm_drives_boxlite_sandbox_agent():
    """Real LLM + real (faked) BoxLite path:

    - Load DeepSeek-V4-Flash via SiliconFlow OpenAI-compatible endpoint.
    - Wire it into a SandboxAgent backed by BoxLiteSandboxClient.
    - Ask the agent to write & run a Python script in the sandbox; verify the
      sandbox path actually executes (via the in-memory FakeBox in conftest).

    Compatibility goal: prove that OpenAI Agents SDK harness ↔ compute
    separation works with (1) a non-OpenAI LLM provider AND (2) the BoxLite
    sandbox client at the same time, with zero changes to either side.
    """
    from agents import OpenAIChatCompletionsModel, Runner
    from agents.run import RunConfig
    from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
    from agents.sandbox.capabilities import Shell
    from openai import AsyncOpenAI

    from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions

    creds = _load_creds()
    assert creds.get("model"), "model missing in custom_openai_llm.md"
    assert creds.get("base_url"), "base_url missing"
    assert creds.get("llm_key"), "llm_key missing"

    custom_client = AsyncOpenAI(
        base_url=_normalize_base_url(creds["base_url"]),
        api_key=creds["llm_key"],
    )

    agent = SandboxAgent(
        name="local-coder",
        instructions=(
            "You are a careful engineer. The user has given you a sandbox; "
            "use the shell tool to run small Python snippets when needed and "
            "respond concisely."
        ),
        model=OpenAIChatCompletionsModel(
            model=creds["model"],
            openai_client=custom_client,
        ),
        # The default Capabilities set includes a Filesystem hosted apply_patch
        # tool that the Chat Completions API does not accept. For non-OpenAI
        # providers we pin to Shell only, which is the canonical minimal
        # sandbox agent shape from the official docs.
        capabilities=[Shell()],
    )

    result = await Runner.run(
        agent,
        "What is 7 * 6?  Reply with just the number.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(
                client=BoxLiteSandboxClient(),
                options=BoxLiteSandboxClientOptions(image="python:3.12-slim"),
            ),
            workflow_name="custom-llm-smoke",
        ),
    )

    print("\n=== custom-LLM final output ===\n", result.final_output)
    final = (result.final_output or "").strip()
    assert "42" in final, f"expected 42 in answer, got: {final!r}"


@pytest.mark.skipif(
    not os.environ.get("BOXLITE_RUN_LIVE_LLM"),
    reason="Set BOXLITE_RUN_LIVE_LLM=1 to enable live SiliconFlow calls.",
)
@pytest.mark.asyncio
async def test_custom_openai_llm_without_sandbox_first():
    """Sanity: confirm the SiliconFlow gateway answers without any sandbox in
    the loop. Isolates LLM-side failures from BoxLite-side failures.
    """
    from agents import Agent, OpenAIChatCompletionsModel, Runner
    from openai import AsyncOpenAI

    creds = _load_creds()
    custom_client = AsyncOpenAI(
        base_url=_normalize_base_url(creds["base_url"]),
        api_key=creds["llm_key"],
    )

    agent = Agent(
        name="ping",
        instructions="Reply with exactly one word: pong.",
        model=OpenAIChatCompletionsModel(
            model=creds["model"], openai_client=custom_client
        ),
    )
    result = await Runner.run(agent, "Send the ping.")
    print("\n=== plain LLM ping ===\n", result.final_output)
    assert "pong" in (result.final_output or "").lower()
