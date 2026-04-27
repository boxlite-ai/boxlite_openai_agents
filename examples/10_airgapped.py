"""Air-gapped agent — BoxLite's headline differentiator.

No official OpenAI tutorial covers this end-to-end because the hosted
sandbox providers cannot deliver it: by definition an air-gapped agent
cannot phone home. The only way the OpenAI Agents SDK reaches an
air-gapped deployment is via a local sandbox backend — that is BoxLite.

What this example proves:

1. Sandbox **egress is deny-all by default** (`egress_allowlist=()`).
   Anything the agent tries to fetch from the public internet hangs / errors.
2. SDK **tracing is disabled** so the harness itself never reaches out to
   `api.openai.com`.
3. The model can be a local LiteLLM-driven Ollama endpoint or an internal
   gateway — pick whichever you have on the secure side. (Code is shown for
   both; uncomment the one you need.)

Run::

    # Option A — internal OpenAI-compatible gateway:
    python examples/10_airgapped.py

    # Option B — Ollama locally:
    pip install 'openai-agents[litellm]'
    ollama pull qwen2.5-coder
    python examples/10_airgapped.py
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from agents import OpenAIChatCompletionsModel, Runner, set_tracing_disabled
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
    for suffix in ("/chat/completions",):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


async def main() -> None:
    # ---- 1. Stop the SDK from exporting traces to OpenAI ----
    set_tracing_disabled(True)

    # ---- 2. Pick a model the secure environment can actually reach ----
    creds = load_creds()
    model = OpenAIChatCompletionsModel(
        model=creds["model"],
        openai_client=AsyncOpenAI(
            base_url=normalize_base_url(creds["base_url"]),
            api_key=creds["llm_key"],
        ),
    )
    # # Or, fully local via LiteLLM + Ollama:
    # from agents.extensions.models.litellm_model import LitellmModel
    # model = LitellmModel(model="ollama/qwen2.5-coder", api_key="")

    # ---- 3. Sandbox: deny-all egress (the default) ----
    options = BoxLiteSandboxClientOptions(
        image="python:3.12-slim",
        egress_allowlist=(),     # explicit: no outbound network from the guest
    )

    agent = SandboxAgent(
        name="Air-gapped Coder",
        instructions=(
            "You MUST use the exec_command tool. Run the requested shell "
            "command exactly as written and reply with only the stdout."
        ),
        capabilities=[Shell()],
        model=model,
    )

    result = await Runner.run(
        agent,
        "Run `echo offline-ok` in the sandbox and report the stdout.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(client=BoxLiteSandboxClient(), options=options),
            workflow_name="boxlite-airgapped-demo",
        ),
        max_turns=4,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
