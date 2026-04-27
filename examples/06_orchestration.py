"""Orchestration — handoff and ``agent.as_tool`` with BoxLite.

Maps to https://developers.openai.com/api/docs/guides/agents/orchestration

Two patterns shown:

1. **Handoff**: a non-sandbox triage agent routes workspace-heavy work to a
   ``SandboxAgent`` running on BoxLite. After the handoff the SandboxAgent
   becomes the active agent for the rest of the run.
2. **agent.as_tool**: an outer SandboxAgent calls another SandboxAgent as a
   tool. Each can have its own ``BoxLiteSandboxClient`` instance — meaning
   each gets its own MicroVM, so workspaces are fully isolated even within
   one ``Runner.run``.

Run::

    export OPENAI_API_KEY=sk-...
    python examples/06_orchestration.py
"""

from __future__ import annotations

import asyncio

from agents import Agent, Runner
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


async def main() -> None:
    # Two independent BoxLite clients = two isolated MicroVM tenants.
    client_main = BoxLiteSandboxClient()
    options = BoxLiteSandboxClientOptions(image="python:3.12-slim")

    # --- Specialist sandbox agent ---
    renewal_analyst = SandboxAgent(
        name="Renewal Analyst",
        instructions=(
            "You analyse renewal materials in a sandbox. Use the shell tool to "
            "inspect /workspace and summarise findings briefly."
        ),
        capabilities=[Shell()],
    )

    # --- Triage agent (no sandbox) routes to the analyst when needed ---
    triage = Agent(
        name="Triage",
        instructions=(
            "If the user asks about renewals, hand off to the Renewal Analyst. "
            "Otherwise, answer directly."
        ),
        handoffs=[renewal_analyst],
    )

    result = await Runner.run(
        triage,
        "Hand off to the Renewal Analyst and ask it to confirm the sandbox is reachable "
        "by running `echo handoff-ok`.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(client=client_main, options=options),
            workflow_name="boxlite-orchestration-demo",
        ),
        max_turns=8,
    )
    print(f"final_output={result.final_output!r}")
    print(f"last_agent={result.last_agent.name!r}")


if __name__ == "__main__":
    asyncio.run(main())
