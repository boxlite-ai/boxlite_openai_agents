"""Three-line BoxLite swap-in for OpenAI Agents SDK.

Run::

    pip install boxlite-openai-agents
    export OPENAI_API_KEY=sk-...
    python examples/01_quickstart.py

Requires KVM on Linux or Hypervisor.framework on macOS. The first run pulls
the OCI image; subsequent runs are warm in well under a second.
"""

from __future__ import annotations

import asyncio

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


async def main() -> None:
    agent = SandboxAgent(
        name="local-coder",
        instructions="You are a careful engineer. Use the sandbox to run code.",
    )

    result = await Runner.run(
        agent,
        "Write fizzbuzz.py for n=15 and run it. Print the output.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(
                client=BoxLiteSandboxClient(),
                options=BoxLiteSandboxClientOptions(
                    image="python:3.12-slim",
                    # Empty allowlist means deny-all outbound network. Add
                    # specific hosts here if your agent needs them.
                    egress_allowlist=(),
                ),
            ),
        ),
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
