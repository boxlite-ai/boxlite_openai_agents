"""Function tools — harness-side helpers running alongside a BoxLite sandbox.

Maps to https://developers.openai.com/api/docs/guides/agents/tools

Important boundary: ``function_tool`` callables run in your **harness
process**, not inside the sandbox MicroVM. That means:

- Tools have full access to your application's secrets, network, and DB.
- Sandbox is for the agent's code and untrusted artifacts only.
- Cloud retrieval (FileSearch, WebSearch, hosted MCP, etc.) lives at the
  harness layer and is unaffected by the sandbox boundary.

Run::

    export OPENAI_API_KEY=sk-...
    python examples/07_function_tool.py
"""

from __future__ import annotations

import asyncio

from agents import Runner, function_tool
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


# ---- Harness-side tool: a fake internal directory lookup ----
@function_tool
def lookup_user(employee_id: str) -> str:
    """Resolve an employee ID to a display name. Returns 'Lin' for emp_42."""
    catalogue = {"emp_42": "Lin", "emp_07": "Marcus"}
    return catalogue.get(employee_id, "unknown")


async def main() -> None:
    agent = SandboxAgent(
        name="Renewal Coordinator",
        instructions=(
            "You can look up employees with the lookup_user tool, and you can "
            "run shell commands inside a sandbox via the shell tool. Use the "
            "right tool for each step. Reply with one short sentence."
        ),
        capabilities=[Shell()],
        tools=[lookup_user],
    )

    result = await Runner.run(
        agent,
        "Look up emp_42 and tell me the display name.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(
                client=BoxLiteSandboxClient(),
                options=BoxLiteSandboxClientOptions(image="python:3.12-slim"),
            ),
            workflow_name="boxlite-tool-demo",
        ),
        max_turns=4,
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
