"""Tracing — wrap a BoxLite-backed run inside a `trace` context.

Maps to https://openai.github.io/openai-agents-python/tracing/

Every sandbox operation (exec, read, write, persist, hydrate, ...) emits a
standard SDK span. With BoxLite the spans carry ``backend_id="boxlite"`` so
you can filter your dashboard by sandbox provider. The trace tree is
structurally identical to a Docker / E2B run for the same agent, which means
your existing observability (LangSmith, Langfuse, AgentOps, OTLP exporters)
keeps working.

**Air-gapped reminder**: by default the SDK exports traces to OpenAI. If your
deployment must not phone home, see README §3 — `set_tracing_disabled(True)`
or wire up a local OTLP exporter via `add_trace_processor`.

Run::

    export OPENAI_API_KEY=sk-...
    python examples/09_tracing.py
"""

from __future__ import annotations

import asyncio

from agents import Runner, trace
from agents.run import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


async def main() -> None:
    agent = SandboxAgent(
        name="Traced Reviewer",
        instructions=(
            "Use the shell tool to run `echo trace-ok` and reply with the stdout."
        ),
        capabilities=[Shell()],
    )

    with trace("renewal-week"):
        result = await Runner.run(
            agent,
            "Run the readiness check.",
            run_config=RunConfig(
                sandbox=SandboxRunConfig(
                    client=BoxLiteSandboxClient(),
                    options=BoxLiteSandboxClientOptions(image="python:3.12-slim"),
                ),
                workflow_name="boxlite-tracing-demo",
            ),
            max_turns=4,
        )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
