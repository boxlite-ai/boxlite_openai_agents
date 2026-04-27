"""Capabilities — Shell + Memory across two runs.

Maps to the *Capabilities* portion of
https://developers.openai.com/api/docs/guides/agents/sandboxes

The first turn writes a short memory note. The second turn resumes the same
sandbox session and proves the note survived. With BoxLite the persistence
goes through native QCOW2 copy-on-write, so the resume is sub-second on a
warm host.

Note: ``Capabilities.default()`` includes ``Filesystem()`` which exposes a
hosted ``apply_patch`` tool. That tool is incompatible with the OpenAI
Chat Completions API used by every non-OpenAI provider — see README
§ Production runbook #1. We pin to ``[Shell(), Memory()]`` here for
maximum portability.

Run::

    export OPENAI_API_KEY=sk-...
    python examples/04_capabilities.py
"""

from __future__ import annotations

import asyncio
import json

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Memory, Shell

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


async def main() -> None:
    agent = SandboxAgent(
        name="Memory-enabled reviewer",
        instructions=(
            "You are inspecting a renewal packet. Use the shell tool to look at "
            "any files under /workspace, then write a short note to memory "
            "summarizing one unresolved blocker. Keep responses concise."
        ),
        default_manifest=Manifest(entries={}),
        capabilities=[Shell(), Memory()],
    )

    client = BoxLiteSandboxClient()
    options = BoxLiteSandboxClientOptions(image="python:3.12-slim")

    # ---- Turn 1: gather a finding, persist memory ----
    session = await client.create(manifest=agent.default_manifest, options=options)
    async with session:
        first = await Runner.run(
            agent,
            "Record one renewal blocker for Northwind: 'security questionnaire pending'.",
            run_config=RunConfig(
                sandbox=SandboxRunConfig(session=session),
                workflow_name="boxlite-capabilities-demo",
            ),
        )
        await session.persist_workspace()
        # capture the state for cross-turn resume
        state = session._inner.state if hasattr(session, "_inner") else session.state
        blob = json.dumps(client.serialize_session_state(state))

    print("Turn 1:", first.final_output)

    # ---- Turn 2: resume from snapshot, recall the note ----
    restored = client.deserialize_session_state(json.loads(blob))
    resumed = await client.resume(restored)
    try:
        async with resumed:
            second = await Runner.run(
                agent,
                "What blocker did you record earlier? Reply in one sentence.",
                run_config=RunConfig(
                    sandbox=SandboxRunConfig(session=resumed),
                    workflow_name="boxlite-capabilities-demo",
                ),
            )
        print("Turn 2:", second.final_output)
    finally:
        await client.delete(resumed)


if __name__ == "__main__":
    asyncio.run(main())
