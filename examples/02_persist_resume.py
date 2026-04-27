"""Demonstrates BoxLite's native QCOW2 CoW snapshot through the OpenAI Agents
SDK ``persist_workspace()`` / ``resume()`` lifecycle.

The first turn installs ``pandas`` and snapshots. The second turn resumes from
that snapshot — no reinstall. On a warm host the resume completes in well
under a second because the SDK calls ``box.snapshots().restore()`` directly
rather than tar-extracting the workspace.
"""

from __future__ import annotations

import asyncio
import json

from agents.sandbox.manifest import Manifest

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


async def main() -> None:
    client = BoxLiteSandboxClient()
    options = BoxLiteSandboxClientOptions(image="python:3.12-slim")

    # ---- Turn 1: install something heavy and snapshot ----
    session = await client.create(manifest=Manifest(entries={}), options=options)
    async with session:
        await session.exec("pip", "install", "--quiet", "pandas")
        await session.persist_workspace()
        state = session._inner.state if hasattr(session, "_inner") else session.state
        serialized = client.serialize_session_state(state)

    # Persist anywhere — Redis, S3, a local file. JSON-safe by construction.
    payload = json.dumps(serialized)

    # ---- Turn 2: resume from the snapshot, no reinstall ----
    restored_state = client.deserialize_session_state(json.loads(payload))
    resumed = await client.resume(restored_state)
    async with resumed:
        result = await resumed.exec("python", "-c", "import pandas; print(pandas.__version__)")
        print("pandas version on resume:", result.stdout.decode().strip())

    await client.delete(resumed)


if __name__ == "__main__":
    asyncio.run(main())
