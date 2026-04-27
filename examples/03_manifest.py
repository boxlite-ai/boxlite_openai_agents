"""Manifest entries — drop-in equivalent of the official Sandbox Agents tutorial.

Mirrors the renewal-packet example from
https://developers.openai.com/api/docs/guides/agents/sandboxes
and swaps the local provider for BoxLite. Demonstrates three of the four
non-cloud entry types you will reach for first:

    File       inline bytes (helper data, prompts, output sentinels)
    Dir        empty directory the agent will populate
    LocalDir   a folder on the host materialized into the guest workspace

``GitRepo`` is shown in commented form — it requires the guest to have ``git``
installed and the host to have egress to the repo's git host.

Cloud mounts (``S3Mount`` / ``GCSMount`` / ``R2Mount`` / ``AzureBlobMount`` /
``BoxMount``) raise ``MountConfigError`` on BoxLite v0.1 — see README §
Production runbook for the recommended workarounds.

Run::

    export OPENAI_API_KEY=sk-...
    python examples/03_manifest.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from agents.sandbox.entries import Dir, File, LocalDir

from boxlite_openai_agents import BoxLiteSandboxClient, BoxLiteSandboxClientOptions


async def main() -> None:
    # Stage some host data so LocalDir has something real to mount.
    with tempfile.TemporaryDirectory(prefix="boxlite-example-") as host_dir:
        host = Path(host_dir)
        (host / "row_001.csv").write_text("id,amount\n1,100\n")
        (host / "row_002.csv").write_text("id,amount\n2,200\n")

        manifest = Manifest(
            entries={
                "account_brief.md": File(
                    content=(
                        b"# Northwind Health\n"
                        b"- Segment: Mid-market healthcare analytics provider.\n"
                        b"- Renewal date: 2026-04-15.\n"
                    ),
                ),
                "skills":   Dir(),                       # empty output dir
                "data":     LocalDir(src=host),          # host folder → guest /workspace/data
                # "schemas": GitRepo(host="github.com", repo="owner/schemas", ref="main"),
            },
        )

        agent = SandboxAgent(
            name="Renewal Packet Analyst",
            instructions=(
                "Use the shell tool to inspect /workspace before answering. "
                "Cite the file paths backing each statement."
            ),
            default_manifest=manifest,
            capabilities=[Shell()],     # required for non-OpenAI LLMs; safe default
        )

        result = await Runner.run(
            agent,
            "List the data files and quote the renewal date from the brief.",
            run_config=RunConfig(
                sandbox=SandboxRunConfig(
                    client=BoxLiteSandboxClient(),
                    options=BoxLiteSandboxClientOptions(image="python:3.12-slim"),
                ),
                workflow_name="boxlite-manifest-demo",
            ),
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
