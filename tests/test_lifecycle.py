"""End-to-end lifecycle on the fake BoxLite backend.

Covers: create → exec → write/read → persist (native CoW) → delete → resume.
"""

from __future__ import annotations

import io

import pytest
from agents.sandbox.manifest import Manifest

from boxlite_openai_agents import (
    BoxLiteSandboxClient,
    BoxLiteSandboxClientOptions,
    BoxLiteSandboxSessionState,
)


@pytest.mark.asyncio
async def test_create_exec_running_delete():
    client = BoxLiteSandboxClient()
    session = await client.create(
        manifest=Manifest(entries={}),
        options=BoxLiteSandboxClientOptions(),
    )
    inner = session._inner if hasattr(session, "_inner") else session
    try:
        async with session:
            assert await session.running() is True
            result = await session.exec("echo", "hello")
            assert result.exit_code == 0
            assert b"hello" in result.stdout
    finally:
        await client.delete(session)


@pytest.mark.asyncio
async def test_write_then_read_roundtrip():
    client = BoxLiteSandboxClient()
    session = await client.create(
        manifest=Manifest(entries={}),
        options=BoxLiteSandboxClientOptions(),
    )
    try:
        async with session:
            from pathlib import Path

            await session.write(Path("/workspace/note.txt"), io.BytesIO(b"hello-world"))
            stream = await session.read(Path("/workspace/note.txt"))
            try:
                assert stream.read() == b"hello-world"
            finally:
                stream.close()
    finally:
        await client.delete(session)


@pytest.mark.asyncio
async def test_native_cow_persist_and_resume_roundtrip():
    """persist_workspace should hand back our magic-prefixed ref;
    serialize/deserialize must round-trip; resume must reattach."""
    from pathlib import Path

    from boxlite_openai_agents._internal.magic import BOXLITE_SNAPSHOT_MAGIC

    client = BoxLiteSandboxClient()
    session = await client.create(
        manifest=Manifest(entries={}),
        options=BoxLiteSandboxClientOptions(),
    )
    try:
        async with session:
            # Touch the workspace so we can verify post-resume.
            await session.write(Path("/workspace/marker"), io.BytesIO(b"v1"))

            ref_stream = await session.persist_workspace()
            blob = ref_stream.getvalue() if hasattr(ref_stream, "getvalue") else ref_stream.read()
            assert blob.startswith(BOXLITE_SNAPSHOT_MAGIC), (
                "persist_workspace should return a native CoW magic-prefixed ref"
            )

        # Serialize → deserialize via the SDK contract.
        inner = session._inner if hasattr(session, "_inner") else session
        payload = client.serialize_session_state(inner.state)
        restored_state = client.deserialize_session_state(payload)
        assert isinstance(restored_state, BoxLiteSandboxSessionState)
        assert restored_state.box_id == inner.state.box_id

        # Resume reattaches to the same fake box.
        resumed = await client.resume(restored_state)
        try:
            async with resumed:
                stream = await resumed.read(Path("/workspace/marker"))
                try:
                    assert stream.read() == b"v1"
                finally:
                    stream.close()
        finally:
            await client.delete(resumed)
    finally:
        # Original session was already cleaned up by the resume path.
        pass


@pytest.mark.asyncio
async def test_delete_is_idempotent_when_box_is_gone():
    client = BoxLiteSandboxClient()
    session = await client.create(
        manifest=Manifest(entries={}),
        options=BoxLiteSandboxClientOptions(),
    )
    await client.delete(session)
    # Second delete must not raise.
    await client.delete(session)


@pytest.mark.asyncio
async def test_exec_nonzero_exit_does_not_raise():
    """exec() should return a non-zero exit code, not raise."""
    client = BoxLiteSandboxClient()
    session = await client.create(
        manifest=Manifest(entries={}),
        options=BoxLiteSandboxClientOptions(),
    )
    try:
        async with session:
            result = await session.exec("false")
            assert result.exit_code == 1
    finally:
        await client.delete(session)
