"""Manifest materialization unit tests."""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest
from agents.sandbox.errors import InvalidManifestPathError, MountConfigError

from boxlite_openai_agents.manifest_apply import (
    _resolve_under_root,
    _materialize_entry,
)


def test_absolute_path_rejected():
    with pytest.raises(InvalidManifestPathError):
        _resolve_under_root(PurePosixPath("/workspace"), "/etc/passwd")


def test_parent_traversal_rejected():
    with pytest.raises(InvalidManifestPathError):
        _resolve_under_root(PurePosixPath("/workspace"), "../etc/passwd")


def test_normal_path_resolves_under_root():
    out = _resolve_under_root(PurePosixPath("/workspace"), "subdir/file.txt")
    assert str(out) == "/workspace/subdir/file.txt"


@pytest.mark.asyncio
async def test_unsupported_mount_entry_raises():
    class FakeS3Mount:
        bucket = "x"

    # Type name contains "Mount" so we route to the v0.2-deferred branch.
    FakeS3Mount.__name__ = "S3Mount"

    class FakeBox:
        async def exec(self, *a, **kw):  # pragma: no cover - not reached
            raise AssertionError("should not exec on rejected entry")

    with pytest.raises(MountConfigError):
        await _materialize_entry(FakeBox(), PurePosixPath("/workspace/data"), FakeS3Mount())
