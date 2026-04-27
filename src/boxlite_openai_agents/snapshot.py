"""``BoxLiteSnapshot`` — native QCOW2 CoW snapshot wrapper.

Why custom: the default ``LocalSnapshot`` from upstream serializes the full
workspace as a tar file on the host. BoxLite-core already implements QCOW2
backing-file forks (`Box.snapshots()`), so we wrap the native API and expose a
``SnapshotBase`` subclass that round-trips a small JSON ref through the
``persist`` / ``restore`` interface.

The byte stream we hand back from ``restore()`` is the same magic-prefixed ref
blob that ``BoxLiteSandboxSession.persist_workspace()`` produces, so the
session's ``hydrate_workspace()`` can detect it and take the native restore
path. This mirrors E2B's pattern (``_E2B_SANDBOX_SNAPSHOT_MAGIC``).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Literal

from agents.sandbox.errors import (
    SnapshotNotRestorableError,
    SnapshotPersistError,
    SnapshotRestoreError,
)
from agents.sandbox.session.dependencies import Dependencies
from agents.sandbox.snapshot import SnapshotBase, SnapshotSpec

from ._internal.magic import encode_snapshot_ref, try_decode_snapshot_ref


class BoxLiteSnapshot(SnapshotBase):
    """Native CoW snapshot ref managed by BoxLite-core.

    The ``id`` field is the BoxLite-side snapshot name; ``box_id`` lets us locate
    the source box on resume. The ``persist`` / ``restore`` methods round-trip
    the magic-prefixed JSON envelope so the session can detect the native path.
    """

    type: Literal["boxlite-snapshot"] = "boxlite-snapshot"

    box_id: str
    snapshot_name: str
    core_version: str = ""

    async def persist(
        self,
        data: io.IOBase,
        *,
        dependencies: Dependencies | None = None,
    ) -> None:
        # The session already created the snapshot inside BoxLite-core before
        # producing this stream. Persist is a no-op for native-CoW refs; we
        # only sanity-check the stream actually carries our magic.
        _ = dependencies
        ref = try_decode_snapshot_ref(data)
        if ref is None:
            raise SnapshotPersistError(
                snapshot_id=self.id,
                path=Path(f"<boxlite:{self.box_id}>"),
                cause=ValueError("expected a BoxLite native snapshot ref blob"),
            )

    async def restore(
        self,
        *,
        dependencies: Dependencies | None = None,
    ) -> io.IOBase:
        _ = dependencies
        try:
            payload: dict[str, Any] = {
                "kind": "native",
                "box_id": self.box_id,
                "snapshot_name": self.snapshot_name,
                "core_version": self.core_version,
            }
            return encode_snapshot_ref(payload)
        except Exception as e:
            raise SnapshotRestoreError(
                snapshot_id=self.id,
                path=Path(f"<boxlite:{self.box_id}>"),
                cause=e,
            ) from e

    async def restorable(
        self,
        *,
        dependencies: Dependencies | None = None,
    ) -> bool:
        # Without a live ``Boxlite`` runtime in scope here we cannot ping the
        # backing snapshot directly; the session will surface a
        # ``SnapshotNotRestorableError`` if the snapshot is gone at resume time.
        _ = dependencies
        return bool(self.snapshot_name)


class BoxLiteSnapshotSpec(SnapshotSpec):
    """Companion spec used by ``RunConfig`` when callers want a fresh snapshot
    bound at session creation time.
    """

    type: Literal["boxlite-snapshot-spec"] = "boxlite-snapshot-spec"

    box_id: str
    snapshot_name: str
    core_version: str = ""

    def build(self, snapshot_id: str) -> SnapshotBase:
        return BoxLiteSnapshot(
            id=snapshot_id,
            box_id=self.box_id,
            snapshot_name=self.snapshot_name,
            core_version=self.core_version,
        )


def raise_not_restorable(snapshot_id: str, box_id: str) -> None:
    """Helper used by the session when a hydrate ref points at a vanished box."""
    raise SnapshotNotRestorableError(
        snapshot_id=snapshot_id,
        path=Path(f"<boxlite:{box_id}>"),
    )
