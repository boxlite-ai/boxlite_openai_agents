"""Magic header used to identify a native BoxLite snapshot reference inside the
``persist_workspace()`` byte stream.

Modeled after E2B's ``_E2B_SANDBOX_SNAPSHOT_MAGIC``: when the upstream SDK
expects an ``io.IOBase`` representing the workspace, we instead return a
small JSON ref blob prefixed with this magic. ``hydrate_workspace()`` peeks at
the prefix and dispatches to the native CoW restore path or falls back to tar
extraction.
"""

from __future__ import annotations

import io
import json
from typing import Any

BOXLITE_SNAPSHOT_MAGIC = b"BOXLITE_SANDBOX_SNAPSHOT_V1\n"


def encode_snapshot_ref(payload: dict[str, Any]) -> io.BytesIO:
    """Return an ``io.BytesIO`` carrying ``BOXLITE_SNAPSHOT_MAGIC + json``.

    Callers feed this directly back to the SDK as the result of
    ``persist_workspace()``.
    """
    blob = BOXLITE_SNAPSHOT_MAGIC + json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return io.BytesIO(blob)


def try_decode_snapshot_ref(data: io.IOBase) -> dict[str, Any] | None:
    """Peek at ``data`` for the magic prefix and, if present, return the JSON ref.

    On miss, the stream is rewound to its original position so the caller can
    fall through to tar extraction without losing bytes.
    """
    start = data.tell() if data.seekable() else None
    head = data.read(len(BOXLITE_SNAPSHOT_MAGIC))
    if head != BOXLITE_SNAPSHOT_MAGIC:
        if start is not None:
            data.seek(start)
        return None
    body = data.read()
    return json.loads(body.decode("utf-8"))
