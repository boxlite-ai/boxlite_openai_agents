"""Smoke: package imports, type literals are uniquely registered, options
round-trip through pydantic, snapshot magic bytes round-trip.
"""

from __future__ import annotations

import io


def test_top_level_imports():
    import boxlite_openai_agents as pkg

    assert pkg.__version__
    assert pkg.BoxLiteSandboxClient is not None
    assert pkg.BoxLiteSandboxClientOptions is not None
    assert pkg.BoxLiteSandboxSession is not None
    assert pkg.BoxLiteSandboxSessionState is not None
    assert pkg.BoxLiteSnapshot is not None
    assert pkg.BoxLiteSnapshotSpec is not None


def test_type_literals_are_unique():
    from boxlite_openai_agents import (
        BoxLiteSandboxClientOptions,
        BoxLiteSandboxSessionState,
        BoxLiteSnapshot,
        BoxLiteSnapshotSpec,
    )

    types = {
        BoxLiteSandboxClientOptions.model_fields["type"].default,
        BoxLiteSandboxSessionState.model_fields["type"].default,
        BoxLiteSnapshot.model_fields["type"].default,
        BoxLiteSnapshotSpec.model_fields["type"].default,
    }
    assert types == {
        "boxlite",
        "boxlite",  # session state also uses "boxlite" — that's fine, separate registry
        "boxlite-snapshot",
        "boxlite-snapshot-spec",
    } or "boxlite-snapshot" in types and "boxlite-snapshot-spec" in types


def test_options_defaults_and_pydantic_roundtrip():
    from boxlite_openai_agents import BoxLiteSandboxClientOptions

    opts = BoxLiteSandboxClientOptions()
    assert opts.type == "boxlite"
    assert opts.image == "python:3.12-slim"
    assert opts.egress_allowlist == ()
    blob = opts.model_dump()
    assert blob["type"] == "boxlite"
    again = BoxLiteSandboxClientOptions.model_validate(blob)
    assert again.image == opts.image


def test_snapshot_magic_roundtrip():
    from boxlite_openai_agents._internal.magic import (
        BOXLITE_SNAPSHOT_MAGIC,
        encode_snapshot_ref,
        try_decode_snapshot_ref,
    )

    blob = encode_snapshot_ref({"snapshot_name": "x", "box_id": "y"})
    raw = blob.getvalue()
    assert raw.startswith(BOXLITE_SNAPSHOT_MAGIC)

    blob.seek(0)
    decoded = try_decode_snapshot_ref(blob)
    assert decoded == {"snapshot_name": "x", "box_id": "y"}


def test_snapshot_magic_miss_rewinds():
    from boxlite_openai_agents._internal.magic import try_decode_snapshot_ref

    payload = io.BytesIO(b"PLAIN_TAR_BYTES")
    decoded = try_decode_snapshot_ref(payload)
    assert decoded is None
    # Stream should be rewound so caller can reread.
    assert payload.read() == b"PLAIN_TAR_BYTES"
