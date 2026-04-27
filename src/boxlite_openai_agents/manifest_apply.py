"""Materialize a SDK ``Manifest`` into a live BoxLite guest workspace.

The SDK's ``Manifest.entries`` is a dict of ``BaseEntry`` subclasses keyed by
relative path. Each subclass knows how to apply itself (``entry.apply``), but
that contract is async and operates against a session abstraction. For the
provider adapter we instead inspect entry types directly so we can use
``Box.copy_in`` for the local-file fast path and ``Box.exec`` for everything
else.

Mount entries (S3/GCS/...) are deferred to v0.2; we surface a clear error for
those today rather than silently dropping them.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from agents.sandbox.errors import (
    InvalidManifestPathError,
    MountConfigError,
)
from agents.sandbox.manifest import Manifest

if TYPE_CHECKING:
    from boxlite import Box


def _resolve_under_root(root: PurePosixPath, rel: str | os.PathLike[str]) -> PurePosixPath:
    """Reject absolute paths and parent-traversal attempts; return joined path."""
    rel_str = str(rel)
    if rel_str.startswith("/"):
        raise InvalidManifestPathError(rel=rel_str, reason="absolute")
    candidate = (root / rel_str).resolve() if False else PurePosixPath(root, rel_str)
    # Manual escape check (PurePosixPath doesn't actually resolve symlinks).
    parts: list[str] = []
    for part in PurePosixPath(rel_str).parts:
        if part == "..":
            if not parts:
                raise InvalidManifestPathError(rel=rel_str, reason="escape_root")
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return PurePosixPath(root, *parts) if parts else PurePosixPath(root)


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _shquote(s: str) -> str:
    return _shell_quote(s)


async def _exec_or_raise(box: "Box", argv: list[str]) -> None:
    execution = await box.exec(argv)
    result = await execution.wait()
    code = getattr(result, "code", lambda: 0)()
    if code != 0:
        stderr = (
            getattr(result, "stderr", lambda: b"")() if callable(getattr(result, "stderr", None)) else b""
        )
        raise RuntimeError(
            f"manifest apply failed: argv={argv!r} code={code} stderr={stderr!r}"
        )


async def apply_manifest(box: "Box", manifest: Manifest | None) -> None:
    """Materialize ``manifest`` into ``box``.

    Idempotent for already-existing directories; safe to call after a resume
    that reattached to a live box (in which case the SDK passes only ephemeral
    entries via the base session's preserved-workspace flow).
    """
    if manifest is None:
        return

    root = PurePosixPath(manifest.root or "/workspace")
    await _exec_or_raise(box, ["mkdir", "-p", str(root)])

    for rel_key, entry in manifest.entries.items():
        target = _resolve_under_root(root, rel_key)
        await _materialize_entry(box, target, entry)


async def _materialize_entry(box: "Box", target: PurePosixPath, entry: Any) -> None:
    """Dispatch a single ``BaseEntry`` subclass to the right BoxLite primitive.

    We avoid ``isinstance`` against the SDK classes to keep the import surface
    minimal; instead we look at the entry's class name (which is part of its
    public contract — every entry registers a ``type`` literal upstream).
    """
    name = type(entry).__name__

    if name in {"Dir", "Directory"}:
        # Dir is recursive: it carries `.children` of nested BaseEntry. Create
        # the directory itself, then descend into children.
        await _exec_or_raise(box, ["mkdir", "-p", str(target)])
        children = _entry_attr(entry, "children", default={}) or {}
        for child_rel, child_entry in children.items():
            child_target = _resolve_under_root(target, str(child_rel))
            await _materialize_entry(box, child_target, child_entry)
        return

    if name == "File":
        # Inline content lives on the entry as ``content`` (bytes).
        content = _entry_attr(entry, "content", default=b"")
        await _exec_or_raise(box, ["mkdir", "-p", str(target.parent)])
        await _write_inline(box, target, content)
        return

    if name in {"LocalFile", "LocalDir"}:
        # Both LocalFile and LocalDir use `.src: Path` per the SDK schema; the
        # field was previously assumed to be `.path`, which broke materialization.
        src = _entry_attr(entry, "src", "path", default=None)
        if src is None:
            raise FileNotFoundError(f"local entry {name} has no src path")
        host_path = Path(src)
        if not host_path.exists():
            raise FileNotFoundError(f"local entry {name} missing on host: {host_path}")
        await _exec_or_raise(box, ["mkdir", "-p", str(target.parent)])
        await box.copy_in(str(host_path), str(target))
        return

    if name == "GitRepo":
        # SDK schema: `.host` (default github.com), `.repo` (owner/name), `.ref`,
        # `.subpath` (optional). Compose the clone URL ourselves.
        host = _entry_attr(entry, "host", default="github.com") or "github.com"
        repo = _entry_attr(entry, "repo")
        ref = _entry_attr(entry, "ref", default=None)
        subpath = _entry_attr(entry, "subpath", default=None)
        url = f"https://{host}/{repo}.git"
        await _exec_or_raise(box, ["mkdir", "-p", str(target.parent)])
        clone_target = str(target) if not subpath else "/tmp/_boxlite_git_clone"
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", str(ref)]
        cmd += [url, clone_target]
        await _exec_or_raise(box, cmd)
        if subpath:
            await _exec_or_raise(
                box,
                [
                    "sh",
                    "-c",
                    f"mv {_shquote(clone_target)}/{_shquote(str(subpath))} "
                    f"{_shquote(str(target))} && rm -rf {_shquote(clone_target)}",
                ],
            )
        return

    if "Mount" in name:
        # S3Mount / R2Mount / GCSMount / AzureBlobMount / BoxMount.
        # Deferred to v0.2; surface a clean error rather than silently no-op.
        raise MountConfigError(
            message=(
                f"BoxLite v0.1 does not yet support manifest entry `{name}`; "
                "use a LocalDir or GitRepo entry instead, or pin the package "
                "to v0.2+ once cloud mounts ship."
            ),
            context={"backend": "boxlite", "entry_type": name},
        )

    # Unknown subclass — never silently skip; surface clearly so users can pin
    # a newer adapter version that supports the new entry type.
    raise NotImplementedError(
        f"BoxLite v0.1 does not yet handle manifest entry type `{name}` "
        f"(at relpath {target!s})"
    )


def _entry_attr(entry: Any, *names: str, default: Any = ...) -> Any:
    for name in names:
        if hasattr(entry, name):
            return getattr(entry, name)
    if default is ...:
        raise AttributeError(f"entry {type(entry).__name__} has none of {names}")
    return default


async def _write_inline(box: "Box", target: PurePosixPath, content: str | bytes) -> None:
    """Write ``content`` to ``target`` inside the guest via copy_in."""
    if isinstance(content, str):
        data = content.encode("utf-8")
    elif isinstance(content, (bytes, bytearray)):
        data = bytes(content)
    else:
        raise TypeError(f"File entry content must be str/bytes, got {type(content).__name__}")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        await box.copy_in(tmp_path, str(target))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
