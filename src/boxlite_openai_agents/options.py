"""``BoxLiteSandboxClientOptions`` — runtime configuration for a BoxLite session.

Subclasses :class:`agents.sandbox.session.sandbox_client.BaseSandboxClientOptions`,
which auto-registers the class globally via the ``type`` literal default.
"""

from __future__ import annotations

from typing import Literal

from agents.sandbox.session.sandbox_client import BaseSandboxClientOptions
from pydantic import Field


class BoxLiteSandboxClientOptions(BaseSandboxClientOptions):
    """Per-session BoxLite settings passed at ``client.create()``.

    Mirrors the shape of ``DockerSandboxClientOptions`` and
    ``E2BSandboxClientOptions`` so users can swap providers via a one-line edit.
    """

    type: Literal["boxlite"] = "boxlite"

    image: str = Field(
        default="python:3.12-slim",
        description="OCI image reference for the guest rootfs.",
    )
    cpus: int = Field(default=2, ge=1, le=64)
    memory_mib: int = Field(default=1024, ge=64)
    workspace_root: str = Field(
        default="/workspace",
        description="Guest path used as the SDK workspace root.",
    )
    egress_allowlist: tuple[str, ...] = Field(
        default=(),
        description=(
            "Domains/IPs allowed for outbound network traffic. Empty (the "
            "default) means deny-all — recommended for untrusted agent code."
        ),
    )
    auto_remove: bool = Field(
        default=True,
        description=(
            "Whether the underlying BoxLite box is removed when the session is "
            "deleted. Set False if you want to inspect a failed session."
        ),
    )
    boot_timeout_s: float = Field(default=30.0, gt=0)
