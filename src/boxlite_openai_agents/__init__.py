"""``boxlite-openai-agents`` — local MicroVM sandbox provider for OpenAI Agents SDK.

Public surface (mirrors the shape of the official 7 hosted providers):

    >>> from boxlite_openai_agents import (
    ...     BoxLiteSandboxClient,
    ...     BoxLiteSandboxClientOptions,
    ... )
"""

from __future__ import annotations

from .client import BoxLiteSandboxClient
from .options import BoxLiteSandboxClientOptions
from .session import BoxLiteSandboxSession
from .snapshot import BoxLiteSnapshot, BoxLiteSnapshotSpec
from .state import BoxLiteSandboxSessionState

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BoxLiteSandboxClient",
    "BoxLiteSandboxClientOptions",
    "BoxLiteSandboxSession",
    "BoxLiteSandboxSessionState",
    "BoxLiteSnapshot",
    "BoxLiteSnapshotSpec",
]
