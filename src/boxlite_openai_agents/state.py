"""``BoxLiteSandboxSessionState`` — JSON-serializable handle for resume."""

from __future__ import annotations

from typing import Literal

from agents.sandbox.session import SandboxSessionState


class BoxLiteSandboxSessionState(SandboxSessionState):
    """State payload that survives across ``serialize_session_state`` round-trips.

    ``box_id`` lets ``client.resume()`` reattach to the same MicroVM if it is
    still alive, and otherwise hydrate a replacement from ``self.snapshot``.
    """

    type: Literal["boxlite"] = "boxlite"

    box_id: str = ""
    box_name: str = ""
    workspace_root: str = "/workspace"
    image: str = ""
