"""``BoxLiteSandboxClient`` — OpenAI Agents SDK provider entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.sandbox.errors import WorkspaceStartError
from agents.sandbox.manifest import Manifest
from agents.sandbox.session import SandboxSession, SandboxSessionState
from agents.sandbox.session.dependencies import Dependencies
from agents.sandbox.session.manager import Instrumentation
from agents.sandbox.session.sandbox_client import BaseSandboxClient
from agents.sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot

from .options import BoxLiteSandboxClientOptions
from .session import BoxLiteSandboxSession
from .snapshot import BoxLiteSnapshot
from .state import BoxLiteSandboxSessionState

if TYPE_CHECKING:
    from boxlite import Box, Boxlite


class BoxLiteSandboxClient(BaseSandboxClient[BoxLiteSandboxClientOptions]):
    """Local, embedded MicroVM sandbox client for the OpenAI Agents SDK.

    Construct without arguments for the typical case; the underlying
    ``Boxlite`` runtime is created lazily on the first call. Pass an existing
    runtime via ``runtime=`` for tests or shared singletons.
    """

    backend_id: str = "boxlite"
    supports_default_options: bool = True

    def __init__(
        self,
        *,
        runtime: "Boxlite | None" = None,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        self._runtime = runtime
        self._instrumentation = instrumentation
        self._dependencies = dependencies

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: BoxLiteSandboxClientOptions,
    ) -> SandboxSession:
        runtime = await self._ensure_runtime()
        manifest = manifest or Manifest(entries={})

        try:
            box = await self._spawn_box(runtime, options)
        except Exception as e:
            raise WorkspaceStartError(path=Path(options.workspace_root)) from e

        # SDK's resolve_snapshot needs an id to bind a SnapshotSpec → SnapshotBase.
        # We use the BoxLite snapshot name we'd produce on the first persist().
        snapshot_id = f"oai-{box.id}"
        resolved_snapshot = resolve_snapshot(snapshot, snapshot_id)

        state = BoxLiteSandboxSessionState(
            type="boxlite",
            snapshot=resolved_snapshot,
            manifest=manifest,
            exposed_ports=tuple(_collect_exposed_ports(manifest)),
            box_id=box.id,
            box_name=getattr(box, "name", "") or "",
            workspace_root=options.workspace_root,
            image=options.image,
        )

        inner = BoxLiteSandboxSession(box=box, state=state, reused_existing_box=False)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        if not isinstance(state, BoxLiteSandboxSessionState):
            # Best effort: upcast through the registry.
            state = SandboxSessionState.parse(state)
            if not isinstance(state, BoxLiteSandboxSessionState):
                raise TypeError(
                    f"BoxLiteSandboxClient.resume requires BoxLiteSandboxSessionState, "
                    f"got {type(state).__name__}"
                )

        runtime = await self._ensure_runtime()

        # Try to reattach to the original box.
        box = None
        reused = False
        if state.box_id:
            try:
                box = await runtime.get(state.box_id)
            except Exception:
                box = None
        if box is not None:
            reused = True
        else:
            # Recreate the box; the session will hydrate from state.snapshot
            # via the upstream snapshot-restore lifecycle.
            options = BoxLiteSandboxClientOptions(
                image=state.image or "python:3.12-slim",
                workspace_root=state.workspace_root or "/workspace",
            )
            box = await self._spawn_box(runtime, options)
            # Tell the base it must re-prepare workspace.
            state.workspace_root_ready = False

        inner = BoxLiteSandboxSession(box=box, state=state, reused_existing_box=reused)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def delete(self, session: SandboxSession) -> SandboxSession:
        # Drill through the SDK wrapper to the underlying inner session.
        inner = getattr(session, "_inner", session)
        if not isinstance(inner, BoxLiteSandboxSession):
            return session

        runtime = await self._ensure_runtime()
        box_id = inner.state.box_id
        if not box_id:
            return session

        try:
            await runtime.remove(box_id, force=True)
        except Exception as e:
            # Mirror Docker client: if the box is already gone, swallow.
            if _looks_like_not_found(e):
                return session
            raise
        return session

    def deserialize_session_state(
        self, payload: dict[str, object]
    ) -> SandboxSessionState:
        return BoxLiteSandboxSessionState.model_validate(payload)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _ensure_runtime(self) -> "Boxlite":
        if self._runtime is None:
            try:
                from boxlite import Boxlite  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover - import guard
                raise RuntimeError(
                    "boxlite-openai-agents requires the `boxlite` package. "
                    "Install it via `pip install boxlite`."
                ) from e
            self._runtime = await Boxlite.default()  # type: ignore[attr-defined]
        return self._runtime

    async def _spawn_box(
        self, runtime: "Boxlite", options: BoxLiteSandboxClientOptions
    ) -> "Box":
        from boxlite import BoxOptions, NetworkConfig  # type: ignore[import-not-found]

        network = NetworkConfig(
            allow_net=list(options.egress_allowlist),
        )
        box_opts = BoxOptions(
            image=options.image,
            cpus=options.cpus,
            memory_mib=options.memory_mib,
            auto_remove=options.auto_remove,
            network=network,
        )
        box = await runtime.create(box_opts)
        await box.start()
        return box


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _collect_exposed_ports(manifest: Manifest) -> tuple[int, ...]:
    """Collect declared exposed ports from a Manifest, if the version exposes them.

    The SDK Manifest may carry an ``exposed_ports`` field on some versions or
    an ``environment`` extra on others. We tolerate both shapes and degrade to
    an empty tuple if neither is present.
    """
    direct = getattr(manifest, "exposed_ports", None)
    if isinstance(direct, (list, tuple)):
        return tuple(int(p) for p in direct if isinstance(p, int))
    env = getattr(manifest, "environment", None)
    if env is not None:
        ports = getattr(env, "exposed_ports", None)
        if isinstance(ports, (list, tuple)):
            return tuple(int(p) for p in ports if isinstance(p, int))
    return ()


def _looks_like_not_found(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return any(
        hint in msg or hint in name
        for hint in ("notfound", "not_found", "no such", "missing")
    )


__all__ = ["BoxLiteSandboxClient"]
