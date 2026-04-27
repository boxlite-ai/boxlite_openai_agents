"""``BoxLiteSandboxSession`` — adapter that drives a live BoxLite ``Box``.

This subclass deliberately does *not* override ``ls`` / ``rm`` / ``mkdir`` /
``extract`` / ``apply_patch``: the upstream ``BaseSandboxSession`` already
implements them via ``_exec_internal``, and BoxLite's own SDK uses the same
"everything goes through exec" design — overriding would only duplicate code
and add bugs.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.sandbox.errors import (
    ExecNonZeroError,
    ExecTimeoutError,
    PtySessionNotFoundError,
)
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.session.pty_types import PtyExecUpdate, allocate_pty_process_id
from agents.sandbox.types import ExecResult, ExposedPortEndpoint, User

from ._internal.magic import (
    BOXLITE_SNAPSHOT_MAGIC,
    encode_snapshot_ref,
    try_decode_snapshot_ref,
)
from .errors import (
    map_exec_transport_errors,
    map_read_errors,
    map_start_errors,
    map_stop_errors,
    map_write_errors,
)
from .manifest_apply import apply_manifest
from .state import BoxLiteSandboxSessionState

if TYPE_CHECKING:
    from boxlite import Box


class BoxLiteSandboxSession(BaseSandboxSession):
    """One BoxLite MicroVM session bound to one ``SandboxSessionState``.

    Lifecycle ordering (enforced by upstream ``BaseSandboxSession``):
        ``start`` → ``_ensure_backend_started`` → ``_prepare_backend_workspace``
        → ``_start_workspace`` → ``_after_start``;
        ``stop`` → ``_before_stop`` → ``_persist_snapshot`` → ``_after_stop``;
        ``shutdown`` → ``_shutdown_backend``;
        ``aclose`` → ``stop`` → ``shutdown`` → close deps.

    The client owns the BoxLite ``Box`` lifecycle; this class never calls
    ``runtime.remove(...)``. That is ``client.delete()``'s job.
    """

    state: BoxLiteSandboxSessionState

    def __init__(
        self,
        *,
        box: "Box",
        state: BoxLiteSandboxSessionState,
        reused_existing_box: bool = False,
    ) -> None:
        self._box = box
        self.state = state
        self._reused_existing_box = reused_existing_box
        self._pty_lock = asyncio.Lock()
        # session_id (int) → (Execution, stdin)
        self._pty_executions: dict[int, Any] = {}

    def supports_pty(self) -> bool:
        return True

    @property
    def box(self) -> "Box":
        return self._box

    # ------------------------------------------------------------------
    # Lifecycle hooks (called by BaseSandboxSession)
    # ------------------------------------------------------------------

    async def _ensure_backend_started(self) -> None:
        with map_start_errors(Path(self.state.workspace_root)):
            info = await self._box.info()
            status = getattr(info, "status", None)
            running = (
                str(status).lower() == "running" if status is not None else False
            )
            if not running:
                await self._box.start()

            # Critical gotcha #2: tell base session whether the workspace files
            # may already be in place (reattach) or must be re-applied (fresh).
            self._set_start_state_preserved(
                workspace=self._reused_existing_box,
                system=self._reused_existing_box,
            )

    async def _prepare_backend_workspace(self) -> None:
        # Make sure /workspace exists; idempotent.
        await self._exec_or_raise(["mkdir", "-p", self.state.workspace_root])
        # The base will mark workspace_root_ready=True on success; we materialize
        # the manifest only on a fresh backend (preserved=False).
        if not self._reused_existing_box:
            await apply_manifest(self._box, self.state.manifest)

    async def _after_start_failed(self) -> None:
        # Best-effort: stop the box so the user is not left with a half-booted
        # MicroVM. delete() is the client's job.
        try:
            await self._box.stop()
        except Exception:
            pass

    async def _before_stop(self) -> None:
        await self.pty_terminate_all()

    async def _shutdown_backend(self) -> None:
        # We do not delete the box here — that's client.delete(). Just make
        # sure the guest is stopped so resources are released.
        with map_stop_errors(Path(self.state.workspace_root)):
            try:
                info = await self._box.info()
                status = getattr(info, "status", None)
                if status is not None and str(status).lower() == "running":
                    await self._box.stop()
            except Exception:
                # Box might already be gone (e.g., delete() raced); ignore.
                pass

    # ------------------------------------------------------------------
    # exec / pty / read / write / running
    # ------------------------------------------------------------------

    async def _exec_internal(
        self,
        *command: str | os.PathLike[str],
        timeout: float | None = None,
    ) -> ExecResult:
        argv = [str(arg) for arg in command]
        try:
            with map_exec_transport_errors(argv):
                execution = await self._box.exec(argv, timeout=timeout)
            result = await execution.wait()
        except asyncio.TimeoutError as e:
            raise ExecTimeoutError(command=tuple(argv), timeout_s=timeout or 0.0) from e
        except Exception as e:
            # If wait() hit a transport-layer issue (FFI/RPC) classify it.
            name = type(e).__name__.lower()
            if "timeout" in name:
                raise ExecTimeoutError(command=tuple(argv), timeout_s=timeout or 0.0) from e
            raise

        stdout, stderr, exit_code = _drain_exec_result(result)
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    async def pty_exec_start(
        self,
        *command: str | os.PathLike[str],
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
        tty: bool = False,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        # The SDK's PTY contract for non-streaming backends: run the command
        # to completion synchronously, then return a single PtyExecUpdate
        # with the full output and exit_code. ``yield_time_s`` /
        # ``max_output_tokens`` shape streamed responses; for our blocking
        # implementation they are advisory and we cap output at the limit.
        argv = self._prepare_exec_command(*command, shell=shell, user=user)
        async with self._pty_lock:
            process_id = allocate_pty_process_id(self._pty_executions)
            with map_exec_transport_errors(argv):
                execution = await self._box.exec(
                    argv,
                    timeout=timeout,
                    tty=True,
                    user=_user_name(user),
                )
            self._pty_executions[process_id] = execution

        try:
            result = await execution.wait()
        except Exception as e:
            async with self._pty_lock:
                self._pty_executions.pop(process_id, None)
            raise

        stdout, stderr, exit_code = _drain_exec_result(result)
        output = stdout + stderr
        if max_output_tokens is not None and len(output) > max_output_tokens * 4:
            output = output[: max_output_tokens * 4]

        async with self._pty_lock:
            self._pty_executions.pop(process_id, None)

        return PtyExecUpdate(
            process_id=process_id,
            output=output,
            exit_code=exit_code,
            original_token_count=None,
        )

    async def pty_write_stdin(
        self,
        *,
        session_id: int,
        chars: str,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        _ = (yield_time_s, max_output_tokens)
        async with self._pty_lock:
            execution = self._pty_executions.get(session_id)
        if execution is None:
            raise PtySessionNotFoundError(session_id=session_id)
        stdin = execution.stdin()
        write = stdin.write
        data = chars.encode("utf-8") if isinstance(chars, str) else chars
        write_result = write(data)
        if asyncio.iscoroutine(write_result):
            await write_result
        return PtyExecUpdate(
            process_id=session_id,
            output=b"",
            exit_code=None,
            original_token_count=None,
        )

    async def pty_terminate_all(self) -> None:
        async with self._pty_lock:
            executions = list(self._pty_executions.items())
            self._pty_executions.clear()
        for _sid, execution in executions:
            try:
                kill = execution.kill()
                if asyncio.iscoroutine(kill):
                    await kill
            except Exception:
                # Best-effort; PTY processes are about to die with the VM anyway.
                pass

    async def read(
        self,
        path: Path,
        *,
        user: str | User | None = None,
    ) -> io.IOBase:
        _ = user  # BoxLite reads don't honor `user`; the file is fetched as host.
        with map_read_errors(path):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            try:
                await self._box.copy_out(str(path), tmp_path)
                # Hand back a file object the caller will close.
                return open(tmp_path, "rb")  # noqa: SIM115 — caller closes
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    async def write(
        self,
        path: Path,
        data: io.IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        _ = user
        payload = data.read()
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(
                f"write() data stream must yield bytes, got {type(payload).__name__}"
            )
        with map_write_errors(path, context="copy_in"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(payload)
                tmp_path = tmp.name
            try:
                # Ensure parent dir exists; this matches Docker session behavior.
                parent = str(Path(path).parent)
                if parent and parent != ".":
                    await self._exec_or_raise(["mkdir", "-p", parent])
                await self._box.copy_in(tmp_path, str(path))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def running(self) -> bool:
        try:
            info = await self._box.info()
        except Exception:
            return False
        status = getattr(info, "status", None)
        if status is None:
            return False
        return str(status).lower() == "running"

    # ------------------------------------------------------------------
    # persist / hydrate (native CoW + tar fallback)
    # ------------------------------------------------------------------

    async def persist_workspace(self) -> io.IOBase:
        """Snapshot the workspace.

        Default path: native QCOW2 CoW snapshot via ``box.snapshots().create``.
        We hand the SDK a magic-prefixed JSON ref instead of inline tar bytes;
        ``hydrate_workspace()`` decodes that ref on the way back in.

        Fallback: if the manifest forces a tar fallback (e.g. ephemeral mounts
        that don't survive native snapshot), we use ``box.export(...)``.
        """
        if self._native_snapshot_requires_tar_fallback():
            return await self._persist_workspace_via_tar()

        snapshot_name = f"oai-{self.state.session_id.hex}"
        try:
            snapshots = self._box.snapshots()
            await snapshots.create(snapshot_name)
        except Exception as e:
            raise _wrap_snapshot_persist_error(e, snapshot_name) from e

        ref = {
            "kind": "native",
            "box_id": self._box.id,
            "snapshot_name": snapshot_name,
            "core_version": _safe_boxlite_version(),
        }
        return encode_snapshot_ref(ref)

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        ref = try_decode_snapshot_ref(data)
        if ref is None:
            await self._hydrate_workspace_via_tar(data)
            return

        snapshot_name = ref.get("snapshot_name")
        if not isinstance(snapshot_name, str):
            raise ValueError("BoxLite snapshot ref missing snapshot_name")

        # Native CoW restore requires the box to be stopped first.
        try:
            info = await self._box.info()
            status = getattr(info, "status", None)
            if status is not None and str(status).lower() == "running":
                await self._box.stop()
            snapshots = self._box.snapshots()
            await snapshots.restore(snapshot_name)
            await self._box.start()
        except Exception as e:
            raise _wrap_snapshot_restore_error(e, snapshot_name) from e

    async def _persist_workspace_via_tar(self) -> io.IOBase:
        """Fallback path: ``box.export(...)`` the workspace as a tar archive.

        Note: BoxLite's export lays out a ``.boxlite`` archive; we tar the
        workspace dir inside the guest instead so the bytes are SDK-portable
        (workspace-relative members, no backend-specific prefix).
        """
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # Tar the workspace from inside the guest, then stream out.
            guest_tar = "/tmp/boxlite-workspace.tar"
            await self._exec_or_raise(
                [
                    "sh",
                    "-c",
                    f"cd {_shquote(self.state.workspace_root)} && "
                    f"tar -cf {_shquote(guest_tar)} . ",
                ]
            )
            await self._box.copy_out(guest_tar, tmp_path)
            return open(tmp_path, "rb")  # noqa: SIM115 — caller closes
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    async def _hydrate_workspace_via_tar(self, data: io.IOBase) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp.write(data.read())
            tmp_path = tmp.name
        try:
            guest_tar = "/tmp/boxlite-workspace-restore.tar"
            await self._box.copy_in(tmp_path, guest_tar)
            await self._exec_or_raise(["mkdir", "-p", self.state.workspace_root])
            await self._exec_or_raise(
                [
                    "sh",
                    "-c",
                    f"tar -xf {_shquote(guest_tar)} -C {_shquote(self.state.workspace_root)}",
                ]
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Port resolution
    # ------------------------------------------------------------------

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        # BoxLite networking publishes mappings via NetworkConfig.port_mappings
        # which we record in self.state.exposed_ports. The mapping is symmetric
        # (host_port == guest_port) by default, with localhost host ip.
        from agents.sandbox.errors import ExposedPortUnavailableError

        if port not in self.state.exposed_ports:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="not_configured",
                context={"backend": "boxlite"},
            )
        try:
            info = await self._box.info()
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "boxlite", "detail": "info_failed"},
                cause=e,
            ) from e

        # ``info`` should expose a ``port_mappings`` (guest → host) dict.
        mappings = getattr(info, "port_mappings", None) or {}
        host_port: int | None = None
        if isinstance(mappings, dict):
            host_port = mappings.get(port) or mappings.get(str(port))
            if isinstance(host_port, str) and host_port.isdigit():
                host_port = int(host_port)
        if not isinstance(host_port, int):
            host_port = port  # default 1:1 mapping when backend doesn't surface one

        return ExposedPortEndpoint(host="127.0.0.1", port=host_port)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _exec_or_raise(self, argv: list[str]) -> None:
        result = await self._exec_internal(*argv)
        if result.exit_code != 0:
            raise ExecNonZeroError(result, command=tuple(argv))


def _drain_exec_result(result: Any) -> tuple[bytes, bytes, int]:
    """Normalize a BoxLite ``ExecResult`` into ``(stdout, stderr, exit_code)``.

    BoxLite returns either methods (``result.stdout()``) or attributes
    depending on the binding revision; we tolerate both.
    """

    def _read(field: str) -> bytes:
        attr = getattr(result, field, b"")
        if callable(attr):
            attr = attr()
        if attr is None:
            return b""
        if isinstance(attr, str):
            return attr.encode("utf-8")
        if isinstance(attr, (bytes, bytearray)):
            return bytes(attr)
        return str(attr).encode("utf-8")

    stdout = _read("stdout")
    stderr = _read("stderr")
    code_attr = getattr(result, "code", None)
    if callable(code_attr):
        exit_code = int(code_attr())
    elif isinstance(code_attr, int):
        exit_code = code_attr
    else:
        exit_code = int(getattr(result, "exit_code", 0))
    return stdout, stderr, exit_code


def _user_name(user: str | User | None) -> str | None:
    if user is None:
        return None
    if isinstance(user, str):
        return user or None
    return getattr(user, "name", None) or None


def _safe_boxlite_version() -> str:
    try:
        import boxlite

        return getattr(boxlite, "__version__", "")
    except Exception:
        return ""


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _wrap_snapshot_persist_error(exc: BaseException, snapshot_name: str) -> Exception:
    from agents.sandbox.errors import SnapshotPersistError

    return SnapshotPersistError(
        snapshot_id=snapshot_name,
        path=Path(f"<boxlite-snapshot:{snapshot_name}>"),
        cause=exc,
    )


def _wrap_snapshot_restore_error(exc: BaseException, snapshot_name: str) -> Exception:
    from agents.sandbox.errors import SnapshotRestoreError

    return SnapshotRestoreError(
        snapshot_id=snapshot_name,
        path=Path(f"<boxlite-snapshot:{snapshot_name}>"),
        cause=exc,
    )


__all__ = ["BoxLiteSandboxSession"]
