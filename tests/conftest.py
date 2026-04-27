"""Test setup: wire up a fake ``boxlite`` module so unit tests can run on a host
without KVM or HVF.

The fake mirrors the public surface our adapter actually touches (Boxlite,
BoxOptions, NetworkConfig, Box, Execution, SnapshotHandle), so the adapter
exercises real code paths against an in-memory backend. Real-VM integration
tests live elsewhere and require a host with /dev/kvm or HVF.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


# ----------------------------------------------------------------------
# Fake boxlite implementation
# ----------------------------------------------------------------------


class FakeExecResult:
    def __init__(self, stdout: bytes, stderr: bytes, code: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._code = code

    def stdout(self) -> bytes:
        return self._stdout

    def stderr(self) -> bytes:
        return self._stderr

    def code(self) -> int:
        return self._code


class FakeStdin:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def write(self, data: bytes) -> None:
        self.buffer += data


class FakeExecution:
    def __init__(self, fn) -> None:
        self._fn = fn
        self._stdin = FakeStdin()
        self._killed = False
        self._result: FakeExecResult | None = None

    def stdin(self) -> FakeStdin:
        return self._stdin

    async def wait(self) -> FakeExecResult:
        if self._result is None:
            self._result = self._fn()
        return self._result

    async def kill(self) -> None:
        self._killed = True


class FakeSnapshotInfo:
    def __init__(self, name: str) -> None:
        self.name = name
        self.created_at = time.time()


class FakeSnapshotHandle:
    def __init__(self, box: "FakeBox") -> None:
        self._box = box

    async def create(self, name: str) -> FakeSnapshotInfo:
        snapshot_dir = self._box._snapshots_dir / name
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        # Copy current workspace to snapshot dir.
        if self._box._workspace.exists():
            shutil.copytree(
                self._box._workspace,
                snapshot_dir / "workspace",
                dirs_exist_ok=True,
            )
        info = FakeSnapshotInfo(name)
        self._box._snapshots[name] = info
        return info

    async def list(self) -> list[FakeSnapshotInfo]:
        return list(self._box._snapshots.values())

    async def get(self, name: str) -> FakeSnapshotInfo | None:
        return self._box._snapshots.get(name)

    async def restore(self, name: str) -> None:
        if name not in self._box._snapshots:
            raise KeyError(f"snapshot not found: {name}")
        snapshot_workspace = self._box._snapshots_dir / name / "workspace"
        if self._box._workspace.exists():
            shutil.rmtree(self._box._workspace)
        if snapshot_workspace.exists():
            shutil.copytree(snapshot_workspace, self._box._workspace)
        else:
            self._box._workspace.mkdir(parents=True, exist_ok=True)

    async def remove(self, name: str) -> None:
        info = self._box._snapshots.pop(name, None)
        if info is not None:
            shutil.rmtree(self._box._snapshots_dir / name, ignore_errors=True)


class FakeBoxInfo:
    def __init__(self, status: str, port_mappings: dict[int, int]) -> None:
        self.status = status
        self.port_mappings = port_mappings


class FakeBox:
    def __init__(self, runtime: "FakeBoxlite", options: Any) -> None:
        self.id = uuid.uuid4().hex
        self.name = ""
        self._runtime = runtime
        self._options = options
        self._status = "Created"
        self._root = Path(tempfile.mkdtemp(prefix=f"fakebox-{self.id}-"))
        self._workspace = self._root / "rootfs" / "workspace"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir = self._root / "snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots: dict[str, FakeSnapshotInfo] = {}
        self._removed = False
        self.exec_log: list[list[str]] = []

    async def start(self) -> None:
        self._status = "running"

    async def stop(self) -> None:
        self._status = "stopped"

    async def info(self) -> FakeBoxInfo:
        return FakeBoxInfo(self._status, port_mappings={})

    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        tty: bool = False,
        user: str | None = None,
    ) -> FakeExecution:
        self.exec_log.append(list(argv))
        # Trivially simulate a few commands so manifest_apply / lifecycle work.
        result = self._run_argv(argv)
        return FakeExecution(lambda: result)

    async def copy_in(self, host: str, guest: str) -> None:
        target = self._guest_to_host(guest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(host).is_dir():
            shutil.copytree(host, target, dirs_exist_ok=True)
        else:
            shutil.copy2(host, target)

    async def copy_out(self, guest: str, host: str) -> None:
        source = self._guest_to_host(guest)
        if not source.exists():
            raise FileNotFoundError(guest)
        if source.is_dir():
            shutil.copytree(source, host, dirs_exist_ok=True)
        else:
            shutil.copy2(source, host)

    def snapshots(self) -> FakeSnapshotHandle:
        return FakeSnapshotHandle(self)

    # ------------- internal helpers -------------
    def _guest_to_host(self, guest: str) -> Path:
        # Map /workspace/... to <root>/rootfs/workspace/...
        guest_path = Path(guest)
        if not guest_path.is_absolute():
            guest_path = Path("/") / guest_path
        return self._root / "rootfs" / Path(*guest_path.parts[1:])

    def _run_argv(self, argv: list[str]) -> FakeExecResult:
        # Best-effort minimal shell.
        if not argv:
            return FakeExecResult(b"", b"", 0)

        if argv[0] == "mkdir" and "-p" in argv:
            target = argv[-1]
            try:
                self._guest_to_host(target).mkdir(parents=True, exist_ok=True)
                return FakeExecResult(b"", b"", 0)
            except OSError as e:
                return FakeExecResult(b"", str(e).encode(), 1)

        if argv[0] == "true":
            return FakeExecResult(b"", b"", 0)
        if argv[0] == "false":
            return FakeExecResult(b"", b"", 1)
        if argv[0] == "echo":
            payload = (" ".join(argv[1:]) + "\n").encode()
            return FakeExecResult(payload, b"", 0)
        if argv[0] == "sh" and len(argv) >= 3 and argv[1] in ("-c", "-lc"):
            return self._run_shell(argv[2])

        # Default: pretend it ran.
        return FakeExecResult(b"", b"", 0)

    def _run_shell(self, script: str) -> FakeExecResult:
        # Support our tar-based persist/hydrate fallback by shelling out on host.
        import subprocess

        # Translate the workspace root marker so host tar sees the right dir.
        # The script references absolute guest paths; we rewrite /workspace and
        # /tmp/boxlite-workspace*.tar to host paths.
        host_workspace = str(self._workspace)
        host_tar = str(self._root / "boxlite-workspace.tar")
        host_restore = str(self._root / "boxlite-workspace-restore.tar")
        rewritten = (
            script
            .replace("/workspace", host_workspace)
            .replace("/tmp/boxlite-workspace.tar", host_tar)
            .replace("/tmp/boxlite-workspace-restore.tar", host_restore)
        )
        try:
            result = subprocess.run(
                rewritten,
                shell=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            return FakeExecResult(result.stdout, result.stderr, result.returncode)
        except Exception as e:
            return FakeExecResult(b"", str(e).encode(), 1)


@dataclass
class FakeNetworkConfig:
    allow_net: list[str] = field(default_factory=list)


@dataclass
class FakeBoxOptions:
    image: str = "python:3.12-slim"
    cpus: int = 2
    memory_mib: int = 1024
    auto_remove: bool = True
    network: FakeNetworkConfig | None = None
    env: dict[str, str] | None = None


class FakeBoxlite:
    def __init__(self) -> None:
        self._boxes: dict[str, FakeBox] = {}

    @classmethod
    async def default(cls) -> "FakeBoxlite":
        return cls()

    async def create(self, options: FakeBoxOptions) -> FakeBox:
        box = FakeBox(self, options)
        self._boxes[box.id] = box
        return box

    async def get(self, box_id: str) -> FakeBox | None:
        return self._boxes.get(box_id)

    async def remove(self, box_id: str, *, force: bool = False) -> None:
        box = self._boxes.pop(box_id, None)
        if box is None:
            class _NotFound(Exception):
                pass
            raise _NotFound(f"box not found: {box_id}")
        box._removed = True
        shutil.rmtree(box._root, ignore_errors=True)


def _install_fake_boxlite() -> ModuleType:
    """Install a fake ``boxlite`` module so the adapter can import without KVM.
    Returns the module so tests can grab a runtime reference.
    """
    if "boxlite" in sys.modules and isinstance(sys.modules["boxlite"], ModuleType) and getattr(
        sys.modules["boxlite"], "_FAKE", False
    ):
        return sys.modules["boxlite"]

    mod = ModuleType("boxlite")
    mod._FAKE = True
    mod.__version__ = "0.8.2-fake"
    mod.Boxlite = FakeBoxlite
    mod.Box = FakeBox
    mod.BoxOptions = FakeBoxOptions
    mod.NetworkConfig = FakeNetworkConfig
    sys.modules["boxlite"] = mod
    return mod


@pytest.fixture(autouse=True)
def fake_boxlite():
    mod = _install_fake_boxlite()
    yield mod


@pytest.fixture
async def fake_runtime(fake_boxlite):
    return await FakeBoxlite.default()
