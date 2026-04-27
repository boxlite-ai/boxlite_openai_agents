"""Journey-test fixtures: a recorder that captures per-stage outputs and a
session-level finalizer that writes the report to disk after pytest finishes.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest


_REPORT_PATH_JSON = (
    Path(__file__).resolve().parent.parent.parent
    / "integration_design"
    / "journey-report.json"
)


@pytest.fixture(scope="session")
def journey_state() -> dict[str, Any]:
    return {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().sysname + " " + os.uname().release,
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "stages": [],
    }


@pytest.fixture
def journey_recorder(request, journey_state):
    def record(label: str, output: str) -> None:
        journey_state["stages"].append(
            {
                "test": request.node.name,
                "label": label,
                "output": output,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return record


def pytest_sessionfinish(session, exitstatus):
    state = getattr(session, "_journey_state", None)
    if state is None:
        # The fixture never instantiated (no live tests collected).
        return
    state["exit_status"] = exitstatus
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _REPORT_PATH_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    """Stash the session-scoped journey_state on the session object so
    pytest_sessionfinish can find it.
    """
    pass


@pytest.fixture(autouse=True)
def _stash_state(request, journey_state):
    request.session._journey_state = journey_state
    yield
