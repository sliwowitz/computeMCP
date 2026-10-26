# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: ISC
"""Apptainer teardown must not hide failures or discard live endpoint state."""

import subprocess

import pytest

from test_provision_shell import (
    CONTAINER,
    PROVISION,
    _base_apptainer_env,
    _write_stub,
    stubs as stubs,
)

TARGET = "stop-test"
INSTANCE = "computemcp-stop-test"
PORT = 2222
ENDPOINT = f"127.0.0.1:{PORT}\n"
FAILURE = 23

APPTAINER = '''#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$TEST_LOG"
case "$*" in
    'instance list')
        if [ "$TEST_SCENARIO" = list-failure ]; then
            echo 'cannot list instances' >&2
            exit 23
        fi
        echo 'INSTANCE NAME PID IP IMAGE'
        echo 'computemcp-stop-test-other 456 other image'
        if [ -f "$TEST_RUNNING" ]; then
            echo 'computemcp-stop-test 123 local image'
        fi
        ;;
    'instance stop computemcp-stop-test')
        if [ "$TEST_SCENARIO" = stop-failure ]; then
            echo 'cannot stop instance' >&2
            exit 23
        fi
        rm "$TEST_RUNNING"
        ;;
    *) echo "unexpected runtime command: $*" >&2; exit 99 ;;
esac
'''


@pytest.mark.parametrize("script", [CONTAINER, PROVISION], ids=["container", "provision"])
@pytest.mark.parametrize("action", ["stop", "close"])
@pytest.mark.parametrize("scenario", ["running", "absent", "list-failure", "stop-failure"])
def test_apptainer_stop(script, action, scenario, tmp_path, stubs):
    """Run the real helpers against a stub; no installed runtime is invoked."""
    storage = tmp_path / "storage"
    state = storage / TARGET / "state"
    state.mkdir(parents=True)
    (state / "mode").write_text("direct\n")
    endpoint = state / "container.endpoint"
    endpoint.write_text(ENDPOINT)
    running = tmp_path / "running"
    if scenario != "absent":
        running.touch()
    log = tmp_path / "runtime.log"
    _write_stub(stubs.direct.bin / "apptainer", APPTAINER)
    env = _base_apptainer_env(TARGET, storage, PORT, INSTANCE)
    env.update(
        PATH=stubs.direct.shell_path,
        TEST_SCENARIO=scenario,
        TEST_LOG=str(log),
        TEST_RUNNING=str(running),
    )

    def stop():
        return subprocess.run(
            ["bash", str(script), action], env=env, text=True,
            capture_output=True, timeout=10,
        )

    result = stop()
    if scenario.endswith("failure"):
        assert result.returncode == FAILURE, result.stderr
        assert ("cannot list" if scenario == "list-failure" else "cannot stop") in result.stderr
        assert endpoint.read_text() == ENDPOINT
        assert running.exists()
        assert "Stopped direct container" not in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert not endpoint.exists()
        assert not running.exists()
        repeated = stop()
        assert repeated.returncode == 0, repeated.stderr

    calls = log.read_text().splitlines()
    assert calls.count(f"instance stop {INSTANCE}") == int(scenario in {"running", "stop-failure"})
