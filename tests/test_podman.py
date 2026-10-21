# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: ISC
"""Podman configuration and shared lifecycle, using stub CLIs only."""

import fcntl
import os
import shlex
import subprocess

import pytest

from compute_mcp.config import parse_config
from compute_mcp.setup import collect_target, render_target_block
from test_config import base_raw
from test_provision_shell import (
    CONTAINER,
    DOCKER_STUB,
    PROVISION,
    _base_env,
    _cleanup,
    _free_port,
    _published_port,
    _write_stub,
    stubs as stubs,
)
from test_setup import _wizard, _write_and_load

LOOPBACK = "127.0.0.1"
NVIDIA_DEVICE = "nvidia.com/gpu=all"
PODMAN_IMAGE = "docker.io/library/ubuntu:24.04"
TARGET = "runtime-test"
CONTAINER_NAME = f"computemcp-{TARGET}"


def test_podman_config():
    """Podman is accepted without changing the runtime contract."""
    target = parse_config(base_raw(container={"runtime": "podman"})).targets["hal"]
    assert target.container.runtime == "podman"


def test_podman_wizard_round_trip(tmp_path):
    """The wizard accepts Podman and supplies a non-interactive image reference."""
    key = tmp_path / "key"
    answers = collect_target(
        _wizard(
            [
                TARGET,
                "tunnel",
                TARGET,
                "agent",
                str(key),
                "",
                "y",
                "y",
                "podman",
                str(tmp_path / "storage"),
                "",
                "nvidia",
                "y",
                str(tmp_path / "bundle"),
                "",
            ]
        ),
        set(),
    )
    config, _ = _write_and_load(tmp_path, answers)
    assert config.targets[TARGET].container.runtime == "podman"
    assert config.targets[TARGET].container.image == PODMAN_IMAGE
    answers.container_image = None
    assert PODMAN_IMAGE in render_target_block(answers)


@pytest.fixture(params=("docker", "podman"))
def runtime_env(request, tmp_path, stubs):
    """Record the actual executable used; never invoke an installed runtime."""
    runtime = request.param
    stub = DOCKER_STUB.replace("printf 'docker %s\\n'", "printf 'podman %s\\n'")
    _write_stub(stubs.direct.bin / "podman", stub)
    env = _base_env(TARGET, tmp_path / "storage", _free_port())
    env.update(
        COMPUTEMCP_CONTAINER_RUNTIME=runtime,
        DOCKER_STUB_LOG=str(tmp_path / "runtime.log"),
        DOCKER_STUB_STATE=str(tmp_path / "runtime-state"),
        DOCKER_STUB_DOCKERFILE_LOG=str(tmp_path / "Dockerfile"),
        DOCKER_STUB_PUBLISHED_PORT=str(
            _published_port(int(env["COMPUTEMCP_CONTAINER_PORT"]))
        ),
        PATH=stubs.direct.shell_path,
    )
    try:
        yield env
    finally:
        _cleanup()


def _run(script, action, env):
    """Run the shipped helper against the stub environment."""
    result = subprocess.run(
        ["bash", str(script), action],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("gpu", [False, True])
def test_runtime_lifecycle(runtime_env, tmp_path, gpu):
    """Build once, reuse, refresh the published port, stop and restart."""
    env = runtime_env
    runtime = env["COMPUTEMCP_CONTAINER_RUNTIME"]
    env["COMPUTEMCP_GPU_VENDORS"] = "nvidia" if gpu else ""
    endpoint = f"{LOOPBACK}:{env['DOCKER_STUB_PUBLISHED_PORT']}"
    state = tmp_path / "storage" / TARGET / "state"
    for _ in range(2):
        assert f"ENDPOINT {endpoint}" in _run(PROVISION, "provision", env)
    assert (state / "container.runtime").read_text().strip() == runtime

    trace = (tmp_path / "runtime.log").read_text().splitlines()
    assert all(line.startswith(f"{runtime} ") for line in trace)
    assert sum(line.startswith(f"{runtime} build ") for line in trace) == 1
    assert sum(line.startswith(f"{runtime} start ") for line in trace) == 1
    (create,) = [
        shlex.split(line) for line in trace if line.startswith(f"{runtime} create ")
    ]
    assert (
        create[create.index("--publish") + 1]
        == f"{LOOPBACK}::{env['COMPUTEMCP_CONTAINER_PORT']}"
    )
    assert ("--userns" in create) == (runtime == "podman")
    assert ("relabel=private" in create[create.index("--mount") + 1]) == (
        runtime == "podman"
    )
    assert f"AGENT_UID={os.getuid()}" in create
    assert f"AGENT_GID={os.getgid()}" in create
    if runtime == "podman":
        assert create[create.index("--userns") + 1] == "keep-id"
        assert create[create.index("--user") + 1] == "root"
        assert "--gpus" not in create
        assert (NVIDIA_DEVICE in create) == gpu
        assert ("NVIDIA_VISIBLE_DEVICES=void" in create) == gpu
        assert ("label=disable" in create) == gpu
    else:
        assert ("--gpus" in create) == gpu
    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert "openssh-server" in dockerfile
    assert "ENTRYPOINT" in dockerfile

    # Status without the original gateway environment must select the saved runtime,
    # not Docker merely because both executables are installed.
    restored = {
        key: value for key, value in env.items() if not key.startswith("COMPUTEMCP_")
    }
    restored.update(
        COMPUTEMCP_STATE_DIR=str(state), COMPUTEMCP_CONTAINER_NAME=CONTAINER_NAME
    )
    (state / "container.endpoint").write_text(f"{LOOPBACK}:1\n")
    assert endpoint in _run(PROVISION, "status", restored)
    assert (state / "container.endpoint").read_text().strip() == endpoint
    _run(CONTAINER, "update", env)
    _run(CONTAINER, "fingerprint", env)
    _run(PROVISION, "stop", restored)
    assert not (state / "container.endpoint").exists()
    assert f"ENDPOINT {endpoint}" in _run(PROVISION, "provision", env)
    _run(PROVISION, "close", restored)
    trace = (tmp_path / "runtime.log").read_text()
    assert trace.count(f"{runtime} start {CONTAINER_NAME}") == 2
    assert trace.count(f"{runtime} stop {CONTAINER_NAME}") == 2


def test_podman_refuses_foreign_container(runtime_env, tmp_path):
    """Ownership checks still protect pre-existing containers."""
    env = runtime_env
    _run(PROVISION, "provision", env)
    owner = tmp_path / "runtime-state" / f"ctr-{CONTAINER_NAME}" / "owner"
    owner.write_text(str(os.getuid() + 1))
    result = subprocess.run(
        ["bash", str(CONTAINER), "stop"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "expected owner/system labels" in result.stderr


def test_provision_releases_lock_with_container_running(runtime_env, tmp_path):
    """A persistent runtime monitor must not keep the provisioner's lock alive."""
    _run(PROVISION, "provision", runtime_env)
    lock_path = tmp_path / "storage" / TARGET / "state" / "provision.lock"
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_runtime_preserves_parent_locks_without_inheriting_them(runtime_env, tmp_path):
    """Only the caller owns the job (8) and provisioning (9) lock lifetimes."""
    _run(CONTAINER, "build", runtime_env)
    job_lock = tmp_path / "job-run.lock"
    provision_lock = tmp_path / "provision.lock"
    result = subprocess.run(
        [
            "bash",
            "-euc",
            """
            exec 8>"$1" 9>"$2"
            flock -n 8
            flock -n 9
            bash "$3" start
            if flock -n "$1" true; then echo 'parent job lock lost' >&2; exit 1; fi
            if flock -n "$2" true; then echo 'parent provision lock lost' >&2; exit 1; fi
            """,
            "lock-owner",
            str(job_lock),
            str(provision_lock),
            str(CONTAINER),
        ],
        env=runtime_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for path in (job_lock, provision_lock):
        with path.open("r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_busy_provisioner_fails_without_starting_runtime(runtime_env, tmp_path):
    """Contention is reported promptly, independently of allocation wait time."""
    state = tmp_path / "storage" / TARGET / "state"
    state.mkdir(parents=True)
    lock_path = state / "provision.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run(
            ["bash", str(PROVISION), "provision"],
            env=runtime_env,
            capture_output=True,
            text=True,
            timeout=20,
        )
    assert result.returncode == 75
    assert f"Target {TARGET} busy" in result.stderr
    assert str(lock_path) in result.stderr
    assert "within 10s" in result.stderr
    assert not (tmp_path / "runtime.log").exists()


def test_runtime_stop_failure_reaches_provisioner(runtime_env, tmp_path, stubs):
    """Refresh can only fail closed if the bundle reports runtime failures."""
    _run(PROVISION, "provision", runtime_env)
    runtime = stubs.direct.bin / runtime_env["COMPUTEMCP_CONTAINER_RUNTIME"]
    source = runtime.read_text()
    assert "    stop)\n" in source
    runtime.write_text(
        source.replace(
            "    stop)\n",
            '    stop)\n        echo "runtime stop failed" >&2; exit 23\n',
        )
    )
    result = subprocess.run(
        ["bash", str(PROVISION), "stop"],
        env=runtime_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 23
    assert "runtime stop failed" in result.stderr
