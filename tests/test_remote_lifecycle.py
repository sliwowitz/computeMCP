# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: ISC
"""Remote deadlines use real shell processes, never a container runtime."""

import asyncio
import fcntl
import os
import shlex
import signal
import subprocess
from types import SimpleNamespace

import pytest

from compute_mcp.tunnel import TunnelError, _run_bounded_command


class LocalRoute:
    """Execute the SSH command locally to exercise its process-group boundary."""

    async def run(self, command, **kwargs):
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            capture_output=True,
            timeout=kwargs["timeout"],
        )
        return SimpleNamespace(
            exit_status=result.returncode,
            exit_signal=("KILL", False, "", "")
            if result.returncode == -signal.SIGKILL
            else None,
            stdout=result.stdout,
            stderr=result.stderr,
        )


async def test_remote_deadline_preserves_shell_quoting():
    result = await _run_bounded_command(
        LocalRoute(),
        shlex.join(["printf", "%s", "a 'quoted' value"]),
        2,
    )
    assert result.exit_status == 0
    assert result.stdout == b"a 'quoted' value"


@pytest.mark.parametrize("ignore_term", [False, True])
async def test_remote_deadline_releases_child_locks_but_leaves_detached_monitor(
    tmp_path,
    ignore_term,
):
    lock = tmp_path / "provision.lock"
    monitor_pid = tmp_path / "monitor.pid"
    monitor = shlex.join(
        [
            "bash",
            "-c",
            f"exec 9>&-; echo $$ > {shlex.quote(str(monitor_pid))}; exec sleep 60",
        ]
    )
    command = f"""
        exec 9>{shlex.quote(str(lock))}
        flock 9
        setsid {monitor} >/dev/null 2>&1 </dev/null &
        {"trap '' TERM" if ignore_term else ":"}
        sleep 60 &
        wait
    """
    try:
        with pytest.raises(TunnelError, match="timeout limit: 1s"):
            await _run_bounded_command(LocalRoute(), command, 1)
        # The helper AND its waiting children no longer retain the lock.
        with lock.open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.kill(int(monitor_pid.read_text()), 0)
    finally:
        if monitor_pid.exists():
            os.kill(int(monitor_pid.read_text()), signal.SIGKILL)


async def test_ssh_deadline_reports_unconfirmed_cleanup():
    class SilentRoute:
        async def run(self, command, **kwargs):
            raise TimeoutError

    with pytest.raises(TunnelError, match="2s; cleanup could not be confirmed"):
        await _run_bounded_command(SilentRoute(), "true", 2)
