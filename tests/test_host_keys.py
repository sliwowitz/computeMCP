# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: ISC
"""Verify default known-hosts checking on both direct and tunneled route hops."""

from contextlib import nullcontext

import asyncssh
import pytest

from compute_mcp.config import SSHConfig
from compute_mcp.ssh_backend import SSHError
from compute_mcp.tunnel import _dial_hop

LOCALHOST = "127.0.0.1"


class RouteServer(asyncssh.SSHServer):
    """Local test server requiring no user authentication."""

    def begin_auth(self, username):
        return False

    def connection_requested(self, dest_host, dest_port, orig_host, orig_port):
        return dest_host == LOCALHOST


@pytest.mark.parametrize("through_jump", [False, True])
@pytest.mark.parametrize("trusted", [False, True])
async def test_default_known_hosts(tmp_path, through_jump, trusted):
    host_key = asyncssh.generate_private_key("ssh-ed25519")
    async with asyncssh.listen(
        LOCALHOST, 0, server_factory=RouteServer, server_host_keys=[host_key],
    ) as server:
        port = server.get_port()
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        recorded = host_key if trusted else asyncssh.generate_private_key("ssh-ed25519")
        (ssh_dir / "known_hosts").write_bytes(
            f"[{LOCALHOST}]:{port} ".encode() + recorded.export_public_key()
        )
        info = {"hostname": LOCALHOST, "port": port, "user": "tester"}

        async def dial(tunnel=None, pin=None):
            return await _dial_hop(
                info, tunnel=tunnel, client_keys=[], prompter=None, pin=pin,
                ssh=SSHConfig(), name="test", fallback_user="tester",
            )

        jump = await dial(pin=host_key.get_fingerprint()) if through_jump else None
        async with jump if jump else nullcontext():
            if trusted:
                async with await dial(tunnel=jump):
                    pass
            else:
                with pytest.raises(SSHError, match="Host key is not trusted"):
                    await dial(tunnel=jump)
