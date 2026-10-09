# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: ISC
"""Exercise agent signing over real local SSH connections, without private files."""

import errno
import os
import re
import shutil
import socket
import subprocess
from contextlib import asynccontextmanager

import asyncssh
import pytest

from compute_mcp.bundle import BundleError, public_key_for
from compute_mcp.config import BundleConfig, ConfigError, SSHConfig, TargetConfig, TransportConfig
from compute_mcp.ssh_auth import AgentError, agent_socket, authentication_keys
from compute_mcp.ssh_backend import SSHBackend, SSHError, connect_ssh, dial_route
from compute_mcp.tunnel import _dial_hop, _resolve_route

LOCALHOST = "127.0.0.1"
TEST_PORT = 22
TIMEOUT = 5


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """Start an isolated agent; never alter the operator's agent."""
    if not shutil.which("ssh-agent") or not shutil.which("ssh-add"):
        pytest.skip("OpenSSH ssh-agent and ssh-add are required")
    path = str(tmp_path / "agent.sock")
    result = subprocess.run(
        ["ssh-agent", "-s", "-a", path],
        capture_output=True,
        text=True,
        check=True,
        timeout=TIMEOUT,
    )
    pid = re.search(r"SSH_AGENT_PID=(\d+)", result.stdout).group(1)
    env = {**os.environ, "SSH_AUTH_SOCK": path, "SSH_AGENT_PID": pid}
    monkeypatch.setenv("SSH_AUTH_SOCK", path)

    def command(*args):
        return subprocess.run(
            ["ssh-add", *args],
            env=env,
            capture_output=True,
            check=True,
            timeout=TIMEOUT,
        )

    def add(name):
        key = asyncssh.generate_private_key("ssh-ed25519")
        private = tmp_path / name
        private.write_bytes(key.export_private_key())
        private.chmod(0o600)
        public = tmp_path / (name + ".pub")
        public.write_bytes(key.export_public_key())
        command(str(private))
        private.unlink()
        return key, public

    try:
        yield path, add, command
    finally:
        subprocess.run(["ssh-agent", "-k"], env=env, capture_output=True, timeout=TIMEOUT)


@asynccontextmanager
async def login_server(key):
    """Accept one client key and provide a real server host key."""

    class Server(asyncssh.SSHServer):
        def begin_auth(self, username):
            return True

        def public_key_auth_supported(self):
            return True

        def validate_public_key(self, username, offered):
            return offered.public_data == key.public_data

        def connection_requested(self, dest_host, dest_port, orig_host, orig_port):
            return dest_host == LOCALHOST

    host_key = asyncssh.generate_private_key("ssh-ed25519")
    async with asyncssh.listen(
        LOCALHOST,
        0,
        server_factory=Server,
        server_host_keys=[host_key],
    ) as listener:
        yield listener.get_port(), host_key


async def route_connection(port, **kwargs):
    """Dial an isolated test server using the production route primitive."""
    return await dial_route(
        name="test",
        host=LOCALHOST,
        port=port,
        username="tester",
        client_keys=kwargs.pop("client_keys", None),
        passphrase=None,
        prompter=None,
        host_key_sha256=kwargs.pop("host_key_sha256", None),
        known_hosts=kwargs.pop("known_hosts", None),
        agent_path=kwargs.pop("agent_path", "SSH_AUTH_SOCK"),
        host_key_algorithms=(),
        connect_timeout=TIMEOUT,
        **kwargs,
    )


@pytest.mark.parametrize("hop", ["route", "container"])
async def test_agent_only_connection_without_private_file(agent, hop):
    _, add, _ = agent
    key, public = add("container")
    async with login_server(key) as (port, host_key):
        pin = host_key.get_fingerprint()
        if hop == "route":
            conn = await route_connection(
                port,
                client_keys=[str(public)],
                identities_only=True,
                host_key_sha256=pin,
            )
        else:
            target = TargetConfig(
                name="test",
                user="tester",
                client_public_key=str(public),
                agent_path="SSH_AUTH_SOCK",
                host_key_sha256=pin,
                transport=TransportConfig(kind="direct", remote_host=LOCALHOST, remote_port=port),
            )
            conn = await SSHBackend()._dial(target, LOCALHOST, port)
        async with conn:
            assert not conn.is_closed()


async def test_unselected_agent_identity_is_not_offered(agent):
    _, add, _ = agent
    accepted, _ = add("accepted")
    _, other_public = add("other")
    async with login_server(accepted) as (port, host_key):
        with pytest.raises(SSHError, match="Permission denied"):
            await route_connection(
                port,
                client_keys=[str(other_public)],
                identities_only=True,
                host_key_sha256=host_key.get_fingerprint(),
            )
        async with await route_connection(port, host_key_sha256=host_key.get_fingerprint()):
            pass  # Without IdentitiesOnly, the accepted identity is available.


async def test_proxyjump_uses_each_hops_agent_settings(agent, tmp_path, monkeypatch):
    path, add, _ = agent
    jump_key, jump_public = add("jump")
    route_key, route_public = add("route")
    async with login_server(jump_key) as (jump_port, jump_host_key):
        async with login_server(route_key) as (route_port, route_host_key):
            (tmp_path / ".ssh").mkdir()
            hosts = tmp_path / ".ssh" / "known_hosts"
            hosts.write_bytes(
                f"[{LOCALHOST}]:{jump_port} ".encode() + jump_host_key.export_public_key()
            )

            async def config(alias, ssh):
                is_jump = alias == "jump"
                return "\n".join(
                    [
                        f"hostname {LOCALHOST}",
                        "user tester",
                        f"port {jump_port if is_jump else route_port}",
                        f"identityfile {jump_public if is_jump else route_public}",
                        f"identityagent {path if is_jump else 'SSH_AUTH_SOCK'}",
                        "identitiesonly yes",
                        "" if is_jump else "proxyjump jump",
                    ]
                )

            monkeypatch.setattr("compute_mcp.tunnel._ssh_g", config)
            route = await _resolve_route("route", SSHConfig())
            jump = route["jumps"][0]
            assert jump["agent_path"] == path
            assert route["agent_path"] == "SSH_AUTH_SOCK"
            options = {
                "prompter": None, "ssh": SSHConfig(), "name": "test", "fallback_user": "tester",
                "agent_path": "SSH_AUTH_SOCK",
            }
            async with await _dial_hop(
                jump,
                tunnel=None,
                client_keys=list(jump["identityfiles"]),
                pin=None,
                **options,
            ) as jump_conn:
                async with await _dial_hop(
                    route,
                    tunnel=jump_conn,
                    client_keys=list(route["identityfiles"]),
                    pin=route_host_key.get_fingerprint(),
                    **options,
                ) as route_conn:
                    assert not route_conn.is_closed()


async def test_reconnect_reads_current_identities_and_never_loads_default_files(agent, tmp_path):
    _, add, command = agent
    key, public = add("first")
    default_dir = tmp_path / ".ssh"
    default_dir.mkdir()
    (default_dir / "id_ed25519").write_bytes(key.export_private_key())
    command("-D")
    with pytest.raises(AgentError, match="no matching identity"):
        async with authentication_keys(
            [str(public)], agent_path="SSH_AUTH_SOCK", identities_only=True,
        ):
            pytest.fail("must not use a default private file")
    next_key, _ = add("next")
    async with authentication_keys([], agent_path="SSH_AUTH_SOCK") as keys:
        assert [item.public_data for item in keys] == [next_key.public_data]


async def test_disabled_agent_does_not_offer_loaded_keys(agent):
    key, _ = agent[1]("key")
    async with login_server(key) as (port, host_key):
        with pytest.raises(SSHError, match="Permission denied"):
            await route_connection(
                port,
                agent_path="none",
                host_key_sha256=host_key.get_fingerprint(),
            )


async def test_explicit_socket_works_without_environment(agent, monkeypatch):
    path, add, _ = agent
    key, _ = add("key")
    monkeypatch.delenv("SSH_AUTH_SOCK")
    async with authentication_keys([], agent_path=path) as keys:
        assert [item.public_data for item in keys] == [key.public_data]


async def test_missing_agent_reports_actionable_error(tmp_path):
    with pytest.raises(SSHError, match="SSH agent unavailable"):
        await route_connection(TEST_PORT, agent_path=str(tmp_path / "missing.sock"))


async def test_explicit_environment_agent_requires_socket():
    with pytest.raises(SSHError, match="set SSH_AUTH_SOCK"):
        await route_connection(TEST_PORT, agent_path="SSH_AUTH_SOCK")


async def test_empty_agent_does_not_block_interactive_auth(agent):
    async with authentication_keys([], agent_path="SSH_AUTH_SOCK", interactive=True) as keys:
        assert keys is None


def test_socket_environment_reference(tmp_path, monkeypatch):
    path = str(tmp_path / "socket")
    monkeypatch.setenv("MY_AGENT", path)
    assert agent_socket("$MY_AGENT") == path
    assert agent_socket("none") == ""


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError(errno.ECONNREFUSED, "refused"),
        OSError(errno.EHOSTUNREACH, "unreachable"),
        socket.gaierror(socket.EAI_NONAME, "unknown host"),
        asyncssh.ChannelOpenError(asyncssh.OPEN_CONNECT_FAILED, "jump could not connect"),
    ],
)
async def test_network_errors_include_generic_remedy(monkeypatch, error):
    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(asyncssh, "connect", fail)
    with pytest.raises(
        SSHError, match="Check network connectivity and the configured address/port"
    ):
        await connect_ssh(LOCALHOST, port=TEST_PORT, name="test")


async def test_missing_private_file_is_not_a_network_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        await connect_ssh(
            LOCALHOST,
            port=TEST_PORT,
            name="test",
            client_keys=[str(tmp_path / "missing")],
        )


@pytest.mark.parametrize("with_key", [False, True])
async def test_omitted_agent_path_preserves_native_auth_options(tmp_path, monkeypatch, with_key):
    monkeypatch.setenv("SSH_AUTH_SOCK", str(tmp_path / "unavailable.sock"))
    paths = [str(tmp_path / "private")] if with_key else None

    async def capture(*args, **kwargs):
        return kwargs

    monkeypatch.setattr(asyncssh, "connect", capture)
    options = await connect_ssh(LOCALHOST, port=TEST_PORT, name="test", client_keys=paths)
    assert options["client_keys"] == paths
    assert "agent_path" not in options
    assert "agent_forwarding" not in options


def test_bundle_accepts_public_only_identity(tmp_path):
    key = asyncssh.generate_private_key("ssh-ed25519")
    public = tmp_path / "container.pub"
    public.write_bytes(key.export_public_key())
    target = TargetConfig(
        name="test",
        user="tester",
        client_public_key=str(public),
        transport=TransportConfig(kind="tunnel", ssh_targets=("login",)),
        bundle=BundleConfig(source="computemcp-container", deploy_dir=str(tmp_path / "deploy")),
    )
    assert public_key_for(target) == key.export_public_key().decode().strip()
    public.write_text("not a public key")
    with pytest.raises(BundleError, match="public key"):
        public_key_for(target)


@pytest.mark.parametrize("field", ["agent_path", "client_public_key"])
@pytest.mark.parametrize("value", ["", False, 42])
def test_new_auth_fields_require_nonempty_strings(field, value):
    with pytest.raises(ConfigError):
        TargetConfig(
            name="test",
            user="tester",
            **{field: value},
            transport=TransportConfig(kind="direct", remote_host=LOCALHOST, remote_port=TEST_PORT),
        )
