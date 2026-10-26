# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: ISC
"""Select local SSH identities without exporting or forwarding agent keys."""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import asyncssh


class AgentError(RuntimeError):
    """The requested SSH agent cannot supply an authentication identity."""


def agent_socket(value: str | None = None) -> str:
    """Resolve an OpenSSH IdentityAgent value, defaulting to SSH_AUTH_SOCK."""
    if value == "none":
        return ""
    if value in (None, "SSH_AUTH_SOCK"):
        return os.environ.get("SSH_AUTH_SOCK", "")
    if value.startswith("$") and not value.startswith("${"):
        value = os.environ.get(value[1:], "")
    return os.path.expanduser(os.path.expandvars(value))


def public_identity(path: str) -> asyncssh.SSHKey | None:
    """Read a public identity without opening its private half."""
    for candidate in (path + ".pub", path):
        try:
            return asyncssh.read_public_key(Path(candidate).expanduser())
        except (OSError, asyncssh.KeyImportError):
            continue
    return None


@asynccontextmanager
async def authentication_keys(
    identity_files: list[str] | tuple[str, ...],
    *,
    agent_path: str | None = None,
    identities_only: bool = False,
    passphrase: str | None = None,
    timeout: float = 10.0,
    interactive: bool = False,
):
    """Use legacy key arguments unless an agent is explicitly selected."""
    if agent_path is None:
        yield list(identity_files) or None
        return
    paths = [os.path.expanduser(path) for path in identity_files]
    socket_path = agent_socket(agent_path)
    if not socket_path:
        if agent_path != "none":
            raise AgentError(
                f"SSH agent unavailable: {agent_path!r} resolves to no socket; "
                "set SSH_AUTH_SOCK in the gateway's environment or configure an explicit "
                "agent_path. A systemd service does not read your shell startup files"
            )
        yield paths or None
        return

    async with AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(timeout):
                agent = await stack.enter_async_context(asyncssh.connect_agent(socket_path))
                keys = await agent.get_keys()
        except TimeoutError as exc:
            raise AgentError(
                f"SSH agent at {socket_path!r} did not respond within {timeout:g}s; "
                "check the agent is running and responsive"
            ) from exc
        except (OSError, asyncssh.Error, ValueError) as exc:
            if isinstance(exc, FileNotFoundError):
                detail = "socket does not exist"
            elif isinstance(exc, ConnectionRefusedError):
                detail = "no agent is listening on the socket"
            elif isinstance(exc, PermissionError):
                detail = "the gateway user cannot access the socket"
            else:
                detail = str(exc) or type(exc).__name__
            raise AgentError(
                f"SSH agent unavailable at {socket_path!r}: {detail}; "
                "check the configured socket, agent service and socket permissions"
            ) from exc

        agent_empty = not keys
        identities = [public_identity(path) for path in paths]
        allowed = {key.public_data for key in identities if key is not None}
        if identities_only:
            keys = [key for key in keys if key.public_data in allowed]

        offered = {key.public_data for key in keys}
        remaining = [
            path
            for path, public in zip(paths, identities, strict=True)
            if public is None or public.public_data not in offered
        ]
        # Public-only selectors cannot sign without their agent identity.
        local_keys = asyncssh.load_keypairs(
            remaining, passphrase=passphrase, skip_public=True,
            ignore_encrypted=passphrase is None,
        )
        selected = [*keys, *local_keys]
        if not selected and not interactive:
            detail = (
                "no identities; load or unlock the required key in this agent"
                if agent_empty else
                "no matching identity; check IdentityFile/IdentitiesOnly or client_public_key "
                "and load or unlock the required key"
            )
            raise AgentError(
                f"SSH agent {socket_path!r} has {detail}. "
                "Inspect this agent with SSH_AUTH_SOCK set to this socket and ssh-add -l"
            )
        yield selected or None
