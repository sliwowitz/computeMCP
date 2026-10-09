# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: ISC
"""Keep SSH tests away from the operator's keys and authentication agent."""

import pytest


@pytest.fixture(autouse=True)
def isolated_ssh_environment(tmp_path, monkeypatch):
    """Use temporary SSH configuration and require tests to opt into an agent."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
