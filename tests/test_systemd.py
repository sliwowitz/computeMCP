# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: ISC
"""Keep the shipped user service compatible with host SSH configuration."""

from configparser import ConfigParser
from pathlib import Path

UNIT_PATH = Path(__file__).resolve().parents[1] / "systemd/compute-mcp-gateway.service"


def test_user_service_preserves_ssh_config_ownership() -> None:
    """PrivateTmp's implied user namespace breaks OpenSSH ownership checks."""
    unit = ConfigParser(interpolation=None)
    unit.optionxform = str
    unit.read_string(UNIT_PATH.read_text())
    service = unit["Service"]

    assert not service.getboolean("PrivateTmp")
    assert service.getboolean("NoNewPrivileges")
