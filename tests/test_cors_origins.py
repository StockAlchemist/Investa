"""Only this server's own networks may make credentialed cross-origin calls."""

import os
import re
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from server.main import build_local_origin_regex  # noqa: E402


def _allows(pattern: str, origin: str) -> bool:
    return re.match(pattern, origin) is not None


def test_own_tailnet_is_allowed_and_a_strangers_is_not():
    pattern = build_local_origin_regex("tail1234.ts.net")
    assert _allows(pattern, "https://muon.tail1234.ts.net")
    assert _allows(pattern, "https://muon.tail1234.ts.net:8443")
    # Anyone can get a *.ts.net host through Tailscale Funnel.
    assert not _allows(pattern, "https://evil.tail9999.ts.net")
    assert not _allows(pattern, "https://muon.tail1234.ts.net.evil.com")


def test_cloud_run_hosts_are_not_trusted_by_default():
    pattern = build_local_origin_regex("tail1234.ts.net")
    assert not _allows(pattern, "https://anything.a.run.app")


def test_local_networks_still_work():
    pattern = build_local_origin_regex("tail1234.ts.net")
    for origin in (
        "http://localhost:3000",
        "http://192.168.1.20:3000",
        "http://100.101.102.103:3000",
        "http://muon.local:3000",
    ):
        assert _allows(pattern, origin), origin


def test_unknown_tailnet_falls_back_rather_than_locking_out():
    assert _allows(build_local_origin_regex(None), "https://muon.tail1234.ts.net")
