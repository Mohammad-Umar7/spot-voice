"""Pre-flight checks.

These run on a demo floor with a real robot in front of you, so the rule is that
a check must never raise -- a broken check has to report itself and let the
others through, not take the whole report down.
"""

from __future__ import annotations

import socket
import types
from pathlib import Path

import pytest

from spot_voice import preflight
from spot_voice.preflight import (
    Check,
    _host_of,
    can_reach,
    check_map,
    check_model_provider,
    check_robot,
    check_vision_provider,
    run_all,
)


def make_config(**overrides):
    """A config-shaped object with sane defaults for the checks."""
    base = dict(
        mock_robot=False,
        spot_ip="192.168.33.180",
        graph_path=None,
        llm_provider="anthropic",
        llm_api_key="key",
        llm_model="claude-sonnet-4-6",
        vision_provider="anthropic",
        gemini_api_key="",
        gemini_model="gemini-2.0-flash",
        mic_device_name="",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ----------------------------------------------------------------------
# Host parsing


@pytest.mark.parametrize(
    "value,expected",
    [
        ("192.168.33.180", "192.168.33.180"),
        ("192.168.33.180:443", "192.168.33.180"),
        ("https://api.anthropic.com", "api.anthropic.com"),
        ("api.groq.com/v1", "api.groq.com"),
        ("", ""),
    ],
)
def test_host_parsing_accepts_ips_hosts_and_urls(value, expected):
    assert _host_of(value) == expected


# ----------------------------------------------------------------------
# Reachability


def test_a_refused_connection_reports_the_host_is_up(monkeypatch):
    # The distinction matters: "refused" means right network, wrong port;
    # "timed out" usually means wrong network entirely.
    def refuse(*_args, **_kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(socket, "create_connection", refuse)
    ok, detail = can_reach("10.0.0.1", 443)
    assert ok is False
    assert "host is up" in detail


def test_a_timeout_is_reported_as_such(monkeypatch):
    def time_out(*_args, **_kwargs):
        raise socket.timeout

    monkeypatch.setattr(socket, "create_connection", time_out)
    ok, detail = can_reach("10.0.0.1", 443, timeout=1.0)
    assert ok is False
    assert "timed out" in detail


def test_dns_failure_is_reported_distinctly(monkeypatch):
    def no_dns(*_args, **_kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "create_connection", no_dns)
    ok, detail = can_reach("nope.invalid", 443)
    assert ok is False
    assert "did not resolve" in detail


def test_a_successful_connect_is_closed_again(monkeypatch):
    closed = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed.append(True)

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: FakeSocket())
    ok, _detail = can_reach("192.168.33.180", 443)
    assert ok is True
    assert closed == [True]


# ----------------------------------------------------------------------
# Individual checks


def test_mock_mode_needs_no_robot():
    result = check_robot(make_config(mock_robot=True))
    assert result.ok and "MOCK_ROBOT" in result.detail


def test_a_missing_spot_ip_is_caught_before_any_connect():
    result = check_robot(make_config(spot_ip=""))
    assert result.ok is False
    assert "SPOT_IP" in result.detail


def test_an_unreachable_robot_suggests_the_dhcp_explanation(monkeypatch):
    monkeypatch.setattr(preflight, "can_reach", lambda *a, **k: (False, "timed out"))
    result = check_robot(make_config())
    assert result.ok is False
    assert "DHCP" in result.fix


def test_a_missing_model_key_is_reported_without_a_network_call(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("should not have tried to connect")

    monkeypatch.setattr(preflight, "can_reach", explode)
    result = check_model_provider(make_config(llm_api_key=""))
    assert result.ok is False
    assert "ANTHROPIC_API_KEY" in result.fix


def test_an_unreachable_model_api_explains_the_second_interface(monkeypatch):
    monkeypatch.setattr(preflight, "can_reach", lambda *a, **k: (False, "timed out"))
    result = check_model_provider(make_config())
    assert result.ok is False
    assert "second" in result.fix


def test_anthropic_vision_needs_no_separate_check():
    result = check_vision_provider(make_config(vision_provider="anthropic"))
    assert result.ok and "itself" in result.detail


def test_no_vision_provider_is_a_pass_but_says_what_it_costs():
    result = check_vision_provider(make_config(vision_provider="none"))
    assert result.ok
    assert "not be described" in result.detail


def test_gemini_without_a_key_fails():
    result = check_vision_provider(
        make_config(vision_provider="gemini", gemini_api_key="")
    )
    assert result.ok is False


def test_a_missing_map_directory_is_caught(tmp_path: Path):
    result = check_map(make_config(graph_path=tmp_path / "nope"))
    assert result.ok is False
    assert "does not exist" in result.detail


def test_pointing_one_level_too_high_is_caught(tmp_path: Path):
    # The classic mistake: GRAPH_PATH set to the parent of downloaded_graph.
    (tmp_path / "downloaded_graph").mkdir()
    result = check_map(make_config(graph_path=tmp_path))
    assert result.ok is False
    assert "downloaded_graph" in result.fix


def test_a_valid_map_counts_its_snapshots(tmp_path: Path):
    (tmp_path / "graph").write_bytes(b"")
    snapshots = tmp_path / "waypoint_snapshots"
    snapshots.mkdir()
    for index in range(3):
        (snapshots / f"snap{index}").write_bytes(b"")

    result = check_map(make_config(graph_path=tmp_path))
    assert result.ok
    assert "3 waypoint snapshots" in result.detail


# ----------------------------------------------------------------------
# The whole report


def test_run_all_returns_one_result_per_check(monkeypatch):
    monkeypatch.setattr(preflight, "can_reach", lambda *a, **k: (True, "reachable"))
    results = run_all(make_config(mock_robot=True))
    assert len(results) == len(preflight.ALL_CHECKS)
    assert all(isinstance(result, Check) for result in results)


def test_a_check_that_raises_does_not_take_down_the_report(monkeypatch):
    def explode(_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(preflight, "ALL_CHECKS", (explode, check_map))
    results = run_all(make_config())

    assert len(results) == 2
    assert results[0].ok is False
    assert "boom" in results[0].detail
    assert results[1].ok is True  # the other check still ran


# ----------------------------------------------------------------------
# Finding the robot when its DHCP address moves


@pytest.mark.parametrize(
    "address,expected",
    [
        ("192.168.33.137", "192.168.33"),
        ("192.168.80.3", "192.168.80"),
        ("10.0.0.1", "10.0.0"),
        ("not-an-ip", None),
        ("192.168.33", None),
        ("192.168.33.999", None),
        ("", None),
    ],
)
def test_subnet_is_derived_only_from_a_real_ipv4(address, expected):
    from spot_voice.preflight import subnet_of

    assert subnet_of(address) == expected


def test_the_sweep_covers_the_whole_usable_range(monkeypatch):
    from spot_voice.preflight import find_robots

    probed = []

    def record(host, _port, timeout=0.0):
        probed.append(host)
        return False, "no"

    monkeypatch.setattr(preflight, "can_reach", record)
    find_robots("192.168.33", workers=8)

    assert len(probed) == 254
    assert "192.168.33.1" in probed
    assert "192.168.33.254" in probed
    assert "192.168.33.0" not in probed  # network address
    assert "192.168.33.255" not in probed  # broadcast


def test_results_come_back_in_numeric_not_lexical_order(monkeypatch):
    from spot_voice.preflight import find_robots

    live = {"192.168.33.137", "192.168.33.9", "192.168.33.80"}
    monkeypatch.setattr(
        preflight,
        "can_reach",
        lambda host, _port, timeout=0.0: (host in live, ""),
    )

    # Lexical sorting would put .137 before .80, which reads as wrong.
    assert find_robots("192.168.33", workers=8) == [
        "192.168.33.9",
        "192.168.33.80",
        "192.168.33.137",
    ]


def test_an_empty_sweep_is_not_an_error(monkeypatch):
    from spot_voice.preflight import find_robots

    monkeypatch.setattr(preflight, "can_reach", lambda *a, **k: (False, ""))
    assert find_robots("192.168.33", workers=4) == []


def test_certificate_hints_identify_a_spot():
    from spot_voice.preflight import looks_like_spot

    assert looks_like_spot("Boston Dynamics, spot-BD-1347000") is True
    assert looks_like_spot("BOSDYN internal CA") is True
    assert looks_like_spot("TP-Link Router, admin") is False
    assert looks_like_spot(None) is False
    assert looks_like_spot("") is False


def test_tls_identity_returns_none_rather_than_raising(monkeypatch):
    # A host that refuses the handshake must not take the sweep down.
    from spot_voice.preflight import tls_identity

    def refuse(*_args, **_kwargs):
        raise ConnectionResetError

    monkeypatch.setattr(socket, "create_connection", refuse)
    assert tls_identity("192.168.33.1") is None
