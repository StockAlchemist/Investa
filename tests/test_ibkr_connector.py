"""Flex Web Service error handling.

IBKR answers a SendRequest with either a reference code or a `<Status>Fail`
envelope carrying an ErrorCode. "Statement could not be generated at this time"
(1009) is a wait-and-retry condition, not a server fault, so it must retry and
then surface as `IBKRBusyError`; a bad token must fail immediately.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from ibkr_connector import IBKRBusyError, IBKRConnector, IBKRError  # noqa: E402


def _fail(code, message):
    return (
        '<FlexStatementResponse timestamp="25 August, 2026">'
        f"<Status>Fail</Status><ErrorCode>{code}</ErrorCode>"
        f"<ErrorMessage>{message}</ErrorMessage></FlexStatementResponse>"
    )


_SUCCESS = (
    '<FlexStatementResponse timestamp="25 August, 2026">'
    "<Status>Success</Status><ReferenceCode>1234567890</ReferenceCode>"
    "<Url>https://ndcdyn.interactivebrokers.com/GetStatement</Url>"
    "</FlexStatementResponse>"
)

_BUSY = _fail(
    1009, "Statement could not be generated at this time. Please try again shortly."
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("ibkr_connector.time.sleep", lambda _s: None)


def _connector(responses, monkeypatch):
    calls = []

    def fake_request(self, url, params):
        calls.append(url)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(IBKRConnector, "_make_request", fake_request)
    return IBKRConnector(token="tok", query_id="qid"), calls


def test_busy_retries_then_raises_busy(monkeypatch):
    conn, calls = _connector([_BUSY], monkeypatch)
    with pytest.raises(IBKRBusyError) as excinfo:
        conn.request_report()
    assert excinfo.value.code == "1009"
    assert "try again shortly" in str(excinfo.value)
    assert len(calls) == 3  # initial attempt + two backoff retries
    assert not excinfo.value.is_config_error


def test_busy_that_clears_returns_reference_code(monkeypatch):
    conn, calls = _connector([_BUSY, _SUCCESS], monkeypatch)
    ref, url = conn.request_report()
    assert ref == "1234567890"
    assert url.endswith("GetStatement")
    assert len(calls) == 2


def test_invalid_token_fails_without_retrying(monkeypatch):
    conn, calls = _connector([_fail(1015, "Token is invalid.")], monkeypatch)
    with pytest.raises(IBKRError) as excinfo:
        conn.request_report()
    assert excinfo.value.is_config_error
    assert not isinstance(excinfo.value, IBKRBusyError)
    assert len(calls) == 1


def test_unreachable_service_raises_ibkr_error(monkeypatch):
    conn, _ = _connector([None], monkeypatch)
    with pytest.raises(IBKRError) as excinfo:
        conn.request_report()
    assert not isinstance(excinfo.value, IBKRBusyError)
    assert "Flex Web Service" in str(excinfo.value)


def test_missing_credentials_is_a_config_error(monkeypatch):
    # Never reaches the network: a blank token is rejected before the request.
    monkeypatch.setattr(
        IBKRConnector,
        "_make_request",
        lambda self, url, params: pytest.fail("should not call IBKR"),
    )
    conn = IBKRConnector(token="tok", query_id="qid")
    conn.token = ""
    with pytest.raises(IBKRError) as excinfo:
        conn.request_report()
    assert excinfo.value.is_config_error


def test_download_stops_on_terminal_error(monkeypatch):
    conn, calls = _connector([_fail(1017, "Reference code is invalid.")], monkeypatch)
    with pytest.raises(IBKRError) as excinfo:
        conn.download_report("ref", "https://example.com/stmt")
    assert not isinstance(excinfo.value, IBKRBusyError)
    assert len(calls) == 1


def test_download_waits_out_generation_in_progress(monkeypatch):
    in_progress = _fail(
        1019, "Statement generation in progress. Please try again shortly."
    )
    report = '<FlexQueryResponse queryName="Activity"></FlexQueryResponse>'
    conn, calls = _connector([in_progress, report], monkeypatch)
    assert conn.download_report("ref", "https://example.com/stmt") == report
    assert len(calls) == 2
