"""Tests for the persistent market-data fetch worker pool.

These exercise the request/response transport, timeout, crash-restart, memory
recycling, and stray-stdout protection deterministically, against a tiny fake
``--serve`` script instead of the real yfinance worker (no network).
"""

import os
import sys
import tempfile
import textwrap
import time

import pytest

import market_data
from market_data import _PersistentFetchWorker, _WORKER_RESULT_MARKER

# A minimal stand-in for market_data_worker.py --serve. Speaks the same
# marker-prefixed, newline-delimited JSON protocol.
_FAKE_WORKER = textwrap.dedent(
    f'''
    import sys, json, os, time
    MARKER = {_WORKER_RESULT_MARKER!r}
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        req = json.loads(line)
        if req.get("crash"):
            os._exit(1)
        if req.get("sleep"):
            time.sleep(req["sleep"])
        if req.get("garbage_stdout"):
            sys.stdout.write("noise that is not a response\\n")
            sys.stdout.flush()
        resp = {{"status": "success", "pid": os.getpid(), "echo": req.get("n")}}
        sys.stdout.write(MARKER + json.dumps(resp) + "\\n")
        sys.stdout.flush()
    '''
)


@pytest.fixture
def fake_worker(monkeypatch):
    fd, path = tempfile.mkstemp(suffix="_fake_worker.py")
    os.write(fd, _FAKE_WORKER.encode())
    os.close(fd)
    monkeypatch.setattr(market_data, "_build_worker_command", lambda: [sys.executable, path])
    worker = _PersistentFetchWorker()
    yield worker
    worker._kill()
    os.remove(path)


def test_worker_reuses_one_process(fake_worker):
    pids = []
    for n in range(4):
        resp = fake_worker.run({"n": n}, timeout=5)
        assert resp["status"] == "success"
        assert resp["echo"] == n
        pids.append(resp["pid"])
    # All four requests were served by the same long-lived process.
    assert len(set(pids)) == 1


def test_worker_recycles_after_max_requests(fake_worker, monkeypatch):
    monkeypatch.setattr(market_data, "_MAX_REQUESTS_PER_WORKER", 2)
    first = fake_worker.run({"n": 1}, timeout=5)["pid"]
    second = fake_worker.run({"n": 2}, timeout=5)["pid"]
    # Same process for the first two; the 2nd hits the cap and kills the worker.
    assert first == second
    assert not fake_worker._alive()
    # Next call transparently restarts with a fresh process.
    third = fake_worker.run({"n": 3}, timeout=5)["pid"]
    assert third != first


def test_worker_restarts_after_crash(fake_worker):
    assert fake_worker.run({"n": 1}, timeout=5)["status"] == "success"
    # A crash mid-request surfaces as None (caller substitutes an empty result).
    assert fake_worker.run({"crash": True}, timeout=5) is None
    # The pool transparently restarts the worker on the next request.
    assert fake_worker.run({"n": 2}, timeout=5)["status"] == "success"


def test_worker_timeout_returns_none_and_recovers(fake_worker):
    start = time.time()
    result = fake_worker.run({"sleep": 5}, timeout=0.3)
    elapsed = time.time() - start
    assert result is None
    assert elapsed < 3  # did not wait for the full 5s sleep
    assert not fake_worker._alive()
    # A late response from the killed worker must not leak into the next request.
    assert fake_worker.run({"n": 99}, timeout=5)["echo"] == 99


def test_stray_stdout_does_not_desync_protocol(fake_worker):
    resp = fake_worker.run({"n": 7, "garbage_stdout": True}, timeout=5)
    assert resp["status"] == "success"
    assert resp["echo"] == 7


def test_run_isolated_fetch_uses_persistent_pool(fake_worker, monkeypatch):
    """End-to-end: _run_isolated_fetch routes through the pool and parses the
    response. The fake worker returns no file, so history yields an empty frame."""
    monkeypatch.setattr(market_data, "_PERSISTENT_WORKER_ENABLED", True)
    monkeypatch.setattr(market_data, "_WORKER_POOL", None)
    # Pool of fake workers (fixture already redirected the worker command).
    import queue as _queue

    pool = _queue.Queue()
    pool.put(_PersistentFetchWorker())
    pool.put(_PersistentFetchWorker())
    monkeypatch.setattr(market_data, "_WORKER_POOL", pool)

    result = market_data._run_isolated_fetch(["AAPL"], task="history")
    assert result.empty  # fake worker returns {"status":"success"} with no file

    market_data.shutdown_fetch_workers()
