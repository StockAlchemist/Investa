"""Index quotes ask the worker for minimal info.

Indices carry no fund data, analyst estimates or earnings, but a full info
fetch asks for them anyway: 10-13 s per index, so the dashboard's three
indices ran past the request's 30 s timeout and the worker was killed.
"""

import pandas as pd

import market_data


def test_index_info_is_minimal(monkeypatch, tmp_path):
    provider = market_data.MarketDataProvider.__new__(market_data.MarketDataProvider)
    provider._index_quotes_memory_cache = market_data.BoundedTTLCache(
        max_size=4, default_ttl=60.0
    )
    provider.current_cache_file = str(tmp_path / "quotes.json")
    calls = []

    def fetch(tickers, **kwargs):
        calls.append(kwargs)
        if kwargs["task"] == "info":
            return {t: {"regularMarketPrice": 100.0, "shortName": t} for t in tickers}
        return pd.DataFrame()

    monkeypatch.setattr(market_data, "_run_isolated_fetch", fetch)
    quotes = provider.get_index_quotes(["^GSPC", "^DJI"])
    info = [c for c in calls if c["task"] == "info"]
    assert info == [{"task": "info", "timeout": 30, "minimal": True}]
    assert {q["price"] for q in quotes.values()} == {100.0}
