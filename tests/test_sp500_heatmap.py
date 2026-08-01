import json
import re
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from server.routes.market import (
    _build_sp500_heatmap_sync,
    _dedupe_share_classes,
    _dividend_yield_fraction,
    _fetch_monthly_closes,
)

# Wikipedia lists both share classes of a dual-class issuer under one CIK.
CONSTITUENTS = [
    {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Information Technology",
        "sub_industry": "Consumer Electronics",
        "cik": "320193",
    },
    {
        "symbol": "GOOGL",
        "name": "Alphabet Inc. (Class A)",
        "sector": "Communication Services",
        "sub_industry": "Interactive Media",
        "cik": "1652044",
    },
    {
        "symbol": "GOOG",
        "name": "Alphabet Inc. (Class C)",
        "sector": "Communication Services",
        "sub_industry": "Interactive Media",
        "cik": "1652044",
    },
    {
        "symbol": "FOXA",
        "name": "Fox Corporation (Class A)",
        "sector": "Communication Services",
        "sub_industry": "Broadcasting",
        "cik": "1754301",
    },
    {
        "symbol": "FOX",
        "name": "Fox Corporation (Class B)",
        "sector": "Communication Services",
        "sub_industry": "Broadcasting",
        "cik": "1754301",
    },
]

PERIODS = 130
LATEST = 100.0 + PERIODS - 1  # value of the final monthly bar

# Frozen so the lookbacks are exact arithmetic rather than a function of when
# the suite runs. Month end, with the last bar being that month, so a clean
# N-year lookback lands exactly N*12 bars back.
TODAY = pd.Timestamp("2026-07-31").date()
FRAME_END = "2026-07-01"


@pytest.fixture(autouse=True)
def _isolated_fetch_env(tmp_path_factory):
    """Keep tests off the real history cache and off the production backoff.

    Without the path override every test would read (and overwrite) the live
    500-symbol frame on disk, so the suite would pass on production data
    instead of its own fixtures.
    """
    cache_dir = tmp_path_factory.mktemp("heatmap-history")
    with patch("server.routes.market._HEATMAP_RETRY_BACKOFF", 0), patch(
        "server.routes.market._history_cache_path",
        side_effect=lambda interval: str(cache_dir / f"{interval}.pkl"),
    ):
        yield


def _monthly_frame(symbols, end=FRAME_END, periods=PERIODS):
    """A worker-shaped monthly fetch: (ticker, field) columns, month-START labels.

    Matches what `_run_isolated_fetch` returns — the worker downloads with
    group_by="ticker", so the ticker is the outer level.

    The label understates the bar: the row labelled 2026-07-01 carries July's
    *closing* price. Prices rise by 1 per month so a lookback's expected value
    is unambiguous.
    """
    index = pd.date_range(end=pd.Timestamp(end), periods=periods, freq="MS")
    fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    columns = pd.MultiIndex.from_product([symbols, fields], names=["Ticker", None])
    data = {
        (s, f): [100.0 + i for i in range(periods)] for s in symbols for f in fields
    }
    return pd.DataFrame(data, index=index, columns=columns)


def _run(
    tmp_path,
    constituents=None,
    hist=None,
    quotes=None,
    screener=None,
    fundamentals=None,
    today=TODAY,
):
    """Drive the real builder against a fake Yahoo/screener/fundamentals layer."""
    constituents = CONSTITUENTS if constituents is None else constituents
    symbols = [c["symbol"] for c in constituents]
    quotes = (
        quotes
        if quotes is not None
        else {s: {"price": 200.0, "changesPercentage": 1.5} for s in symbols}
    )
    screener = (
        screener
        if screener is not None
        else {s: {"market_cap": 1e12, "pe_ratio": 30.0} for s in symbols}
    )
    hist = _monthly_frame(symbols) if hist is None else hist

    # Real files, so the on-disk fundamentals read is exercised rather than mocked.
    for sym, info in (fundamentals or {}).items():
        (tmp_path / f"{sym}.json").write_text(json.dumps({"ticker_info": info}))

    mdp = MagicMock()
    mdp.get_current_quotes.return_value = (quotes, {}, {}, None, None)
    mdp._get_symbol_fundamentals_path.side_effect = lambda s: str(
        tmp_path / f"{s}.json"
    )

    with (
        patch(
            "server.screener_service.get_sp500_constituents", return_value=constituents
        ),
        patch("server.routes.market._get_heatmap_mdp", return_value=mdp),
        patch(
            "server.routes.market.get_cached_screener_results", return_value=screener
        ),
        patch("server.routes.market.get_est_today", return_value=today),
        patch("market_data._run_isolated_fetch", return_value=hist),
    ):
        return {item["symbol"]: item for item in _build_sp500_heatmap_sync()}


def _run_with(tmp_path, fetch):
    """`_run`, but with a caller-supplied worker stub so calls can be counted."""
    mdp = MagicMock()
    mdp._get_symbol_fundamentals_path.side_effect = lambda s: str(tmp_path / f"{s}.json")
    with patch("market_data._run_isolated_fetch", side_effect=fetch), patch(
        "server.screener_service.get_sp500_constituents", return_value=CONSTITUENTS
    ), patch("server.routes.market._get_heatmap_mdp", return_value=mdp), patch(
        "server.routes.market.get_cached_screener_results", return_value={}
    ), patch("server.routes.market.get_est_today", return_value=TODAY):
        return _build_sp500_heatmap_sync()


class TestShareClassDedupe:
    def test_collapses_dual_class_lines_by_cik(self):
        kept = [c["symbol"] for c in _dedupe_share_classes(CONSTITUENTS)]
        assert kept == ["AAPL", "GOOGL", "FOXA"], (
            "Yahoo reports full company market cap against both share classes, "
            "so keeping both would draw Alphabet at twice its true weight"
        )

    def test_end_to_end_drops_secondary_classes(self, tmp_path):
        assert set(_run(tmp_path)) == {"AAPL", "GOOGL", "FOXA"}

    def test_rows_without_a_cik_are_all_kept(self):
        rows = [{"symbol": "AAPL", "cik": ""}, {"symbol": "MSFT", "cik": ""}]
        assert len(_dedupe_share_classes(rows)) == 2


class TestDividendYield:
    def test_percent_encoded_yield_is_normalized_to_a_fraction(self):
        # Real Yahoo payload: KO reports dividendYield=2.4 meaning 2.4%.
        info = {"dividendYield": 2.4, "dividendRate": 2.12}
        assert _dividend_yield_fraction(info, 87.59) == pytest.approx(0.0242, rel=1e-2)

    def test_resolves_against_price_rather_than_guessing_from_magnitude(self):
        # AAPL reports dividendYield=0.35 — also percent, not a fraction. The
        # two encodings overlap in range, so only rate/price settles it.
        info = {"dividendYield": 0.35, "dividendRate": 1.08}
        assert _dividend_yield_fraction(info, 308.91) == pytest.approx(0.0035, rel=1e-2)

    def test_falls_back_to_trailing_yield_which_is_always_a_fraction(self):
        info = {"dividendYield": 2.4, "trailingAnnualDividendYield": 0.0235}
        assert _dividend_yield_fraction(info, None) == pytest.approx(0.0235)

    def test_no_dividend_data_is_none_not_zero(self):
        assert _dividend_yield_fraction({}, 100.0) is None

    def test_endpoint_ships_a_fraction(self, tmp_path):
        data = _run(
            tmp_path,
            fundamentals={"AAPL": {"dividendYield": 2.4, "dividendRate": 2.12}},
        )
        # Resolved against the price the payload itself reports — a fraction,
        # not Yahoo's 2.4 and not 240.
        expected = 2.12 / data["AAPL"]["price"]
        assert data["AAPL"]["dividend_yield"] == pytest.approx(expected, rel=1e-6)
        assert 0 < data["AAPL"]["dividend_yield"] < 0.25


class TestPerformanceLookbacks:
    def test_one_year_spans_twelve_months_not_eleven(self, tmp_path):
        # Monthly bars are labelled month-start but carry the month's close, so
        # comparing labels directly used to make "1Y" span only 11 months.
        # The series rises by 1/month, so a true 1Y lookback is exactly 12 back.
        data = _run(tmp_path)
        past = LATEST - 12
        assert data["AAPL"]["1y_change_pct"] == pytest.approx((LATEST - past) / past)

    def test_multi_year_lookbacks_land_on_the_right_month(self, tmp_path):
        data = _run(tmp_path)
        for key, months in (
            ("3y_change_pct", 36),
            ("5y_change_pct", 60),
            ("10y_change_pct", 120),
        ):
            past = LATEST - months
            assert data["AAPL"][key] == pytest.approx((LATEST - past) / past), key

    def test_ytd_measures_from_the_prior_december_close(self, tmp_path):
        data = _run(tmp_path)
        december = LATEST - 7  # 2025-12-01 bar carries December's close
        assert data["AAPL"]["ytd_change_pct"] == pytest.approx(
            (LATEST - december) / december
        )

    def test_mid_month_uses_the_last_close_at_or_before_the_anniversary(self, tmp_path):
        # Asked on 15 Aug, the closest prior monthly close to 15 Aug last year is
        # July's — August's close (end of month) falls *after* the anniversary,
        # so using it would measure a window shorter than a year.
        data = _run(tmp_path, today=pd.Timestamp("2026-08-15").date())
        past = LATEST - 12  # the 2025-07-01 bar, 12 bars before the 2026-07-01 bar
        assert data["AAPL"]["1y_change_pct"] == pytest.approx((LATEST - past) / past)

    def test_fetch_window_reaches_past_the_ten_year_cutoff(self, tmp_path):
        # period="10y" returns exactly 120 monthly bars starting *after* the
        # 10-year mark, so the earliest close sits inside the window and the 10Y
        # column silently resolves to None for every stock.
        mdp = MagicMock()
        mdp.get_current_quotes.return_value = ({"AAPL": {"price": 200.0}}, {}, {}, None, None)
        mdp._get_symbol_fundamentals_path.side_effect = lambda s: str(tmp_path / f"{s}.json")

        with patch("market_data._run_isolated_fetch", return_value=_monthly_frame(["AAPL"])) as fetch, \
             patch("server.screener_service.get_sp500_constituents", return_value=CONSTITUENTS[:1]), \
             patch("server.routes.market._get_heatmap_mdp", return_value=mdp), \
             patch("server.routes.market.get_cached_screener_results", return_value={}), \
             patch("server.routes.market.get_est_today", return_value=TODAY):
            _build_sp500_heatmap_sync()

        monthly = [c for c in fetch.call_args_list if c.kwargs.get("interval") == "1mo"]
        assert monthly, "the long-horizon fetch must ask for monthly bars"
        start = pd.Timestamp(monthly[0].kwargs["start"])
        assert start < pd.Timestamp(TODAY) - pd.DateOffset(years=10)

    def test_ten_year_change_resolves(self, tmp_path):
        data = _run(tmp_path)
        past = LATEST - 120
        assert data["AAPL"]["10y_change_pct"] == pytest.approx((LATEST - past) / past)

    def test_a_recently_listed_symbol_yields_none_not_zero(self, tmp_path):
        # Real case: FDXF and HONA both first traded mid-2026, so they have no
        # 2025 year-end close and YTD is genuinely unanswerable for them.
        symbols = [c["symbol"] for c in _dedupe_share_classes(CONSTITUENTS)]
        data = _run(tmp_path, hist=_monthly_frame(symbols, periods=3))
        assert data["AAPL"]["ytd_change_pct"] is None
        assert data["AAPL"]["10y_change_pct"] is None
        # Price and the 1-day change still come through, off the last two bars.
        assert data["AAPL"]["price"] == 102.0
        assert data["AAPL"]["change_pct"] == pytest.approx(1 / 101 * 100)


class TestTimezoneResilience:
    def test_tz_aware_index_does_not_raise(self, tmp_path):
        hist = _monthly_frame(["AAPL", "GOOGL", "FOXA"])
        hist.index = hist.index.tz_localize("UTC")
        data = _run(hist=hist, tmp_path=tmp_path)
        assert data["AAPL"]["1y_change_pct"] is not None


class TestHistoryFetch:
    """The history fetch must go through the isolated worker.

    A direct in-process ``yf.download`` races the refresh worker and the
    portfolio's own fetches — yfinance keeps module-level state across threads
    and has no rate-limit memory — and symbols come back empty. Observed live:
    348 of 500 constituents reading n/a for every period at once.
    """

    def test_routes_through_the_isolated_worker(self, tmp_path):
        with patch("market_data._run_isolated_fetch", return_value=_monthly_frame(["AAPL"])) as fetch, \
             patch("yfinance.download") as direct:
            _fetch_monthly_closes(["AAPL"], "2016-05-01", "2026-07-31")
        assert fetch.called
        assert not direct.called, "must not call yfinance in-process"
        assert fetch.call_args.kwargs["interval"] == "1mo"
        assert fetch.call_args.kwargs["task"] == "history"

    def test_retries_symbols_that_come_back_empty(self):
        # Yahoo drops symbols sporadically under load; a second pass over the
        # stragglers is far cheaper than a map where a third of tiles read n/a.
        calls = []

        def flaky(batch, **kwargs):
            calls.append(list(batch))
            if len(calls) == 1:
                return _monthly_frame(["AAPL"])  # BBB missing on the first pass
            return _monthly_frame(["BBB"])

        with patch("market_data._run_isolated_fetch", side_effect=flaky):
            closes = _fetch_monthly_closes(["AAPL", "BBB"], "2016-05-01", "2026-07-31")

        assert calls[1] == ["BBB"], "retry should ask only for the stragglers"
        assert set(closes.columns) == {"AAPL", "BBB"}

    def test_refuses_a_catastrophically_degraded_fetch(self):
        # The caller caches for up to an hour, so returning a mostly-empty frame
        # would pin every period column at n/a long after Yahoo recovered.
        # Raising leaves the previous good payload in place.
        symbols = [f"S{i}" for i in range(10)]
        with patch("market_data._run_isolated_fetch", return_value=_monthly_frame(["S0"])):
            with pytest.raises(RuntimeError, match="covered only"):
                _fetch_monthly_closes(symbols, "2016-05-01", "2026-07-31")

    def test_partial_coverage_above_the_floor_still_builds(self):
        symbols = [f"S{i}" for i in range(10)]
        with patch("market_data._run_isolated_fetch", return_value=_monthly_frame(symbols[:8])):
            closes = _fetch_monthly_closes(symbols, "2016-05-01", "2026-07-31")
        assert closes.shape[1] == 8

    def test_a_rebuild_stays_within_a_small_number_of_worker_fetches(self, tmp_path):
        # Observed live: a rebuild issuing ~36 worker fetches (25 quote chunks,
        # a 500-symbol intraday pull, then the history) got throttled partway
        # through and returned "covered only 107/500". Price and the 1-day
        # change now come off the daily frame instead, so the whole build is
        # two chunked history fetches.
        calls = []

        def counting(tickers, **kwargs):
            calls.append(kwargs.get("interval"))
            return _monthly_frame(list(tickers))

        mdp = MagicMock()
        mdp._get_symbol_fundamentals_path.side_effect = lambda s: str(tmp_path / f"{s}.json")

        with patch("market_data._run_isolated_fetch", side_effect=counting), \
             patch("server.screener_service.get_sp500_constituents", return_value=CONSTITUENTS), \
             patch("server.routes.market._get_heatmap_mdp", return_value=mdp), \
             patch("server.routes.market.get_cached_screener_results", return_value={}), \
             patch("server.routes.market.get_est_today", return_value=TODAY):
            _build_sp500_heatmap_sync()

        assert sorted(set(calls)) == ["1d", "1mo"], f"unexpected fetches: {calls}"
        assert len(calls) <= 4, f"{len(calls)} worker fetches for 3 symbols"
        # The quote path is what made this expensive; it must stay unused.
        assert not mdp.get_current_quotes.called

    def test_price_and_day_change_come_from_the_daily_bars(self, tmp_path):
        # Dropping get_current_quotes must not drop the two fields it supplied.
        data = _run(tmp_path)
        assert data["AAPL"]["price"] == LATEST
        # The synthetic series steps by 1 a bar, so the day change is 1/prev.
        assert data["AAPL"]["change_pct"] == pytest.approx(1 / (LATEST - 1) * 100)

    def test_history_is_reused_rather_than_refetched_every_rebuild(self, tmp_path):
        # Ten years of monthly bars only change when a month closes. Re-fetching
        # 500 symbols of them every five minutes was the load that got the
        # rebuild throttled in the first place.
        calls = []

        def counting(tickers, **kwargs):
            calls.append(kwargs.get("interval"))
            return _monthly_frame(list(tickers))

        with patch("market_data._run_isolated_fetch", side_effect=counting):
            _run_with(tmp_path, counting)
            first = len(calls)
            _run_with(tmp_path, counting)

        assert first > 0
        assert len(calls) == first, "the second rebuild refetched cached history"

    def test_a_throttled_refresh_cannot_reduce_coverage(self):
        # Yahoo silently omits symbols it is throttling. Replacing the frame
        # turns one bad minute into a map where most tiles read n/a, so a fresh
        # fetch is merged over the previous one and can only add coverage.
        symbols = [f"S{i}" for i in range(10)]
        with patch("market_data._run_isolated_fetch", return_value=_monthly_frame(symbols)):
            full = _fetch_monthly_closes(symbols, "2016-05-01", "2026-07-31")
        assert full.shape[1] == 10

        with patch("server.routes.market._HISTORY_CACHE_TTL", {"1mo": 0}), \
             patch("market_data._run_isolated_fetch", return_value=_monthly_frame(["S0"])):
            after = _fetch_monthly_closes(symbols, "2016-05-01", "2026-07-31")
        assert after.shape[1] == 10, "a throttled refresh dropped symbols from the frame"

    def test_merging_survives_a_timezone_mismatch(self):
        # The worker returns tz-aware bars for some intervals and naive for
        # others, so a cached frame can disagree with a fresh one. Left alone,
        # pandas refuses to join them and the whole fetch fails.
        symbols = ["AAA", "BBB"]
        aware = _monthly_frame(symbols)
        aware.index = aware.index.tz_localize("UTC")
        with patch("market_data._run_isolated_fetch", return_value=aware):
            _fetch_monthly_closes(symbols, "2016-05-01", "2026-07-31")

        with patch("server.routes.market._HISTORY_CACHE_TTL", {"1mo": 0}), \
             patch("market_data._run_isolated_fetch", return_value=_monthly_frame(symbols)):
            merged = _fetch_monthly_closes(symbols, "2016-05-01", "2026-07-31")
        assert merged.shape[1] == 2
        assert merged.index.tz is None

    def test_a_total_history_failure_raises_rather_than_serving_nulls(self, tmp_path):
        # An empty fetch must not become a cached map where every period reads
        # n/a; raising keeps the last good payload in front of the user.
        with pytest.raises(RuntimeError):
            _run(tmp_path, hist=pd.DataFrame())

    def test_a_chunk_failing_does_not_sink_the_whole_fetch(self):
        def one_bad(batch, **kwargs):
            if "AAPL" in batch:
                raise RuntimeError("worker timeout")
            return _monthly_frame(list(batch))

        with patch("server.routes.market._HEATMAP_HISTORY_CHUNK", 1), \
             patch("market_data._run_isolated_fetch", side_effect=one_bad):
            closes = _fetch_monthly_closes(["AAPL", "BBB", "CCC"], "2016-05-01", "2026-07-31")
        assert set(closes.columns) == {"BBB", "CCC"}


class TestClientParity:
    """Every field the payload ships must be consumed by both clients.

    CLAUDE.md requires feature parity across web, macOS and iOS. A metric added
    to the backend and wired into only one client is invisible on the others,
    and a renamed field silently becomes n/a — neither shows up as a failure
    anywhere else.
    """

    ROOT = Path(__file__).resolve().parent.parent

    def _payload_fields(self, tmp_path) -> set:
        return set(_run(tmp_path)["AAPL"])

    def test_web_client_reads_every_payload_field(self, tmp_path):
        src = (self.ROOT / "web_app" / "lib" / "api.ts").read_text()
        block = src.split("export interface SP500HeatmapItem {", 1)[1].split("\n}", 1)[0]
        declared = set(re.findall(r'^\s*"?([a-z0-9_]+)"?\??:', block, re.MULTILINE))
        missing = self._payload_fields(tmp_path) - declared
        assert not missing, f"web SP500HeatmapItem is missing {sorted(missing)}"

    def test_swift_client_reads_every_payload_field(self, tmp_path):
        src = (self.ROOT / "macos_app" / "Investa" / "Models" / "SP500HeatmapItem.swift").read_text()
        decoded = set(re.findall(r'(?:\bd\(|raw\[)\s*"([a-z0-9_]+)"', src))
        missing = self._payload_fields(tmp_path) - decoded
        assert not missing, f"Swift SP500HeatmapItem is missing {sorted(missing)}"

    def test_metric_tables_agree_across_clients(self):
        web = (self.ROOT / "web_app" / "lib" / "metrics.ts").read_text()
        swift = (self.ROOT / "macos_app" / "Investa" / "Features" / "Markets" / "SP500HeatmapView.swift").read_text()

        web_labels = set(re.findall(r"label:\s*'([^']+)'", web))
        swift_labels = set(re.findall(r'case\s+\w+\s*=\s*"([^"]+)"', swift))
        # Swift's enum also carries non-metric cases (sizing modes); compare the
        # metric labels the web table defines.
        assert web_labels <= swift_labels, (
            f"macOS/iOS is missing metrics: {sorted(web_labels - swift_labels)}"
        )

    def test_every_metric_field_exists_in_the_payload(self, tmp_path):
        web = (self.ROOT / "web_app" / "lib" / "metrics.ts").read_text()
        table = web.split("export const METRICS", 1)[1].split("\n];", 1)[0]
        fields = set(re.findall(r"field:\s*'([^']+)'", table))
        unknown = fields - self._payload_fields(tmp_path)
        assert not unknown, f"web METRICS reference fields the API never sends: {sorted(unknown)}"


class TestMarketCapFallback:
    def test_falls_back_to_the_fundamentals_blob_when_the_screener_is_empty(
        self, tmp_path
    ):
        # Without this the tile has no size and `cap` mode drops the stock,
        # so an unswept screener DB would render an empty heatmap.
        data = _run(
            tmp_path,
            screener={},
            fundamentals={"AAPL": {"marketCap": 3.4e12, "trailingPE": 37.4}},
        )
        assert data["AAPL"]["market_cap"] == 3.4e12
        assert data["AAPL"]["pe_ratio"] == 37.4
