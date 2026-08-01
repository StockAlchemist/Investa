"""The metric block the stock detail window reads (`key_metrics`).

Same computation the S&P 500 heatmap runs over the whole index — these tests pin
the per-symbol path and the units, which are the part clients cannot infer: a
fraction shown as a percentage is off by 100x and still looks plausible.
"""

import re
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from server.routes.market import _fundamental_metrics, _key_metrics_for_symbol

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 3, 31)

# Filed annual EPS: 2.00 -> 4.00 over five years, and 3.00 -> 4.00 over three.
EPS_FACTS = {
    "EarningsPerShareDiluted": {
        "2021-03-31": 2.0,
        "2023-03-31": 3.0,
        "2026-03-31": 4.0,
    }
}

INFO = {
    "marketCap": 1_000_000_000.0,
    "trailingPE": 20.0,
    "forwardPE": 17.0,
    "freeCashflow": 50_000_000.0,
    "bookValue": 10.0,
    "sharesOutstanding": 20_000_000.0,
    "totalDebt": 100_000_000.0,
    "netIncomeToCommon": 60_000_000.0,
    "dividendRate": 2.0,
    "volume": 3_000_000,
    "averageVolume": 1_500_000,
    "returnOnEquity": 0.2,
    "debtToEquity": 45.0,
}


def _metrics(info=None, facts=None, price=40.0, today=TODAY):
    merged = {**INFO, **(info or {})}
    return _fundamental_metrics(
        merged,
        facts or {},
        market_cap=merged.get("marketCap"),
        pe_ratio=merged.get("trailingPE"),
        price=price,
        today=today,
    )


class TestUnits:
    """Which fields are fractions and which are percent points.

    Both conventions are in the payload deliberately (the filed leverage ratios
    are percent points because that is how the filings express them), so the
    split is a contract, not an accident.
    """

    def test_dividend_yield_is_a_fraction_resolved_against_price(self):
        # Yahoo's own `dividendYield` flips between percent and fraction, so it
        # is settled against rate/price instead of trusted.
        assert _metrics()["dividend_yield"] == 0.05

    def test_margins_and_returns_stay_fractions(self):
        m = _metrics({"grossMargins": 0.48, "returnOnAssets": 0.27})
        assert m["gross_margin"] == 0.48
        assert m["roa"] == 0.27

    def test_leverage_ratios_are_percent_points(self):
        m = _metrics(facts={
            "LongTermDebtNoncurrent": {"2026-03-31": 300.0},
            "StockholdersEquity": {"2026-03-31": 600.0},
        })
        assert m["debt_equity"] == 45.0        # Yahoo's, passed through
        assert m["lt_debt_equity"] == 50.0     # filed, scaled to match

    def test_negative_book_equity_reports_no_leverage_ratio(self):
        # The ratio flips sign rather than growing, so a number here would be
        # actively misleading.
        m = _metrics(facts={
            "LongTermDebtNoncurrent": {"2026-03-31": 300.0},
            "StockholdersEquity": {"2026-03-31": -600.0},
        })
        assert m["lt_debt_equity"] is None


class TestDerivedFigures:
    def test_price_to_free_cash_flow(self):
        assert _metrics()["p_fcf"] == 20.0

    def test_no_free_cash_flow_means_no_multiple(self):
        assert _metrics({"freeCashflow": 0})["p_fcf"] is None
        assert _metrics({"freeCashflow": None})["p_fcf"] is None

    def test_roic_uses_book_equity_plus_debt(self):
        # 60m over (10 * 20m book + 100m debt) = 20%.
        assert _metrics()["roic"] == 0.2

    def test_roic_absent_when_the_book_value_is(self):
        assert _metrics({"bookValue": None})["roic"] is None

    def test_relative_volume_falls_back_to_the_regular_market_field(self):
        assert _metrics()["relative_volume"] == 2.0
        m = _metrics({"volume": None, "regularMarketVolume": 750_000})
        assert m["relative_volume"] == 0.5


class TestFiledHistory:
    def test_growth_comes_from_filed_annuals(self):
        m = _metrics(facts=EPS_FACTS)
        # 3.00 -> 4.00 over three years, 2.00 -> 4.00 over five.
        assert round(m["eps_growth_3y"], 4) == round((4 / 3) ** (1 / 3) - 1, 4)
        assert round(m["eps_growth_5y"], 4) == round(2 ** (1 / 5) - 1, 4)

    def test_no_filings_means_no_growth_rate_rather_than_a_guess(self):
        m = _metrics(facts={})
        assert m["eps_growth_3y"] is None
        assert m["eps_growth_5y"] is None
        assert m["sales_growth_3y"] is None
        # The trailing-twelve-month figures still come through.
        assert m["pe_ratio"] == 20.0

    def test_growth_through_a_loss_is_not_reported(self):
        m = _metrics(facts={
            "EarningsPerShareDiluted": {"2023-03-31": -1.0, "2026-03-31": 4.0}
        })
        assert m["eps_growth_3y"] is None


class TestEarnings:
    def test_eps_surprise_is_converted_to_a_fraction(self):
        # Yahoo stashes this one in percent points; everything else on the wire
        # is a fraction.
        m = _metrics({"_earnings_history": {
            "2025-12-31": {"eps_actual": 1.0, "surprise_pct": 4.0},
            "2026-03-31": {"eps_actual": 2.0, "surprise_pct": 6.74},
        }})
        assert m["eps_surprise"] == 0.0674

    def test_days_to_earnings_counts_from_the_given_market_day(self):
        ts = datetime(2026, 4, 12, tzinfo=timezone.utc).timestamp()
        assert _metrics({"earningsTimestamp": ts})["earnings_days"] == 12

    def test_a_report_already_given_counts_negative(self):
        ts = datetime(2026, 3, 29, tzinfo=timezone.utc).timestamp()
        assert _metrics({"earningsTimestamp": ts})["earnings_days"] == -2


class TestPerSymbolPath:
    def test_reads_the_filed_history_of_the_symbols_own_filer(self):
        with (
            patch("server.routes.market._cik_for_symbol", return_value="0000320193"),
            patch(
                "server.routes.market._edgar_annual_facts",
                return_value={"AAPL": EPS_FACTS},
            ) as facts,
        ):
            m = _key_metrics_for_symbol("AAPL", {**INFO, "regularMarketPrice": 40.0})

        assert facts.call_args[0][0] == [{"symbol": "AAPL", "cik": "0000320193"}]
        assert m["eps_growth_5y"] is not None
        assert m["dividend_yield"] == 0.05

    def test_a_symbol_with_no_filings_still_gets_the_rest_of_the_block(self):
        # Foreign private issuers and recent listings legitimately have no CIK;
        # that costs the filed-history fields, not the whole panel.
        with patch("server.routes.market._cik_for_symbol", return_value=None):
            m = _key_metrics_for_symbol("SHOP", {**INFO, "regularMarketPrice": 40.0})

        assert m["eps_growth_3y"] is None
        assert m["p_fcf"] == 20.0
        assert m["roic"] == 0.2

    def test_days_to_earnings_is_counted_on_the_exchange_clock(self):
        # Investa runs on a Bangkok clock that is up to a day ahead of New York,
        # so a server-local count would read a day short for a US listing.
        ts = datetime(2026, 4, 12, tzinfo=timezone.utc).timestamp()
        info = {"earningsTimestamp": ts, "exchangeTimezoneName": "America/New_York"}
        with (
            patch("server.routes.market._cik_for_symbol", return_value=None),
            patch("server.calendar_events.market_today", return_value=TODAY) as today,
        ):
            m = _key_metrics_for_symbol("AAPL", info)

        today.assert_called_once_with(info)
        assert m["earnings_days"] == 12


class TestClientParity:
    """The clients render this block from a shared catalogue of field names.

    A field renamed on one side silently becomes "n/a" on the other, which looks
    like missing data rather than a bug — so the names are asserted, not trusted.
    """

    def _block_fields(self) -> set:
        return set(_metrics(facts=EPS_FACTS))

    def test_the_web_catalogue_only_references_fields_the_backend_sends(self):
        src = (ROOT / "web_app" / "lib" / "metrics.ts").read_text()
        table = src.split("export const METRICS", 1)[1].split("\n];", 1)[0]
        # Performance metrics come off the heatmap payload, which carries price
        # history this per-symbol block deliberately does not.
        panel = [
            line for line in table.splitlines()
            if "group: 'Performance'" not in line and "field:" in line
        ]
        fields = {re.search(r"field:\s*'([^']+)'", line).group(1) for line in panel}
        unknown = fields - self._block_fields()
        assert not unknown, f"the detail panel reads fields the API never sends: {sorted(unknown)}"

    def test_the_swift_client_reads_every_field_of_the_block(self):
        src = (ROOT / "macos_app" / "Investa" / "Models" / "StockKeyMetrics.swift").read_text()
        read = set(re.findall(r'field:\s*"([a-z0-9_]+)"', src))
        missing = self._block_fields() - read
        assert not missing, f"macOS/iOS never reads {sorted(missing)}"

    def test_the_two_catalogues_agree_on_every_scale(self):
        """A metric judged against different midpoints reads as a different
        company on each client — the same figure, one green and one red."""
        web = (ROOT / "web_app" / "lib" / "metrics.ts").read_text()
        swift = (ROOT / "macos_app" / "Investa" / "Models" / "StockKeyMetrics.swift").read_text()

        web_scales = {}
        table = web.split("export const METRICS", 1)[1].split("\n];", 1)[0]
        for line in table.splitlines():
            if "group: 'Performance'" in line or "field:" not in line:
                continue
            field = re.search(r"field:\s*'([^']+)'", line).group(1)
            mid = float(re.search(r"mid:\s*([-\d.e]+)", line).group(1))
            clamp = float(re.search(r"clamp:\s*([\d.e]+)", line).group(1))
            web_scales[field] = (mid, clamp, "inverted: true" in line)

        swift_scales = {}
        for block in re.findall(r'\.init\(field:.*?\),\n', swift, re.DOTALL):
            field = re.search(r'field:\s*"([^"]+)"', block).group(1)
            mid = float(re.search(r"mid:\s*([-\d.e]+)", block).group(1))
            clamp = float(re.search(r"clamp:\s*([\d.e]+)", block).group(1))
            swift_scales[field] = (mid, clamp, "inverted: true" in block)

        assert swift_scales == web_scales
