import XCTest
@testable import Investa

final class ModelDecodingTests: XCTestCase {

    /// The data-quality flag mirrors `web_app/lib/api.ts` and the
    /// `/api/data_quality` response. The snake_case key and the severity
    /// fallback are the two things that would fail silently.
    func testDataQualityFlagDecoding() throws {
        let json = """
        {
            "symbols": {
                "BYND": {
                    "symbol": "BYND",
                    "severity": "high",
                    "findings": 3,
                    "kinds": ["unapplied", "mixed"],
                    "occurred_on": "2026-08-14",
                    "detail": "A 0.0333 split is on record, but the stored prices do not reflect it."
                }
            },
            "count": 1,
            "scanned": true
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(DataQualityResponse.self, from: json)
        let flag = try XCTUnwrap(decoded.symbols["BYND"])

        XCTAssertEqual(flag.severity, .high)
        XCTAssertEqual(flag.findings, 3)
        XCTAssertEqual(flag.kinds, ["unapplied", "mixed"])
        XCTAssertEqual(flag.occurredOn, "2026-08-14", "occurred_on must map to occurredOn")
        XCTAssertTrue(decoded.scanned)
    }

    /// A severity this build does not know about must read as the milder one,
    /// so an older client understates rather than alarms.
    func testUnknownSeverityDegradesToMedium() throws {
        let json = """
        {"symbols": {"X": {"symbol": "X", "severity": "catastrophic",
         "findings": 1, "kinds": [], "occurred_on": null, "detail": null}},
         "count": 1, "scanned": true}
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(DataQualityResponse.self, from: json)

        XCTAssertEqual(decoded.symbols["X"]?.severity, .medium)
        XCTAssertNil(decoded.symbols["X"]?.occurredOn)
    }

    func testAppSettingsDecodingAndComputedProperties() throws {
        let json = """
        {
            "available_currencies": ["USD", "EUR", "THB"],
            "account_groups": {
                "Retirement": ["Roth IRA", "401k"],
                "Taxable": ["Brokerage Main", "Margin"]
            },
            "account_group_order": ["Taxable", "Retirement"],
            "display_currency": "USD",
            "selected_accounts": ["Brokerage Main"],
            "benchmarks": ["S&P 500", "NASDAQ"],
            "show_closed": false,
            "manual_overrides": {
                "AAPL": 175.50,
                "NVDA": { "price": 120.0, "currency": "USD" }
            },
            "account_closure_dates": {
                "Old Account": "2024-01-01"
            },
            "account_currency_map": {
                "SET": "THB",
                "Brokerage Main": "USD"
            },
            "account_cash_mode_map": {
                "Brokerage Main": "Auto"
            },
            "gemini_api_key": "test_gemini_key",
            "fmp_api_key": "test_fmp_key",
            "sec_th_api_key": "test_sec_key",
            "bot_api_key": "test_bot_key",
            "tiingo_api_key": "test_tiingo_key"
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let settings = try decoder.decode(AppSettings.self, from: json)

        XCTAssertEqual(settings.displayCurrency, "USD")
        XCTAssertEqual(settings.availableCurrencies, ["USD", "EUR", "THB"])
        XCTAssertEqual(settings.accountGroupOrder, ["Taxable", "Retirement"])
        XCTAssertEqual(settings.accountGroups?["Retirement"], ["Roth IRA", "401k"])
        XCTAssertEqual(settings.accountGroups?["Taxable"], ["Brokerage Main", "Margin"])

        // Test allAccounts computed property respecting accountGroupOrder
        let allAccounts = settings.allAccounts
        XCTAssertEqual(allAccounts, ["Brokerage Main", "Margin", "Roth IRA", "401k"])

        // Test manualOverridePrices computed property (both number and object forms)
        let prices = settings.manualOverridePrices
        XCTAssertEqual(prices["AAPL"], 175.50)
        XCTAssertEqual(prices["NVDA"], 120.0)

        // Test closure dates & mappings
        XCTAssertEqual(settings.accountClosureDates?["Old Account"], "2024-01-01")
        XCTAssertEqual(settings.accountCurrencyMap?["SET"], "THB")
        XCTAssertEqual(settings.accountCashModeMap?["Brokerage Main"], "Auto")

        // Test API keys decoding
        XCTAssertEqual(settings.geminiApiKey, "test_gemini_key")
        XCTAssertEqual(settings.fmpApiKey, "test_fmp_key")
        XCTAssertEqual(settings.secThApiKey, "test_sec_key")
        XCTAssertEqual(settings.botApiKey, "test_bot_key")
        XCTAssertEqual(settings.tiingoApiKey, "test_tiingo_key")
    }

    func testHoldingDecoding() throws {
        let json = """
        {
            "Symbol": "AAPL",
            "Description": "Apple Inc.",
            "Quantity": 50.0,
            "Price": 180.0,
            "Market Value (USD)": 9000.0,
            "Cost Basis": 7500.0,
            "Unrealized Gain": 1500.0,
            "Total Return %": 20.0,
            "Account": "Taxable Main",
            "Local Currency": "USD",
            "Sector": "Technology"
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let holding = try decoder.decode(Holding.self, from: json)

        XCTAssertEqual(holding.symbol, "AAPL")
        XCTAssertEqual(holding.quantity, 50.0)
        XCTAssertEqual(holding.account, "Taxable Main")
        XCTAssertEqual(holding.sector, "Technology")
        XCTAssertEqual(holding.marketValue(currency: "USD"), 9000.0)
        XCTAssertEqual(holding.double("Unrealized Gain"), 1500.0)
    }

    func testTransactionDecoding() throws {
        let json = """
        {
            "Date": "2024-05-15",
            "Type": "Buy",
            "Symbol": "MSFT",
            "Quantity": 10.0,
            "Price/Share": 420.0,
            "Total Amount": -4200.0,
            "Commission": 1.50,
            "Account": "Retirement",
            "Local Currency": "USD"
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let tx = try decoder.decode(Transaction.self, from: json)

        XCTAssertEqual(tx.symbol, "MSFT")
        XCTAssertEqual(tx.type, "Buy")
        XCTAssertEqual(tx.quantity, 10.0)
        XCTAssertEqual(tx.pricePerShare, 420.0)
        XCTAssertEqual(tx.totalAmount, -4200.0)
        XCTAssertEqual(tx.commission, 1.50)
        XCTAssertEqual(tx.cashImpact, .outflow)
    }

    func testDividendDecoding() throws {
        let json = """
        {
            "Symbol": "KO",
            "Date": "2024-06-01",
            "DividendAmountLocal": 48.50,
            "DividendAmountDisplayCurrency": 48.50,
            "Account": "Taxable Main",
            "LocalCurrency": "USD"
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let div = try decoder.decode(Dividend.self, from: json)

        XCTAssertEqual(div.symbol, "KO")
        XCTAssertEqual(div.amountLocal, 48.50)
        XCTAssertEqual(div.amountDisplay, 48.50)
        XCTAssertEqual(div.account, "Taxable Main")
    }

    func testCapitalGainDecoding() throws {
        let json = """
        {
            "Symbol": "NVDA",
            "Date": "2024-01-10",
            "Type": "Sell",
            "Quantity": 20.0,
            "Total Proceeds (Display)": 10000.0,
            "Total Cost Basis (Display)": 3000.0,
            "Realized Gain (Display)": 7000.0,
            "Account": "Brokerage"
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let gain = try decoder.decode(CapitalGain.self, from: json)

        XCTAssertEqual(gain.symbol, "NVDA")
        XCTAssertEqual(gain.quantity, 20.0)
        XCTAssertEqual(gain.proceedsDisplay, 10000.0)
        XCTAssertEqual(gain.realizedGainDisplay, 7000.0)
    }

    func testAPIErrorDescriptions() {
        let unauthorized = APIError.unauthorized
        XCTAssertTrue(unauthorized.errorDescription?.contains("expired") == true)

        let http = APIError.http(status: 404, detail: "Resource not found")
        XCTAssertEqual(http.errorDescription, "Resource not found")

        let httpFallback = APIError.http(status: 500, detail: nil)
        XCTAssertEqual(httpFallback.errorDescription, "Request failed (HTTP 500).")

        let invalid = APIError.invalidURL
        XCTAssertEqual(invalid.errorDescription, "The server address is invalid.")
    }

    /// The AI review travels on every ranked row. `rank` is the blended
    /// position and `base_rank` the stored one; the shift between them is what
    /// the row marks, and an unreviewed company must decode as nil, not 0.
    func testBuffettRankRowDecodesTheAIReview() throws {
        let json = """
        {
            "total": 2,
            "ai_weight": 0.2,
            "rows": [
                {"symbol": "DECK", "model": "generic", "rank": 1, "base_rank": 3,
                 "ai_moat": 8.5, "ai_financial_strength": 9.0, "ai_predictability": 7.0,
                 "ai_growth": 8.0, "ai_rating": 8.125, "ai_score": 94.0},
                {"symbol": "WTM", "model": "insurer", "rank": 5, "base_rank": 5,
                 "ai_rating": null, "ai_score": null}
            ]
        }
        """.data(using: .utf8)!

        let page = try JSONDecoder().decode(BuffettRankPage.self, from: json)
        let reviewed = page.rows[0]
        let unreviewed = page.rows[1]

        XCTAssertEqual(reviewed.baseRank, 3)
        XCTAssertEqual(reviewed.rankShift, 2, "rose from 3 to 1")
        XCTAssertEqual(reviewed.aiMoat, 8.5)
        XCTAssertEqual(reviewed.aiScore, 94.0)
        XCTAssertNil(unreviewed.aiScore)
        XCTAssertNil(unreviewed.rankShift, "an unmoved company carries no marker")
    }

    func testAIReviewWeightOnlyHonoursPresets() {
        XCTAssertEqual(AIReviewWeight.normalised(0.3), 0.3)
        XCTAssertEqual(AIReviewWeight.normalised(0.37), AIReviewWeight.defaultValue)
        XCTAssertEqual(AIReviewWeight.label(0), "Off")
        XCTAssertEqual(AIReviewWeight.label(0.2), "20%")
        XCTAssertEqual(AIReviewWeight.queryItem(0.5).value, "0.5")
        // The full range is offered: the review alone can decide the order.
        XCTAssertEqual(AIReviewWeight.presets.last, 1.0)
        XCTAssertEqual(AIReviewWeight.label(1), "100%")
        XCTAssertEqual(AIReviewWeight.queryItem(1).value, "1.0")
    }

    /// Favorites is marked by the server, never inferred from the name, and an
    /// older server that omits the flag must decode as "not favorites".
    func testWatchlistMetaAndMembershipDecoding() throws {
        let lists = """
        [{"id": 7, "name": "Favorites", "created_at": "2026-09-25", "is_favorites": true},
         {"id": 1, "name": "My Watchlist", "created_at": "2026-01-01"}]
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode([WatchlistMeta].self, from: lists)
        XCTAssertEqual(decoded[0].isFavorites, true)
        XCTAssertNil(decoded[1].isFavorites)

        let membership = """
        {"symbol": "AAPL", "watchlist_ids": [1, 7]}
        """.data(using: .utf8)!
        let parsed = try JSONDecoder().decode(WatchlistMembership.self, from: membership)
        XCTAssertEqual(parsed.watchlistIDs, [1, 7])
    }
}
