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
            }
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
}
