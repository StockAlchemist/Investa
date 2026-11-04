#if os(iOS)
import SwiftUI
import XCTest
@testable import Investa

/// Dashboard cards must fit an iPhone's width at the app's Dynamic Type size.
/// One card that demands more widens the dashboard's whole ScrollView and
/// pushes every card off the right edge of the screen.
@MainActor
final class PhoneCardWidthTests: XCTestCase {
    /// A 402 pt iPhone (17 / 17 Pro) less the dashboard's 14 pt padding each side.
    private let offered: CGFloat = 402 - 28

    private func width<V: View>(of view: V, size: DynamicTypeSize = .xLarge) -> CGFloat {
        let host = UIHostingController(rootView: view
            .environment(\.horizontalSizeClass, .compact)
            .dynamicTypeSize(size))
        return host.sizeThatFits(in: CGSize(width: offered, height: .greatestFiniteMagnitude)).width
    }

    private func decode<T: Decodable>(_ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    private func day(_ offset: Int) -> String {
        let d = MarketTime.localCalendar.date(byAdding: .day, value: offset, to: Date())!
        return MarketTime.isoFormatter().string(from: d)
    }

    func testUpcomingEventsFitsPhone() throws {
        let earnings: [EarningsEvent] = try decode("""
        [{"symbol":"GOOGL","name":"Alphabet Inc.","earnings_date":"\(day(-4))","status":"reported",
          "eps_actual":12.34,"surprise_pct":-12.5,"market_timezone":"America/New_York"},
         {"symbol":"BRK-B","name":"Berkshire Hathaway","earnings_date":"\(day(3))","status":"estimated",
          "eps_estimate":5.12,"market_timezone":"America/New_York"},
         {"symbol":"CPALL.BK","name":"CP All","earnings_date":"\(day(30))","status":"confirmed",
          "eps_estimate":0.53,"market_timezone":"Asia/Bangkok"}]
        """)
        let dividends: [DividendEvent] = try decode("""
        [{"symbol":"SCHD","name":"Schwab US Dividend Equity ETF","dividend_date":"\(day(5))",
          "ex_dividend_date":"\(day(2))","amount":1234.56,"status":"estimated","market_timezone":"America/New_York"}]
        """)
        let card = UpcomingEventsCard(dividends: dividends, earnings: earnings, currency: "USD")
        for size in [DynamicTypeSize.large, .xLarge, .xxLarge, .xxxLarge] {
            XCTAssertLessThanOrEqual(width(of: card, size: size), offered, "at \(size)")
        }
    }
    private var metrics: Metrics {
        get throws {
            try decode("""
            {"market_value":1800268.71,"day_change_display":32418.0,"day_change_percent":1.83,
             "cumulative_twr":2898.08,"annualized_twr":14.99,"portfolio_mwr":15.93,
             "unrealized_gain":612345.67,"realized_gain":234567.89,"total_gain":846913.56,
             "total_return_pct":89.12,"dividends":123456.78,"cash_balance":45678.9,
             "max_drawdown":-34.56,"sharpe_ratio":1.23,"ytd_return":21.34,"volatility_ann":18.9,
             "beta":1.05,"est_annual_income_display":23456.78,"cost_basis":1012345.67}
            """)
        }
    }

    private var holdings: [Holding] {
        get throws {
            let rows = [("AMZN", 4.18, 345678.9), ("SCBRMS&P500", -2.31, 234567.8), ("BRK-B", 1.08, 123456.7),
                        ("GOOGL", -0.42, 98765.4), ("CPALL.BK", 3.3, 54321.0), ("SCHD", 0.1, 43210.9)]
            let json = rows.map { sym, pct, mv in
                """
                {"Symbol":"\(sym)","Description":"\(sym) Holdings Company Incorporated","Quantity":1234.5678,
                 "Account":"IBKR Taxable Account","Market Value (USD)":\(mv),"Day Change %":\(pct),
                 "Day Change (USD)":\(mv * pct / 100),"Unreal. Gain %":\(pct * 10),"Total Return %":\(pct * 12),
                 "Unrealized Gain (USD)":\(mv / 3),"Cost Basis (USD)":\(mv / 1.5),"Sector":"Consumer Cyclical",
                 "Local Currency":"USD","lots":[]}
                """
            }.joined(separator: ",")
            return try decode("[\(json)]")
        }
    }

    private var indices: [IndexQuote] {
        get throws {
            try decode("""
            [{"name":"Dow Jones Industrial Average","price":47562.87,"change":475.02,"changesPercentage":1.01},
             {"name":"NASDAQ Composite","price":23724.96,"change":189.22,"changesPercentage":0.80},
             {"name":"S&P 500","price":6851.97,"change":40.86,"changesPercentage":0.60}]
            """)
        }
    }

    func testOverviewCardsFitPhone() throws {
        let (m, h, i) = (try metrics, try holdings, try indices)
        let cards: [(String, AnyView)] = [
            ("hero", AnyView(PortfolioHeroCard(metrics: m, currency: "USD", longHistory: [], isMarketOpen: true))),
            ("today", AnyView(TodayStripCard(holdings: h, currency: "USD", portfolioDayPct: 1.83,
                                             indices: i, onSelectSymbol: { _ in }))),
            ("insights", AnyView(ActionableInsightsCard(holdings: h, currency: "USD"))),
            ("composition", AnyView(PortfolioCompositionCard(holdings: h, currency: "USD"))),
        ]
        for (name, card) in cards {
            for size in [DynamicTypeSize.large, .xLarge, .xxLarge] {
                XCTAssertLessThanOrEqual(width(of: card, size: size), offered, "\(name) at \(size)")
            }
        }
    }
}
#endif
