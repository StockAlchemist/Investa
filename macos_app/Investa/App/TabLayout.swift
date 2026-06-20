import Foundation

/// One toggleable section in a tab's Layout configurator.
struct LayoutItem: Identifiable, Hashable {
    let id: String
    let title: String
    /// Optional group header shown in the Layout menu (Performance tab only).
    let group: String?
    init(_ id: String, _ title: String, group: String? = nil) { self.id = id; self.title = title; self.group = group }
}

/// Swift port of the web `lib/layout_registry.ts` + `lib/dashboard_constants.ts`.
/// Defines which sections each tab can show/hide and their default visibility.
enum TabLayout {
    // Performance (web "performance") — DEFAULT_ITEMS, grouped.
    static let performanceItems: [LayoutItem] = [
        LayoutItem("portfolioHero", "Portfolio Hero", group: "Overview"),
        LayoutItem("todayStrip", "Market Today", group: "Overview"),
        LayoutItem("dashboardEvents", "Upcoming Events", group: "Insights & Events"),
        LayoutItem("dashboardInsights", "Actionable Insights", group: "Insights & Events"),
        LayoutItem("totalReturn", "Total Return", group: "Returns"),
        LayoutItem("unrealizedGL", "Unrealized G/L", group: "Returns"),
        LayoutItem("realizedGain", "Realized Gain", group: "Returns"),
        LayoutItem("annualTWR", "Total TWR", group: "Returns"),
        LayoutItem("mwr", "IRR (MWR)", group: "Returns"),
        LayoutItem("ytdDividends", "Total Dividends", group: "Income & Cash"),
        LayoutItem("dividendYield", "Dividend Yield %", group: "Income & Cash"),
        LayoutItem("ytdReturn", "YTD Return", group: "Income & Cash"),
        LayoutItem("cashBalance", "Cash Balance", group: "Income & Cash"),
        LayoutItem("fxGL", "FX Gain/Loss", group: "Costs & FX"),
        LayoutItem("fees", "Fees", group: "Costs & FX"),
        LayoutItem("taxes", "Taxes", group: "Costs & FX"),
        LayoutItem("portfolioDonut", "Portfolio Composition", group: "Charts"),
        LayoutItem("performanceGraph", "Performance Graph", group: "Charts"),
        LayoutItem("riskMetrics", "Risk Analytics", group: "Risk & Attribution"),
        LayoutItem("sectorContribution", "Sector Contribution", group: "Risk & Attribution"),
        LayoutItem("topContributors", "Top Contributors", group: "Risk & Attribution"),
    ]
    static let performanceInitialVisible: Set<String> = [
        "portfolioHero", "todayStrip", "dashboardEvents", "dashboardInsights",
        "totalReturn", "unrealizedGL", "realizedGain", "annualTWR", "mwr",
        "ytdDividends", "dividendYield", "ytdReturn", "cashBalance", "fxGL", "fees", "taxes",
        "portfolioDonut", "performanceGraph",
    ]

    static let allocationItems: [LayoutItem] = [
        LayoutItem("holdingsTable", "Holdings Table"),
        LayoutItem("concentrationKpis", "Concentration KPIs"),
        LayoutItem("categoryDrift", "Category Drift (Asset, Sector, Country)"),
        LayoutItem("stockDrift", "Stock Drift"),
        LayoutItem("rebalanceHelper", "Rebalance Helper"),
        LayoutItem("treemap", "Treemap"),
        LayoutItem("donutCharts", "Donut Charts"),
    ]
    static let assetChangeItems: [LayoutItem] = [
        LayoutItem("kpiStrip", "KPI Strip"),
        LayoutItem("returnsChart", "Returns Chart"),
        LayoutItem("monthlyHeatmap", "Monthly Heatmap"),
        LayoutItem("drawdownTimeline", "Drawdown Timeline"),
        LayoutItem("benchmarkScoreboard", "Benchmark Scoreboard"),
        LayoutItem("sectorAttribution", "Sector Contribution"),
        LayoutItem("topContributors", "Top Contributors"),
    ]
    static let capitalGainsItems: [LayoutItem] = [
        LayoutItem("unrealizedTax", "Unrealized Tax Lots"),
        LayoutItem("capitalGainsKpis", "Realized Gains KPIs"),
        LayoutItem("annualCapitalGains", "Annual Realized Gains"),
        LayoutItem("capitalGainsTransactions", "Realized Gains Transactions"),
    ]
    static let dividendItems: [LayoutItem] = [
        LayoutItem("incomeKpis", "Income KPIs"),
        LayoutItem("incomeProjector", "Income Projector"),
        LayoutItem("dividendCalendar", "Dividend Calendar"),
        LayoutItem("topPayers", "Top Payers"),
        LayoutItem("byAccount", "By Account"),
        LayoutItem("annualDividends", "Annual Dividends"),
        LayoutItem("dividendTransactions", "Dividend Transactions"),
    ]

    static func items(for section: AppSection) -> [LayoutItem] {
        switch section {
        case .performance: return performanceItems
        case .allocation: return allocationItems
        case .assetChange: return assetChangeItems
        case .capitalGains: return capitalGainsItems
        case .dividend: return dividendItems
        default: return []
        }
    }

    static func hasLayout(_ section: AppSection) -> Bool { !items(for: section).isEmpty }

    static func sectionTitle(for section: AppSection) -> String {
        switch section {
        case .performance: return "Dashboard Elements"
        case .allocation: return "Portfolio Sections"
        case .assetChange: return "Performance Sections"
        case .capitalGains: return "Capital Gains Sections"
        case .dividend: return "Income Sections"
        default: return "Elements"
        }
    }

    /// Default-visible ids per tab (performance keeps a curated subset; others = all).
    static func defaultVisible(for section: AppSection) -> Set<String> {
        if section == .performance { return performanceInitialVisible }
        return Set(items(for: section).map(\.id))
    }
}
