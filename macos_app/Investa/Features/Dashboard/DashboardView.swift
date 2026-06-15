import SwiftUI

/// Performance tab — mirrors the web app's Dashboard composition and order
/// (lib/dashboard_constants.ts DEFAULT_ITEMS + page.tsx).
struct DashboardView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = DashboardViewModel()
    @State private var detail: SymbolID?
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var isPhone: Bool { hSize == .compact }
    #else
    private var isPhone: Bool { false }
    #endif

    private var cur: String { appState.displayCurrency }

    var body: some View {
        VStack(spacing: 0) {
            // On iPhone the nav bar already shows the "Dashboard" title.
            if !isPhone { header; Divider() }
            ScrollView {
                VStack(spacing: 16) {
                    if let error = viewModel.errorMessage { errorBanner(error) }

                    // Overview
                    if vis("portfolioHero") {
                        PortfolioHeroCard(metrics: viewModel.metrics, currency: cur, longHistory: viewModel.heroHistory)
                    }
                    if vis("todayStrip") {
                        TodayStripCard(holdings: viewModel.holdings, currency: cur,
                                       portfolioDayPct: viewModel.metrics?.dayChangePercent,
                                       indices: viewModel.indices,
                                       onSelectSymbol: { detail = SymbolID(id: $0) })
                    }

                    // Insights & Events
                    if anyInsightsVisible { sectionDivider("Insights & Events") }
                    eventsAndInsights

                    // Compact metric grid (visible scalar cards, in DEFAULT_ITEMS order)
                    if anyMetricVisible { sectionDivider("Key Metrics") }
                    metricGrid

                    // Analytics
                    if anyAnalyticsVisible { sectionDivider("Analytics") }
                    if vis("portfolioDonut") {
                        PortfolioCompositionCard(holdings: viewModel.holdings, currency: cur)
                    }
                    if vis("performanceGraph") {
                        PerformanceChartView(points: viewModel.history, currency: cur,
                                             benchmarks: appState.benchmarks, period: $appState.period,
                                             customFrom: $appState.customFrom, customTo: $appState.customTo)
                    }
                    if vis("riskMetrics") {
                        if let health = viewModel.health {
                            twoColumn(PortfolioHealthCard(health: health), RiskMetricsCard(risk: viewModel.risk))
                        } else {
                            RiskMetricsCard(risk: viewModel.risk)
                        }
                    }
                    attributionRow
                }
                .padding(isPhone ? 14 : 20)
            }
        }
        .macMinSize(width: 900, height: 600)
        .task {
            if !appState.didLoadSettings { await appState.loadSettings() }
            reload()
        }
        .onChange(of: selectionSignature) { _, _ in reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: cur) }
    }

    private var header: some View {
        HStack {
            Text("Dashboard").font(.title2.bold())
            if viewModel.isLoading { ProgressView().controlSize(.small) }
            Spacer()
            Button { reload() } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                .buttonStyle(.borderless)
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
    }

    private func sectionDivider(_ label: String) -> some View {
        HStack(spacing: 12) {
            Rectangle().fill(.quaternary).frame(height: 1)
            Text(label).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                .textCase(.uppercase).tracking(2).fixedSize()
            Rectangle().fill(.quaternary).frame(height: 1)
        }
        .padding(.top, 4)
    }

    @ViewBuilder private func twoColumn<L: View, R: View>(_ left: L, _ right: R) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 16) { left; right }
            VStack(spacing: 16) { left; right }
        }
    }

    private func vis(_ id: String) -> Bool { appState.isVisible(.performance, id) }
    private var anyInsightsVisible: Bool { vis("dashboardEvents") || vis("dashboardInsights") }
    private var anyMetricVisible: Bool { metricOrder.contains(where: vis) }
    private var anyAnalyticsVisible: Bool {
        vis("portfolioDonut") || vis("performanceGraph") || vis("riskMetrics") || vis("sectorContribution") || vis("topContributors")
    }

    @ViewBuilder private var eventsAndInsights: some View {
        let events = vis("dashboardEvents")
        let insights = vis("dashboardInsights")
        if events && insights {
            twoColumn(UpcomingEventsCard(dividends: viewModel.dividendEvents, currency: cur,
                                         onSelectSymbol: { detail = SymbolID(id: $0) }),
                      ActionableInsightsCard(holdings: viewModel.holdings, currency: cur,
                                             targets: appState.targetAllocation))
        } else if events {
            UpcomingEventsCard(dividends: viewModel.dividendEvents, currency: cur,
                               onSelectSymbol: { detail = SymbolID(id: $0) })
        } else if insights {
            ActionableInsightsCard(holdings: viewModel.holdings, currency: cur,
                                   targets: appState.targetAllocation)
        }
    }

    @ViewBuilder private var attributionRow: some View {
        if let attr = viewModel.attribution {
            let sector = vis("sectorContribution")
            let top = vis("topContributors")
            if sector && top {
                twoColumn(SectorAttributionCard(attribution: attr, currency: cur), topContributors(attr))
            } else if sector {
                SectorAttributionCard(attribution: attr, currency: cur)
            } else if top {
                topContributors(attr)
            }
        }
    }

    private func topContributors(_ attr: Attribution) -> some View {
        TopContributorsCard(attribution: attr, currency: cur,
                            accounts: appState.accountsQuery, showClosed: appState.showClosed,
                            onSelectSymbol: { detail = SymbolID(id: $0) })
    }

    private let metricOrder = ["totalReturn", "unrealizedGL", "realizedGain", "annualTWR", "mwr",
                               "ytdDividends", "dividendYield", "ytdReturn", "cashBalance", "fxGL", "fees", "taxes"]

    // Compact metric grid: the web's 12 scalar cards, filtered by Layout visibility, in order.
    private var metricGrid: some View {
        let m = viewModel.metrics
        func pct(_ v: Double?) -> String? { v == nil ? nil : Fmt.percent(v) }
        // YTD Return: prefer the risk-metrics value (a fraction), fall back to the
        // summary metric (already a percent) — mirrors the web Dashboard.
        let ytdReturn: Double? = viewModel.risk?.ytdReturn.map { $0 * 100 } ?? m?.ytdReturn
        let byId: [String: MetricCard] = [
            "totalReturn": MetricCard(title: "Total Return", value: Fmt.currency(m?.totalGain, code: cur),
                       subtitle: pct(m?.totalReturnPct), tint: Fmt.tint(for: m?.totalGain)),
            "unrealizedGL": MetricCard(title: "Unrealized G/L", value: Fmt.currency(m?.unrealizedGain, code: cur),
                       tint: Fmt.tint(for: m?.unrealizedGain)),
            "realizedGain": MetricCard(title: "Realized Gain", value: Fmt.currency(m?.realizedGain, code: cur),
                       tint: Fmt.tint(for: m?.realizedGain)),
            "annualTWR": MetricCard(title: "Total TWR", value: Fmt.percent(m?.cumulativeTWR),
                       subtitle: m?.annualizedTWR != nil ? "\(Fmt.percent(m?.annualizedTWR)) p.a." : nil,
                       tint: Fmt.tint(for: m?.cumulativeTWR)),
            "mwr": MetricCard(title: "IRR (MWR)", value: Fmt.percent(m?.portfolioMWR), subtitle: "p.a.",
                       tint: Fmt.tint(for: m?.portfolioMWR)),
            "ytdDividends": MetricCard(title: "Total Dividends", value: Fmt.currency(m?.dividends, code: cur), tint: .up),
            "dividendYield": MetricCard(title: "Dividend Yield", value: Fmt.percent(m?.dividendReturnCumulative),
                       subtitle: m?.dividendReturnAnnualized != nil ? "\(Fmt.percent(m?.dividendReturnAnnualized)) p.a." : nil,
                       accent: Theme.brand),
            "ytdReturn": MetricCard(title: "YTD Return", value: Fmt.percent(ytdReturn), tint: Fmt.tint(for: ytdReturn)),
            "cashBalance": MetricCard(title: "Cash Balance", value: Fmt.currency(m?.cashBalance, code: cur),
                       accent: Theme.brand),
            "fxGL": MetricCard(title: "FX Gain/Loss", value: Fmt.currency(m?.fxGainLossDisplay, code: cur),
                       subtitle: pct(m?.fxGainLossPct), tint: Fmt.tint(for: m?.fxGainLossDisplay)),
            "fees": MetricCard(title: "Fees", value: Fmt.currency(m?.commissions, code: cur), tint: .down),
            "taxes": MetricCard(title: "Taxes", value: Fmt.currency(m?.taxes, code: cur), tint: .down),
        ]
        let cards = metricOrder.filter { vis($0) }.compactMap { byId[$0] }
        // macOS: a fixed 6-column grid (the 12 metric cards land in two rows of 6);
        // iPhone: adaptive so cards stay legible on the narrow width.
        let columns: [GridItem] = isPhone
            ? [GridItem(.adaptive(minimum: 150), spacing: 12)]
            : Array(repeating: GridItem(.flexible(), spacing: 12), count: 6)
        return LazyVGrid(columns: columns, spacing: 12) {
            ForEach(cards) { MetricCardView(card: $0) }
        }
    }

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(message)
            Spacer()
            Button("Retry") { reload() }
        }
        .font(.callout)
        .padding(12)
        .background(.red.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(.red)
    }

    private var selectionSignature: String {
        let accounts = appState.selectedAccounts.sorted().joined(separator: ",")
        let benches = appState.benchmarks.sorted().joined(separator: ",")
        let custom = appState.period == .custom ? "\(appState.customFromYMD)>\(appState.customToYMD)" : ""
        return "\(cur)|\(appState.period.rawValue)|\(custom)|\(appState.showClosed)|\(accounts)|\(benches)"
    }

    private func reload() {
        viewModel.reload(currency: cur, accounts: appState.accountsQuery, period: appState.period,
                         showClosed: appState.showClosed, benchmarks: appState.benchmarks,
                         customFrom: appState.period == .custom ? appState.customFromYMD : nil,
                         customTo: appState.period == .custom ? appState.customToYMD : nil)
    }
}
