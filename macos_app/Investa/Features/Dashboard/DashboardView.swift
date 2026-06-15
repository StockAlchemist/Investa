import SwiftUI

/// Performance tab — mirrors the web app's Dashboard composition and order
/// (lib/dashboard_constants.ts DEFAULT_ITEMS + page.tsx).
struct DashboardView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = DashboardViewModel()
    @State private var detail: SymbolID?

    private var cur: String { appState.displayCurrency }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(spacing: 16) {
                    ControlBarView()
                    if let error = viewModel.errorMessage { errorBanner(error) }

                    // Overview
                    PortfolioHeroCard(metrics: viewModel.metrics, currency: cur, longHistory: viewModel.heroHistory)
                    TodayStripCard(holdings: viewModel.holdings, currency: cur,
                                   portfolioDayPct: viewModel.metrics?.dayChangePercent,
                                   indices: viewModel.indices,
                                   onSelectSymbol: { detail = SymbolID(id: $0) })

                    // Insights & Events
                    twoColumn(
                        UpcomingEventsCard(dividends: viewModel.dividendEvents, currency: cur),
                        ActionableInsightsCard(holdings: viewModel.holdings, currency: cur)
                    )

                    // Compact metric grid (the 12 scalar cards, in DEFAULT_ITEMS order)
                    metricGrid

                    // Analytics divider
                    analyticsDivider

                    // Analytics widgets
                    PortfolioCompositionCard(holdings: viewModel.holdings, currency: cur)
                    PerformanceChartView(points: viewModel.history, currency: cur,
                                         benchmarks: appState.benchmarks, period: $appState.period)
                    if let health = viewModel.health {
                        twoColumn(PortfolioHealthCard(health: health), RiskMetricsCard(risk: viewModel.risk))
                    } else {
                        RiskMetricsCard(risk: viewModel.risk)
                    }
                    if let attr = viewModel.attribution {
                        twoColumn(SectorAttributionCard(attribution: attr, currency: cur),
                                  TopContributorsCard(attribution: attr, currency: cur))
                    }
                }
                .padding(20)
            }
        }
        .frame(minWidth: 900, minHeight: 600)
        .task {
            if !appState.didLoadSettings { await appState.loadSettings() }
            reload()
        }
        .onChange(of: selectionSignature) { _, _ in reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
        .sheet(item: $detail) { StockDetailView(symbol: $0.id) }
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

    private var analyticsDivider: some View {
        HStack(spacing: 12) {
            Rectangle().fill(.quaternary).frame(height: 1)
            Text("ANALYTICS").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).tracking(2)
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

    // Compact metric grid: exactly the web's 12 scalar cards in DEFAULT_ITEMS order.
    private var metricGrid: some View {
        let m = viewModel.metrics
        func pct(_ v: Double?) -> String? { v == nil ? nil : Fmt.percent(v) }
        let cards: [MetricCard] = [
            MetricCard(title: "Total Return", value: Fmt.currency(m?.totalGain, code: cur),
                       subtitle: pct(m?.totalReturnPct), tint: Fmt.tint(for: m?.totalGain)),
            MetricCard(title: "Unrealized G/L", value: Fmt.currency(m?.unrealizedGain, code: cur),
                       tint: Fmt.tint(for: m?.unrealizedGain)),
            MetricCard(title: "Realized Gain", value: Fmt.currency(m?.realizedGain, code: cur),
                       tint: Fmt.tint(for: m?.realizedGain)),
            MetricCard(title: "Total TWR", value: Fmt.percent(m?.cumulativeTWR),
                       subtitle: m?.annualizedTWR != nil ? "\(Fmt.percent(m?.annualizedTWR)) p.a." : nil,
                       tint: Fmt.tint(for: m?.cumulativeTWR)),
            MetricCard(title: "IRR (MWR)", value: Fmt.percent(m?.portfolioMWR), subtitle: "p.a.",
                       tint: Fmt.tint(for: m?.portfolioMWR)),
            MetricCard(title: "Total Dividends", value: Fmt.currency(m?.dividends, code: cur), tint: .green),
            MetricCard(title: "Dividend Yield", value: Fmt.percent(m?.dividendReturnCumulative),
                       subtitle: m?.dividendReturnAnnualized != nil ? "\(Fmt.percent(m?.dividendReturnAnnualized)) p.a." : nil),
            MetricCard(title: "YTD Return", value: Fmt.percent(m?.ytdReturn), tint: Fmt.tint(for: m?.ytdReturn)),
            MetricCard(title: "Cash Balance", value: Fmt.currency(m?.cashBalance, code: cur)),
            MetricCard(title: "FX Gain/Loss", value: Fmt.currency(m?.fxGainLossDisplay, code: cur),
                       subtitle: pct(m?.fxGainLossPct), tint: Fmt.tint(for: m?.fxGainLossDisplay)),
            MetricCard(title: "Fees", value: Fmt.currency(m?.commissions, code: cur), tint: .red),
            MetricCard(title: "Taxes", value: Fmt.currency(m?.taxes, code: cur), tint: .red),
        ]
        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 165), spacing: 12)], spacing: 12) {
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
        return "\(cur)|\(appState.period.rawValue)|\(appState.showClosed)|\(accounts)|\(benches)"
    }

    private func reload() {
        viewModel.reload(currency: cur, accounts: appState.accountsQuery, period: appState.period,
                         showClosed: appState.showClosed, benchmarks: appState.benchmarks)
    }
}
