import SwiftUI

struct DashboardView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = DashboardViewModel()

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(spacing: 16) {
                    ControlBarView()
                    if let error = viewModel.errorMessage {
                        errorBanner(error)
                    }
                    metricGrid
                    PerformanceChartView(points: viewModel.history, currency: appState.displayCurrency)
                    HoldingsTableView(holdings: viewModel.holdings, currency: appState.displayCurrency)
                }
                .padding(20)
            }
        }
        .frame(minWidth: 900, minHeight: 600)
        .task {
            if !appState.didLoadSettings {
                await appState.loadSettings()
            }
            reload()
        }
        .onChange(of: selectionSignature) { _, _ in reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in
            reload()
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text("Dashboard")
                .font(.title2.bold())
            if viewModel.isLoading {
                ProgressView().controlSize(.small)
            }
            Spacer()
            Button { reload() } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                .buttonStyle(.borderless)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    // MARK: - Metric cards

    private var metricGrid: some View {
        let m = viewModel.metrics
        let cur = appState.displayCurrency
        let cards: [MetricCard] = [
            MetricCard(
                title: "Market Value",
                value: Fmt.currency(m?.marketValue, code: cur)
            ),
            MetricCard(
                title: "Day Change",
                value: Fmt.currency(m?.dayChangeDisplay, code: cur),
                subtitle: Fmt.percent(m?.dayChangePercent),
                tint: Fmt.tint(for: m?.dayChangeDisplay)
            ),
            MetricCard(
                title: "Total Gain",
                value: Fmt.currency(m?.totalGain, code: cur),
                subtitle: Fmt.percent(m?.totalReturnPct),
                tint: Fmt.tint(for: m?.totalGain)
            ),
            MetricCard(
                title: "Annualized TWR",
                value: Fmt.percent(m?.annualizedTWR),
                subtitle: m?.portfolioMWR != nil ? "MWR \(Fmt.percent(m?.portfolioMWR))" : nil,
                tint: Fmt.tint(for: m?.annualizedTWR)
            ),
        ]
        return LazyVGrid(
            columns: [GridItem(.adaptive(minimum: 180), spacing: 12)],
            spacing: 12
        ) {
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

    // MARK: - Reload

    /// A value that changes whenever any selection affecting the data changes.
    private var selectionSignature: String {
        let accounts = appState.selectedAccounts.sorted().joined(separator: ",")
        return "\(appState.displayCurrency)|\(appState.period.rawValue)|\(appState.showClosed)|\(accounts)"
    }

    private func reload() {
        viewModel.reload(
            currency: appState.displayCurrency,
            accounts: appState.accountsQuery,
            period: appState.period,
            showClosed: appState.showClosed
        )
    }
}
