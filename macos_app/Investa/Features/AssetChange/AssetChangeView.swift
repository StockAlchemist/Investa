import SwiftUI

@MainActor
final class AssetChangeViewModel: ObservableObject {
    /// period name → rows (each row a dynamic-keyed dict).
    @Published var data: [String: [[String: JSONValue]]] = [:]
    @Published var metrics: Metrics?
    @Published var risk: RiskMetrics?
    @Published var attribution: Attribution?
    @Published var history: [PerformancePoint] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: APIClient
    private var task: Task<Void, Never>?

    init(api: APIClient = .shared) { self.api = api }

    func reload(currency: String, accounts: [String]?, showClosed: Bool, period: Period, benchmarks: [String]) {
        task?.cancel()
        task = Task { [weak self] in
            await self?.load(currency: currency, accounts: accounts, showClosed: showClosed,
                             period: period, benchmarks: benchmarks)
        }
    }

    private func load(currency: String, accounts: [String]?, showClosed: Bool, period: Period, benchmarks: [String]) async {
        isLoading = true; errorMessage = nil
        defer { isLoading = false }
        let accountItems = APIClient.arrayQuery("accounts", accounts)
        let curItem = URLQueryItem(name: "currency", value: currency)
        let closedItem = URLQueryItem(name: "show_closed", value: showClosed ? "true" : "false")

        async let dataR: [String: [[String: JSONValue]]] = api.get(
            "/asset_change", query: [curItem, closedItem] + accountItems + APIClient.arrayQuery("benchmarks", benchmarks))
        async let summaryR: SummaryResponse = api.get("/summary", query: [curItem, closedItem] + accountItems)
        async let riskR: RiskMetrics = api.get("/risk_metrics", query: [curItem, closedItem] + accountItems)
        async let attrR: Attribution = api.get("/attribution", query: [curItem, closedItem] + accountItems)
        // Benchmark scoreboard + drawdown use the daily history with benchmarks overlaid.
        async let histR: [PerformancePoint] = api.get(
            "/history",
            query: [curItem, URLQueryItem(name: "period", value: "1y"), URLQueryItem(name: "interval", value: "1d")]
                + accountItems + APIClient.arrayQuery("benchmarks", benchmarks))

        do {
            data = try await dataR
        } catch is CancellationError { return }
        catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
        metrics = (try? await summaryR)?.metrics
        risk = try? await riskR
        attribution = try? await attrR
        history = (try? await histR) ?? []
    }
}

/// Performance tab — mirrors the web "Asset Change" tab (AssetChange.tsx),
/// whose layout group is labelled "Performance Sections".
struct AssetChangeView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = AssetChangeViewModel()
    @State private var detail: SymbolID?
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    @Environment(\.verticalSizeClass) private var vSize
    #endif

    private var cur: String { appState.displayCurrency }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Performance").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            if let error = viewModel.errorMessage {
                Text(error).foregroundStyle(.red).font(.callout).padding(12)
            }
            ScrollView {
                VStack(spacing: 20) {
                    if vis("kpiStrip") {
                        PerfKpiStrip(data: viewModel.data, metrics: viewModel.metrics,
                                     risk: viewModel.risk, benchmarks: appState.benchmarks)
                    }
                    if vis("returnsChart") { ReturnsChart(data: viewModel.data, currency: cur) }
                    if vis("monthlyHeatmap") { MonthlyHeatmap(data: viewModel.data) }
                    drawdownBenchmarkRow
                    attributionRow
                }
                .padding(20)
            }
        }
        .macMinSize(width: 820, height: 560)
        .task(id: signature) { reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: cur) }
    }

    private func vis(_ id: String) -> Bool { appState.isVisible(.assetChange, id) }

    @ViewBuilder private var drawdownBenchmarkRow: some View {
        let dd = vis("drawdownTimeline"); let bs = vis("benchmarkScoreboard")
        if dd && bs {
            twoColumn(DrawdownTimeline(history: viewModel.history), BenchmarkScoreboard(history: viewModel.history))
        } else if dd { DrawdownTimeline(history: viewModel.history) }
        else if bs { BenchmarkScoreboard(history: viewModel.history) }
    }

    @ViewBuilder private var attributionRow: some View {
        if let attr = viewModel.attribution {
            let sector = vis("sectorAttribution"); let top = vis("topContributors")
            if sector && top {
                twoColumn(SectorAttributionCard(attribution: attr, currency: cur), topContributors(attr))
            } else if sector { SectorAttributionCard(attribution: attr, currency: cur) }
            else if top { topContributors(attr) }
        }
    }

    private func topContributors(_ attr: Attribution) -> some View {
        TopContributorsCard(attribution: attr, currency: cur,
                            accounts: appState.accountsQuery, showClosed: appState.showClosed,
                            onSelectSymbol: { detail = SymbolID(id: $0) })
    }

    @ViewBuilder private func twoColumn<L: View, R: View>(_ left: L, _ right: R) -> some View {
        #if os(iOS)
        if hSize == .compact && vSize == .regular {
            VStack(spacing: 20) { left; right }
        } else {
            HStack(alignment: .top, spacing: 20) { left; right }
        }
        #else
        HStack(alignment: .top, spacing: 20) { left; right }
        #endif
    }

    private var signature: String {
        let b = appState.benchmarks.sorted().joined(separator: ",")
        return "\(cur)|\(appState.showClosed)|\(appState.selectedAccounts.sorted().joined(separator: ","))|\(b)"
    }
    private func reload() {
        viewModel.reload(currency: cur, accounts: appState.accountsQuery, showClosed: appState.showClosed,
                         period: appState.period, benchmarks: appState.benchmarks)
    }
}
