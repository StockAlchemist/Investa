import SwiftUI

@MainActor
final class AllocationViewModel: ObservableObject {
    @Published var holdings: [Holding] = []
    /// Per-holding period price returns (%): symbol → period ("1m"/"3m"/…) → value.
    /// Powers the performance heatmap; null backend values are dropped.
    @Published var holdingReturns: [String: [String: Double]] = [:]
    /// bucket → name → target %.
    @Published var targets: [String: [String: Double]] = [:]
    // Starts true so the first render shows a loading state, not an empty one,
    // before the initial `.task` fires.
    @Published var isLoading = true
    @Published var errorMessage: String?

    private let api: APIClient
    private var task: Task<Void, Never>?

    init(api: APIClient = .shared) { self.api = api }

    func reload(currency: String, accounts: [String]?, showClosed: Bool) {
        task?.cancel()
        task = Task { [weak self] in await self?.load(currency: currency, accounts: accounts, showClosed: showClosed) }
    }

    private func load(currency: String, accounts: [String]?, showClosed: Bool) async {
        isLoading = true; errorMessage = nil
        defer { isLoading = false }
        async let holdingsR: [Holding] = api.get(
            "/holdings",
            query: [URLQueryItem(name: "currency", value: currency),
                    URLQueryItem(name: "show_closed", value: showClosed ? "true" : "false")]
                + APIClient.arrayQuery("accounts", accounts))
        async let settingsR: AppSettings = api.get("/settings")
        do {
            holdings = try await holdingsR
        } catch is CancellationError { return }
        catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
        if let s = try? await settingsR { targets = s.targetAllocation ?? [:] }
        await loadReturns()
    }

    /// Fetch period price returns for the current holdings (best-effort; the
    /// heatmap degrades gracefully when this is empty).
    private func loadReturns() async {
        let syms = Array(Set(holdings.map(\.symbol))).sorted()
        guard !syms.isEmpty else { holdingReturns = [:]; return }
        let resp: [String: [String: JSONValue]]? = try? await api.get(
            "/holdings/returns", query: APIClient.arrayQuery("symbols", syms))
        guard let resp else { return }
        var out: [String: [String: Double]] = [:]
        for (sym, periods) in resp {
            out[sym] = periods.compactMapValues { $0.doubleValue }
        }
        holdingReturns = out
    }

    /// Persist edited targets for one bucket, merging with the rest.
    func saveTargets(bucket: String, values: [String: Double]) async {
        var merged = targets
        merged[bucket] = values
        targets = merged
        struct Update: Encodable { let target_allocation: [String: [String: Double]] }
        let _: StatusResponse? = try? await api.send(
            method: "POST", path: "/settings/update", body: Update(target_allocation: merged))
    }
}

/// Shared bucketing helpers used across the Portfolio sections.
enum PortfolioBucket {
    static func isUnknown(_ v: String?) -> Bool {
        guard let v else { return true }
        let s = v.trimmingCharacters(in: .whitespaces).uppercased()
        return s.isEmpty || s == "-" || s == "NONE" || s == "NULL" || s == "UNKNOWN" || s.hasPrefix("N/A") || s.hasPrefix("UNKNOWN")
    }
    static func value(_ h: Holding, key: String) -> String {
        let raw: String?
        switch key {
        case "Country": raw = h.string("geography") ?? h.string("Country")
        case "Symbol": raw = h.symbol
        default: raw = h.string(key)
        }
        return isUnknown(raw) ? "Unknown" : raw!
    }
}

struct AllocationView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = AllocationViewModel()
    @State private var detail: SymbolID?

    private var cur: String { appState.displayCurrency }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Portfolio").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            if viewModel.holdings.isEmpty {
                // Distinguish "still loading" from "genuinely empty" — otherwise
                // the in-flight initial fetch reads as "No holdings".
                if viewModel.isLoading {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Loading holdings…").font(.callout).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ContentUnavailableView("No holdings", systemImage: "chart.pie")
                }
            } else {
                GeometryReader { geo in
                ScrollView {
                    // space-y-6 between sections, matching Allocation.tsx.
                    VStack(spacing: 24) {
                        if vis("holdingsTable") { HoldingsTableView(holdings: viewModel.holdings, currency: cur) }
                        if vis("concentrationKpis") { ConcentrationKpiStrip(holdings: viewModel.holdings, currency: cur) }

                        // Category drift — grid-cols-1 md:grid-cols-2 xl:grid-cols-3 (up to 3-up).
                        if vis("categoryDrift") {
                            LazyVGrid(columns: flexColumns(count: geo.size.width >= 1100 ? 3 : (geo.size.width >= 720 ? 2 : 1)), spacing: 24) {
                                AllocationDriftCard(holdings: viewModel.holdings, currency: cur,
                                                    bucketKey: "quoteType", settingsBucket: "quoteType",
                                                    title: "Asset Type — drift vs target", vm: viewModel)
                                AllocationDriftCard(holdings: viewModel.holdings, currency: cur,
                                                    bucketKey: "Sector", settingsBucket: "sector",
                                                    title: "Sector — drift vs target", vm: viewModel)
                                AllocationDriftCard(holdings: viewModel.holdings, currency: cur,
                                                    bucketKey: "Country", settingsBucket: "country",
                                                    title: "Country — drift vs target", vm: viewModel)
                            }
                        }

                        if vis("stockDrift") {
                            AllocationDriftCard(holdings: viewModel.holdings, currency: cur,
                                                bucketKey: "Symbol", settingsBucket: "symbol",
                                                title: "Stocks — drift vs target", vm: viewModel, scrollable: true)
                        }

                        if vis("rebalanceHelper") { RebalanceHelperCard(holdings: viewModel.holdings, currency: cur, vm: viewModel) }
                        if vis("treemap") { PortfolioTreemapView(holdings: viewModel.holdings, currency: cur,
                                                                  onSelectSymbol: { detail = SymbolID(id: $0) }) }
                        if vis("holdingsHeatmap") {
                            HoldingsHeatmapView(holdings: viewModel.holdings, currency: cur,
                                                returns: viewModel.holdingReturns,
                                                onSelectSymbol: { detail = SymbolID(id: $0) })
                        }

                        // Donut charts — grid-cols-1 md:grid-cols-2 (exactly 2-up).
                        if vis("donutCharts") {
                            LazyVGrid(columns: flexColumns(count: geo.size.width >= 720 ? 2 : 1), spacing: 24) {
                                AllocationDonutChart(title: "By Asset Type", holdings: viewModel.holdings, currency: cur, bucketKey: "quoteType")
                                AllocationDonutChart(title: "By Sector", holdings: viewModel.holdings, currency: cur, bucketKey: "Sector")
                                AllocationDonutChart(title: "By Industry", holdings: viewModel.holdings, currency: cur, bucketKey: "Industry")
                                AllocationDonutChart(title: "By Country", holdings: viewModel.holdings, currency: cur, bucketKey: "Country")
                            }
                        }
                    }
                    .padding(16)
                }
                }
            }
        }
        .macMinSize(width: 820, height: 560)
        .task(id: signature) { reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: cur) }
    }

    private func vis(_ id: String) -> Bool { appState.isVisible(.allocation, id) }

    private func flexColumns(count: Int) -> [GridItem] {
        Array(repeating: GridItem(.flexible(), spacing: 24), count: max(1, count))
    }

    private var signature: String {
        "\(cur)|\(appState.showClosed)|\(appState.selectedAccounts.sorted().joined(separator: ","))"
    }
    private func reload() {
        viewModel.reload(currency: cur, accounts: appState.accountsQuery, showClosed: appState.showClosed)
    }
}
