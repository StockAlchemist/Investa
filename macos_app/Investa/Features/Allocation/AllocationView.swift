import SwiftUI

@MainActor
final class AllocationViewModel: ObservableObject {
    @Published var holdings: [Holding] = []
    /// bucket → name → target %.
    @Published var targets: [String: [String: Double]] = [:]
    @Published var isLoading = false
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
            if viewModel.holdings.isEmpty && !viewModel.isLoading {
                ContentUnavailableView("No holdings", systemImage: "chart.pie")
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
                        if vis("treemap") { PortfolioTreemapView(holdings: viewModel.holdings, currency: cur) }

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
        .frame(minWidth: 820, minHeight: 560)
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
