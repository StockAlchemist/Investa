import SwiftUI

@MainActor
final class CapitalGainsViewModel: ObservableObject {
    @Published var gains: [CapitalGain] = []
    @Published var holdings: [Holding] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: APIClient
    private var task: Task<Void, Never>?

    init(api: APIClient = .shared) { self.api = api }

    func reload(currency: String, accounts: [String]?) {
        task?.cancel()
        task = Task { [weak self] in await self?.load(currency: currency, accounts: accounts) }
    }

    private func load(currency: String, accounts: [String]?) async {
        isLoading = true; errorMessage = nil
        defer { isLoading = false }
        let q = [URLQueryItem(name: "currency", value: currency)] + APIClient.arrayQuery("accounts", accounts)
        async let gainsR: [CapitalGain] = api.get("/capital_gains", query: q)
        async let holdingsR: [Holding] = api.get("/holdings", query: q)
        do {
            gains = try await gainsR
        } catch is CancellationError { return }
        catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
        holdings = (try? await holdingsR) ?? []
    }
}

/// Capital Gains tab — mirrors the web capital_gains tab (UnrealizedTaxView + CapitalGains.tsx).
struct CapitalGainsView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = CapitalGainsViewModel()
    @State private var selectedYear: String?

    private var cur: String { appState.displayCurrency }

    /// Realized gains filtered to the selected year (KPIs + table reflect this).
    private var filteredGains: [CapitalGain] {
        guard let y = selectedYear else { return viewModel.gains }
        return viewModel.gains.filter { $0.date.hasPrefix(y) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Capital Gains").font(.title2.bold())
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
                    if vis("unrealizedTax") { UnrealizedTaxSection(holdings: viewModel.holdings, currency: cur) }
                    if vis("capitalGainsKpis") { CapitalGainsKpiStrip(gains: filteredGains, currency: cur) }
                    if vis("annualCapitalGains") { AnnualRealizedGainsCard(gains: viewModel.gains, currency: cur, selectedYear: $selectedYear) }
                    if vis("capitalGainsTransactions") { RealizedGainsTable(gains: filteredGains, currency: cur) }
                }
                .padding(20)
            }
        }
        .macMinSize(width: 820, height: 560)
        .task(id: signature) { reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
    }

    private func vis(_ id: String) -> Bool { appState.isVisible(.capitalGains, id) }

    private var signature: String {
        "\(cur)|\(appState.selectedAccounts.sorted().joined(separator: ","))"
    }
    private func reload() {
        viewModel.reload(currency: cur, accounts: appState.accountsQuery)
    }
}
