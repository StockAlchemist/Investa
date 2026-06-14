import SwiftUI

@MainActor
final class CapitalGainsViewModel: ObservableObject {
    @Published var gains: [CapitalGain] = []
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
        do {
            let items: [CapitalGain] = try await api.get(
                "/capital_gains",
                query: [URLQueryItem(name: "currency", value: currency)]
                    + APIClient.arrayQuery("accounts", accounts)
            )
            if Task.isCancelled { return }
            gains = items
        } catch is CancellationError {
        } catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }
}

/// Sortable presentation row resolved for the active currency.
private struct GainRow: Identifiable {
    let id: String
    let date: String
    let symbol: String
    let account: String
    let quantity: Double
    let proceeds: Double
    let costBasis: Double
    let realizedGain: Double

    init(_ g: CapitalGain) {
        id = g.id; date = g.date; symbol = g.symbol; account = g.account
        quantity = g.quantity; proceeds = g.proceedsDisplay
        costBasis = g.costBasisDisplay; realizedGain = g.realizedGainDisplay
    }
}

struct CapitalGainsView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = CapitalGainsViewModel()
    @State private var sortOrder = [KeyPathComparator(\GainRow.date, order: .reverse)]

    private var cur: String { appState.displayCurrency }
    private var rows: [GainRow] { viewModel.gains.map(GainRow.init).sorted(using: sortOrder) }
    private var totalGain: Double { viewModel.gains.reduce(0) { $0 + $1.realizedGainDisplay } }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Capital Gains").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
                Text("Realized: \(Fmt.currency(totalGain, code: cur))")
                    .font(.headline).foregroundStyle(Fmt.tint(for: totalGain))
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            if viewModel.gains.isEmpty && !viewModel.isLoading {
                ContentUnavailableView("No realized gains", systemImage: "arrow.up.right")
            } else {
                Table(rows, sortOrder: $sortOrder) {
                    TableColumn("Date", value: \.date) { Text($0.date) }
                    TableColumn("Symbol", value: \.symbol) { Text($0.symbol).fontWeight(.medium) }
                    TableColumn("Account", value: \.account) { Text($0.account) }
                    TableColumn("Qty", value: \.quantity) {
                        Text(Fmt.number($0.quantity)).monospacedDigit()
                    }
                    TableColumn("Proceeds", value: \.proceeds) {
                        Text(Fmt.currency($0.proceeds, code: cur)).monospacedDigit()
                    }
                    TableColumn("Cost Basis", value: \.costBasis) {
                        Text(Fmt.currency($0.costBasis, code: cur)).monospacedDigit()
                    }
                    TableColumn("Realized Gain", value: \.realizedGain) { row in
                        Text(Fmt.currency(row.realizedGain, code: cur)).monospacedDigit()
                            .foregroundStyle(Fmt.tint(for: row.realizedGain))
                    }
                }
            }
        }
        .frame(minWidth: 700, minHeight: 500)
        .task(id: signature) { reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
    }

    private var signature: String {
        "\(appState.displayCurrency)|\(appState.selectedAccounts.sorted().joined(separator: ","))"
    }
    private func reload() {
        viewModel.reload(currency: appState.displayCurrency, accounts: appState.accountsQuery)
    }
}
