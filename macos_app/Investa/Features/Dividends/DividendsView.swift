import SwiftUI

@MainActor
final class DividendsViewModel: ObservableObject {
    @Published var dividends: [Dividend] = []
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
            let items: [Dividend] = try await api.get(
                "/dividends",
                query: [URLQueryItem(name: "currency", value: currency)]
                    + APIClient.arrayQuery("accounts", accounts)
            )
            if Task.isCancelled { return }
            dividends = items
        } catch is CancellationError {
        } catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }
}

struct DividendsView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = DividendsViewModel()
    @State private var sortOrder = [KeyPathComparator(\Dividend.date, order: .reverse)]

    private var rows: [Dividend] { viewModel.dividends.sorted(using: sortOrder) }
    private var cur: String { appState.displayCurrency }

    private var total: Double { viewModel.dividends.reduce(0) { $0 + $1.amountDisplay } }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Dividends").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
                Text("Total: \(Fmt.currency(total, code: cur))")
                    .font(.headline).foregroundStyle(.green)
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            if viewModel.dividends.isEmpty && !viewModel.isLoading {
                ContentUnavailableView("No dividends", systemImage: "dollarsign.circle")
            } else {
                Table(rows, sortOrder: $sortOrder) {
                    TableColumn("Date", value: \.date) { Text($0.date) }
                    TableColumn("Symbol", value: \.symbol) { Text($0.symbol).fontWeight(.medium) }
                    TableColumn("Account", value: \.account) { Text($0.account) }
                    TableColumn("Amount (Local)", value: \.amountLocal) { d in
                        Text(Fmt.currency(d.amountLocal, code: d.localCurrency)).monospacedDigit()
                    }
                    TableColumn("Amount (\(cur))", value: \.amountDisplay) { d in
                        Text(Fmt.currency(d.amountDisplay, code: cur)).monospacedDigit()
                            .foregroundStyle(.green)
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
