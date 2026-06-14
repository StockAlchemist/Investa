import SwiftUI

@MainActor
final class WatchlistViewModel: ObservableObject {
    @Published var items: [WatchlistItem] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    /// MVP uses the default watchlist (id 1), matching the web client default.
    private let watchlistId = 1
    private let api: APIClient
    private var task: Task<Void, Never>?

    init(api: APIClient = .shared) { self.api = api }

    func reload(currency: String) {
        task?.cancel()
        task = Task { [weak self] in await self?.load(currency: currency) }
    }

    private func load(currency: String) async {
        isLoading = true; errorMessage = nil
        defer { isLoading = false }
        do {
            let result: [WatchlistItem] = try await api.get(
                "/watchlist",
                query: [
                    URLQueryItem(name: "currency", value: currency),
                    URLQueryItem(name: "id", value: String(watchlistId)),
                ]
            )
            if Task.isCancelled { return }
            items = result
        } catch is CancellationError {
        } catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }

    func add(symbol: String, currency: String) async {
        let clean = symbol.trimmingCharacters(in: .whitespaces).uppercased()
        guard !clean.isEmpty else { return }
        struct Body: Encodable { let symbol: String; let note: String; let watchlist_id: Int }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/watchlist",
                body: Body(symbol: clean, note: "", watchlist_id: watchlistId)
            )
            await load(currency: currency)
        } catch let error as APIError {
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }

    func remove(symbol: String, currency: String) async {
        do {
            let _: StatusResponse = try await api.send(
                method: "DELETE", path: "/watchlist/\(symbol)",
                query: [URLQueryItem(name: "id", value: String(watchlistId))]
            )
            items.removeAll { $0.symbol == symbol }
        } catch let error as APIError {
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }
}

struct WatchlistView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = WatchlistViewModel()
    @State private var newSymbol = ""
    @State private var sortOrder = [KeyPathComparator(\WatchlistItem.symbol)]

    private var cur: String { appState.displayCurrency }
    private var rows: [WatchlistItem] { viewModel.items.sorted(using: sortOrder) }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Watchlist").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
                TextField("Add symbol…", text: $newSymbol)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 140)
                    .onSubmit(add)
                Button("Add", action: add)
                    .buttonStyle(.borderedProminent)
                    .disabled(newSymbol.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            if viewModel.items.isEmpty && !viewModel.isLoading {
                ContentUnavailableView("Watchlist is empty", systemImage: "star",
                                       description: Text("Add a symbol above."))
            } else {
                Table(rows, sortOrder: $sortOrder) {
                    TableColumn("Symbol", value: \.symbol) { Text($0.symbol).fontWeight(.medium) }
                        .width(min: 70, ideal: 90)
                    TableColumn("Name") { Text($0.name ?? "—").lineLimit(1) }
                        .width(min: 120, ideal: 200)
                    TableColumn("Price") { item in
                        Text(item.price.map { Fmt.currency($0, code: item.currency ?? cur) } ?? "—")
                            .monospacedDigit()
                    }
                    .width(min: 80, ideal: 110)
                    TableColumn("Day %") { item in
                        Text(Fmt.percent(item.dayChangePct))
                            .monospacedDigit()
                            .foregroundStyle(Fmt.tint(for: item.dayChangePct))
                    }
                    .width(min: 70, ideal: 90)
                    TableColumn("") { item in
                        Button { remove(item.symbol) } label: { Image(systemName: "trash") }
                            .buttonStyle(.borderless).foregroundStyle(.red)
                    }
                    .width(40)
                }
            }
        }
        .frame(minWidth: 700, minHeight: 500)
        .task(id: cur) { viewModel.reload(currency: cur) }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in
            viewModel.reload(currency: cur)
        }
    }

    private func add() {
        let symbol = newSymbol
        newSymbol = ""
        Task { await viewModel.add(symbol: symbol, currency: cur) }
    }

    private func remove(_ symbol: String) {
        Task { await viewModel.remove(symbol: symbol, currency: cur) }
    }
}
