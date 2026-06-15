import SwiftUI

@MainActor
final class ScreenerViewModel: ObservableObject {
    @Published var results: [ScreenerResult] = []
    @Published var watchlists: [WatchlistMeta] = []
    @Published var isLoading = false
    @Published var isRefreshing = false
    @Published var reviewingSymbol: String?
    @Published var reviews: [String: ScreenReview] = [:]
    @Published var errorMessage: String?

    private let api: APIClient
    init(api: APIClient = .shared) { self.api = api }

    func loadWatchlists() async {
        watchlists = (try? await api.get("/watchlists")) ?? []
    }

    private func dedupe(_ items: [ScreenerResult]) -> [ScreenerResult] {
        var seen = Set<String>(); var out: [ScreenerResult] = []
        for r in items where !seen.contains(r.symbol) { seen.insert(r.symbol); out.append(r) }
        return out
    }

    func run(universe: ScreenerUniverse, watchlistId: String?, manualSymbols: [String], prompt: String) async {
        isLoading = true; errorMessage = nil; results = []
        defer { isLoading = false; isRefreshing = false }
        do {
            if universe == .narrative {
                struct Body: Encodable { let prompt: String }
                let data: [ScreenerResult] = try await api.send(method: "POST", path: "/screener/narrative", body: Body(prompt: prompt))
                results = dedupe(data)
                return
            }
            // Phase 1 — fast (cache only).
            let fast: [ScreenerResult] = try await api.send(
                method: "POST", path: "/screener/run",
                body: ScreenerRequest(universe_type: universe.rawValue,
                                      universe_id: universe == .watchlist ? watchlistId : nil,
                                      manual_symbols: manualSymbols, fast_mode: true))
            if !fast.isEmpty { results = dedupe(fast); isLoading = false; isRefreshing = true }
            // Phase 2 — fresh (live data).
            let fresh: [ScreenerResult] = try await api.send(
                method: "POST", path: "/screener/run",
                body: ScreenerRequest(universe_type: universe.rawValue,
                                      universe_id: universe == .watchlist ? watchlistId : nil,
                                      manual_symbols: manualSymbols, fast_mode: false))
            results = dedupe(fresh)
        } catch let error as APIError {
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }

    func review(_ symbol: String, force: Bool = false) async {
        if !force, reviews[symbol] != nil { return }
        reviewingSymbol = symbol
        defer { reviewingSymbol = nil }
        do {
            let data: ScreenReview = try await api.send(
                method: "POST", path: "/screener/review/\(symbol)",
                query: [URLQueryItem(name: "force", value: force ? "true" : "false")])
            reviews[symbol] = data
        } catch { /* surfaced inline */ }
    }
}

struct ScreenerView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = ScreenerViewModel()

    @State private var universe: ScreenerUniverse = .watchlist
    @State private var watchlistId = ""
    @State private var manualText = ""
    @State private var prompt = ""

    private var cur: String { appState.displayCurrency }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    inputCard
                    if viewModel.isRefreshing {
                        Label("Updating live prices…", systemImage: "binoculars")
                            .font(.caption).foregroundStyle(.secondary)
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(.background.secondary, in: Capsule())
                    }
                    if let error = viewModel.errorMessage {
                        Text(error).foregroundStyle(.red).font(.callout)
                    }
                    ScreenerResultsView(viewModel: viewModel, currency: cur)
                }
                .padding(20)
            }
        }
        .frame(minWidth: 820, minHeight: 560)
        .task {
            await viewModel.loadWatchlists()
            if watchlistId.isEmpty { watchlistId = viewModel.watchlists.first.map { String($0.id) } ?? "" }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Market Explorer").font(.title2.bold())
                .foregroundStyle(LinearGradient(colors: [.cyan, .blue], startPoint: .leading, endPoint: .trailing))
            Text("Identify opportunities using intrinsic-value models and AI fundamental audits.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 20).padding(.vertical, 12)
    }

    // MARK: - Input card

    private var inputCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Initial Parameters", systemImage: "line.3.horizontal.decrease.circle").font(.headline)
            HStack(alignment: .bottom, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Universe").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    Picker("", selection: $universe) {
                        ForEach(ScreenerUniverse.allCases) { Text($0.label).tag($0) }
                    }.labelsHidden().frame(width: 240)
                }
                if universe == .watchlist {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Target Portfolio").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        Picker("", selection: $watchlistId) {
                            ForEach(viewModel.watchlists) { Text($0.name).tag(String($0.id)) }
                        }.labelsHidden().frame(width: 180)
                    }
                }
                if universe == .narrative {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("AI Search Prompt", systemImage: "sparkles").font(.caption.weight(.semibold)).foregroundStyle(.cyan)
                        TextField("e.g. high-growth tech with margin of safety > 20%", text: $prompt)
                            .textFieldStyle(.roundedBorder).frame(minWidth: 280)
                    }
                }
                if universe == .manual {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Manual Symbols").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        TextField("e.g. AAPL, MSFT, NVDA", text: $manualText)
                            .textFieldStyle(.roundedBorder).frame(minWidth: 240)
                    }
                }
                Spacer()
                Button(action: run) {
                    if viewModel.isLoading {
                        HStack { ProgressView().controlSize(.small); Text("AI is analyzing…") }
                    } else {
                        Label(universe == .narrative ? "Search with AI" : "Execute Screen",
                              systemImage: universe == .narrative ? "sparkles" : "arrow.clockwise")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isLoading || (universe == .watchlist && watchlistId.isEmpty))
            }
            Label(universe == .narrative
                  ? "Narrative Search uses AI to translate natural language into a query. Works best on cached stocks."
                  : "Screening large universes may take 1–5 min on the first run to build the cache. Subsequent runs are instant.",
                  systemImage: universe == .narrative ? "sparkles" : "info.circle")
                .font(.caption2).foregroundStyle(.secondary)
                .padding(10).frame(maxWidth: .infinity, alignment: .leading)
                .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func run() {
        let symbols = manualText.split(whereSeparator: { $0 == "," || $0 == "\n" })
            .map { $0.trimmingCharacters(in: .whitespaces).uppercased() }.filter { !$0.isEmpty }
        Task { await viewModel.run(universe: universe, watchlistId: watchlistId, manualSymbols: symbols, prompt: prompt) }
    }
}
