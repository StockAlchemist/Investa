import SwiftUI

@MainActor
final class BuffettRankViewModel: ObservableObject {
    @Published private(set) var rows: [BuffettRankRow] = []
    @Published private(set) var exclusions: [BuffettExclusion] = []
    @Published private(set) var run: BuffettRankRun?
    /// Whether the run lookup has come back at all. Before it does, "no run
    /// yet" and "still loading" look identical and must not be confused: one is
    /// a permanent state with an instruction attached, the other is a spinner.
    @Published private(set) var runResolved = false
    @Published private(set) var isLoading = false
    @Published private(set) var isLoadingMore = false
    @Published private(set) var errorMessage: String?

    /// Nil means "all models". Filtering happens server-side because the
    /// ranking is paged out of a snapshot, not held in memory.
    @Published private(set) var model: BuffettModel?
    @Published private(set) var showingExclusions = false

    /// Applied server-side across the whole run. Filtering the loaded rows
    /// instead would only ever search what is already on screen, so a company
    /// ranked below that would appear not to exist.
    @Published var search = ""
    @Published private(set) var totalMatches = 0

    static let pageSize = 100

    private let api: APIClient
    private var searchTask: Task<Void, Never>?
    private var loadedPages = 0
    private var hasStarted = false
    /// Bumped whenever the query changes. A page that lands after its filter
    /// has moved on belongs to a list nobody is looking at any more, and
    /// appending it would mix two result sets into one.
    private var generation = 0

    init(api: APIClient = .shared) { self.api = api }

    var hasRun: Bool { run != nil }
    var loadedCount: Int { showingExclusions ? exclusions.count : rows.count }
    var hasMore: Bool { loadedCount < totalMatches }
    var trimmedSearch: String { search.trimmingCharacters(in: .whitespaces) }

    /// What the list is showing out of what matched — the count the pager used
    /// to carry, now that the list runs on instead of paging.
    var rangeCaption: String {
        let noun = showingExclusions ? "excluded" : "ranked"
        let total = Fmt.number(Double(totalMatches), fractionDigits: 0)
        if hasMore {
            return "Showing \(Fmt.number(Double(loadedCount), fractionDigits: 0)) of \(total) \(noun)"
        }
        return totalMatches == 1 ? "1 \(noun) company" : "All \(total) \(noun) companies"
    }

    func start() async {
        guard !hasStarted else { return }
        hasStarted = true
        // A 404 here is the normal state before the first batch run has
        // completed, so it must not surface as an error.
        run = try? await api.get("/buffett-rank/latest")
        runResolved = true
        await reload()
    }

    /// Replaces the list from the first page. Every filter change goes through
    /// here: keeping the loaded rows across a filter change would leave the old
    /// result set on screen under the new heading.
    func reload() async {
        generation += 1
        let token = generation
        isLoading = true
        errorMessage = nil
        defer { if token == generation { isLoading = false } }
        await fetch(page: 0, replacing: true, token: token)
    }

    /// Appends the next page. Called both by the button and by the footer
    /// appearing, so reaching the end of the list is itself the request.
    func loadMore() async {
        guard !isLoading, !isLoadingMore, hasMore else { return }
        let token = generation
        isLoadingMore = true
        defer { if token == generation { isLoadingMore = false } }
        await fetch(page: loadedPages, replacing: false, token: token)
    }

    private func fetch(page: Int, replacing: Bool, token: Int) async {
        var query = [
            URLQueryItem(name: "limit", value: String(Self.pageSize)),
            URLQueryItem(name: "offset", value: String(page * Self.pageSize)),
        ]
        if !trimmedSearch.isEmpty { query.append(URLQueryItem(name: "search", value: trimmedSearch)) }

        do {
            if showingExclusions {
                // Searchable too: when a company is missing from the ranking,
                // finding out why is the immediate next question.
                let result: BuffettExclusionPage = try await api.get("/buffett-rank/exclusions", query: query)
                guard token == generation else { return }
                exclusions = replacing ? result.rows : exclusions + result.rows
                totalMatches = result.total
            } else {
                if let model { query.append(URLQueryItem(name: "model", value: model.rawValue)) }
                let result: BuffettRankPage = try await api.get("/buffett-rank", query: query)
                guard token == generation else { return }
                rows = replacing ? result.rows : rows + result.rows
                totalMatches = result.total
            }
            loadedPages = page + 1
        } catch let error as APIError {
            guard token == generation else { return }
            errorMessage = error.errorDescription
        } catch {
            guard token == generation else { return }
            errorMessage = error.localizedDescription
        }
    }

    /// Debounced so typing does not fire a request per keystroke.
    func searchChanged() {
        searchTask?.cancel()
        searchTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 250_000_000)
            guard !Task.isCancelled, let self else { return }
            await self.reload()
        }
    }

    func select(model newModel: BuffettModel?) async {
        guard model != newModel else { return }
        model = newModel
        rows = []
        await reload()
    }

    func setShowingExclusions(_ showing: Bool) async {
        guard showingExclusions != showing else { return }
        showingExclusions = showing
        totalMatches = 0
        await reload()
    }
}

/// The Buffett & value ranking of every US listing.
///
/// Two lists behind one search: the companies that were ranked, and the (much
/// larger) set that failed a quality gate. Each ranked row carries the evidence
/// for its own position — five quality percentiles and the two yields the value
/// half is made of — so the ranking can be argued with rather than trusted.
struct BuffettRankView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = BuffettRankViewModel()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                BuffettRankHero(run: viewModel.run)

                if viewModel.hasRun {
                    BuffettRankControls(viewModel: viewModel)
                    if viewModel.showingExclusions { exclusionNote }
                    content
                } else if viewModel.runResolved {
                    noRunCard
                } else {
                    ProgressView().frame(maxWidth: .infinity).padding(.vertical, 40)
                }
            }
            .padding(Theme.gutter)
        }
        .task { await viewModel.start() }
    }

    // MARK: - Body states

    @ViewBuilder
    private var content: some View {
        if let message = viewModel.errorMessage, viewModel.loadedCount == 0 {
            Label(message, systemImage: "exclamationmark.triangle")
                .appFont(.callout)
                .foregroundStyle(.red)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 30)
        } else if viewModel.isLoading && viewModel.loadedCount == 0 {
            ProgressView().frame(maxWidth: .infinity).padding(.vertical, 40)
        } else if viewModel.loadedCount == 0 {
            emptyState
        } else if viewModel.showingExclusions {
            BuffettExclusionList(viewModel: viewModel) { appState.openStock($0) }
        } else {
            BuffettRankList(viewModel: viewModel) { appState.openStock($0) }
        }
    }

    private var emptyState: some View {
        let term = viewModel.trimmedSearch
        return VStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .appFont(.title3)
                .foregroundStyle(.tertiary)
            Text(term.isEmpty
                 ? (viewModel.showingExclusions ? "No exclusions in this run." : "No companies in this run.")
                 : "Nothing matches “\(term)”.")
                .appFont(.callout)
                .multilineTextAlignment(.center)
            if !term.isEmpty && !viewModel.showingExclusions {
                Text("It may have failed a quality gate — check the Excluded list.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    /// Before the first batch run there is nothing to show and nothing the app
    /// can do about it — ranking every US listing is a job measured in minutes,
    /// so it is started from the command line, never from a tap here.
    private var noRunCard: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.bar.doc.horizontal")
                .appFont(.title)
                .foregroundStyle(.tertiary)
            Text("No ranking run yet").appFont(.headline)
            Text("Build the first snapshot with `python src/buffett_rank_worker.py`, then reopen this tab.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity)
        .padding(28)
        .card()
    }

    private var exclusionNote: some View {
        Text("""
             Most of the listed market is excluded, which is expected when ranking every listing \
             rather than an index. A gate only fires on something the filings actually show — \
             missing data never fails a company, it lowers its confidence instead.
             """)
            .appFont(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .card()
    }
}

// MARK: - Lists

/// Kept as its own view, not a `@ViewBuilder` on the screen: a single `body`
/// holding the whole page builds one enormous view type, which overflows the
/// stack on iPhone before it ever renders.
private struct BuffettRankList: View {
    @ObservedObject var viewModel: BuffettRankViewModel
    let onOpen: (String) -> Void

    var body: some View {
        LazyVStack(alignment: .leading, spacing: 8) {
            ForEach(viewModel.rows) { row in
                BuffettRankRowCard(row: row) { onOpen(row.symbol) }
            }
            BuffettListFooter(viewModel: viewModel)
        }
    }
}

private struct BuffettExclusionList: View {
    @ObservedObject var viewModel: BuffettRankViewModel
    let onOpen: (String) -> Void

    var body: some View {
        LazyVStack(alignment: .leading, spacing: 8) {
            ForEach(viewModel.exclusions) { item in
                BuffettExclusionCard(item: item) { onOpen(item.symbol) }
            }
            BuffettListFooter(viewModel: viewModel)
        }
    }
}

/// The end of the list: how much of the match set is on screen, and the next
/// page. Replaces the previous/next pager, which on a phone meant scrolling a
/// hundred rows to reach two buttons.
private struct BuffettListFooter: View {
    @ObservedObject var viewModel: BuffettRankViewModel

    var body: some View {
        VStack(spacing: 8) {
            Text(viewModel.rangeCaption)
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)

            if viewModel.hasMore {
                if viewModel.isLoadingMore {
                    ProgressView().controlSize(.small)
                } else {
                    Button("Load \(BuffettRankViewModel.pageSize) more") {
                        Task { await viewModel.loadMore() }
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        // Reaching the end of the list is the request for the next page; the
        // button stays for anyone who lands here while a load is in flight.
        .onAppear { Task { await viewModel.loadMore() } }
    }
}
