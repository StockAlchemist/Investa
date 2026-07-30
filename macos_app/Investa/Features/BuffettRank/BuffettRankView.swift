import SwiftUI

@MainActor
final class BuffettRankViewModel: ObservableObject {
    @Published var rows: [BuffettRankRow] = []
    @Published var exclusions: [BuffettExclusion] = []
    @Published var run: BuffettRankRun?
    @Published var isLoading = false
    @Published var errorMessage: String?

    /// Nil means "all models". Filtering happens server-side because the
    /// ranking is paged out of a snapshot, not held in memory.
    @Published var model: BuffettModel?
    @Published var page = 0
    @Published var showingExclusions = false

    /// Applied server-side across the whole run. Filtering the loaded page
    /// instead would only ever search the ~100 rows already on screen, so a
    /// company ranked below that would appear not to exist.
    @Published var search = ""
    @Published var totalMatches = 0

    static let pageSize = 100

    private let api: APIClient
    private var searchTask: Task<Void, Never>?

    init(api: APIClient = .shared) { self.api = api }

    var isLastPage: Bool { (page + 1) * Self.pageSize >= totalMatches }

    func loadRun() async {
        // A 404 here is the normal state before the first batch run has
        // completed, so it must not surface as an error.
        run = try? await api.get("/buffett-rank/latest")
    }

    func load() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        var query = [
            URLQueryItem(name: "limit", value: String(Self.pageSize)),
            URLQueryItem(name: "offset", value: String(page * Self.pageSize)),
        ]

        let term = search.trimmingCharacters(in: .whitespaces)
        if !term.isEmpty { query.append(URLQueryItem(name: "search", value: term)) }

        do {
            if showingExclusions {
                // Searchable too: when a company is missing from the ranking,
                // finding out why is the immediate next question.
                let page: BuffettExclusionPage = try await api.get("/buffett-rank/exclusions", query: query)
                exclusions = page.rows
                totalMatches = page.total
            } else {
                if let model { query.append(URLQueryItem(name: "model", value: model.rawValue)) }
                let page: BuffettRankPage = try await api.get("/buffett-rank", query: query)
                rows = page.rows
                totalMatches = page.total
            }
        } catch let error as APIError {
            errorMessage = error.errorDescription
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Debounced so typing does not fire a request per keystroke. Any change
    /// resets to the first page: staying on page 4 of a result set that now has
    /// one match would show an empty list.
    func searchChanged() {
        searchTask?.cancel()
        searchTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 250_000_000)
            guard !Task.isCancelled, let self else { return }
            self.page = 0
            await self.load()
        }
    }

    func select(model newModel: BuffettModel?) async {
        model = newModel
        page = 0
        await load()
    }

    func setShowingExclusions(_ showing: Bool) async {
        showingExclusions = showing
        page = 0
        await load()
    }

    func changePage(by delta: Int) async {
        page = max(0, page + delta)
        await load()
    }
}

struct BuffettRankView: View {
    @StateObject private var viewModel = BuffettRankViewModel()
    @State private var detail: SymbolID?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                controls

                if viewModel.isLoading && viewModel.rows.isEmpty && viewModel.exclusions.isEmpty {
                    ProgressView().frame(maxWidth: .infinity).padding(.vertical, 40)
                } else if let message = viewModel.errorMessage {
                    Label(message, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 30)
                } else if viewModel.showingExclusions {
                    exclusionList
                } else {
                    rankList
                }

                pager
            }
            .padding(20)
        }
        .navigationTitle("Rankings")
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: "USD") }
        .task {
            await viewModel.loadRun()
            await viewModel.load()
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Buffett & Value Ranking")
                .font(.title2.bold())

            Text("""
                 Every US-listed common stock, scored 60% on business quality and 40% on value. \
                 Quality gates run first — a company that fails one is excluded rather than ranked \
                 low, because cheapness never rescues a broken business.
                 """)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if let run = viewModel.run {
                HStack(spacing: 20) {
                    statistic("Ranked", run.rankedCount)
                    statistic("Excluded", run.excludedCount)
                    statistic("Universe", run.universeSize)
                }
                .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .card()
    }

    private func statistic(_ label: String, _ value: Int?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value.map(String.init) ?? "—").font(.headline)
        }
    }

    // MARK: - Controls

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Button("Ranked") { Task { await viewModel.setShowingExclusions(false) } }
                    .buttonStyle(.borderedProminent)
                    .opacity(viewModel.showingExclusions ? 0.5 : 1)

                Button("Excluded") { Task { await viewModel.setShowingExclusions(true) } }
                    .buttonStyle(.bordered)
                    .opacity(viewModel.showingExclusions ? 1 : 0.5)
            }

            if !viewModel.showingExclusions {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        modelChip(nil, label: "All models")
                        ForEach(BuffettModel.allCases) { model in
                            modelChip(model, label: model.label)
                        }
                    }
                }
            }

            // Search serves both tabs: when a company is missing from the
            // ranking, looking it up in the excluded list is the very next
            // thing you want to do.
            HStack(spacing: 8) {
                TextField(viewModel.showingExclusions ? "Search excluded stocks…" : "Search all ranked stocks…",
                          text: $viewModel.search)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 320)
                    .onChange(of: viewModel.search) { _, _ in viewModel.searchChanged() }

                if !viewModel.search.trimmingCharacters(in: .whitespaces).isEmpty {
                    Text("\(viewModel.totalMatches) match\(viewModel.totalMatches == 1 ? "" : "es")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func modelChip(_ model: BuffettModel?, label: String) -> some View {
        let selected = viewModel.model == model
        return Button(label) { Task { await viewModel.select(model: model) } }
            .buttonStyle(.bordered)
            .tint(selected ? .accentColor : .secondary)
    }

    // MARK: - Ranked list

    @ViewBuilder
    private var rankList: some View {
        if viewModel.rows.isEmpty {
            let term = viewModel.search.trimmingCharacters(in: .whitespaces)
            VStack(spacing: 6) {
                Text(term.isEmpty ? "No companies on this page." : "No ranked company matches “\(term)”.")
                    .font(.callout)
                if !term.isEmpty {
                    Text("It may have been excluded by a quality gate — check the Excluded tab.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 40)
        } else {
            LazyVStack(spacing: 8) {
                ForEach(viewModel.rows) { row in
                    Button { detail = SymbolID(id: row.symbol) } label: {
                        rankRow(row)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    /// Rank column and logo sizes, shared by the row and by the indent that
    /// keeps the pillar grid aligned under the company name on Mac.
    private var rankColumnWidth: CGFloat { 30 }
    private var logoSize: CGFloat { isPhoneLayout ? 30 : 34 }

    private func rankRow(_ row: BuffettRankRow) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(row.rank.map(String.init) ?? "—")
                    .font(.callout.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .frame(width: rankColumnWidth, alignment: .trailing)

                // The logo is what makes a 100-row list scannable: a company is
                // recognised by its mark long before its ticker is read.
                StockIcon(symbol: row.symbol, size: logoSize)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(row.symbol).font(.headline)
                        if row.model != .generic {
                            Text(row.model.shortLabel)
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 5).padding(.vertical, 1)
                                .background(Color.secondary.opacity(0.15), in: Capsule())
                        }
                    }
                    if let name = row.name, !name.isEmpty {
                        // Two lines rather than one: at phone width a single
                        // line ends in an ellipsis for most of the market.
                        Text(name)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                Spacer(minLength: 8)

                VStack(alignment: .trailing, spacing: 2) {
                    HStack(spacing: 3) {
                        Text(score(row.compositeScore))
                            .font(.title3.monospacedDigit().weight(.semibold))
                            .foregroundStyle(tint(row.compositeScore))
                        // Flagged next to the score it modified, so a company
                        // demoted for thin data is distinguishable from one
                        // that is honestly mediocre.
                        if row.isConfidenceReduced {
                            Image(systemName: "arrowtriangle.down.fill")
                                .font(.system(size: 9))
                                .foregroundStyle(.orange)
                        }
                    }
                    Text("Q \(score(row.qualityScore)) · V \(score(row.valueScore))")
                        .font(.subheadline.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .layoutPriority(1)
            }

            pillarGrid(row)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    /// The pillar breakdown, always the same six columns.
    ///
    /// The last column is the earnings yield, the heaviest input to the value
    /// score. It renders as a dash where the company reported no earnings for
    /// the period. Dropping the column instead re-spaced every other one, so
    /// scanning a single metric down the list meant re-finding its position on
    /// each card.
    private func pillarGrid(_ row: BuffettRankRow) -> some View {
        HStack(spacing: 10) {
            ForEach(row.pillars, id: \.label) { pillar in
                pillarCell(score(pillar.value), pillarLabel(pillar.label), tint(pillar.value))
            }
            pillarCell(
                row.earningsYield.map { Fmt.percent($0 / 100) } ?? "—",
                "E/P",
                row.earningsYield.map { $0 > 0 ? Color.green : .red } ?? .secondary
            )
        }
        .padding(.leading, isPhoneLayout ? 0 : rankColumnWidth + logoSize + 20)
    }

    private func pillarCell(_ value: String, _ label: String, _ tone: Color) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.callout.monospacedDigit().weight(.semibold))
                .foregroundStyle(tone)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Exclusions

    private var exclusionList: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("""
                 Most of the listed market is excluded, which is expected when ranking every listing \
                 rather than an index. A gate only fires on something the filings actually show — \
                 missing data never fails a company, it lowers its confidence instead.
                 """)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .card()

            if viewModel.exclusions.isEmpty {
                Text(viewModel.search.trimmingCharacters(in: .whitespaces).isEmpty
                     ? "No exclusions on this page."
                     : "No excluded company matches “\(viewModel.search.trimmingCharacters(in: .whitespaces))”.")
                    .font(.callout)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 30)
            }

            LazyVStack(spacing: 6) {
                ForEach(viewModel.exclusions) { item in
                    HStack(alignment: .top, spacing: 10) {
                        // Same identity treatment as the ranked list, so the two
                        // tabs read as one list seen two ways.
                        StockIcon(symbol: item.symbol, size: isPhoneLayout ? 26 : 30)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.symbol).font(.headline)
                            if let name = item.name, !name.isEmpty {
                                Text(name)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        .frame(width: isPhoneLayout ? 120 : 150, alignment: .leading)

                        VStack(alignment: .leading, spacing: 3) {
                            ForEach(item.reasonList, id: \.self) { reason in
                                Text(reason)
                                    .font(.caption)
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(Color.secondary.opacity(0.12), in: Capsule())
                            }
                        }

                        Spacer()

                        Text(item.periodCount.map { "\($0)y" } ?? "—")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .card()
                }
            }
        }
    }

    // MARK: - Paging

    private var pager: some View {
        let size = BuffettRankViewModel.pageSize
        let first = viewModel.page * size + 1
        return HStack {
            Button("Previous") { Task { await viewModel.changePage(by: -1) } }
                .disabled(viewModel.page == 0)
            Spacer()
            Text(viewModel.showingExclusions
                 ? "Rows \(first)–\(first + size - 1)"
                 : "\(viewModel.totalMatches == 0 ? 0 : first)–\(min(first + size - 1, viewModel.totalMatches)) of \(viewModel.totalMatches)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button("Next") { Task { await viewModel.changePage(by: 1) } }
                // The exclusions list has no match count, so its Next stays
                // enabled and simply lands on an empty page at the end.
                .disabled(!viewModel.showingExclusions && viewModel.isLastPage)
        }
        .buttonStyle(.bordered)
    }

    // MARK: - Formatting

    /// Full labels on Mac, abbreviated where a phone column cannot hold them.
    private func pillarLabel(_ label: String) -> String {
        guard isPhoneLayout else { return label }
        return label == "Predictable" ? "Predict." : label
    }

    private func score(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.0f", value)
    }

    /// One ramp for every percentile score, since they all share a 0–100 scale.
    private func tint(_ value: Double?) -> Color {
        guard let value else { return .secondary }
        if value >= 70 { return .green }
        if value >= 50 { return .cyan }
        if value >= 30 { return .orange }
        return .red
    }
}
