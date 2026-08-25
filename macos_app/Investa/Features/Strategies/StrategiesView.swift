import SwiftUI

@MainActor
final class StrategiesViewModel: ObservableObject {
    @Published var catalogue: [StrategyDefinition] = []
    @Published var defaultID: String?
    @Published var selectedID: String?
    @Published var allocation: StrategyAllocation?
    @Published var isLoading = false
    @Published var isBuilding = false
    @Published var errorMessage: String?

    /// Amount the allocation is sized against. Defaults to the portfolio's own
    /// market value so the buy list is in the user's units rather than a round
    /// number they have to mentally rescale.
    @Published var capital: Double = 100_000

    private let api: APIClient

    init(api: APIClient = .shared) { self.api = api }

    var selected: StrategyDefinition? {
        guard let id = selectedID ?? defaultID else { return nil }
        return catalogue.first { $0.id == id }
    }

    /// Size the allocation against the portfolio's own market value.
    ///
    /// Read here rather than from `AppState`, which does not publish it: a
    /// failure just leaves the default in place, since a wrong default is only
    /// a starting number the user can overwrite.
    func adoptPortfolioValue(currency: String) async {
        let summary: SummaryResponse? = try? await api.get(
            "/summary/headline", query: [URLQueryItem(name: "currency", value: currency)]
        )
        if let value = summary?.metrics?.marketValue, value > 0 {
            capital = value.rounded()
        }
    }

    func loadCatalogue() async {
        guard catalogue.isEmpty else { return }
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            let result: StrategyCatalogue = try await api.get("/strategies")
            catalogue = result.strategies
            defaultID = result.default
        } catch {
            errorMessage = "Could not load strategies."
        }
    }

    func buildAllocation() async {
        guard let strategy = selected, capital > 0 else { return }
        isBuilding = true
        defer { isBuilding = false }
        do {
            allocation = try await api.get(
                "/strategies/\(strategy.id)/allocation",
                query: [URLQueryItem(name: "capital", value: String(format: "%.2f", capital))]
            )
        } catch {
            allocation = nil
            errorMessage = "Could not build the allocation."
        }
    }

    func select(_ strategy: StrategyDefinition) async {
        guard selectedID != strategy.id else { return }
        selectedID = strategy.id
        allocation = nil
        await buildAllocation()
    }
}

/// The strategy catalogue and today's sized allocation.
///
/// These are fixed *rules*, not suggestions the app generates: each was
/// backtested point-in-time (`scripts/buffett_strategy_search.py`,
/// `docs/quality_strategy_report.pdf`), and the returns shown assume the rule is
/// followed mechanically. Each strategy's risks render beside its record for
/// that reason — a CAGR without the drawdown next to it is the number that
/// gets people hurt. Nothing in this screen places an order.
struct StrategiesView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var model = StrategiesViewModel()
    @State private var capitalText = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                intro

                if model.isLoading {
                    ProgressView().frame(maxWidth: .infinity).padding(.vertical, 30)
                } else if let error = model.errorMessage, model.catalogue.isEmpty {
                    Text(error).foregroundStyle(.secondary).padding(.vertical, 20)
                } else {
                    catalogueGrid
                    if let strategy = model.selected {
                        detail(strategy)
                    }
                }
            }
            .padding(Theme.gutter)
        }
        .task {
            await model.loadCatalogue()
            await model.adoptPortfolioValue(currency: appState.displayCurrency)
            capitalText = Fmt.regroup(String(format: "%.0f", model.capital))
            await model.buildAllocation()
        }
    }


    private var intro: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Strategies").appFont(.title2.weight(.bold))
            Text("Fixed rules, not live suggestions. Each was backtested point-in-time; "
                 + "the returns below assume the rule is followed mechanically. "
                 + "Nothing here places an order.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("**Individual stocks only — no leverage, no funds.** Each strategy holds "
                 + "the ranked companies directly and stays fully invested; there is no "
                 + "defensive mode and no index or cash ETF. None of them out-returns the "
                 + "NASDAQ-100 over 2013–2025.")
                .appFont(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var catalogueGrid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 260), spacing: 12)], spacing: 12) {
            ForEach(model.catalogue) { strategy in
                Button {
                    Task { await model.select(strategy) }
                } label: {
                    StrategyCard(
                        strategy: strategy,
                        isSelected: strategy.id == (model.selectedID ?? model.defaultID)
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    @ViewBuilder
    private func detail(_ strategy: StrategyDefinition) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            header(strategy)

            rulesAndRisks(strategy)

            if model.isBuilding {
                ProgressView().frame(maxWidth: .infinity).padding(.vertical, 12)
            }

            ForEach(model.allocation?.warnings ?? [], id: \.self) { warning in
                Label(warning, systemImage: "info.circle")
                    .appFont(.caption)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.orange.opacity(0.12)))
            }

            ForEach(model.allocation?.sleeves ?? []) { sleeve in
                SleeveSection(sleeve: sleeve,
                              currency: appState.displayCurrency,
                              ageDays: model.allocation?.rankingAgeDays,
                              isStale: model.allocation?.rankingIsStale ?? false) {
                    appState.openStock($0)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    /// The allocation's title and the amount it is sized against.
    ///
    /// Side by side only where the title fits on one line: sharing a phone's
    /// width broke the name mid-phrase over three lines and squeezed the Apply
    /// button until its own label wrapped. `ViewThatFits` measures the
    /// side-by-side row at its one-line ideal width and falls back to stacking.
    private func header(_ strategy: StrategyDefinition) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .bottom, spacing: 12) {
                title(strategy)
                Spacer(minLength: 12)
                capitalField
            }
            VStack(alignment: .leading, spacing: 10) {
                title(strategy)
                capitalField
            }
        }
    }

    private func title(_ strategy: StrategyDefinition) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(strategy.name) — today's allocation")
                .appFont(.headline)
                .fixedSize(horizontal: false, vertical: true)
            Text("Weights are the rule; share counts are indicative and move with price.")
                .appFont(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var capitalField: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("AMOUNT TO ALLOCATE")
                .appFont(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                .lineLimit(1).minimumScaleFactor(0.8)
            HStack(spacing: 8) {
                // The symbol sits outside the field: it says which currency the
                // amount is in without becoming text the user has to type past
                // or delete.
                Text(Fmt.symbol(appState.displayCurrency))
                    .appFont(.callout.weight(.medium))
                    .foregroundStyle(.secondary)
                TextField("100,000", text: Binding(
                    get: { capitalText },
                    set: { capitalText = Fmt.regroup($0) }
                ))
                    .textFieldStyle(.roundedBorder)
                    // Fixed on Mac — wide enough for a grouped eight-figure sum —
                    // but stretched across the phone's stacked row so the amount
                    // is never clipped by a narrow box.
                    .frame(idealWidth: 150, maxWidth: isPhoneLayout ? .infinity : 150)
                    .monospacedDigit()
                    .onSubmit { applyCapital() }
                // The label is a word, not a paragraph: it keeps its intrinsic
                // width instead of being compressed into two lines.
                Button("Apply") { applyCapital() }
                    .lineLimit(1)
                    .fixedSize()
            }
        }
    }

    private func applyCapital() {
        guard let parsed = Fmt.ungroup(capitalText), parsed > 0 else { return }
        let value = parsed.rounded()
        model.capital = value
        // Settle the field on the applied figure, so what is displayed is what
        // the sleeves below were sized against.
        capitalText = Fmt.regroup(String(format: "%.0f", value))
        Task { await model.buildAllocation() }
    }

    private func rulesAndRisks(_ strategy: StrategyDefinition) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 20) {
                rules(strategy).frame(maxWidth: .infinity, alignment: .leading)
                risks(strategy).frame(maxWidth: .infinity, alignment: .leading)
            }
            VStack(alignment: .leading, spacing: 14) {
                rules(strategy)
                risks(strategy)
            }
        }
    }

    private func rules(_ strategy: StrategyDefinition) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("THE RULE").appFont(.caption2.weight(.semibold)).foregroundStyle(.secondary)

            let ranking = strategy.ranking
            Label {
                Text("Top \(ranking.topN) of the quality ranking at "
                     + "\(Int(ranking.qualityWeight * 100))% quality, equally weighted"
                     + (ranking.maxPerSector.map { ", max \($0) per industry" } ?? "")
                     + (ranking.minMarketCap.map { ", min $\(Int($0 / 1_000_000_000))B market cap" } ?? "")
                     + ". \(ranking.rebalance).")
                    .appFont(.caption).fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "chart.line.uptrend.xyaxis").foregroundStyle(.secondary)
            }

            Label {
                Text("Fully invested at all times — no market timing, no cash position, "
                     + "no funds.")
                    .appFont(.caption).fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "checkmark.shield").foregroundStyle(.secondary)
            }

            BacktestStats(backtest: strategy.backtest)
        }
    }

    private func risks(_ strategy: StrategyDefinition) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("WHAT IT COSTS YOU").appFont(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            ForEach(strategy.risks, id: \.self) { risk in
                Label {
                    Text(risk).appFont(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "exclamationmark.shield").foregroundStyle(.orange)
                }
            }
        }
    }

}

private struct StrategyCard: View {
    let strategy: StrategyDefinition
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                if isSelected {
                    Image(systemName: "checkmark")
                        .appFont(.caption.weight(.bold))
                        .foregroundStyle(Theme.brand)
                        .accessibilityHidden(true)
                }
                Text(strategy.name).appFont(.subheadline.weight(.semibold))
                    .foregroundStyle(isSelected ? Theme.brand : .primary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 6)
                if strategy.isDefault {
                    Text("Recommended")
                        .appFont(.caption2.weight(.medium))
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Capsule().fill(Color.secondary.opacity(0.15)))
                }
            }
            Text(strategy.summary)
                .appFont(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            BacktestStats(backtest: strategy.backtest, compact: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cardRadius)
                .fill(isSelected ? Theme.brand.opacity(0.06) : .clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cardRadius)
                .strokeBorder(isSelected ? Theme.brand : .clear, lineWidth: 2)
        )
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

private struct BacktestStats: View {
    let backtest: StrategyBacktest
    var compact = false

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            stat("CAGR \(backtest.window ?? "")", pct(backtest.cagr), .up)
            stat("Max drawdown", pct(backtest.maxDrawdown), .down)
            stat("Sharpe", backtest.sharpe.map { String(format: "%.2f", $0) } ?? "—", nil)
            if !compact {
                stat("Volatility", pct(backtest.volatility), nil)
            }
        }
    }

    private func stat(_ label: String, _ value: String, _ tone: Color?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            // A stat is unreadable without its label, so a tight column shrinks
            // the label rather than truncating it to "Max drawdo…".
            Text(label).appFont(.caption2).foregroundStyle(.secondary)
                .lineLimit(1).minimumScaleFactor(0.75)
            Text(value).appFont(.callout.weight(.bold)).monospacedDigit()
                .foregroundStyle(tone ?? .primary)
        }
    }

    private func pct(_ value: Double?) -> String {
        value.map { String(format: "%.1f%%", $0) } ?? "—"
    }
}

private struct SleeveSection: View {
    @Environment(\.appFontScale) private var fontScale
    private var cols: SleeveWidths { SleeveWidths(scale: fontScale) }
    let sleeve: StrategySleeve
    let currency: String
    let ageDays: Int?
    let isStale: Bool
    let onSelect: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // The sleeve's name and its provenance sit on one line where they
            // fit, and on two where they don't — never half-wrapped through the
            // middle of the weight-and-date phrase.
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .firstTextBaseline) {
                    Text(sleeve.label).appFont(.subheadline.weight(.semibold))
                    Spacer(minLength: 8)
                    provenance
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(sleeve.label).appFont(.subheadline.weight(.semibold))
                    provenance
                }
            }
            columnHeader
            Divider()

            if sleeve.positions.isEmpty {
                Text("No positions — the sleeve could not be built.")
                    .appFont(.caption).foregroundStyle(.secondary).padding(.vertical, 6)
            } else {
                ForEach(sleeve.positions) { position in
                    Button { onSelect(position.symbol) } label: {
                        PositionRow(position: position, currency: currency)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    /// The snapshot's date is provenance, not decoration: it says how stale the
    /// ranked list is, which the reader cannot infer.
    private var provenance: some View {
        HStack(spacing: 4) {
            Text("\(Int(sleeve.weight * 100))% · \(Fmt.currency(sleeve.amount, code: currency))"
                 + (sleeve.rankedAt.map { " · ranked \(MarketTime.formatted($0))" } ?? ""))
                .foregroundStyle(.secondary)
            if let ageDays {
                Text(ageDays == 0 ? "(today)" : "(\(ageDays)d ago)")
                    .foregroundStyle(isStale ? .orange : .secondary)
                    .fontWeight(isStale ? .semibold : .regular)
            }
        }
        .appFont(.caption).monospacedDigit()
        .lineLimit(1).minimumScaleFactor(0.75)
    }

    /// Column labels, so the trailing numbers are identifiable without having
    /// to infer them from their format — the web table has a header row.
    private var columnHeader: some View {
        HStack(spacing: cols.spacing) {
            Text("SYMBOL")
            Spacer(minLength: 6)
            if SleeveColumns.showsIndustry {
                Text("INDUSTRY").frame(width: cols.industry, alignment: .leading)
            }
            Text(isPhoneLayout ? "WT" : "WEIGHT")
                .frame(width: cols.weight, alignment: .trailing)
            Text("AMOUNT").frame(width: cols.amount, alignment: .trailing)
            Text(sleeve.priceSource == .snapshot
                 ? (isPhoneLayout ? "STORED" : "PRICE (STORED)")
                 : "PRICE")
                .frame(width: cols.price, alignment: .trailing)
            Text("SHARES").frame(width: cols.shares, alignment: .trailing)
        }
        .appFont(.caption2.weight(.semibold))
        .foregroundStyle(.secondary)
        .lineLimit(1)
    }
}

private extension View {
    /// A fixed-width numeric column that never wraps: outsized values (a
    /// four-digit share price, a five-digit share count) shrink to fit the
    /// column instead of breaking onto a second line.
    func numericCell(width: CGFloat) -> some View {
        lineLimit(1)
            .minimumScaleFactor(0.6)
            .frame(width: width, alignment: .trailing)
    }
}

/// Fixed widths for the sleeve table's numeric columns. At Mac spacing the
/// four columns total ~306pt with gaps — wider than the space a ~390pt phone
/// leaves inside its card, which crushed the symbol column to an ellipsis.
/// The sleeve table's columns at the window's font scale. The constants below
/// are sized for unscaled type; on a wide window the type grows, and a column
/// that doesn't grow with it just pushes its number back down through
/// `minimumScaleFactor` — a bigger table of the same small figures.
private struct SleeveWidths {
    let scale: CGFloat
    var spacing: CGFloat { SleeveColumns.spacing * scale }
    var industry: CGFloat { SleeveColumns.industry * scale }
    var weight: CGFloat { SleeveColumns.weight * scale }
    var amount: CGFloat { SleeveColumns.amount * scale }
    var price: CGFloat { SleeveColumns.price * scale }
    var shares: CGFloat { SleeveColumns.shares * scale }
}

private enum SleeveColumns {
    static let spacing: CGFloat = isPhoneLayout ? 6 : 10
    static let weight: CGFloat = isPhoneLayout ? 38 : 48
    static let amount: CGFloat = isPhoneLayout ? 80 : 92
    static let price: CGFloat = isPhoneLayout ? 50 : 62
    static let shares: CGFloat = isPhoneLayout ? 46 : 64

    /// The industry a name was capped under — the same column the web table
    /// shows, and hidden on the same terms: the web hides it below `md`, and a
    /// phone has no room for a sixth column beside five that already crowd the
    /// symbol. On a phone the industry still appears, as the row's subtitle.
    static let industry: CGFloat = 120
    static var showsIndustry: Bool { !isPhoneLayout }
}

private struct PositionRow: View {
    @Environment(\.appFontScale) private var fontScale
    private var cols: SleeveWidths { SleeveWidths(scale: fontScale) }
    let position: StrategyPosition
    let currency: String

    /// The company, under its ticker.
    private var subtitle: String? { position.name ?? position.note }

    /// The industry, where the table has no column for it.
    ///
    /// A phone cannot fit a sixth column — the web table hides this one below
    /// `md` for the same reason — but the industry is the fact the
    /// three-per-industry cap is about, so it rides under the name rather than
    /// being dropped from the narrow layout entirely.
    private var inlineIndustry: String? {
        SleeveColumns.showsIndustry ? nil : position.industry
    }

    var body: some View {
        HStack(spacing: cols.spacing) {
            StockIcon(symbol: position.symbol, size: isPhoneLayout ? 20 : 24)
            VStack(alignment: .leading, spacing: 1) {
                Text(position.symbol).appFont(.caption.weight(.semibold))
                    .lineLimit(1).fixedSize(horizontal: true, vertical: false)
                if let subtitle {
                    Text(subtitle).appFont(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }
                if let inlineIndustry {
                    Text(inlineIndustry).appFont(.caption2).foregroundStyle(.tertiary).lineLimit(1)
                }
            }
            Spacer(minLength: 6)
            if SleeveColumns.showsIndustry {
                // The industry the three-per-industry cap was applied against,
                // so a reader can see the concentration the rule allowed rather
                // than having to recognise it from the tickers.
                Text(position.industry ?? "—")
                    .appFont(.caption2).foregroundStyle(.secondary)
                    .lineLimit(1).minimumScaleFactor(0.75)
                    .frame(width: cols.industry, alignment: .leading)
            }
            Text("\(String(format: "%.1f", position.weight * 100))%")
                .appFont(.caption2).foregroundStyle(.secondary).monospacedDigit()
                .numericCell(width: cols.weight)
            Text(Fmt.currency(position.amount, code: currency))
                .appFont(.caption).monospacedDigit()
                .numericCell(width: cols.amount)
            Text(position.price.map { String(format: "%.2f", $0) } ?? "—")
                .appFont(.caption).monospacedDigit().foregroundStyle(.secondary)
                .numericCell(width: cols.price)
            Text(position.shares.map { "\($0)" } ?? "—")
                .appFont(.caption).monospacedDigit().foregroundStyle(.secondary)
                .numericCell(width: cols.shares)
        }
        .padding(.vertical, 3)
    }
}
