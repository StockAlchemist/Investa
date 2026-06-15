import SwiftUI
import Charts

/// Identifiable wrapper so a bare symbol string can drive `.sheet(item:)`.
struct SymbolID: Identifiable, Hashable {
    let id: String
}

/// Detailed stock view presented as a sheet. Mirrors the web StockDetailModal:
/// fundamentals, price chart, intrinsic value, earnings, and on-demand AI review.
struct StockDetailView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL
    @StateObject private var viewModel: StockDetailViewModel

    private let periods = ["1m", "3m", "6m", "1y", "5y"]

    init(symbol: String) {
        _viewModel = StateObject(wrappedValue: StockDetailViewModel(symbol: symbol))
    }

    private var f: Fundamentals? { viewModel.fundamentals }
    private var cur: String { f?.currency ?? "USD" }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    priceChart
                    fundamentalsGrid
                    if let iv = viewModel.intrinsic { intrinsicSection(iv) }
                    if let summary = f?.summary, !summary.isEmpty { aboutSection(summary) }
                    if !viewModel.earnings.isEmpty { earningsSection }
                    financialsSection
                    aiSection
                }
                .padding(20)
            }
        }
        .frame(width: 720, height: 720)
        .task { await viewModel.loadAll() }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.symbol).font(.title.bold())
                if let name = f?.name { Text(name).foregroundStyle(.secondary).lineLimit(1) }
                if let sector = f?.sector {
                    Text(sector).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            VStack(alignment: .trailing) {
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Text(Fmt.currency(f?.price, code: cur)).font(.title2.weight(.semibold))
            }
            Button { dismiss() } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary).font(.title2)
        }
        .padding(20)
    }

    private var priceChart: some View {
        let dated = viewModel.history.compactMap { p -> (Date, Double)? in
            guard let d = p.parsedDate else { return nil }; return (d, p.value)
        }
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Price").font(.headline)
                Spacer()
                Picker("Period", selection: $viewModel.period) {
                    ForEach(periods, id: \.self) { Text($0.uppercased()).tag($0) }
                }
                .pickerStyle(.segmented).fixedSize()
                .onChange(of: viewModel.period) { _, _ in Task { await viewModel.loadHistory() } }
            }
            if dated.isEmpty {
                ContentUnavailableView("No price data", systemImage: "chart.xyaxis.line")
                    .frame(height: 200)
            } else {
                Chart(dated, id: \.0) { item in
                    LineMark(x: .value("Date", item.0), y: .value("Price", item.1))
                        .foregroundStyle(.tint).interpolationMethod(.monotone)
                }
                .frame(height: 220)
            }
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private var fundamentalsGrid: some View {
        let items: [(String, String)] = [
            ("Market Cap", Fmt.number(f?.marketCap, fractionDigits: 0)),
            ("Trailing P/E", Fmt.number(f?.trailingPE)),
            ("Forward P/E", Fmt.number(f?.forwardPE)),
            ("Dividend Yield", Fmt.percent(f?.dividendYield)),
            ("Beta", Fmt.number(f?.beta)),
            ("52W High", Fmt.currency(f?.high52, code: cur)),
            ("52W Low", Fmt.currency(f?.low52, code: cur)),
            ("Exchange", f?.exchange ?? "—"),
        ]
        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
            ForEach(items, id: \.0) { item in
                VStack(alignment: .leading, spacing: 3) {
                    Text(item.0).font(.caption).foregroundStyle(.secondary)
                    Text(item.1).font(.callout.weight(.medium))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func intrinsicSection(_ iv: IntrinsicValueResponse) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Valuation").font(.headline)
            HStack {
                valueBox("Intrinsic Value", Fmt.currency(iv.averageIntrinsicValue, code: cur))
                valueBox("Current Price", Fmt.currency(iv.currentPrice, code: cur))
                valueBox("Margin of Safety", Fmt.percent(iv.marginOfSafetyPct),
                         tint: Fmt.tint(for: iv.marginOfSafetyPct))
            }
            if let dcf = iv.models?.dcf?.intrinsicValue {
                Text("DCF: \(Fmt.currency(dcf, code: cur))").font(.caption).foregroundStyle(.secondary)
            }
            if let graham = iv.models?.graham?.intrinsicValue {
                Text("Graham: \(Fmt.currency(graham, code: cur))").font(.caption).foregroundStyle(.secondary)
            }
            if let note = iv.valuationNote, !note.isEmpty {
                Text(note).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func valueBox(_ title: String, _ value: String, tint: Color = .primary) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.weight(.semibold)).foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func aboutSection(_ summary: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("About").font(.headline)
                Spacer()
                if let site = f?.website, let url = URL(string: site) {
                    Button("Website") { openURL(url) }.font(.caption)
                }
            }
            Text(summary).font(.callout).foregroundStyle(.secondary)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private var earningsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Earnings").font(.headline)
            ForEach(viewModel.earnings.prefix(6)) { e in
                HStack {
                    Text(e.date).frame(width: 110, alignment: .leading)
                    Spacer()
                    if let est = e.epsEstimate { Text("Est \(Fmt.number(est))").font(.caption).foregroundStyle(.secondary) }
                    if let act = e.epsActual { Text("Act \(Fmt.number(act))").font(.caption) }
                    if let s = e.surprisePct {
                        Text(Fmt.percent(s)).font(.caption).foregroundStyle(Fmt.tint(for: s))
                    }
                }
                .padding(.vertical, 3)
                Divider()
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private var financialsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Financials & Ratios").font(.headline)
                Spacer()
                if viewModel.isLoadingFinancials {
                    ProgressView().controlSize(.small)
                } else if viewModel.financials == nil {
                    Button("Load") { Task { await viewModel.loadFinancials() } }
                }
            }
            if let ratios = viewModel.ratios?.valuation, !ratios.isEmpty {
                Text("Valuation Ratios").font(.subheadline.weight(.medium))
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 8) {
                    ForEach(ratios.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                        HStack {
                            Text(key).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            Spacer()
                            Text(value.doubleValue.map { Fmt.number($0) } ?? (value.stringValue ?? "—"))
                                .font(.caption.weight(.medium)).monospacedDigit()
                        }
                    }
                }
            }
            if let fin = viewModel.financials?.financials, !fin.index.isEmpty {
                statementTable("Income Statement", fin)
            }
            if let bs = viewModel.financials?.balanceSheet, !bs.index.isEmpty {
                statementTable("Balance Sheet", bs)
            }
            if let cf = viewModel.financials?.cashflow, !cf.index.isEmpty {
                statementTable("Cash Flow", cf)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    /// Renders a statement matrix (line items × periods) as a scrollable grid.
    private func statementTable(_ title: String, _ s: FinancialStatement) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.subheadline.weight(.medium)).padding(.top, 6)
            ScrollView(.horizontal, showsIndicators: true) {
                Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 4) {
                    GridRow {
                        Text("").gridColumnAlignment(.leading)
                        ForEach(Array(s.columns.prefix(5).enumerated()), id: \.offset) { _, col in
                            Text(String(col.prefix(10))).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    ForEach(Array(s.index.enumerated()), id: \.offset) { i, label in
                        GridRow {
                            Text(label).font(.caption).lineLimit(1).gridColumnAlignment(.leading)
                            ForEach(0..<min(5, s.columns.count), id: \.self) { j in
                                let v = (i < s.data.count && j < s.data[i].count) ? s.data[i][j] : nil
                                Text(v.map { Fmt.number($0, fractionDigits: 0) } ?? "—")
                                    .font(.caption).monospacedDigit()
                            }
                        }
                    }
                }
            }
        }
    }

    private var aiSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("AI Analysis").font(.headline)
                Spacer()
                if viewModel.isLoadingAnalysis {
                    ProgressView().controlSize(.small)
                } else {
                    Button(viewModel.analysis == nil ? "Generate" : "Refresh") {
                        Task { await viewModel.loadAnalysis(force: viewModel.analysis != nil) }
                    }
                }
            }
            if let a = viewModel.analysis {
                if let err = a.error, !err.isEmpty {
                    Text(err).foregroundStyle(.red).font(.callout)
                }
                if let card = a.scorecard { scorecardView(card) }
                if let summary = a.summary, !summary.isEmpty {
                    Text(summary).font(.callout)
                }
                if let review = a.aiReview, !review.isEmpty {
                    Text(review).font(.callout).foregroundStyle(.secondary)
                }
            } else if !viewModel.isLoadingAnalysis {
                Text("Generate an AI review of this stock's moat, financial strength, growth, and predictability.")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func scorecardView(_ card: StockAnalysis.Scorecard) -> some View {
        let rows: [(String, Double?)] = [
            ("Moat", card.moat), ("Financial Strength", card.financialStrength),
            ("Predictability", card.predictability), ("Growth", card.growth),
        ]
        return VStack(spacing: 4) {
            ForEach(rows, id: \.0) { name, score in
                HStack {
                    Text(name).frame(width: 140, alignment: .leading).font(.caption)
                    ProgressView(value: max(0, min(score ?? 0, 100)), total: 100)
                    Text(Fmt.number(score, fractionDigits: 0)).font(.caption).monospacedDigit()
                        .frame(width: 30, alignment: .trailing)
                }
            }
        }
    }
}
