import SwiftUI

private enum ScreenSortKey: String { case symbol, price, intrinsic, mos, pe, aiScore }

struct ScreenerResultsView: View {
    @ObservedObject var viewModel: ScreenerViewModel
    let currency: String

    @State private var search = ""
    @State private var minMOS = ""
    @State private var maxPE = ""
    @State private var marketCap = "all"
    @State private var onlyAI = false
    @State private var showFilters = false
    @State private var sortKey: ScreenSortKey = .mos
    @State private var sortAsc = false
    @State private var expanded: String?
    @State private var detail: SymbolID?

    private let capOptions: [(key: String, label: String)] = [
        ("all", "All"), ("mega", "Mega ≥ $200B"), ("large", "Large $10–200B"),
        ("mid", "Mid $2–10B"), ("small", "Small $300M–2B"), ("micro", "Micro < $300M"),
    ]

    // MARK: - Pipeline

    private var filtered: [ScreenerResult] {
        let q = search.lowercased()
        let mos = Double(minMOS); let pe = Double(maxPE)
        return viewModel.results.filter { r in
            if !q.isEmpty, !r.symbol.lowercased().contains(q), !(r.name?.lowercased().contains(q) ?? false) { return false }
            if let mos, (r.marginOfSafety ?? -.infinity) < mos { return false }
            if let pe, (r.peRatio ?? .infinity) > pe { return false }
            if onlyAI, !(r.hasAIReview ?? false) { return false }
            if marketCap != "all" {
                let cap = r.marketCap ?? 0
                switch marketCap {
                case "micro": if cap >= 300_000_000 { return false }
                case "small": if cap < 300_000_000 || cap >= 2_000_000_000 { return false }
                case "mid": if cap < 2_000_000_000 || cap >= 10_000_000_000 { return false }
                case "large": if cap < 10_000_000_000 || cap >= 200_000_000_000 { return false }
                case "mega": if cap < 200_000_000_000 { return false }
                default: break
                }
            }
            return true
        }
    }

    private var sorted: [ScreenerResult] {
        func key(_ r: ScreenerResult) -> Double? {
            switch sortKey {
            case .symbol: return nil
            case .price: return r.price
            case .intrinsic: return r.intrinsicValue
            case .mos: return r.marginOfSafety
            case .pe: return r.peRatio
            case .aiScore: return r.aiScore
            }
        }
        return filtered.sorted { a, b in
            if sortKey == .symbol { return sortAsc ? a.symbol < b.symbol : a.symbol > b.symbol }
            let av = key(a); let bv = key(b)
            if av == nil { return false }; if bv == nil { return true }  // nulls last
            return sortAsc ? av! < bv! : av! > bv!
        }
    }

    private var summary: (count: Int, undervalued: Int, avgMOS: Double?, aiReviewed: Int) {
        let withMOS = filtered.compactMap { $0.marginOfSafety }
        let avg = withMOS.isEmpty ? nil : withMOS.reduce(0, +) / Double(withMOS.count)
        return (filtered.count, withMOS.filter { $0 > 0 }.count, avg, filtered.filter { $0.hasAIReview ?? false }.count)
    }

    // MARK: - Body

    var body: some View {
        if viewModel.results.isEmpty {
            ContentUnavailableView("No results yet", systemImage: "chart.bar",
                                   description: Text("Configure parameters and execute a scan."))
                .frame(minHeight: 200)
        } else {
            LazyVStack(alignment: .leading, spacing: 12) {
                headerRow
                summaryRow
                if showFilters { filterPanel }
                tableHeader
                Divider()
                ForEach(sorted) { row in
                    resultRow(row)
                    if expanded == row.symbol, let rev = viewModel.reviews[row.symbol] { reviewPanel(row.symbol, rev) }
                    Divider()
                }
            }
            .padding(16)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
            .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: currency) }
        }
    }

    private var headerRow: some View {
        HStack {
            Label("Scan Results", systemImage: "target").font(.headline)
            Text("\(filtered.count) of \(viewModel.results.count)").font(.caption).foregroundStyle(.secondary)
            Spacer()
            TextField("Search…", text: $search).textFieldStyle(.roundedBorder).frame(width: 160)
            Button { showFilters.toggle() } label: { Image(systemName: "slider.horizontal.3") }
                .buttonStyle(.bordered).tint(showFilters ? .accentColor : nil)
        }
    }

    private var summaryRow: some View {
        #if os(iOS)
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 24) {
                stat("Results", "\(summary.count)", .primary)
                stat("Undervalued", "\(summary.undervalued)", .green)
                stat("Avg MOS", summary.avgMOS.map { "\($0 >= 0 ? "+" : "")\(String(format: "%.1f", $0))%" } ?? "–",
                     Fmt.tint(for: summary.avgMOS))
                stat("AI Reviewed", "\(summary.aiReviewed) of \(summary.count)", .purple)
                Spacer()
            }
        }
        #else
        HStack(spacing: 24) {
            stat("Results", "\(summary.count)", .primary)
            stat("Undervalued", "\(summary.undervalued)", .green)
            stat("Avg MOS", summary.avgMOS.map { "\($0 >= 0 ? "+" : "")\(String(format: "%.1f", $0))%" } ?? "–",
                 Fmt.tint(for: summary.avgMOS))
            stat("AI Reviewed", "\(summary.aiReviewed) of \(summary.count)", .purple)
            Spacer()
        }
        #endif
    }
    private func stat(_ l: String, _ v: String, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(l).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(v).font(.headline).foregroundStyle(tone)
        }
    }

    private var filterPanel: some View {
        #if os(iOS)
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), spacing: 12)], alignment: .leading, spacing: 12) {
            field("Min MOS %", $minMOS)
            field("Max P/E", $maxPE)
            VStack(alignment: .leading, spacing: 2) {
                Text("Market Cap").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $marketCap) { ForEach(capOptions, id: \.key) { Text($0.label).tag($0.key) } }
                    .labelsHidden()
            }
            Toggle("AI Reviewed", isOn: $onlyAI)
        }
        .padding(12).background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
        #else
        HStack(spacing: 12) {
            field("Min MOS %", $minMOS)
            field("Max P/E", $maxPE)
            VStack(alignment: .leading, spacing: 2) {
                Text("Market Cap").font(.caption2).foregroundStyle(.secondary)
                Picker("", selection: $marketCap) { ForEach(capOptions, id: \.key) { Text($0.label).tag($0.key) } }
                    .labelsHidden().fixedSize()
            }
            Toggle("AI Reviewed", isOn: $onlyAI)
            Spacer()
        }
        .padding(12).background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
        #endif
    }
    private func field(_ label: String, _ binding: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            TextField("", text: binding).textFieldStyle(.roundedBorder).frame(width: 80)
        }
    }

    // MARK: - Table

    @ViewBuilder private var tableHeader: some View {
        #if os(iOS)
        EmptyView()
        #else
        HStack(spacing: 8) {
            sortButton("Asset", .symbol, align: .leading).frame(maxWidth: .infinity, alignment: .leading)
            sortButton("Price", .price, align: .trailing).frame(width: 90, alignment: .trailing)
            sortButton("Intrinsic", .intrinsic, align: .trailing).frame(width: 90, alignment: .trailing)
            sortButton("MOS", .mos, align: .trailing).frame(width: 90, alignment: .trailing)
            sortButton("P/E", .pe, align: .trailing).frame(width: 56, alignment: .trailing)
            sortButton("AI Score", .aiScore, align: .trailing).frame(width: 80, alignment: .trailing)
            Text("AI Audit").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).frame(width: 110, alignment: .trailing)
        }
        #endif
    }
    private func sortButton(_ label: String, _ key: ScreenSortKey, align: Alignment) -> some View {
        Button {
            if sortKey == key { sortAsc.toggle() } else { sortKey = key; sortAsc = false }
        } label: {
            HStack(spacing: 2) {
                Text(label).font(.caption2.weight(.semibold))
                if sortKey == key { Image(systemName: sortAsc ? "chevron.up" : "chevron.down").font(.system(size: 9)) }
            }
            .foregroundStyle(sortKey == key ? Color.accentColor : .secondary)
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder private func resultRow(_ row: ScreenerResult) -> some View {
        #if os(iOS)
        iosResultRow(row)
        #else
        HStack(spacing: 8) {
            Button { detail = SymbolID(id: row.symbol) } label: {
                HStack(spacing: 8) {
                    StockIcon(symbol: row.symbol, size: 26)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(row.symbol).font(.callout.bold())
                        Text(row.name ?? "").font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
            }.buttonStyle(.plain)
            .frame(maxWidth: .infinity, alignment: .leading)

            Text(Fmt.currency(row.price, code: "USD")).monospacedDigit().frame(width: 90, alignment: .trailing)
            Text(row.intrinsicValue.map { Fmt.currency($0, code: "USD") } ?? "-").monospacedDigit()
                .foregroundStyle(.secondary).frame(width: 90, alignment: .trailing)
            mosCell(row.marginOfSafety).frame(width: 90, alignment: .trailing)
            Text(row.peRatio.map { String(format: "%.1f", $0) } ?? "-").monospacedDigit()
                .foregroundStyle(.secondary).frame(width: 56, alignment: .trailing)
            aiScoreCell(row.aiScore).frame(width: 80, alignment: .trailing)
            auditButton(row).frame(width: 110, alignment: .trailing)
        }
        .font(.callout)
        .padding(.vertical, 4)
        #endif
    }

    #if os(iOS)
    private func iosResultRow(_ row: ScreenerResult) -> some View {
        VStack(spacing: 12) {
            HStack {
                Button { detail = SymbolID(id: row.symbol) } label: {
                    HStack(spacing: 8) {
                        StockIcon(symbol: row.symbol, size: 26)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(row.symbol).font(.callout.bold())
                            Text(row.name ?? "").font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                        }
                    }
                }.buttonStyle(.plain)
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(Fmt.currency(row.price, code: "USD")).monospacedDigit().fontWeight(.bold)
                    Text("P/E: \(row.peRatio.map { String(format: "%.1f", $0) } ?? "-")").font(.caption2).foregroundStyle(.secondary)
                }
            }
            Divider()
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("MOS").font(.caption2).foregroundStyle(.secondary)
                    mosCell(row.marginOfSafety)
                }
                Spacer()
                VStack(alignment: .center, spacing: 2) {
                    Text("Intrinsic").font(.caption2).foregroundStyle(.secondary)
                    Text(row.intrinsicValue.map { Fmt.currency($0, code: "USD") } ?? "-").monospacedDigit().font(.caption.bold())
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("AI Score").font(.caption2).foregroundStyle(.secondary)
                    aiScoreCell(row.aiScore)
                }
            }
            auditButton(row).frame(maxWidth: .infinity)
        }
        .padding(.vertical, 8)
    }
    #endif

    @ViewBuilder private func mosCell(_ mos: Double?) -> some View {
        if let mos {
            let tone: Color = mos > 15 ? .green : (mos > 0 ? .cyan : .red)
            HStack(spacing: 2) {
                Image(systemName: mos > 0 ? "arrow.up.right" : "arrow.down.right").font(.system(size: 10))
                Text(String(format: "%.1f%%", mos)).monospacedDigit()
            }.foregroundStyle(tone).fontWeight(.bold)
        } else { Text("N/A").foregroundStyle(.secondary.opacity(0.5)).font(.caption) }
    }

    @ViewBuilder private func aiScoreCell(_ score: Double?) -> some View {
        if let s = score {
            let tone: Color = s >= 8 ? .green : (s >= 6 ? .cyan : (s >= 4 ? .orange : .red))
            Text(String(format: "%.1f/10", s)).monospacedDigit().fontWeight(.bold).foregroundStyle(tone)
        } else { Text("N/A").foregroundStyle(.secondary.opacity(0.5)).font(.caption) }
    }

    private func auditButton(_ row: ScreenerResult) -> some View {
        Button {
            if viewModel.reviews[row.symbol] != nil { expanded = expanded == row.symbol ? nil : row.symbol }
            else { Task { await viewModel.review(row.symbol); expanded = row.symbol } }
        } label: {
            if viewModel.reviewingSymbol == row.symbol {
                HStack(spacing: 4) { ProgressView().controlSize(.small); Text("Analyzing") }
            } else if row.hasAIReview ?? false {
                Label("Review", systemImage: "sparkles")
            } else {
                Label("Analyze", systemImage: "brain")
            }
        }
        .font(.caption2.weight(.bold))
        .buttonStyle(.bordered)
        .tint((row.hasAIReview ?? false) ? .purple : nil)
        .disabled(viewModel.reviewingSymbol == row.symbol)
    }

    private func reviewPanel(_ symbol: String, _ rev: ScreenReview) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("AI Technical & Fundamental Audit", systemImage: "sparkles").font(.subheadline.weight(.bold))
                Spacer()
                Button { Task { await viewModel.review(symbol, force: true) } } label: {
                    Label("Regenerate", systemImage: "arrow.clockwise").font(.caption2)
                }.buttonStyle(.borderless)
            }
            if let card = rev.scorecard, !card.isEmpty {
                FlowChips(items: card.sorted { $0.key < $1.key }.map { (key: $0.key, value: $0.value) })
            }
            if let summary = rev.summary, !summary.isEmpty {
                Text("“\(summary)”").font(.callout).italic().padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
            }
            if let analysis = rev.analysis, !analysis.isEmpty {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 12)], alignment: .leading, spacing: 12) {
                    ForEach(analysis.sorted { $0.key < $1.key }, id: \.key) { k, v in
                        VStack(alignment: .leading, spacing: 3) {
                            Label(k.replacingOccurrences(of: "_", with: " ").capitalized, systemImage: "chevron.right")
                                .font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                            Text(v).font(.caption).foregroundStyle(.primary.opacity(0.85))
                        }
                    }
                }
            }
        }
        .padding(14).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
    }
}

/// Simple wrapping row of scorecard chips.
private struct FlowChips: View {
    let items: [(key: String, value: Double)]
    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 8)], alignment: .leading, spacing: 8) {
            ForEach(items, id: \.key) { item in
                HStack {
                    Text(item.key.replacingOccurrences(of: "_", with: " ").capitalized)
                        .font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(item.value))/10").font(.caption.bold())
                        .foregroundStyle(item.value >= 8 ? .green : (item.value >= 6 ? .cyan : .orange))
                }
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }
}
