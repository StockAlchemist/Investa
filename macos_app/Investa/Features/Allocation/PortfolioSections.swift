import SwiftUI
import Charts

/// Shared section chrome for the Portfolio tab.
private struct Section_<Content: View>: View {
    let title: String
    var trailing: AnyView? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack { Text(title).font(.headline); Spacer(); if let trailing { trailing } }
            content
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
}

private func isCashSymbol(_ s: String) -> Bool {
    let u = s.uppercased(); return u == "$CASH" || u == "CASH" || u.hasPrefix("CASH (")
}

// MARK: - Concentration KPIs (mirrors portfolio/ConcentrationKpiStrip.tsx)

struct ConcentrationKpiStrip: View {
    let holdings: [Holding]
    let currency: String

    private struct Metrics {
        var stockCount = 0; var cashCount = 0
        var largestSymbol: String? = nil; var largestPct: Double? = nil
        var top5: Double? = nil; var top10: Double? = nil
        var effectiveN: Double? = nil; var cashPct = 0.0
    }

    private var m: Metrics {
        var out = Metrics()
        let positions = holdings.map { (sym: $0.symbol, isCash: isCashSymbol($0.symbol), value: max(0, $0.marketValue(currency: currency) ?? 0)) }
            .filter { $0.value > 0 }
        let total = positions.reduce(0) { $0 + $1.value }
        let stocks = positions.filter { !$0.isCash }.sorted { $0.value > $1.value }
        let cash = positions.filter { $0.isCash }
        out.stockCount = stocks.count; out.cashCount = cash.count
        let weights = stocks.map { total > 0 ? $0.value / total : 0 }
        if let largest = stocks.first { out.largestSymbol = largest.sym; out.largestPct = total > 0 ? largest.value / total * 100 : nil }
        func topN(_ n: Int) -> Double { weights.prefix(n).reduce(0, +) * 100 }
        if !stocks.isEmpty { out.top5 = topN(5); out.top10 = topN(10) }
        let hhi = weights.reduce(0) { $0 + $1 * $1 }
        out.effectiveN = hhi > 0 ? 1 / hhi : nil
        out.cashPct = total > 0 ? cash.reduce(0) { $0 + $1.value } / total * 100 : 0
        return out
    }

    var body: some View {
        let mt = m
        let largestTone: Color = (mt.largestPct ?? 0) >= 25 ? .orange : ((mt.largestPct ?? 0) >= 15 ? .primary : .green)
        let effTone: Color = (mt.effectiveN ?? 0) >= 10 ? .green : ((mt.effectiveN ?? 0) >= 5 ? .primary : .orange)
        return Section_(title: "Concentration") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 12)], spacing: 12) {
                tile("Holdings", "\(mt.stockCount)", mt.cashCount > 0 ? "+ \(mt.cashCount) cash" : "stocks & funds", .primary)
                tile("Largest", mt.largestSymbol ?? "–", mt.largestPct.map { String(format: "%.1f%%", $0) }, largestTone)
                tile("Top 5", mt.top5.map { String(format: "%.1f%%", $0) } ?? "–", "of portfolio", .primary)
                tile("Top 10", mt.top10.map { String(format: "%.1f%%", $0) } ?? "–", "of portfolio", .primary)
                tile("Effective N", mt.effectiveN.map { String(format: "%.1f", $0) } ?? "–", "equal-weight equiv.", effTone)
                tile("Cash", String(format: "%.1f%%", mt.cashPct), mt.cashPct > 20 ? "heavy cash drag" : "of portfolio", mt.cashPct > 20 ? .orange : .primary)
            }
        }
    }

    private func tile(_ label: String, _ value: String, _ sub: String?, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(value).font(.title3.bold()).foregroundStyle(tone).lineLimit(1)
            if let sub { Text(sub).font(.caption2).foregroundStyle(.secondary) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Allocation drift (mirrors AllocationDrift.tsx)

struct AllocationDriftCard: View {
    let holdings: [Holding]
    let currency: String
    let bucketKey: String
    let settingsBucket: String
    let title: String
    @ObservedObject var vm: AllocationViewModel
    var scrollable = false

    @State private var editing = false
    @State private var draft: [String: String] = [:]

    private struct Row: Identifiable { let bucket: String; let current: Double; let target: Double; var id: String { bucket }
        var drift: Double { current - target } }

    private var targets: [String: Double] { vm.targets[settingsBucket] ?? [:] }

    private var rows: [Row] {
        var agg: [String: Double] = [:]
        for h in holdings { agg[PortfolioBucket.value(h, key: bucketKey), default: 0] += h.marketValue(currency: currency) ?? 0 }
        let total = agg.values.reduce(0, +)
        let buckets = Set(agg.keys).union(targets.keys)
        return buckets.map { b in Row(bucket: b, current: total > 0 ? (agg[b] ?? 0) / total * 100 : 0, target: targets[b] ?? 0) }
            .sorted { $0.current > $1.current || ($0.current == $1.current && $0.target > $1.target) }
    }

    private var targetSum: Double { targets.values.reduce(0, +) }
    private var draftSum: Double { draft.values.reduce(0) { $0 + (Double($1) ?? 0) } }

    var body: some View {
        Section_(title: title, trailing: AnyView(headerButton)) {
            if rows.isEmpty {
                Text("No holdings to bucket.").foregroundStyle(.secondary)
            } else if targetSum == 0 && !editing {
                Text("Set target % per bucket to see drift from your plan.").font(.callout).foregroundStyle(.secondary)
            } else {
                let content = VStack(spacing: 8) { ForEach(rows) { row(of: $0) } }
                if scrollable {
                    ScrollView { content }.frame(maxHeight: 360)
                } else { content }
            }
        }
    }

    @ViewBuilder private var headerButton: some View {
        if editing {
            HStack(spacing: 6) {
                Text("Σ \(String(format: "%.1f%%", draftSum))")
                    .font(.caption.bold()).foregroundStyle(abs(draftSum - 100) < 0.5 ? .green : .orange)
                Button { commit() } label: { Image(systemName: "checkmark") }.tint(.green)
                Button { editing = false; draft = [:] } label: { Image(systemName: "xmark") }
            }
        } else {
            Button(targetSum > 0 ? "Edit targets" : "Set targets") { startEdit() }
                .font(.caption)
        }
    }

    private func row(of r: Row) -> some View {
        let absDrift = abs(r.drift)
        let alert = absDrift >= 10 && r.target > 0
        let warn = !alert && absDrift >= 5 && r.target > 0
        let tone: Color = r.target == 0 ? .secondary : (alert ? .red : (warn ? .orange : .green))
        return HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(r.bucket).font(.caption.weight(.medium)).lineLimit(1)
                    Spacer()
                    Text("\(String(format: "%.1f", r.current))% / \(String(format: "%.1f", r.target))%")
                        .font(.caption2).foregroundStyle(.secondary).monospacedDigit()
                }
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(.quaternary)
                        Capsule().fill(Color.accentColor).frame(width: geo.size.width * min(1, r.current / 100))
                        if r.target > 0 {
                            Rectangle().fill(.primary.opacity(0.7)).frame(width: 2)
                                .offset(x: geo.size.width * min(1, r.target / 100))
                        }
                    }
                }
                .frame(height: 8)
            }
            if editing {
                TextField("0", text: Binding(get: { draft[r.bucket] ?? "" }, set: { draft[r.bucket] = $0 }))
                    .frame(width: 50).multilineTextAlignment(.trailing).textFieldStyle(.roundedBorder)
            } else {
                HStack(spacing: 2) {
                    if alert { Image(systemName: "exclamationmark.triangle.fill").font(.caption2) }
                    if r.drift != 0 { Image(systemName: r.drift > 0 ? "arrow.up.right" : "arrow.down.right").font(.caption2) }
                    Text("\(r.drift > 0 ? "+" : "")\(String(format: "%.1f", r.drift))").font(.caption.bold()).monospacedDigit()
                }
                .foregroundStyle(tone).frame(width: 64, alignment: .trailing)
            }
        }
        .padding(.vertical, 2)
    }

    private func startEdit() {
        var seed: [String: String] = [:]
        for r in rows { seed[r.bucket] = targets[r.bucket].map { String($0) } ?? "" }
        draft = seed; editing = true
    }
    private func commit() {
        var next: [String: Double] = [:]
        for (k, v) in draft { if let n = Double(v), n > 0 { next[k] = n } }
        editing = false; draft = [:]
        Task { await vm.saveTargets(bucket: settingsBucket, values: next) }
    }
}

// MARK: - Rebalance helper (mirrors portfolio/RebalanceHelper.tsx)

struct RebalanceHelperCard: View {
    let holdings: [Holding]
    let currency: String
    @ObservedObject var vm: AllocationViewModel
    @State private var dim = "quoteType"

    private let dims: [(key: String, label: String)] = [
        ("quoteType", "Asset Type"), ("sector", "Sector"), ("country", "Country"),
    ]
    private var bucketKey: String { dim == "sector" ? "Sector" : (dim == "country" ? "Country" : "quoteType") }

    private struct Row: Identifiable { let bucket: String; let currentPct: Double; let targetPct: Double; let delta: Double; var id: String { bucket } }

    private var data: (rows: [Row], total: Double, hasTargets: Bool) {
        let targets = vm.targets[dim] ?? [:]
        var agg: [String: Double] = [:]
        for h in holdings { agg[PortfolioBucket.value(h, key: bucketKey), default: 0] += h.marketValue(currency: currency) ?? 0 }
        let total = agg.values.reduce(0, +)
        let buckets = Set(agg.keys).union(targets.keys)
        let rows = buckets.map { b -> Row in
            let cv = agg[b] ?? 0
            let tp = targets[b] ?? 0
            return Row(bucket: b, currentPct: total > 0 ? cv / total * 100 : 0, targetPct: tp, delta: tp / 100 * total - cv)
        }.filter { $0.targetPct > 0 || $0.currentPct > 0 }.sorted { abs($0.delta) > abs($1.delta) }
        return (rows, total, targets.values.reduce(0, +) > 0)
    }

    var body: some View {
        let d = data
        Section_(title: "Rebalance Helper", trailing: AnyView(
            Picker("", selection: $dim) { ForEach(dims, id: \.key) { Text($0.label).tag($0.key) } }
                .pickerStyle(.menu).fixedSize())) {
            if !d.hasTargets {
                Text("No targets set for \(dims.first { $0.key == dim }!.label.lowercased()). Set them in the drift card above to see suggested trades.")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                ForEach(d.rows) { r in
                    VStack(spacing: 4) {
                        HStack {
                            Text(r.bucket).fontWeight(.bold).lineLimit(1)
                            Spacer()
                            Text("\(String(format: "%.1f", r.currentPct))% → \(String(format: "%.1f", r.targetPct))%")
                                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                            Group {
                                if abs(r.delta) < max(0.005 * d.total, 0.01) {
                                    Text("On target").foregroundStyle(.secondary)
                                } else {
                                    Text("\(r.delta > 0 ? "Buy " : "Sell ")\(Fmt.currency(abs(r.delta), code: currency))")
                                        .foregroundStyle(r.delta > 0 ? .green : .red)
                                }
                            }
                            .font(.caption.weight(.semibold)).frame(width: 120, alignment: .trailing)
                        }
                        Divider()
                    }
                    .padding(.vertical, 2)
                }
                Text("Trades to align each bucket with its target weight at the current value (\(Fmt.currency(d.total, code: currency))). Within 0.5% is on target.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Treemap (mirrors portfolio/PortfolioTreemap.tsx — strip-treemap layout)

struct PortfolioTreemapView: View {
    let holdings: [Holding]
    let currency: String
    @State private var dim = "Sector"

    private let dims: [(key: String, label: String)] = [
        ("Sector", "Sector"), ("quoteType", "Asset Type"), ("Country", "Country"),
    ]
    private let palette: [Color] = [.blue, .teal, .green, .orange, .purple, .pink, .red, .indigo, .mint, .cyan]

    private struct Leaf: Identifiable { let id = UUID(); let symbol: String; let size: Double; let color: Color }

    private var leaves: [Leaf] {
        var groupTotals: [String: Double] = [:]
        var bySymbol: [String: (val: Double, group: String)] = [:]
        for h in holdings {
            let v = h.marketValue(currency: currency) ?? 0
            guard v > 0, !isCashSymbol(h.symbol) else { continue }
            let g = PortfolioBucket.value(h, key: dim)
            groupTotals[g, default: 0] += v
            if var cur = bySymbol[h.symbol] { cur.val += v; bySymbol[h.symbol] = cur } else { bySymbol[h.symbol] = (v, g) }
        }
        let orderedGroups = groupTotals.sorted { $0.value > $1.value }.map { $0.key }
        var color: [String: Color] = [:]
        for (i, g) in orderedGroups.enumerated() { color[g] = palette[i % palette.count] }
        return bySymbol.map { Leaf(symbol: $0.key, size: $0.value.val, color: color[$0.value.group] ?? .gray) }
            .sorted { $0.size > $1.size }
    }

    var body: some View {
        Section_(title: "Treemap", trailing: AnyView(
            Picker("", selection: $dim) { ForEach(dims, id: \.key) { Text($0.label).tag($0.key) } }
                .pickerStyle(.menu).fixedSize())) {
            let items = leaves
            if items.isEmpty {
                Text("No holdings.").foregroundStyle(.secondary)
            } else {
                GeometryReader { geo in
                    treemap(items, in: geo.size)
                }
                .frame(height: 280)
            }
        }
    }

    /// Strip treemap: pack sorted items into rows whose heights are proportional
    /// to the row's share of the total value.
    private func treemap(_ items: [Leaf], in size: CGSize) -> some View {
        let total = items.reduce(0) { $0 + $1.size }
        let targetRow = max(total / 5, 1) // ~5 rows
        var rows: [[Leaf]] = []; var current: [Leaf] = []; var acc = 0.0
        for leaf in items {
            current.append(leaf); acc += leaf.size
            if acc >= targetRow { rows.append(current); current = []; acc = 0 }
        }
        if !current.isEmpty { rows.append(current) }
        return VStack(spacing: 2) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                let rowSum = row.reduce(0) { $0 + $1.size }
                let h = total > 0 ? size.height * (rowSum / total) : 0
                HStack(spacing: 2) {
                    ForEach(row) { leaf in
                        let w = rowSum > 0 ? size.width * (leaf.size / rowSum) : 0
                        ZStack {
                            Rectangle().fill(leaf.color.gradient)
                            if w > 36 && h > 18 {
                                Text(leaf.symbol).font(.caption2.bold()).foregroundStyle(.white).lineLimit(1)
                            }
                        }
                        .frame(width: max(2, w))
                    }
                }
                .frame(height: max(2, h))
            }
        }
    }
}

// MARK: - Drill-down donut (mirrors portfolio/AllocationPieChart.tsx aggregation)

struct AllocationDonutChart: View {
    let title: String
    let holdings: [Holding]
    let currency: String
    let bucketKey: String

    private var slices: [AllocationSlice] {
        var agg: [String: Double] = [:]
        for h in holdings { agg[PortfolioBucket.value(h, key: bucketKey), default: 0] += h.marketValue(currency: currency) ?? 0 }
        let total = agg.values.reduce(0, +)
        var top: [AllocationSlice] = []; var otherVal = 0.0
        for (name, value) in agg.sorted(by: { $0.value > $1.value }) {
            if total > 0 && value / total >= 0.02 { top.append(AllocationSlice(label: name, value: value)) }
            else { otherVal += value }
        }
        if otherVal > 0 { top.append(AllocationSlice(label: "Other", value: otherVal)) }
        return top
    }

    var body: some View {
        let data = slices
        let total = data.reduce(0) { $0 + $1.value }
        Section_(title: title) {
            if data.isEmpty {
                Text("No data.").foregroundStyle(.secondary)
            } else {
                Chart(data) { s in
                    SectorMark(angle: .value("Value", s.value), innerRadius: .ratio(0.6), angularInset: 1.5)
                        .foregroundStyle(by: .value("Label", s.label)).cornerRadius(3)
                }
                .frame(height: 220)
                ForEach(data.prefix(8)) { s in
                    HStack {
                        Text(s.label).lineLimit(1)
                        Spacer()
                        Text(Fmt.currency(s.value, code: currency)).monospacedDigit().foregroundStyle(.secondary)
                        Text(total > 0 ? String(format: "%.1f%%", s.value / total * 100) : "—")
                            .monospacedDigit().frame(width: 52, alignment: .trailing)
                    }
                    .font(.caption)
                }
            }
        }
    }
}
