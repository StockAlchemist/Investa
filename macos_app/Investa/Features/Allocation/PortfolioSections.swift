import SwiftUI
import Charts

/// Shared section chrome for the Portfolio tab (matches the Dashboard card style).
private struct Section_<Content: View>: View {
    let title: String
    var icon: String? = nil
    var trailing: AnyView? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 7) {
                if let icon { Image(systemName: icon).font(.caption.weight(.semibold)).foregroundStyle(Theme.brand) }
                Text(title).font(.caption.weight(.semibold)).tracking(0.8).textCase(.uppercase).foregroundStyle(.secondary)
                Spacer(minLength: 0)
                if let trailing { trailing }
            }
            Divider()
            content
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }
}

private func isCashSymbol(_ s: String) -> Bool {
    let u = s.uppercased(); return u == "$CASH" || u == "CASH" || u.hasPrefix("CASH (")
}

/// Exact web palettes (AllocationPieChart 10-color, PortfolioTreemap 12-color).
private let allocPalette: [Color] = [
    0x6366f1, 0x06b6d4, 0x10b981, 0xf59e0b, 0xef4444,
    0x8b5cf6, 0xec4899, 0x14b8a6, 0xf97316, 0x84cc16,
].map { Color(hex: $0) }
private let treemapPalette: [Color] = allocPalette + [Color(hex: 0x3b82f6), Color(hex: 0xa855f7)]

/// Squarified treemap (Bruls et al.) — lays sorted values into a rect, returning
/// one frame per value in input order (matches recharts' default Treemap look).
func squarifiedTreemap(_ values: [Double], in rect: CGRect) -> [CGRect] {
    var result = [CGRect](repeating: .zero, count: values.count)
    let total = values.reduce(0, +)
    guard total > 0, rect.width > 0, rect.height > 0 else { return result }
    let scale = (rect.width * rect.height) / total
    let areas = values.map { $0 * scale }
    func worst(_ row: ArraySlice<Double>, _ side: Double) -> Double {
        guard let rmin = row.min(), let rmax = row.max(), side > 0 else { return .infinity }
        let s = row.reduce(0, +); guard s > 0 else { return .infinity }
        let side2 = side * side, s2 = s * s
        return Swift.max(side2 * rmax / s2, s2 / (side2 * rmin))
    }
    var free = rect
    var i = 0
    while i < areas.count {
        let side = Swift.min(free.width, free.height)
        var j = i + 1
        while j < areas.count, worst(areas[i..<(j + 1)], side) <= worst(areas[i..<j], side) { j += 1 }
        let rowSum = areas[i..<j].reduce(0, +)
        let thickness = side > 0 ? rowSum / side : 0
        if free.width >= free.height {
            var y = free.minY
            for k in i..<j {
                let h = thickness > 0 ? areas[k] / thickness : 0
                result[k] = CGRect(x: free.minX, y: y, width: thickness, height: h); y += h
            }
            free = CGRect(x: free.minX + thickness, y: free.minY,
                          width: Swift.max(0, free.width - thickness), height: free.height)
        } else {
            var x = free.minX
            for k in i..<j {
                let w = thickness > 0 ? areas[k] / thickness : 0
                result[k] = CGRect(x: x, y: free.minY, width: w, height: thickness); x += w
            }
            free = CGRect(x: free.minX, y: free.minY + thickness,
                          width: free.width, height: Swift.max(0, free.height - thickness))
        }
        i = j
    }
    return result
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
        let largestTone: Color = (mt.largestPct ?? 0) >= 25 ? .orange : ((mt.largestPct ?? 0) >= 15 ? .primary : .up)
        let effTone: Color = (mt.effectiveN ?? 0) >= 10 ? .up : ((mt.effectiveN ?? 0) >= 5 ? .primary : .orange)
        return Section_(title: "Concentration", icon: "scope") {
            KpiRow(count: 6, minTileWidth: 140) {
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
            Text(value).font(.title3.bold()).monospacedDigit().foregroundStyle(tone).lineLimit(1)
            if let sub { Text(sub).font(.caption2).foregroundStyle(.secondary) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.white.opacity(0.05), lineWidth: 1))
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
        Section_(title: title, icon: "target", trailing: AnyView(headerButton)) {
            if rows.isEmpty {
                EmptyHint(text: "No holdings to bucket.", systemImage: "tray")
            } else if targetSum == 0 && !editing {
                EmptyHint(text: "Set target % per bucket to see drift from your plan.", systemImage: "target")
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
                    .font(.caption.bold()).foregroundStyle(abs(draftSum - 100) < 0.5 ? .up : .orange)
                Button { commit() } label: { Image(systemName: "checkmark") }.tint(.up)
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
        let tone: Color = r.target == 0 ? .secondary : (alert ? .down : (warn ? .orange : .up))
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
        Section_(title: "Rebalance Helper", icon: "arrow.left.arrow.right", trailing: AnyView(
            Picker("", selection: $dim) { ForEach(dims, id: \.key) { Text($0.label).tag($0.key) } }
                .pickerStyle(.menu).fixedSize())) {
            if !d.hasTargets {
                EmptyHint(text: "No targets set for \(dims.first { $0.key == dim }!.label.lowercased()). Set them in the drift card above to see suggested trades.",
                          systemImage: "slider.horizontal.3")
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
                                        .foregroundStyle(r.delta > 0 ? .up : .down)
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

// MARK: - Treemap (mirrors portfolio/PortfolioTreemap.tsx — squarified, clickable)

struct PortfolioTreemapView: View {
    let holdings: [Holding]
    let currency: String
    var onSelectSymbol: (String) -> Void = { _ in }
    @State private var dim = "Sector"
    @State private var hovered: String?

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var compact: Bool { hSize == .compact }
    #else
    private let compact = false
    #endif

    private var dims: [(key: String, label: String)] {
        [
            ("Sector", "Sector"), 
            ("quoteType", compact ? "Type" : "Asset Type"), 
            ("Country", "Country"),
        ]
    }

    private struct Leaf: Identifiable {
        let symbol: String; let size: Double; let group: String; let color: Color; let pct: Double
        var id: String { symbol }
    }

    private var total: Double { holdings.reduce(0) { $0 + max(0, $1.marketValue(currency: currency) ?? 0) } }

    private var leaves: [Leaf] {
        var groupTotals: [String: Double] = [:]
        var bySymbol: [String: (val: Double, group: String)] = [:]
        var tot = 0.0
        for h in holdings {
            let v = max(0, h.marketValue(currency: currency) ?? 0)
            guard v > 0 else { continue }
            tot += v
            let g = PortfolioBucket.value(h, key: dim)
            groupTotals[g, default: 0] += v
            if var cur = bySymbol[h.symbol] { cur.val += v; bySymbol[h.symbol] = cur } else { bySymbol[h.symbol] = (v, g) }
        }
        let ordered = groupTotals.sorted { $0.value > $1.value }.map { $0.key }
        var color: [String: Color] = [:]
        for (i, g) in ordered.enumerated() { color[g] = treemapPalette[i % treemapPalette.count] }
        return bySymbol.map { Leaf(symbol: $0.key, size: $0.value.val, group: $0.value.group,
                                   color: color[$0.value.group] ?? treemapPalette[0],
                                   pct: tot > 0 ? $0.value.val / tot * 100 : 0) }
            .sorted { $0.size > $1.size }
    }

    var body: some View {
        let items = leaves
        Section_(title: "Treemap", icon: "square.grid.3x3.fill", trailing: AnyView(dimToggle)) {
            if items.isEmpty {
                EmptyHint(text: "No holdings to map.", systemImage: "square.grid.3x3")
            } else {
                GeometryReader { geo in
                    let rects = squarifiedTreemap(items.map(\.size), in: CGRect(origin: .zero, size: geo.size))
                    ZStack(alignment: .topLeading) {
                        ForEach(Array(items.enumerated()), id: \.element.id) { idx, leaf in
                            tile(leaf, rects[idx])
                        }
                        if let h = hovered, let idx = items.firstIndex(where: { $0.symbol == h }) {
                            tooltip(items[idx]).position(tooltipPoint(rects[idx], in: geo.size))
                        }
                    }
                }
                .frame(height: 320)
                Text("\(items.count) holdings · \(Fmt.compact(total, code: currency)) · click a tile for detail")
                    .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
            }
        }
    }

    private var dimToggle: some View {
        HStack(spacing: 2) {
            ForEach(dims, id: \.key) { d in
                Button { dim = d.key } label: {
                    Text(d.label).font(.caption.weight(.semibold))
                        .lineLimit(1)
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(dim == d.key ? Theme.brand : Color.clear, in: RoundedRectangle(cornerRadius: 6))
                        .foregroundStyle(dim == d.key ? .white : .secondary)
                }.buttonStyle(.plain)
            }
        }
        .padding(2).background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
    }

    private func tile(_ leaf: Leaf, _ rect: CGRect) -> some View {
        let showLabel = rect.width > 44 && rect.height > 24
        let showPct = rect.width > 60 && rect.height > 38
        return Button { onSelectSymbol(leaf.symbol) } label: {
            ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 3).fill(leaf.color)
                if showLabel {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(leaf.symbol).font(.system(size: 13, weight: .bold)).foregroundStyle(.white).lineLimit(1)
                        if showPct {
                            Text(String(format: "%.1f%%", leaf.pct)).font(.system(size: 11))
                                .foregroundStyle(.white.opacity(0.8)).monospacedDigit()
                        }
                    }
                    .padding(.leading, 6).padding(.top, 5)
                }
            }
            // Inset by 2px so the card background shows through as a border (web stroke).
            .frame(width: max(0, rect.width - 2), height: max(0, rect.height - 2))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .position(x: rect.midX, y: rect.midY)
        .onHover { hovered = $0 ? leaf.symbol : (hovered == leaf.symbol ? nil : hovered) }
    }

    private func tooltip(_ leaf: Leaf) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 5) {
                RoundedRectangle(cornerRadius: 2).fill(leaf.color).frame(width: 8, height: 8)
                Text(leaf.symbol).font(.caption.bold())
            }
            Text(leaf.group).font(.caption2).foregroundStyle(.secondary)
            HStack(spacing: 4) {
                Text(Fmt.compact(leaf.size, code: currency)).fontWeight(.medium)
                Text("· \(String(format: "%.1f%%", leaf.pct))").foregroundStyle(.secondary)
            }.font(.caption2).monospacedDigit()
        }
        .padding(8).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary, lineWidth: 1))
        .fixedSize().allowsHitTesting(false)
    }

    private func tooltipPoint(_ rect: CGRect, in size: CGSize) -> CGPoint {
        CGPoint(x: min(max(rect.midX, 70), max(70, size.width - 70)), y: max(rect.minY - 2, 26))
    }
}

// MARK: - Performance heatmap (mirrors web HoldingsHeatmap.tsx)
// Finviz-style map of holdings: tiles sized by position value, colored by the
// selected period return, grouped by sector/account/country.

struct HoldingsHeatmapView: View {
    let holdings: [Holding]
    let currency: String
    let returns: [String: [String: Double]]
    var onSelectSymbol: (String) -> Void = { _ in }

    @State private var metricKey = "day"
    @State private var group = "Sector"
    @State private var sizeMode = "value"   // "value" | "equal"
    @State private var hovered: String?     // symbol under cursor/finger

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var compact: Bool { hSize == .compact }
    #else
    private let compact = false
    #endif

    private enum Src { case holding, spark, returnsKey }
    private struct Metric { let key: String; let label: String; let src: Src; let field: String; let period: String; let clamp: Double }
    private let metrics: [Metric] = [
        .init(key: "day",    label: "1D",      src: .holding,    field: "Day Change %",     period: "",    clamp: 3),
        .init(key: "7d",     label: "7D",      src: .spark,      field: "",                 period: "",    clamp: 5),
        .init(key: "1m",     label: "1M",      src: .returnsKey, field: "",                 period: "1m",  clamp: 8),
        .init(key: "3m",     label: "3M",      src: .returnsKey, field: "",                 period: "3m",  clamp: 15),
        .init(key: "6m",     label: "6M",      src: .returnsKey, field: "",                 period: "6m",  clamp: 25),
        .init(key: "ytd",    label: "YTD",     src: .returnsKey, field: "",                 period: "ytd", clamp: 25),
        .init(key: "1y",     label: "1Y",      src: .returnsKey, field: "",                 period: "1y",  clamp: 40),
        .init(key: "unreal", label: "Unreal.", src: .holding,    field: "Unreal. Gain %",   period: "",    clamp: 40),
        .init(key: "total",  label: "Total",   src: .holding,    field: "Total Return %",   period: "",    clamp: 50),
        .init(key: "irr",    label: "IRR",     src: .holding,    field: "IRR (%)",          period: "",    clamp: 40),
    ]
    private var metric: Metric { metrics.first { $0.key == metricKey } ?? metrics[0] }

    private let groups: [(key: String, label: String)] = [
        ("Sector", "Sector"), ("Account", "Account"), ("Country", "Country"), ("None", "Flat"),
    ]

    private struct Leaf: Identifiable {
        let symbol: String; var size: Double; var value: Double; let metricVal: Double?; let color: Color; let group: String
        var id: String { symbol }
    }
    private struct GroupBlock: Identifiable {
        let name: String; let leaves: [Leaf]; let weight: Double; let totalValue: Double
        var id: String { name }
    }

    // MARK: derived

    private func metricValue(_ h: Holding) -> Double? {
        switch metric.src {
        case .holding:
            return h.double(metric.field)
        case .spark:
            guard let arr = h.raw["sparkline_7d"]?.arrayValue, arr.count >= 2,
                  let first = arr.first?.doubleValue, let last = arr.last?.doubleValue, first != 0 else { return nil }
            return (last / first - 1) * 100
        case .returnsKey:
            return returns[h.symbol]?[metric.period]
        }
    }

    /// Diverging red→neutral→green; bright, vivid tiles (Finviz-style) — extremes
    /// are a bright red / bright green, easing toward neutral grey near 0%.
    private func heat(_ v: Double?) -> Color {
        guard let v, v.isFinite else { return Color(hue: 0.61, saturation: 0.05, brightness: 0.46) }
        let t = max(-1, min(1, v / metric.clamp))
        let mag = abs(t)
        let hue = (t >= 0 ? 145.0 : 2.0) / 360.0
        return Color(hue: hue, saturation: 0.45 + 0.45 * mag, brightness: 0.66 + 0.20 * mag)
    }

    private var blocks: [GroupBlock] {
        var byGroup: [String: [String: Leaf]] = [:]
        for h in holdings {
            let v = max(0, h.marketValue(currency: currency) ?? 0)
            guard v > 0 else { continue }
            let g = group == "None" ? "All" : PortfolioBucket.value(h, key: group)
            if var existing = byGroup[g]?[h.symbol] {
                existing.value += v
                byGroup[g]?[h.symbol] = existing
            } else {
                let mv = metricValue(h)
                byGroup[g, default: [:]][h.symbol] = Leaf(symbol: h.symbol, size: 0, value: v,
                                                          metricVal: mv, color: heat(mv), group: g)
            }
        }
        let equal = sizeMode == "equal"
        let result = byGroup.map { (name, leafMap) -> GroupBlock in
            var leaves = Array(leafMap.values)
            for i in leaves.indices { leaves[i].size = equal ? 1 : leaves[i].value }
            leaves.sort { $0.size > $1.size }
            let totalValue = leaves.reduce(0) { $0 + $1.value }
            let weight = equal ? Double(leaves.count) : totalValue
            return GroupBlock(name: name, leaves: leaves, weight: weight, totalValue: totalValue)
        }
        return result.sorted { $0.weight > $1.weight }
    }

    private var totalValue: Double { blocks.reduce(0) { $0 + $1.totalValue } }
    private var holdingCount: Int { blocks.reduce(0) { $0 + $1.leaves.count } }

    // MARK: body

    var body: some View {
        let bs = blocks
        Section_(title: "Performance Heatmap", icon: "square.grid.3x3.fill") {
            controls
            if holdingCount == 0 {
                EmptyHint(text: "No holdings to map.", systemImage: "square.grid.3x3")
            } else if group == "None" {
                // Match PortfolioTreemapView's height (320) so the squarified layout
                // — which depends on the area's aspect ratio — is identical.
                blockTreemap(bs.first?.leaves ?? [], showLabel: false, name: "")
                    .frame(height: 320)
            } else {
                let totalW = max(bs.reduce(0) { $0 + $1.weight }, 0.0001)
                VStack(spacing: 6) {
                    ForEach(bs) { b in
                        blockTreemap(b.leaves, showLabel: true, name: b.name)
                            .frame(height: max(58, 540 * b.weight / totalW))
                    }
                }
            }
            legend
        }
    }

    private var metricPicker: some View {
        Picker("", selection: $metricKey) {
            ForEach(metrics, id: \.key) { Text($0.label).tag($0.key) }
        }
        .labelsHidden().pickerStyle(.menu).fixedSize()
    }
    private var groupToggle: some View {
        segmented(options: groups.map { ($0.key, $0.label) }, selection: $group, brand: true, fill: compact)
    }
    private var sizeToggle: some View {
        segmented(options: [("value", "Value"), ("equal", "Equal")], selection: $sizeMode, brand: false)
    }

    @ViewBuilder private var controls: some View {
        if compact {
            // The four group labels won't fit beside everything else on a phone,
            // so stack into two rows and let the group toggle span the width.
            VStack(spacing: 8) {
                HStack(spacing: 8) { metricPicker; Spacer(minLength: 0); sizeToggle }
                groupToggle
            }
        } else {
            HStack(spacing: 8) { metricPicker; Spacer(minLength: 0); groupToggle; sizeToggle }
        }
    }

    private func segmented(options: [(String, String)], selection: Binding<String>, brand: Bool, fill: Bool = false) -> some View {
        HStack(spacing: 2) {
            ForEach(options, id: \.0) { opt in
                Button { selection.wrappedValue = opt.0 } label: {
                    Text(opt.1).font(.caption.weight(.semibold)).lineLimit(1).minimumScaleFactor(0.8)
                        .frame(maxWidth: fill ? .infinity : nil)
                        .padding(.horizontal, fill ? 4 : 8).padding(.vertical, 4)
                        .background(selection.wrappedValue == opt.0 ? (brand ? Theme.brand : Color.accentColor) : Color.clear,
                                    in: RoundedRectangle(cornerRadius: 6))
                        .foregroundStyle(selection.wrappedValue == opt.0 ? .white : .secondary)
                }.buttonStyle(.plain)
            }
        }
        .padding(2).frame(maxWidth: fill ? .infinity : nil).background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
    }

    private func blockTreemap(_ leaves: [Leaf], showLabel: Bool, name: String) -> some View {
        GeometryReader { geo in
            let rects = squarifiedTreemap(leaves.map(\.size), in: CGRect(origin: .zero, size: geo.size))
            ZStack(alignment: .topLeading) {
                ForEach(Array(leaves.enumerated()), id: \.element.id) { idx, leaf in
                    tile(leaf, rects[idx])
                }
                if showLabel {
                    HStack(spacing: 5) {
                        Text(name.uppercased()).font(.system(size: 10, weight: .bold)).tracking(0.4)
                        Text(String(format: "%.0f%%", totalValue > 0 ? leaves.reduce(0) { $0 + $1.value } / totalValue * 100 : 0))
                            .font(.system(size: 10)).foregroundStyle(.white.opacity(0.6)).monospacedDigit()
                    }
                    .foregroundStyle(.white.opacity(0.95))
                    .padding(.horizontal, 5).padding(.vertical, 2)
                    .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 4))
                    .padding(4).allowsHitTesting(false)
                }
                // Floating info card for the hovered/pressed tile in this block.
                if let h = hovered, let idx = leaves.firstIndex(where: { $0.symbol == h }) {
                    tooltip(leaves[idx]).position(tooltipPoint(rects[idx], in: geo.size))
                }
            }
        }
    }

    private func tile(_ leaf: Leaf, _ rect: CGRect) -> some View {
        let showLabel = rect.width > 40 && rect.height > 22
        let showPct = rect.width > 52 && rect.height > 36
        let pctText = leaf.metricVal.map { String(format: "%@%.1f%%", $0 >= 0 ? "+" : "", $0) }
        // Attach interaction to the frame-sized content, BEFORE .position. A
        // positioned view expands to fill its parent, so hover/tap modifiers added
        // after .position would fire across the whole block and the top-most (last)
        // tile would capture every event — making the tooltip show the wrong stock.
        let content = ZStack {
            RoundedRectangle(cornerRadius: 2).fill(leaf.color)
            if showLabel {
                VStack(spacing: 1) {
                    Text(leaf.symbol).font(.system(size: 11, weight: .heavy)).foregroundStyle(.white).lineLimit(1)
                    if showPct, let pctText {
                        Text(pctText).font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.92)).monospacedDigit()
                    }
                }
            }
        }
        .frame(width: max(0, rect.width - 2), height: max(0, rect.height - 2))
        .contentShape(Rectangle())
        .onTapGesture { onSelectSymbol(leaf.symbol) }

        return hoverable(content, symbol: leaf.symbol)
            .position(x: rect.midX, y: rect.midY)
    }

    /// Hover (macOS) / press-and-hold (iOS) reveals the info card, mirroring the
    /// web app's hover tooltip. Applied to the tile's own frame so the active
    /// region matches the tile (see note in `tile`).
    @ViewBuilder private func hoverable(_ content: some View, symbol: String) -> some View {
        #if os(macOS)
        content.onHover { inside in
            if inside { hovered = symbol }
            else if hovered == symbol { hovered = nil }
        }
        #else
        content.onLongPressGesture(minimumDuration: 0.18, maximumDistance: 40, pressing: { pressing in
            if pressing { hovered = symbol }
            else if hovered == symbol { hovered = nil }
        }, perform: {})
        #endif
    }

    /// Floating info card (mirrors web HeatTooltip). Solid background so the bright
    /// tile underneath never bleeds through and washes out the text.
    private func tooltip(_ leaf: Leaf) -> some View {
        let perf = leaf.metricVal.map { String(format: "%@%.2f%%", $0 >= 0 ? "+" : "", $0) } ?? "n/a"
        let perfColor: Color = leaf.metricVal == nil ? .secondary : (leaf.metricVal! >= 0 ? .green : .red)
        return VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 5) {
                RoundedRectangle(cornerRadius: 2).fill(leaf.color).frame(width: 9, height: 9)
                Text(leaf.symbol).font(.caption.bold())
                // In flat ("None") mode the bucket is the synthetic "All" — omit it.
                if group != "None" { Text(leaf.group).font(.caption2).foregroundStyle(.secondary) }
            }
            HStack(spacing: 5) {
                Text(Fmt.compact(leaf.value, code: currency)).fontWeight(.medium)
                Text("·").foregroundStyle(.secondary)
                Text(metric.label).foregroundStyle(.secondary)
                Text(perf).fontWeight(.bold).foregroundStyle(perfColor)
            }.font(.caption2).monospacedDigit()
        }
        .padding(.horizontal, 9).padding(.vertical, 6)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.quaternary, lineWidth: 1))
        .shadow(color: .black.opacity(0.18), radius: 8, y: 2)
        .fixedSize().allowsHitTesting(false)
    }

    private func tooltipPoint(_ rect: CGRect, in size: CGSize) -> CGPoint {
        CGPoint(x: min(max(rect.midX, 80), max(80, size.width - 80)), y: max(rect.minY - 4, 22))
    }

    private var legend: some View {
        HStack {
            Text("−\(Int(metric.clamp))%").font(.system(size: 10)).foregroundStyle(.secondary).monospacedDigit()
            LinearGradient(colors: [-1.0, -0.5, 0, 0.5, 1.0].map { heat($0 * metric.clamp) },
                           startPoint: .leading, endPoint: .trailing)
                .frame(width: 150, height: 10).clipShape(Capsule())
            Text("+\(Int(metric.clamp))%").font(.system(size: 10)).foregroundStyle(.secondary).monospacedDigit()
            Spacer()
            Text("\(holdingCount) holdings · \(Fmt.compact(totalValue, code: currency))")
                .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
        }
    }
}

// MARK: - Drill-down donut (mirrors portfolio/AllocationPieChart.tsx)

struct AllocationDonutChart: View {
    let title: String
    let holdings: [Holding]
    let currency: String
    let bucketKey: String

    @State private var selectedBucket: String?
    @State private var hoverBucket: String?
    @State private var selectedAngle: Double?

    private struct Slice: Identifiable {
        let name: String; let value: Double; let sourceBuckets: [String]; let color: Color
        var id: String { name }
    }
    private struct DrillRow: Identifiable {
        let symbol: String; let name: String; let pctOfBucket: Double; let pctOfPortfolio: Double
        var id: String { symbol }
    }

    private var slices: [Slice] {
        var agg: [String: Double] = [:]
        for h in holdings { agg[PortfolioBucket.value(h, key: bucketKey), default: 0] += max(0, h.marketValue(currency: currency) ?? 0) }
        let sorted = agg.map { (name: $0.key, value: $0.value) }.sorted { $0.value > $1.value }
        let tot = sorted.reduce(0) { $0 + $1.value }
        var top: [(name: String, value: Double, src: [String])] = []
        var otherVal = 0.0; var otherBuckets: [String] = []
        for item in sorted {
            if tot > 0 && item.value / tot >= 0.02 { top.append((item.name, item.value, [])) }
            else { otherVal += item.value; otherBuckets.append(item.name) }
        }
        if otherVal > 0 { top.append(("Other", otherVal, otherBuckets)) }
        return top.enumerated().map { i, t in Slice(name: t.name, value: t.value, sourceBuckets: t.src,
                                                    color: allocPalette[i % allocPalette.count]) }
    }

    private var total: Double { slices.reduce(0) { $0 + $1.value } }
    private var active: Slice? { hoverBucket.flatMap { hb in slices.first { $0.name == hb } } }

    var body: some View {
        let data = slices
        Section_(title: title, icon: "chart.pie", trailing: AnyView(clickHint)) {
            if data.isEmpty {
                EmptyHint(text: "No data.", systemImage: "chart.pie")
                    .frame(maxWidth: .infinity).frame(height: 400)
            } else {
                // The ring fills the card width (square); the legend + drill-down
                // scroll inside a fixed region so the four cards stay a stable 2×2 grid.
                VStack(spacing: 10) {
                    donut(data)
                    ScrollView {
                        VStack(alignment: .leading, spacing: 1) {
                            ForEach(data) { legendRow($0) }
                            if let sel = selectedBucket, let slice = data.first(where: { $0.name == sel }) {
                                drillDown(slice)
                            }
                        }
                    }
                    .frame(height: 160)
                }
            }
        }
    }

    private var clickHint: some View {
        HStack(spacing: 3) {
            Image(systemName: "cursorarrow.click").font(.system(size: 10))
            Text("Click slice").font(.system(size: 11))
        }.foregroundStyle(.secondary)
    }

    private func donut(_ data: [Slice]) -> some View {
        Chart(data) { s in
            // Constant radii: SwiftUI Charts normalizes sector radii to the frame,
            // so growing the active slice would rescale (and visibly shift) all the
            // others. Emphasize via opacity instead — no layout change on hover.
            SectorMark(angle: .value("Value", s.value),
                       innerRadius: .ratio(0.56),
                       outerRadius: .ratio(0.80),
                       angularInset: 1.5)
                .foregroundStyle(s.color)
                .opacity(opacity(s))
        }
        .chartLegend(.hidden)
        .chartAngleSelection(value: $selectedAngle)
        .chartBackground { _ in
            GeometryReader { geo in
                centerLabel.frame(width: geo.size.width, height: geo.size.height)
            }
        }
        // Square the chart to the card's full width so the ring fills it.
        .aspectRatio(1, contentMode: .fit)
        .frame(maxWidth: .infinity)
        .contentShape(Circle())
        // The pointer over the ring drives a STEADY hover highlight. Setting (not
        // toggling) avoids the on/off flicker as the angle updates continuously.
        .onChange(of: selectedAngle) { _, v in
            hoverBucket = v.flatMap { sliceName(forAngle: $0, in: data) }
        }
        // An explicit click on the hovered slice opens/closes its drill-down.
        .onTapGesture {
            guard let h = hoverBucket else { return }
            selectedBucket = (selectedBucket == h) ? nil : h
        }
    }

    private func opacity(_ s: Slice) -> Double {
        if selectedBucket == s.name { return 1 }
        if let h = hoverBucket { return h == s.name ? 1 : 0.35 }
        if selectedBucket != nil { return 0.4 }
        return 1
    }

    private var centerLabel: some View {
        VStack(spacing: 3) {
            if let a = active {
                Text(a.name).font(.caption).textCase(.uppercase).foregroundStyle(.secondary).lineLimit(1)
                Text(Fmt.compact(a.value, code: currency)).font(.title3.bold()).monospacedDigit()
                Text(String(format: "%.1f%%", total > 0 ? a.value / total * 100 : 0))
                    .font(.callout).foregroundStyle(.secondary).monospacedDigit()
            } else {
                Text("Total").font(.caption).textCase(.uppercase).foregroundStyle(.secondary)
                Text(Fmt.compact(total, code: currency)).font(.title3.bold()).monospacedDigit()
                Text("\(slices.count) \(slices.count == 1 ? "group" : "groups")").font(.callout).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(false)
    }

    private func legendRow(_ s: Slice) -> some View {
        let pct = total > 0 ? s.value / total * 100 : 0
        let isSel = selectedBucket == s.name
        let bg: AnyShapeStyle = isSel ? AnyShapeStyle(Theme.brand.opacity(0.15))
            : (hoverBucket == s.name ? AnyShapeStyle(Color.primary.opacity(0.06)) : AnyShapeStyle(.clear))
        return Button { selectedBucket = isSel ? nil : s.name } label: {
            HStack(spacing: 8) {
                RoundedRectangle(cornerRadius: 2).fill(s.color).frame(width: 10, height: 10)
                Text(s.name).lineLimit(1).fontWeight(isSel ? .semibold : .regular)
                Spacer()
                Text(String(format: "%.1f%%", pct)).foregroundStyle(.secondary).fontWeight(.medium).monospacedDigit()
                Text(Fmt.compact(s.value, code: currency)).foregroundStyle(.secondary.opacity(0.7)).monospacedDigit()
                    .frame(minWidth: 52, alignment: .trailing)
            }
            .font(.caption)
            .padding(.horizontal, 8).padding(.vertical, 5)
            .background(bg, in: RoundedRectangle(cornerRadius: 6))
            .overlay { if isSel { RoundedRectangle(cornerRadius: 6).strokeBorder(Theme.brand.opacity(0.3), lineWidth: 1) } }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hoverBucket = $0 ? s.name : (hoverBucket == s.name ? nil : hoverBucket) }
    }

    private func drillDown(_ slice: Slice) -> some View {
        let rows = drillRows(slice)
        return VStack(alignment: .leading, spacing: 8) {
            Divider().padding(.top, 4)
            HStack(spacing: 6) {
                Text("Holdings in").font(.system(size: 11, weight: .semibold)).textCase(.uppercase).foregroundStyle(.secondary)
                Text(slice.name).font(.caption.bold()).lineLimit(1)
                Text("· \(rows.count) \(rows.count == 1 ? "stock" : "stocks")").font(.system(size: 11)).foregroundStyle(.secondary)
                Spacer()
                Button("Close") { selectedBucket = nil }
                    .font(.system(size: 11, weight: .semibold)).textCase(.uppercase).buttonStyle(.plain).foregroundStyle(.secondary)
            }
            if rows.isEmpty {
                Text("No matching holdings.").font(.caption).foregroundStyle(.secondary)
            } else {
                ForEach(rows.prefix(12)) { drillRowView($0) }
                if rows.count > 12 {
                    Text("+ \(rows.count - 12) more").font(.system(size: 11)).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity)
                }
            }
        }
    }

    private func drillRowView(_ r: DrillRow) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(r.symbol).font(.caption.bold()).lineLimit(1)
                    if !r.name.isEmpty { Text(r.name).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1) }
                }
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(.quaternary)
                        Capsule().fill(Theme.brand).frame(width: geo.size.width * min(1, r.pctOfBucket / 100))
                    }
                }.frame(height: 5)
            }
            VStack(alignment: .trailing, spacing: 1) {
                Text(String(format: "%.1f%%", r.pctOfBucket)).font(.caption.bold()).monospacedDigit()
                Text("\(String(format: "%.1f%%", r.pctOfPortfolio)) of total")
                    .font(.system(size: 10)).foregroundStyle(.secondary).monospacedDigit()
            }
        }
    }

    private func drillRows(_ slice: Slice) -> [DrillRow] {
        let bucketSet: Set<String> = slice.sourceBuckets.isEmpty ? [slice.name] : Set(slice.sourceBuckets)
        var bySymbol: [String: (name: String, value: Double)] = [:]
        var order: [String] = []
        for h in holdings {
            let v = max(0, h.marketValue(currency: currency) ?? 0)
            guard v > 0, bucketSet.contains(PortfolioBucket.value(h, key: bucketKey)) else { continue }
            if var ex = bySymbol[h.symbol] { ex.value += v; bySymbol[h.symbol] = ex }
            else { bySymbol[h.symbol] = (h.string("Name") ?? "", v); order.append(h.symbol) }
        }
        let merged = order.compactMap { sym -> (String, String, Double)? in
            bySymbol[sym].map { (sym, $0.name, $0.value) }
        }.sorted { $0.2 > $1.2 }
        let bucketTotal = merged.reduce(0) { $0 + $1.2 }
        let portfolioTotal = total
        return merged.map { DrillRow(symbol: $0.0, name: $0.1,
                                     pctOfBucket: bucketTotal > 0 ? $0.2 / bucketTotal * 100 : 0,
                                     pctOfPortfolio: portfolioTotal > 0 ? $0.2 / portfolioTotal * 100 : 0) }
    }

    private func sliceName(forAngle v: Double, in data: [Slice]) -> String? {
        var cum = 0.0
        for s in data { cum += s.value; if v <= cum { return s.name } }
        return data.last?.name
    }
}
