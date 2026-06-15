import SwiftUI
import Charts

private let cgDayFormatter: DateFormatter = {
    let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
}()
private func cgParseDay(_ s: String) -> Date? { cgDayFormatter.date(from: String(s.prefix(10))) }

private func cgCompact(_ v: Double, _ code: String) -> String {
    let a = abs(v)
    if a >= 1_000_000 { return String(format: "%@%.2fM", v < 0 ? "-" : "", a / 1_000_000) }
    if a >= 10_000 { return String(format: "%@%.1fK", v < 0 ? "-" : "", a / 1_000) }
    return Fmt.currency(v, code: code)
}

private struct CGSection<Content: View>: View {
    let title: String
    var subtitle: String? = nil
    var trailing: AnyView? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline)
                    if let subtitle { Text(subtitle).font(.caption2).foregroundStyle(.secondary) }
                }
                Spacer(); if let trailing { trailing }
            }
            content
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
}

// MARK: - Unrealized tax view (mirrors UnrealizedTaxView.tsx)

struct UnrealizedTaxSection: View {
    let holdings: [Holding]
    let currency: String
    private let minHarvestLoss = 100.0
    @State private var maxCandidates = 10

    private struct CLot: Identifiable {
        let id = UUID(); let symbol: String; let account: String?
        let date: String; let qty: Double; let cost: Double; let value: Double; let gain: Double; let gainPct: Double
        let isLT: Bool; let daysToLong: Int
    }

    private var lots: [CLot] {
        let now = Date(); let cal = Calendar.current
        var out: [CLot] = []
        for h in holdings {
            for raw in h.raw["lots"]?.arrayValue ?? [] {
                guard let dStr = raw["Date"]?.stringValue, let d = cgParseDay(dStr) else { continue }
                let heldDays = cal.dateComponents([.day], from: d, to: now).day ?? 0
                out.append(CLot(symbol: h.symbol, account: h.account, date: String(dStr.prefix(10)),
                                qty: raw["Quantity"]?.doubleValue ?? 0,
                                cost: raw["Cost Basis"]?.doubleValue ?? 0,
                                value: raw["Market Value"]?.doubleValue ?? 0,
                                gain: raw["Unreal. Gain"]?.doubleValue ?? 0,
                                gainPct: raw["Unreal. Gain %"]?.doubleValue ?? 0,
                                isLT: heldDays >= 365, daysToLong: max(0, 365 - heldDays)))
            }
        }
        return out
    }

    var body: some View {
        let all = lots
        let st = all.filter { !$0.isLT }.reduce(0) { $0 + $1.gain }
        let lt = all.filter { $0.isLT }.reduce(0) { $0 + $1.gain }
        let harvest = all.filter { $0.gain < -minHarvestLoss }.sorted { $0.gain < $1.gain }
        let ripening = all.filter { !$0.isLT && $0.daysToLong > 0 && $0.daysToLong <= 30 && $0.gain > 0 }.sorted { $0.daysToLong < $1.daysToLong }
        return VStack(spacing: 12) {
            HStack(spacing: 12) {
                summaryTile("Short-term", st, "Taxed as ordinary income if sold today")
                summaryTile("Long-term", lt, "Taxed at LTCG rate if sold today")
                summaryTile("Total unrealized", st + lt, "\(all.count) tax lots")
            }
            harvestCard(harvest)
            if !ripening.isEmpty { ripeningCard(ripening) }
        }
    }

    private func summaryTile(_ label: String, _ value: Double, _ sub: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
            HStack(spacing: 4) {
                Text("\(value >= 0 ? "+" : "")\(Fmt.currency(value, code: currency))")
                    .font(.title3.bold()).foregroundStyle(value >= 0 ? .green : .red)
                Image(systemName: value >= 0 ? "arrow.up.right" : "arrow.down.right").font(.caption2).foregroundStyle(value >= 0 ? .green : .red)
            }
            Text(sub).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func harvestCard(_ harvest: [CLot]) -> some View {
        CGSection(title: "Tax-loss harvesting candidates",
                  subtitle: "Lots with unrealized loss > \(Fmt.currency(minHarvestLoss, code: currency)). Sorted by deepest loss.",
                  trailing: harvest.count > maxCandidates ? AnyView(Button("Show more (\(harvest.count - maxCandidates))") { maxCandidates += 10 }.font(.caption)) : nil) {
            if harvest.isEmpty {
                Text("No lots with significant unrealized losses — nothing to harvest right now.").foregroundStyle(.secondary)
            } else {
                Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 6) {
                    GridRow {
                        Text("Symbol").gridColumnAlignment(.leading); Text("Acquired").gridColumnAlignment(.leading)
                        Text("Qty"); Text("Cost"); Text("Value"); Text("Loss"); Text("Term").gridColumnAlignment(.leading)
                    }.font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                    Divider()
                    ForEach(harvest.prefix(maxCandidates)) { c in
                        GridRow {
                            (Text(c.symbol).fontWeight(.bold) + Text(c.account.map { "  \($0)" } ?? "").foregroundColor(.secondary))
                                .gridColumnAlignment(.leading)
                            Text(c.date).foregroundStyle(.secondary).gridColumnAlignment(.leading)
                            Text(Fmt.number(c.qty)); Text(Fmt.currency(c.cost, code: currency)).foregroundStyle(.secondary)
                            Text(Fmt.currency(c.value, code: currency))
                            Text("\(Fmt.currency(c.gain, code: currency)) (\(String(format: "%.1f", c.gainPct))%)").foregroundStyle(.red).fontWeight(.bold)
                            termBadge(c.isLT).gridColumnAlignment(.leading)
                        }.font(.caption).monospacedDigit()
                    }
                }
                Label("Watch the wash-sale rule: selling at a loss and rebuying substantially the same security within 30 days disallows the deduction.",
                      systemImage: "info.circle").font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private func ripeningCard(_ ripening: [CLot]) -> some View {
        CGSection(title: "Ripening to long-term within 30 days") {
            ForEach(ripening.prefix(8)) { c in
                HStack {
                    Image(systemName: "exclamationmark.circle").foregroundStyle(.orange)
                    Text(c.symbol).fontWeight(.bold)
                    Text("acquired \(c.date)").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Text("+\(Fmt.currency(c.gain, code: currency))").foregroundStyle(.green).monospacedDigit()
                    Text("\(c.daysToLong)d").foregroundStyle(.orange).fontWeight(.bold).monospacedDigit()
                }.font(.caption)
            }
            Text("Holding ≥30 more days converts these gains to LTCG treatment (typically lower tax).")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func termBadge(_ isLT: Bool) -> some View {
        Text(isLT ? "LT" : "ST").font(.caption2.weight(.bold))
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background((isLT ? Color.green : .orange).opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(isLT ? .green : .orange)
    }
}

// MARK: - Realized-gains KPI strip (mirrors capital-gains/CapitalGainsKpiStrip.tsx)

struct CapitalGainsKpiStrip: View {
    let gains: [CapitalGain]
    let currency: String

    var body: some View {
        var totalGain = 0.0, proceeds = 0.0, cost = 0.0, winSum = 0.0, lossSum = 0.0
        var wins = 0, losses = 0, flat = 0
        var biggestWin: (String, String, Double)?; var biggestLoss: (String, String, Double)?
        for g in gains {
            let gain = g.realizedGainDisplay
            totalGain += gain; proceeds += g.proceedsDisplay; cost += g.costBasisDisplay
            if gain > 0 { wins += 1; winSum += gain; if biggestWin == nil || gain > biggestWin!.2 { biggestWin = (g.symbol, g.date, gain) } }
            else if gain < 0 { losses += 1; lossSum += gain; if biggestLoss == nil || gain < biggestLoss!.2 { biggestLoss = (g.symbol, g.date, gain) } }
            else { flat += 1 }
        }
        let decided = wins + losses
        let winRate: Double? = decided > 0 ? Double(wins) / Double(decided) * 100 : nil
        let returnPct: Double? = cost != 0 ? totalGain / cost * 100 : nil
        return VStack(spacing: 12) {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 12)], spacing: 12) {
                tile("Total Realized", cgCompact(totalGain, currency),
                     returnPct.map { "\($0 >= 0 ? "+" : "")\(String(format: "%.1f", $0))% on cost" } ?? "on cost basis",
                     totalGain >= 0 ? .green : .red)
                tile("Win Rate", winRate.map { String(format: "%.0f%%", $0) } ?? "–",
                     "\(wins) W · \(losses) L\(flat > 0 ? " · \(flat) flat" : "")", (winRate ?? 0) >= 50 ? .green : .orange)
                tile("Avg Win", wins > 0 ? cgCompact(winSum / Double(wins), currency) : "–", "per winning sale", .green)
                tile("Avg Loss", losses > 0 ? cgCompact(lossSum / Double(losses), currency) : "–", "per losing sale", .red)
                tile("Sales", "\(gains.count)", "closing lots", .primary)
                tile("Proceeds", cgCompact(proceeds, currency), "gross sold", .primary)
                tile("Cost Basis", cgCompact(cost, currency), "of sold lots", .primary)
            }
            if biggestWin != nil || biggestLoss != nil {
                HStack(spacing: 12) {
                    if let w = biggestWin { callout("Biggest Win", w, .green, "+") }
                    if let l = biggestLoss { callout("Biggest Loss", l, .red, "") }
                }
            }
        }
    }
    private func tile(_ label: String, _ value: String, _ sub: String, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(value).font(.title3.bold()).foregroundStyle(tone).lineLimit(1)
            Text(sub).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
    }
    private func callout(_ label: String, _ v: (String, String, Double), _ tone: Color, _ prefix: String) -> some View {
        HStack {
            Image(systemName: tone == .green ? "chart.line.uptrend.xyaxis" : "chart.line.downtrend.xyaxis").foregroundStyle(tone)
            VStack(alignment: .leading, spacing: 1) {
                Text(label).font(.caption2.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                Text(v.0).fontWeight(.bold); Text(v.1).font(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(prefix)\(Fmt.currency(v.2, code: currency))").font(.title3.bold()).foregroundStyle(tone)
        }
        .padding(12).frame(maxWidth: .infinity)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
    }
}

// MARK: - Annual realized gains (clickable year filter)

struct AnnualRealizedGainsCard: View {
    let gains: [CapitalGain]
    let currency: String
    @Binding var selectedYear: String?

    private var rows: [(year: String, gain: Double)] {
        var byYear: [String: Double] = [:]
        for g in gains { byYear[String(g.date.prefix(4)), default: 0] += g.realizedGainDisplay }
        return byYear.sorted { $0.key < $1.key }.map { (year: $0.key, gain: $0.value) }
    }

    var body: some View {
        CGSection(title: "Annual Realized Gains",
                  subtitle: selectedYear.map { "Filtered to \($0) — tap the year again to clear" }) {
            let data = rows
            if data.isEmpty {
                Text("No realized gains.").foregroundStyle(.secondary)
            } else {
                Chart(data, id: \.year) { row in
                    BarMark(x: .value("Year", row.year), y: .value("Gain", row.gain))
                        .foregroundStyle(row.gain >= 0 ? Color.green : .red)
                        .opacity(selectedYear == nil || selectedYear == row.year ? 1 : 0.4)
                }
                .frame(height: 240)
                .chartOverlay { proxy in
                    GeometryReader { geo in
                        Rectangle().fill(.clear).contentShape(Rectangle())
                            .onTapGesture { location in
                                if let plotFrame = proxy.plotFrame,
                                   let year: String = proxy.value(atX: location.x - geo[plotFrame].origin.x) {
                                    selectedYear = (selectedYear == year) ? nil : year
                                }
                            }
                    }
                }
                // Year chips as an explicit fallback for selecting.
                HStack {
                    ForEach(data, id: \.year) { row in
                        Button { selectedYear = selectedYear == row.year ? nil : row.year } label: {
                            Text(row.year).font(.caption.weight(.medium))
                                .padding(.horizontal, 8).padding(.vertical, 3)
                                .background(selectedYear == row.year ? Color.accentColor.opacity(0.2) : Color.gray.opacity(0.15), in: Capsule())
                                .foregroundStyle(selectedYear == row.year ? Color.accentColor : Color.gray)
                        }.buttonStyle(.plain)
                    }
                }
            }
        }
    }
}

// MARK: - Realized-gain transactions table

struct CGRow: Identifiable {
    let id: String; let date: String; let symbol: String; let account: String; let type: String
    let quantity: Double; let proceeds: Double; let cost: Double; let gain: Double; let gainPct: Double
    init(_ g: CapitalGain) {
        id = g.id; date = g.date; symbol = g.symbol; account = g.account; type = g.type
        quantity = g.quantity; proceeds = g.proceedsDisplay; cost = g.costBasisDisplay; gain = g.realizedGainDisplay
        gainPct = g.costBasisDisplay != 0 ? g.realizedGainDisplay / g.costBasisDisplay * 100 : 0
    }
}

struct RealizedGainsTable: View {
    let gains: [CapitalGain]
    let currency: String
    @State private var search = ""
    @State private var sortOrder = [KeyPathComparator(\CGRow.date, order: .reverse)]

    private var rows: [CGRow] {
        let q = search.trimmingCharacters(in: .whitespaces).lowercased()
        return gains.map(CGRow.init)
            .filter { q.isEmpty || $0.symbol.lowercased().contains(q) || $0.account.lowercased().contains(q) }
            .sorted(using: sortOrder)
    }

    var body: some View {
        CGSection(title: "Realized Gain Transactions", trailing: AnyView(
            TextField("Search symbol or account…", text: $search).textFieldStyle(.roundedBorder).frame(width: 220))) {
            if rows.isEmpty {
                Text("No realized gains.").foregroundStyle(.secondary)
            } else {
                Table(rows, sortOrder: $sortOrder) {
                    TableColumn("Date", value: \.date) { Text($0.date).foregroundStyle(.secondary) }
                    TableColumn("Symbol", value: \.symbol) { Text($0.symbol).fontWeight(.medium) }
                    TableColumn("Account", value: \.account) { Text($0.account).font(.caption).foregroundStyle(.secondary) }
                    TableColumn("Type", value: \.type) { Text($0.type).font(.caption).foregroundStyle(.secondary) }
                    TableColumn("Qty", value: \.quantity) { Text(Fmt.number($0.quantity)).monospacedDigit() }
                    TableColumn("Proceeds", value: \.proceeds) { Text(Fmt.currency($0.proceeds, code: currency)).monospacedDigit() }
                    TableColumn("Cost Basis", value: \.cost) { Text(Fmt.currency($0.cost, code: currency)).monospacedDigit() }
                    TableColumn("Realized Gain", value: \.gain) { r in
                        Text(Fmt.currency(r.gain, code: currency)).fontWeight(.medium).monospacedDigit().foregroundStyle(Fmt.tint(for: r.gain))
                    }
                    TableColumn("Gain %", value: \.gainPct) { r in
                        Text(Fmt.percent(r.gainPct)).monospacedDigit().foregroundStyle(Fmt.tint(for: r.gainPct))
                    }
                }
                .frame(minHeight: 340)
            }
        }
    }
}
