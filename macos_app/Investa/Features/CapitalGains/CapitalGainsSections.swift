import SwiftUI
import Charts

private let cgDayFormatter: DateFormatter = {
    let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
}()
private func cgParseDay(_ s: String) -> Date? { cgDayFormatter.date(from: String(s.prefix(10))) }

/// "Aug 28" for an acquisition, with the year only when it falls outside the
/// current market year — a lot bought last December is still short-term, and
/// a bare "Dec 12" would read as this year's.
private func cgShortDay(_ iso: String) -> String {
    guard let day = MarketTime.calendarDay(iso), let today = MarketTime.today(timeZone: nil) else { return iso }
    return MarketTime.year(day) == MarketTime.year(today)
        ? MarketTime.shortDay(iso)
        : MarketTime.formatted(iso)
}

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
            #if os(iOS)
            VStack(alignment: .leading, spacing: 8) {
                SectionLabel(title: title)
                if let subtitle { Text(subtitle).appFont(.caption2).foregroundStyle(.secondary) }
                if let trailing { trailing }
            }
            #else
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    SectionLabel(title: title)
                    if let subtitle { Text(subtitle).appFont(.caption2).foregroundStyle(.secondary) }
                }
                Spacer(); if let trailing { trailing }
            }
            #endif
            Divider().opacity(0.6)
            content
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }
}

// MARK: - Unrealized tax view (mirrors UnrealizedTaxView.tsx)

struct UnrealizedTaxSection: View {
    @EnvironmentObject private var appState: AppState
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
        var out: [CLot] = []
        for h in holdings {
            for raw in h.raw["lots"]?.arrayValue ?? [] {
                guard let dStr = raw["Date"]?.stringValue, cgParseDay(dStr) != nil else { continue }
                // Counted on the market's clock, never the device's. Investa runs
                // on a Bangkok calendar that is up to a day ahead of New York, and
                // a day is the whole margin this card is about: it decides both
                // `isLT` and the countdown to the 365-day boundary.
                let heldDays = -(MarketTime.dayDiff(dStr, timeZone: nil) ?? 0)
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
            #if os(iOS)
            // Stack full-width on iPhone so all three tiles are fully visible
            // (matches the web's `grid-cols-1` on small screens); the large,
            // multi-digit currency values don't fit three-across on a phone.
            VStack(spacing: 12) {
                summaryTile("Short-term", st, "Taxed as ordinary income if sold today")
                summaryTile("Long-term", lt, "Taxed at LTCG rate if sold today")
                summaryTile("Total unrealized", st + lt, "\(all.count) tax lots")
            }
            #else
            HStack(spacing: 12) {
                summaryTile("Short-term", st, "Taxed as ordinary income if sold today")
                summaryTile("Long-term", lt, "Taxed at LTCG rate if sold today")
                summaryTile("Total unrealized", st + lt, "\(all.count) tax lots")
            }
            #endif
            harvestCard(harvest)
            if !ripening.isEmpty { ripeningCard(ripening) }
        }
    }

    private func summaryTile(_ label: String, _ value: Double, _ sub: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).appFont(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
            HStack(spacing: 4) {
                Text("\(value >= 0 ? "+" : "")\(Fmt.currency(value, code: currency))")
                    .appFont(.title3.bold()).foregroundStyle(value >= 0 ? .green : .red)
                Image(systemName: value >= 0 ? "arrow.up.right" : "arrow.down.right").appFont(.caption2).foregroundStyle(value >= 0 ? .green : .red)
            }
            Text(sub).appFont(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func harvestCard(_ harvest: [CLot]) -> some View {
        CGSection(title: "Tax-loss harvesting candidates",
                  subtitle: "Lots with unrealized loss > \(Fmt.currency(minHarvestLoss, code: currency)). Sorted by deepest loss.",
                  trailing: harvest.count > maxCandidates ? AnyView(Button("Show more (\(harvest.count - maxCandidates))") { maxCandidates += 10 }.appFont(.caption)) : nil) {
            if harvest.isEmpty {
                Text("No lots with significant unrealized losses — nothing to harvest right now.").foregroundStyle(.secondary)
            } else {
                #if os(iOS)
                LazyVStack(spacing: 12) {
                    ForEach(harvest.prefix(maxCandidates)) { c in
                        VStack(spacing: 8) {
                            HStack {
                                Button {
                                    appState.openStock(c.symbol)
                                } label: {
                                    Text(c.symbol).fontWeight(.bold).foregroundStyle(.indigo)
                                }
                                .buttonStyle(.plain)
                                termBadge(c.isLT)
                                if let acc = c.account {
                                    Text(acc).foregroundStyle(.secondary).appFont(.caption2)
                                }
                                Spacer()
                                Text("\(Fmt.currency(c.gain, code: currency))").foregroundStyle(.red).fontWeight(.bold)
                            }
                            HStack {
                                Text("Acquired \(MarketTime.formatted(c.date))").appFont(.caption2).foregroundStyle(.secondary)
                                Spacer()
                                Text("\(String(format: "%.1f", c.gainPct))%").foregroundStyle(.red).appFont(.caption.weight(.bold))
                            }
                            Divider()
                            HStack(spacing: 0) {
                                VStack(alignment: .leading, spacing: 1) {
                                    Text("Qty").appFont(.caption).foregroundStyle(.secondary)
                                    Text(Fmt.number(c.qty)).appFont(.caption.bold()).lineLimit(1)
                                }.frame(maxWidth: .infinity, alignment: .leading)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text("Cost").appFont(.caption).foregroundStyle(.secondary)
                                    Text(Fmt.currency(c.cost, code: currency)).appFont(.caption.bold()).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.75)
                                }.frame(maxWidth: .infinity, alignment: .leading)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text("Value").appFont(.caption).foregroundStyle(.secondary)
                                    Text(Fmt.currency(c.value, code: currency)).appFont(.caption.bold()).lineLimit(1).minimumScaleFactor(0.75)
                                }.frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .padding(12)
                        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
                        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
                    }
                }
                #else
                Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 6) {
                    GridRow {
                        Text("Symbol").gridColumnAlignment(.leading); Text("Acquired").gridColumnAlignment(.leading)
                        Text("Qty"); Text("Cost"); Text("Value"); Text("Loss"); Text("Term").gridColumnAlignment(.leading)
                    }.appFont(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                    Divider()
                    ForEach(harvest.prefix(maxCandidates)) { c in
                        GridRow {
                            HStack(spacing: 4) {
                                Button {
                                    appState.openStock(c.symbol)
                                } label: {
                                    Text(c.symbol).fontWeight(.bold).foregroundStyle(.indigo)
                                }
                                .buttonStyle(.plain)
                                if let acc = c.account {
                                    Text(acc).foregroundStyle(.secondary)
                                }
                            }
                            .gridColumnAlignment(.leading)
                            Text(MarketTime.formatted(c.date)).foregroundStyle(.secondary).gridColumnAlignment(.leading)
                            Text(Fmt.number(c.qty)); Text(Fmt.currency(c.cost, code: currency)).foregroundStyle(.secondary)
                            Text(Fmt.currency(c.value, code: currency))
                            Text("\(Fmt.currency(c.gain, code: currency)) (\(String(format: "%.1f", c.gainPct))%)").foregroundStyle(.red).fontWeight(.bold)
                            termBadge(c.isLT).gridColumnAlignment(.leading)
                        }.appFont(.caption).monospacedDigit()
                    }
                }
                #endif
                Label("Watch the wash-sale rule: selling at a loss and rebuying substantially the same security within 30 days disallows the deduction.",
                      systemImage: "info.circle").appFont(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    /// Lots days away from long-term treatment.
    ///
    /// Was one run-on line per lot — icon, ticker, "acquired 2025-08-28", gain,
    /// countdown — which at a large type size had nowhere to go: `SCBRMS&P500`
    /// wrapped mid-ticker onto a second line and took the row's alignment with
    /// it. A ticker is an identifier and must never break, so the row is now two
    /// lines by construction, the shape the harvesting card beside it already
    /// uses.
    private func ripeningCard(_ ripening: [CLot]) -> some View {
        CGSection(title: "Ripening to long-term within 30 days") {
            VStack(spacing: 10) {
                ForEach(ripening.prefix(8)) { c in ripeningRow(c) }
            }
            Text("Holding ≥30 more days converts these gains to LTCG treatment (typically lower tax).")
                .appFont(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func ripeningRow(_ c: CLot) -> some View {
        Button {
            appState.openStock(c.symbol)
        } label: {
            HStack(spacing: 10) {
                StockIcon(symbol: c.symbol, size: 26, scalesWithText: true)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 8) {
                        Text(c.symbol).fontWeight(.bold).foregroundStyle(.indigo)
                            // Shrinks, then truncates. Never wraps: half a
                            // ticker on a second line is not the ticker.
                            .lineLimit(1).minimumScaleFactor(0.6)
                        Spacer(minLength: 8)
                        Text("+\(Fmt.currency(c.gain, code: currency))")
                            .fontWeight(.bold).foregroundStyle(.green).monospacedDigit()
                            .lineLimit(1).minimumScaleFactor(0.7)
                    }
                    HStack(spacing: 6) {
                        Text("acquired \(cgShortDay(c.date))")
                            .foregroundStyle(.secondary).lineLimit(1)
                        if let acc = c.account, !acc.isEmpty {
                            Text("· \(acc)").foregroundStyle(.tertiary)
                                .lineLimit(1).layoutPriority(-1)
                        }
                        Spacer(minLength: 8)
                        countdownBadge(c.daysToLong)
                    }
                    .appFont(.caption2)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    /// The countdown carries the card's whole argument — how long the reader has
    /// to wait — so it is a badge rather than the last word of a sentence that
    /// runs off the right edge.
    private func countdownBadge(_ days: Int) -> some View {
        Text("\(days)d")
            .appFont(.caption2.weight(.bold)).monospacedDigit()
            .lineLimit(1)
            .padding(.horizontal, 7).padding(.vertical, 2)
            .background(Color.orange.opacity(0.15), in: Capsule())
            .foregroundStyle(.orange)
    }

    private func termBadge(_ isLT: Bool) -> some View {
        Text(isLT ? "LT" : "ST").appFont(.caption2.weight(.bold))
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background((isLT ? Color.green : .orange).opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(isLT ? .green : .orange)
    }
}

// MARK: - Realized-gains KPI strip (mirrors capital-gains/CapitalGainsKpiStrip.tsx)

struct CapitalGainsKpiStrip: View {
    @EnvironmentObject private var appState: AppState
    let gains: [CapitalGain]
    let currency: String
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    #endif

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
        let returnSub = returnPct.map { "\($0 >= 0 ? "+" : "")\(String(format: "%.1f", $0))% on cost" } ?? "on cost basis"
        let totalTone: Color = totalGain >= 0 ? .green : .red
        // Reused for iPad (regular width) and macOS — Total Realized stays inside
        // the balanced grid with the compact 1.10M value.
        let sevenTileRow = KpiRow(count: 7, minTileWidth: 140) {
            tile("Total Realized", cgCompact(totalGain, currency), returnSub, totalTone)
            tile("Win Rate", winRate.map { String(format: "%.0f%%", $0) } ?? "–",
                 "\(wins) W · \(losses) L\(flat > 0 ? " · \(flat) flat" : "")", (winRate ?? 0) >= 50 ? .green : .orange)
            tile("Avg Win", wins > 0 ? cgCompact(winSum / Double(wins), currency) : "–", "per winning sale", .green)
            tile("Avg Loss", losses > 0 ? cgCompact(lossSum / Double(losses), currency) : "–", "per losing sale", .red)
            tile("Sales", "\(gains.count)", "closing lots", .primary)
            tile("Proceeds", cgCompact(proceeds, currency), "gross sold", .primary)
            tile("Cost Basis", cgCompact(cost, currency), "of sold lots", .primary)
        }
        return VStack(spacing: 12) {
            #if os(iOS)
            if hSize == .compact {
                // iPhone: Total Realized spans the full width and shows the exact
                // amount (two decimals) rather than the compact 1.10M form.
                tile("Total Realized", Fmt.currency(totalGain, code: currency), returnSub, totalTone)
                KpiRow(count: 6, minTileWidth: 140) {
                    tile("Win Rate", winRate.map { String(format: "%.0f%%", $0) } ?? "–",
                         "\(wins) W · \(losses) L\(flat > 0 ? " · \(flat) flat" : "")", (winRate ?? 0) >= 50 ? .green : .orange)
                    tile("Avg Win", wins > 0 ? cgCompact(winSum / Double(wins), currency) : "–", "per winning sale", .green)
                    tile("Avg Loss", losses > 0 ? cgCompact(lossSum / Double(losses), currency) : "–", "per losing sale", .red)
                    tile("Sales", "\(gains.count)", "closing lots", .primary)
                    tile("Proceeds", cgCompact(proceeds, currency), "gross sold", .primary)
                    tile("Cost Basis", cgCompact(cost, currency), "of sold lots", .primary)
                }
            } else {
                sevenTileRow
            }
            #else
            sevenTileRow
            #endif
            if biggestWin != nil || biggestLoss != nil {
                #if os(iOS)
                VStack(spacing: 12) {
                    if let w = biggestWin { callout("Biggest Win", w, .green, "+") }
                    if let l = biggestLoss { callout("Biggest Loss", l, .red, "") }
                }
                #else
                HStack(spacing: 12) {
                    if let w = biggestWin { callout("Biggest Win", w, .green, "+") }
                    if let l = biggestLoss { callout("Biggest Loss", l, .red, "") }
                }
                #endif
            }
        }
    }
    private func tile(_ label: String, _ value: String, _ sub: String, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).appFont(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(value).appFont(.title3.bold()).foregroundStyle(tone).lineLimit(1)
            Text(sub).appFont(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
        .gridTile(alignment: .leading)
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
    }
    private func callout(_ label: String, _ v: (String, String, Double), _ tone: Color, _ prefix: String) -> some View {
        HStack {
            Image(systemName: tone == .green ? "chart.line.uptrend.xyaxis" : "chart.line.downtrend.xyaxis").foregroundStyle(tone)
            VStack(alignment: .leading, spacing: 1) {
                Text(label).appFont(.caption2.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                Button {
                    appState.openStock(v.0)
                } label: {
                    Text(v.0).fontWeight(.bold).foregroundStyle(.indigo)
                }
                .buttonStyle(.plain)
                Text(v.1).appFont(.caption2).foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(prefix)\(Fmt.currency(v.2, code: currency))").appFont(.title3.bold()).foregroundStyle(tone)
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
                  subtitle: selectedYear.map { "Filtered to \($0) — tap the year again to clear" }
                            ?? "Tap a year to filter the transactions below") {
            let data = rows
            if data.isEmpty {
                Text("No realized gains.").foregroundStyle(.secondary)
            } else {
                Chart {
                    // Clean zero baseline (mirrors the web CartesianGrid's 0 line).
                    RuleMark(y: .value("Zero", 0))
                        .foregroundStyle(Color.secondary.opacity(0.35))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                    ForEach(data, id: \.year) { row in
                        BarMark(x: .value("Year", row.year), y: .value("Gain", row.gain))
                            .foregroundStyle(barColor(row.year, row.gain))
                            .cornerRadius(4)
                    }
                }
                .frame(height: 240)
                .chartYScale(domain: yDomain(data))
                // Year labels on the x-axis (thinned when crowded), no vertical grid.
                .chartXAxis {
                    AxisMarks(preset: .aligned, values: labeledYears(data.map(\.year))) { value in
                        AxisValueLabel {
                            if let y = value.as(String.self) {
                                Text(y).appFont(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                // Horizontal dashed gridlines + compact currency labels, like the web.
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine().foregroundStyle(Color.secondary.opacity(0.2))
                        AxisValueLabel {
                            if let v = value.as(Double.self) {
                                Text(cgAxis(v)).appFont(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .chartHoverTooltip(data.map(\.year),
                                   onTap: { i in let y = data[i].year; selectedYear = (selectedYear == y) ? nil : y }) { i in
                    ChartTooltipContent(title: data[i].year,
                                        rows: [ChartTooltipRow(color: data[i].gain >= 0 ? .green : .red,
                                                               label: "Realized Gain",
                                                               value: Fmt.currency(data[i].gain, code: currency)),
                                               ChartTooltipRow(label: "Tap to filter transactions", value: "")])
                }
            }
        }
    }

    /// Bar color matching the web: emerald for gains / red for losses, a darker
    /// shade for the selected year, and a muted fill for the faded (unselected)
    /// years when a filter is active.
    private func barColor(_ year: String, _ gain: Double) -> Color {
        let selected = selectedYear == year
        if selectedYear != nil && !selected { return Color.secondary.opacity(0.25) }
        if gain >= 0 { return selected ? Color(hex: 0x059669) : Color(hex: 0x10B981) }
        return selected ? Color(hex: 0xDC2626) : Color(hex: 0xEF4444)
    }

    /// Compact currency axis label: 1M / 500K / 0 / -500K (matches the web's
    /// `Intl.NumberFormat` compact notation).
    private func cgAxis(_ v: Double) -> String {
        let a = abs(v), sign = v < 0 ? "-" : ""
        func trim(_ x: Double) -> String {
            let s = String(format: "%.1f", x)
            return s.hasSuffix(".0") ? String(s.dropLast(2)) : s
        }
        if a >= 1_000_000 { return "\(sign)\(trim(a / 1_000_000))M" }
        if a >= 1_000 { return "\(sign)\(trim(a / 1_000))K" }
        return "\(sign)\(Int(a))"
    }

    /// Fit the y-axis to the data while always anchoring at 0, so bars use the
    /// full height instead of floating in symmetric empty space.
    private func yDomain(_ data: [(year: String, gain: Double)]) -> ClosedRange<Double> {
        let vals = data.map(\.gain)
        let lo = min(0, vals.min() ?? 0)
        let hi = max(0, vals.max() ?? 0)
        guard hi > lo else { return -1...1 }
        let pad = (hi - lo) * 0.08
        return (lo < 0 ? lo - pad : 0)...(hi > 0 ? hi + pad : 0)
    }

    /// Thin x-axis year labels so they don't overlap on narrow (iPhone) widths;
    /// always keeps the most recent year.
    private func labeledYears(_ years: [String]) -> [String] {
        guard years.count > 8 else { return years }
        let step = Int(ceil(Double(years.count) / 8))
        var out = years.enumerated().filter { $0.offset % step == 0 }.map(\.element)
        if let last = years.last, out.last != last { out.append(last) }
        return out
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
    @EnvironmentObject private var appState: AppState
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
                #if os(iOS)
                LazyVStack(spacing: 12) {
                    ForEach(rows) { row in
                        iosCGRow(row)
                    }
                }
                #else
                Table(rows, sortOrder: $sortOrder) {
                    TableColumn("Date", value: \.date) { Text(MarketTime.formatted($0.date)).foregroundStyle(.secondary) }
                    TableColumn("Symbol", value: \.symbol) { row in
                        Button {
                            appState.openStock(row.symbol)
                        } label: {
                            Text(row.symbol).fontWeight(.bold).foregroundStyle(.indigo)
                        }
                        .buttonStyle(.plain)
                    }
                    TableColumn("Account", value: \.account) { Text($0.account).appFont(.caption).foregroundStyle(.secondary) }
                    TableColumn("Type", value: \.type) { Text($0.type).appFont(.caption).foregroundStyle(.secondary) }
                    TableColumn("Qty", value: \.quantity) { Text(Fmt.number($0.quantity)).monospacedDigit() }
                    TableColumn("Proceeds", value: \.proceeds) { Text(Fmt.currency($0.proceeds, code: currency)).monospacedDigit() }
                    TableColumn("Cost Basis", value: \.cost) { Text(Fmt.currency($0.cost, code: currency)).monospacedDigit() }
                    TableColumn("Realized Gain", value: \.gain) { r in
                        Text(Fmt.currency(r.gain, code: currency)).fontWeight(.medium).monospacedDigit().foregroundStyle(Fmt.tint(for: r.gain))
                    }
                    TableColumn("Gain %", value: \.gainPct) { r in
                        Text(Fmt.percent(r.gainPct, includeSign: true)).monospacedDigit().foregroundStyle(Fmt.tint(for: r.gainPct))
                    }
                }
                .frame(minHeight: 340)
                #endif
            }
        }
    }

    private func iosCGRow(_ r: CGRow) -> some View {
        VStack(spacing: 8) {
            HStack {
                Button {
                    appState.openStock(r.symbol)
                } label: {
                    Text(r.symbol).appFont(.headline).fontWeight(.bold).foregroundStyle(.indigo)
                }
                .buttonStyle(.plain)
                Text(r.type).appFont(.caption.weight(.bold)).padding(.horizontal, 6).padding(.vertical, 2).background(.quaternary, in: Capsule())
                Spacer()
                Text(Fmt.currency(r.gain, code: currency)).fontWeight(.medium).monospacedDigit().foregroundStyle(Fmt.tint(for: r.gain))
            }
            HStack {
                Text(MarketTime.formatted(r.date)).appFont(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(r.account).appFont(.caption2).foregroundStyle(.tertiary)
            }
            Divider()
            HStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Proceeds").appFont(.caption).foregroundStyle(.secondary)
                    Text(Fmt.currency(r.proceeds, code: currency)).appFont(.caption.bold()).monospacedDigit().lineLimit(1).minimumScaleFactor(0.75)
                }.frame(maxWidth: .infinity, alignment: .leading)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Cost").appFont(.caption).foregroundStyle(.secondary)
                    Text(Fmt.currency(r.cost, code: currency)).appFont(.caption.bold()).monospacedDigit().lineLimit(1).minimumScaleFactor(0.75)
                }.frame(maxWidth: .infinity, alignment: .leading)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Gain").appFont(.caption).foregroundStyle(.secondary)
                    Text(Fmt.percent(r.gainPct, includeSign: true)).appFont(.caption.bold()).monospacedDigit().foregroundStyle(Fmt.tint(for: r.gainPct)).lineLimit(1)
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
}
