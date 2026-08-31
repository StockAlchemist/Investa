import SwiftUI
import Charts

private let isoDayFormatter: DateFormatter = {
    let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f
}()
private func parseDay(_ s: String) -> Date? { isoDayFormatter.date(from: String(s.prefix(10))) }

private func compactCurrency(_ v: Double, _ code: String) -> String {
    let a = abs(v)
    if a >= 1_000_000 { return String(format: "%@%.2fM", v < 0 ? "-" : "", a / 1_000_000) }
    if a >= 10_000 { return String(format: "%@%.1fK", v < 0 ? "-" : "", a / 1_000) }
    return Fmt.currency(v, code: code)
}

private struct ISection<Content: View>: View {
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
            Spacer(minLength: 0)
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }
}

// MARK: - Income KPI strip (mirrors income/IncomeKpiStrip.tsx)

struct IncomeKpiStrip: View {
    let dividends: [Dividend]
    let currency: String
    let expectedDividends: Double?
    let dividendYield: Double?

    private struct M { var ytd = 0.0; var priorYtd = 0.0; var trailing12m = 0.0; var trailing12mTax = 0.0; var totalTax = 0.0 }

    private var m: M {
        var out = M()
        let now = Date(); let cal = MarketTime.localCalendar
        let year = cal.component(.year, from: now)
        let priorCutoff = cal.date(byAdding: .year, value: -1, to: now)!
        let oneYearAgo = cal.date(byAdding: .year, value: -1, to: now)!
        for div in dividends {
            guard let d = parseDay(div.date) else { continue }
            let gross = div.amountDisplay; let tax = div.taxDisplay ?? 0
            out.totalTax += tax
            let y = cal.component(.year, from: d)
            if y == year { out.ytd += gross }
            else if y == year - 1 && d <= priorCutoff { out.priorYtd += gross }
            if d >= oneYearAgo { out.trailing12m += gross; out.trailing12mTax += tax }
        }
        return out
    }

    var body: some View {
        let mt = m
        let yoyPct: Double? = mt.priorYtd > 0 ? (mt.ytd - mt.priorYtd) / mt.priorYtd * 100 : nil
        let taxEff: Double? = mt.trailing12m > 0 ? (mt.trailing12m - mt.trailing12mTax) / mt.trailing12m * 100 : nil
        let tileCount = 3 + (expectedDividends != nil ? 1 : 0) + (dividendYield != nil ? 1 : 0) + (taxEff != nil ? 1 : 0)
        // No heading, as the web card has none: this is the first card under a
        // page already titled "Income", and an `ISection(title: "Income")` put
        // that word on screen twice in a row. The tiles name themselves.
        return VStack(alignment: .leading, spacing: 12) {
            KpiRow(count: tileCount, minTileWidth: 150) {
                tile("YTD Received", compactCurrency(mt.ytd, currency),
                     yoyPct.map { "\($0 >= 0 ? "+" : "")\(String(format: "%.1f", $0))% YoY" } ?? "vs prior YTD",
                     yoyPct.map { $0 >= 0 ? Color.green : .red } ?? .secondary)
                tile("Trailing 12M", compactCurrency(mt.trailing12m, currency), "received in last year", .primary)
                tile("Avg Monthly", compactCurrency(mt.trailing12m / 12, currency), "trailing 12M ÷ 12", .primary)
                if let e = expectedDividends { tile("Expected 12M", compactCurrency(e, currency), "forward indicated rate", .green) }
                if let y = dividendYield { tile("Annual Yield", String(format: "%.2f%%", y), "on current portfolio", .primary) }
                if let te = taxEff {
                    tile("Tax Efficiency", String(format: "%.0f%%", te), "\(compactCurrency(mt.totalTax, currency)) paid · 12M",
                         te >= 85 ? .green : (te >= 70 ? .orange : .red))
                }
            }
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }
    private func tile(_ label: String, _ value: String, _ sub: String, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            SectionLabel(title: label)
                .minimumScaleFactor(0.7)
            // A figure shrinks before it truncates — a clipped currency amount
            // is a wrong number, not a tight one.
            Text(value).appFont(.title3.bold()).foregroundStyle(tone)
                .lineLimit(1).minimumScaleFactor(0.6)
            Text(sub).appFont(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .gridTile()
    }
}

// MARK: - Income projector (stacked by symbol — mirrors IncomeProjector.tsx)

struct IncomeProjectorCard: View {
    let income: [ProjectedIncome]
    let currency: String

    private struct Seg: Identifiable { let id = UUID(); let month: String; let symbol: String; let amount: Double }
    private var segs: [Seg] {
        income.flatMap { row in row.segments.map { Seg(month: row.month, symbol: $0.symbol, amount: $0.amount) } }
    }
    @State private var width: CGFloat = 0
    @Environment(\.appFontScale) private var fontScale
    @Environment(\.dynamicTypeSize) private var typeSize

    private var projectorMonths: [String] {
        var seen = Set<String>(); var out: [String] = []
        for s in segs where !seen.contains(s.month) { seen.insert(s.month); out.append(s.month) }
        return out
    }

    /// Whether the twelve months are named by their initials. On a phone they
    /// are: `Jan Feb Mar` twelve times over wants ~310pt, more than the card
    /// has once the y axis is paid for. See `ChartAxis.prefersMonthInitials`.
    private var monthInitials: Bool {
        ChartAxis.prefersMonthInitials(
            count: income.count, width: width, scale: fontScale, typeSize: typeSize
        )
    }

    /// Every month that fits. A band scale draws a mark for every category it
    /// is given, and labels that don't fit overprint into a smear rather than
    /// truncating — so the ones that can't be shown are dropped here. Only
    /// reached where even the initials are too wide to run twelve deep; the
    /// tooltip names the month in full either way.
    private var axisMonths: [String] {
        ChartAxis.ticks(
            income.map(\.month),
            count: ChartAxis.labelCapacity(
                width: width,
                labelWidth: ChartAxis.labelWidth(monthInitials ? "M" : "Mar",
                                                 scale: fontScale, typeSize: typeSize),
                unmeasured: 6,
                cap: 12
            )
        )
    }

    /// `Jan Feb Mar` where there is room, `J F M` where there isn't, with
    /// `.fixedSize()` so the band-scale axis doesn't clip either down to "J…".
    private var monthAxis: some AxisContent {
        AxisMarks(values: axisMonths) { value in
            AxisGridLine()
            AxisValueLabel {
                if let s = value.as(String.self) {
                    // The bucket's own label is "Sep 2026"; the initial comes
                    // from the month it names, not from cutting the string.
                    Text(monthInitials ? MarketTime.monthInitial(yearMonth(s)) : String(s.prefix(3)))
                        .appFont(.caption2)
                        .fixedSize()
                }
            }
        }
    }

    /// The `yyyy-MM` key behind a bucket's display label.
    private func yearMonth(_ label: String) -> String {
        income.first { $0.month == label }?.yearMonth ?? label
    }

    /// One legend entry per paying symbol, which a forty-payer portfolio turns
    /// into a wall of chips taller than the chart it explains. Past what a
    /// couple of rows can carry, the tooltip is the legend — it already lists
    /// the symbols for the month under the finger, largest first.
    private var legendFits: Bool {
        let symbols = Set(segs.map(\.symbol)).count
        guard width > 0 else { return symbols <= 6 }
        return symbols <= max(3, Int(width / 70)) * 2
    }

    var body: some View {
        ISection(title: "Projected 12M Income") {
            if income.isEmpty {
                Text("No projected income.").foregroundStyle(.secondary)
            } else if segs.isEmpty {
                // No per-symbol breakdown — fall back to the monthly total.
                Chart(income) { BarMark(x: .value("Month", $0.month), y: .value("Income", $0.value)).foregroundStyle(.green) }
                    .chartXAxis { monthAxis }
                    .chartHoverTooltip(income.map(\.month)) { i in
                        ChartTooltipContent(title: income[i].month,
                                            rows: [ChartTooltipRow(color: .green, label: "Income",
                                                                   value: Fmt.currency(income[i].value, code: currency))])
                    }
                    .frame(height: 280)
            } else {
                Chart(segs) { s in
                    BarMark(x: .value("Month", s.month), y: .value("Income", s.amount))
                        .foregroundStyle(by: .value("Symbol", s.symbol))
                }
                .chartXAxis { monthAxis }
                .chartLegend(legendFits ? .visible : .hidden)
                .chartHoverTooltip(projectorMonths) { i in
                    let month = projectorMonths[i]
                    let rows = segs.filter { $0.month == month }.sorted { $0.amount > $1.amount }
                    let total = rows.reduce(0) { $0 + $1.amount }
                    var out = rows.prefix(8).map {
                        ChartTooltipRow(label: $0.symbol, value: Fmt.currency($0.amount, code: currency))
                    }
                    out.append(ChartTooltipRow(label: "Total", value: Fmt.currency(total, code: currency)))
                    return ChartTooltipContent(title: month, rows: out)
                }
                .frame(height: 280)
            }
        }
        // Must be the *offered* width. See `readingContainerWidth`.
        .readingContainerWidth { width = $0 }
    }
}

// MARK: - Dividend calendar (3M / 1Y toggle — mirrors DividendCalendar.tsx)

struct DividendCalendarSection: View {
    let events: [DividendEvent]
    let currency: String
    var onSelect: (String) -> Void = { _ in }
    @State private var horizon = "3m"

    /// "Sep 10", with the year only when the payment falls outside the current
    /// year on its own exchange — the 1-year horizon runs into January, where
    /// a bare "Jan 15" would read as the past. ISO strings cost roughly twice
    /// the width for no more meaning in a list that is already sorted by date.
    private func eventDay(_ iso: String, _ zone: String?) -> String {
        guard let day = MarketTime.calendarDay(iso) else { return iso }
        guard let today = MarketTime.today(timeZone: zone),
              MarketTime.year(day) == MarketTime.year(today) else {
            return MarketTime.formatted(iso)
        }
        return MarketTime.shortDay(iso)
    }

    private var filtered: [DividendEvent] {
        // The horizon runs from today on each payment's own exchange, not from the
        // device clock (see `MarketTime`).
        let months = horizon == "3m" ? 3 : 12
        return events
            .filter { MarketTime.isWithin($0.dividendDate, months: months, timeZone: $0.marketTimezone) }
            .sorted { $0.dividendDate < $1.dividendDate }
    }

    var body: some View {
        ISection(title: "Dividend Calendar", trailing: AnyView(
            Picker("", selection: $horizon) { Text("3 Months").tag("3m"); Text("1 Year").tag("1y") }
                .pickerStyle(.segmented).fixedSize())) {
            if filtered.isEmpty {
                Text("No upcoming dividend events found.").foregroundStyle(.secondary)
            } else {
                ForEach(filtered) { ev in
                    Button { onSelect(ev.symbol) } label: {
                        HStack(spacing: 8) {
                            StockIcon(symbol: ev.symbol, size: 26, scalesWithText: true)
                            Text(ev.symbol).fontWeight(.bold)
                                .lineLimit(1).minimumScaleFactor(0.8)
                            if ev.status == "estimated" {
                                Label("est.", systemImage: "clock").appFont(.caption2).foregroundStyle(.orange)
                                    .lineLimit(1)
                            } else {
                                Image(systemName: "checkmark.seal.fill").appFont(.caption2).foregroundStyle(.green)
                            }
                            Spacer(minLength: 8)
                            VStack(alignment: .trailing, spacing: 1) {
                                // An estimated payment has no ex-date yet. The
                                // label still holds its line so the two-line
                                // block keeps a stable height down the list,
                                // but it says so rather than trailing off after
                                // "Ex" with nothing behind it.
                                Text(ev.exDividendDate.isEmpty
                                     ? "Ex \u{2014}"
                                     : "Ex \(eventDay(ev.exDividendDate, ev.marketTimezone))")
                                Text("Pay \(eventDay(ev.dividendDate, ev.marketTimezone))")
                            }
                            .appFont(.caption2).foregroundStyle(.secondary)
                            .lineLimit(1)
                            Text(Fmt.currency(ev.amount, code: currency)).fontWeight(.bold).foregroundStyle(.green)
                                .lineLimit(1).minimumScaleFactor(0.7)
                                .frame(minWidth: 90, alignment: .trailing)
                        }
                    }.buttonStyle(.plain)
                    Divider()
                }
            }
        }
    }
}

// MARK: - Top payers (12M / all — mirrors income/TopPayers.tsx)

struct TopPayersCard: View {
    let dividends: [Dividend]
    let currency: String
    var onSelect: (String) -> Void = { _ in }
    @State private var window = "12m"

    private struct Row: Identifiable { let symbol: String; let gross: Double; let count: Int; var pct: Double; var id: String { symbol } }

    private var rows: [Row] {
        let cutoff = MarketTime.localCalendar.date(byAdding: .year, value: -1, to: Date())!
        var bySym: [String: (gross: Double, count: Int)] = [:]
        for d in dividends {
            if window == "12m", let dt = parseDay(d.date), dt < cutoff { continue }
            var c = bySym[d.symbol] ?? (0, 0); c.gross += d.amountDisplay; c.count += 1; bySym[d.symbol] = c
        }
        let arr = bySym.map { (sym: $0.key, gross: $0.value.gross, count: $0.value.count) }.sorted { $0.gross > $1.gross }
        let total = arr.reduce(0) { $0 + $1.gross }
        return arr.prefix(10).map { Row(symbol: $0.sym, gross: $0.gross, count: $0.count, pct: total > 0 ? $0.gross / total * 100 : 0) }
    }

    var body: some View {
        ISection(title: "Top Dividend Payers", trailing: AnyView(
            Picker("", selection: $window) { Text("12M").tag("12m"); Text("All time").tag("all") }
                .pickerStyle(.segmented).fixedSize())) {
            let data = rows
            if data.isEmpty { Text("No dividends.").foregroundStyle(.secondary) }
            ForEach(Array(data.enumerated()), id: \.element.id) { idx, row in
                Button { onSelect(row.symbol) } label: {
                    HStack(spacing: 10) {
                        Text("\(idx + 1)").appFont(.caption2.bold()).foregroundStyle(.secondary).frame(width: 18, alignment: .trailing)
                        StockIcon(symbol: row.symbol, size: 26, scalesWithText: true)
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text(row.symbol).fontWeight(.bold)
                                Text("· \(row.count) \(row.count == 1 ? "pay" : "pays")").appFont(.caption2).foregroundStyle(.secondary)
                            }
                            GeometryReader { g in
                                ZStack(alignment: .leading) {
                                    Capsule().fill(.quaternary)
                                    Capsule().fill(.green).frame(width: g.size.width * min(1, row.pct / 100))
                                }
                            }.frame(height: 6)
                        }
                        VStack(alignment: .trailing, spacing: 1) {
                            Text(Fmt.currency(row.gross, code: currency)).appFont(.caption.bold()).foregroundStyle(.green)
                            Text(String(format: "%.1f%% of top", row.pct)).appFont(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }.buttonStyle(.plain)
            }
        }
    }
}

// MARK: - By account (12M / all — mirrors income/ByAccount.tsx)

struct ByAccountCard: View {
    let dividends: [Dividend]
    let currency: String
    @State private var window = "12m"

    private struct Row: Identifiable { let account: String; let gross: Double; var id: String { account } }
    private var rows: [Row] {
        let cutoff = MarketTime.localCalendar.date(byAdding: .year, value: -1, to: Date())!
        var byAcc: [String: Double] = [:]
        for d in dividends {
            if window == "12m", let dt = parseDay(d.date), dt < cutoff { continue }
            byAcc[d.account.isEmpty ? "—" : d.account, default: 0] += d.amountDisplay
        }
        return byAcc.map { Row(account: $0.key, gross: $0.value) }.sorted { $0.gross > $1.gross }
    }

    var body: some View {
        let data = rows
        let total = data.reduce(0) { $0 + $1.gross }
        ISection(title: "By Account", trailing: AnyView(
            Picker("", selection: $window) { Text("12M").tag("12m"); Text("All time").tag("all") }
                .pickerStyle(.segmented).fixedSize())) {
            if data.isEmpty {
                Text(window == "12m" ? "No dividends in the last year." : "No dividends.")
                    .foregroundStyle(.secondary)
            }
            ForEach(data) { acc in
                let pct = total > 0 ? acc.gross / total * 100 : 0
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(acc.account).fontWeight(.bold).lineLimit(1).minimumScaleFactor(0.8)
                        Spacer()
                        Text(String(format: "%.1f%%", pct)).appFont(.caption2).foregroundStyle(.secondary)
                        Text(Fmt.currency(acc.gross, code: currency)).appFont(.caption.bold()).foregroundStyle(.green)
                            .lineLimit(1).minimumScaleFactor(0.7)
                    }
                    GeometryReader { g in
                        ZStack(alignment: .leading) {
                            Capsule().fill(.quaternary)
                            Capsule().fill(.cyan).frame(width: g.size.width * min(1, pct / 100))
                        }
                    }.frame(height: 6)
                }
            }
        }
    }
}

// MARK: - Annual dividends (bar + YoY — mirrors Dividend.tsx)

struct AnnualDividendsCard: View {
    let dividends: [Dividend]
    let currency: String
    @Binding var selectedYear: String?

    private struct Row: Identifiable { let year: String; let amount: Double; let yoy: Double?; var id: String { year } }
    private var rows: [Row] {
        var byYear: [String: Double] = [:]
        for d in dividends { byYear[String(d.date.prefix(4)), default: 0] += d.amountDisplay }
        let sorted = byYear.sorted { $0.key < $1.key }
        return sorted.enumerated().map { i, e in
            let prior = i > 0 ? sorted[i-1].value : 0
            return Row(year: e.key, amount: e.value, yoy: i > 0 && prior > 0 ? (e.value - prior) / prior * 100 : nil)
        }
    }

    /// ~6 evenly-spaced years to label (anchored to the latest), so labels don't
    /// crowd into "2(2(2…". `.fixedSize()` on each then prevents the band-scale
    /// axis from clipping "'01" down to "'0".
    private func axisYears(_ data: [Row], target: Int = 6) -> [String] {
        guard data.count > target else { return data.map(\.year) }
        let step = max(1, Int((Double(data.count) / Double(target)).rounded()))
        return data.enumerated().compactMap { i, r in
            (data.count - 1 - i) % step == 0 ? r.year : nil
        }
    }

    var body: some View {
        let data = rows
        // The per-bar YoY% labels overlap once there are many bars — only show
        // them when there's room; the value is always available via the tooltip.
        let showYoY = data.count <= 12
        ISection(title: "Annual Dividends",
                 subtitle: selectedYear.map { "Filtered to \($0) — tap the year again to clear" }
                           ?? "Tap a year to filter the transactions below") {
            if data.isEmpty {
                Text("No dividends.").foregroundStyle(.secondary)
            } else {
                Chart(data) { row in
                    BarMark(x: .value("Year", row.year), y: .value("Amount", row.amount))
                        .foregroundStyle(barColor(row.year))
                        .cornerRadius(4)
                        .annotation(position: .top) {
                            if showYoY, let y = row.yoy {
                                Text("\(y > 0 ? "+" : "")\(String(format: "%.0f", y))%")
                                    .appFont(.caption2.bold()).foregroundStyle(y >= 0 ? .green : .red)
                            }
                        }
                }
                .chartXAxis {
                    AxisMarks(values: axisYears(data)) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel {
                            if let s = value.as(String.self) {
                                Text("'\(s.suffix(2))").appFont(.caption2).fixedSize()
                            }
                        }
                    }
                }
                .chartHoverTooltip(data.map(\.year),
                                   onTap: { i in let y = data[i].year; selectedYear = (selectedYear == y) ? nil : y }) { i in
                    var rows = [ChartTooltipRow(color: .green, label: "Dividends",
                                               value: Fmt.currency(data[i].amount, code: currency))]
                    if let y = data[i].yoy {
                        rows.append(ChartTooltipRow(label: "YoY", value: "\(y > 0 ? "+" : "")\(String(format: "%.1f", y))%"))
                    }
                    rows.append(ChartTooltipRow(label: "Tap to filter transactions", value: ""))
                    return ChartTooltipContent(title: data[i].year, rows: rows)
                }
                .frame(height: 260)
            }
        }
    }

    /// Emerald for dividends, a darker shade for the selected year, and a muted
    /// fill for the other years when a year filter is active — mirrors the web.
    private func barColor(_ year: String) -> Color {
        if selectedYear != nil && selectedYear != year { return Color.secondary.opacity(0.25) }
        return selectedYear == year ? Color(hex: 0x059669) : Color(hex: 0x10B981)
    }
}

// MARK: - Dividend transactions (sortable + search — mirrors Dividend.tsx table)

struct DivRow: Identifiable {
    let id: String; let date: String; let symbol: String; let account: String
    let gross: Double; let tax: Double; let net: Double; let localCurrency: String
    init(_ d: Dividend) {
        id = d.id; date = d.date; symbol = d.symbol; account = d.account
        gross = d.amountDisplay; tax = d.taxDisplay ?? 0; net = d.amountDisplay - (d.taxDisplay ?? 0)
        localCurrency = d.localCurrency
    }
}

struct DividendTransactionsCard: View {
    @EnvironmentObject private var appState: AppState
    let dividends: [Dividend]
    let currency: String
    @State private var search = ""
    @State private var sortOrder = [KeyPathComparator(\DivRow.date, order: .reverse)]

    private var rows: [DivRow] {
        let q = search.trimmingCharacters(in: .whitespaces).lowercased()
        return dividends.map(DivRow.init)
            .filter { q.isEmpty || $0.symbol.lowercased().contains(q) || $0.account.lowercased().contains(q) }
            .sorted(using: sortOrder)
    }

    var body: some View {
        ISection(title: "Dividend Transactions", trailing: AnyView(
            TextField("Search symbol or account…", text: $search).textFieldStyle(.roundedBorder).frame(maxWidth: 220))) {
            if rows.isEmpty {
                Text("No dividend transactions.").foregroundStyle(.secondary)
            } else {
                #if os(iOS)
                LazyVStack(spacing: 12) {
                    ForEach(rows) { row in
                        iosDivRow(row)
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
                    TableColumn("Gross", value: \.gross) { Text(Fmt.currency($0.gross, code: currency)).monospacedDigit().foregroundStyle(.green) }
                    TableColumn("Tax", value: \.tax) { Text($0.tax > 0 ? Fmt.currency($0.tax, code: currency) : "—").monospacedDigit().foregroundStyle(.red) }
                    TableColumn("Net", value: \.net) { Text(Fmt.currency($0.net, code: currency)).fontWeight(.bold).monospacedDigit() }
                }
                .frame(minHeight: 320)
                #endif
            }
        }
    }

    private func iosDivRow(_ r: DivRow) -> some View {
        VStack(spacing: 8) {
            HStack {
                Button {
                    appState.openStock(r.symbol)
                } label: {
                    Text(r.symbol).appFont(.headline).fontWeight(.bold).foregroundStyle(.indigo)
                }
                .buttonStyle(.plain)
                Spacer()
                Text(Fmt.currency(r.net, code: currency)).fontWeight(.bold).monospacedDigit()
            }
            HStack {
                Text(MarketTime.formatted(r.date)).appFont(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(r.account).appFont(.caption2).foregroundStyle(.tertiary)
            }
            Divider()
            HStack {
                Text("Gross").appFont(.caption).foregroundStyle(.secondary)
                Text(Fmt.currency(r.gross, code: currency)).appFont(.caption.bold()).monospacedDigit().foregroundStyle(.green)
                Spacer()
                Text("Tax").appFont(.caption).foregroundStyle(.secondary)
                Text(r.tax > 0 ? Fmt.currency(r.tax, code: currency) : "—").appFont(.caption.bold()).monospacedDigit().foregroundStyle(.red)
            }
        }
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
}
