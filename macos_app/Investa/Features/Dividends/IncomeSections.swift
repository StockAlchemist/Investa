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
    var trailing: AnyView? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { Text(title).font(.headline); Spacer(); if let trailing { trailing } }
            content
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
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
        let now = Date(); let cal = Calendar.current
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
        return ISection(title: "Income") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
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
    }
    private func tile(_ label: String, _ value: String, _ sub: String, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(value).font(.title3.bold()).foregroundStyle(tone).lineLimit(1)
            Text(sub).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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
    private var projectorMonths: [String] {
        var seen = Set<String>(); var out: [String] = []
        for s in segs where !seen.contains(s.month) { seen.insert(s.month); out.append(s.month) }
        return out
    }

    var body: some View {
        ISection(title: "Projected 12M Income") {
            if income.isEmpty {
                Text("No projected income.").foregroundStyle(.secondary)
            } else if segs.isEmpty {
                // No per-symbol breakdown — fall back to the monthly total.
                Chart(income) { BarMark(x: .value("Month", $0.month), y: .value("Income", $0.value)).foregroundStyle(.green) }
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
                .chartLegend(.visible)
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
    }
}

// MARK: - Dividend calendar (3M / 1Y toggle — mirrors DividendCalendar.tsx)

struct DividendCalendarSection: View {
    let events: [DividendEvent]
    let currency: String
    var onSelect: (String) -> Void = { _ in }
    @State private var horizon = "3m"

    private var filtered: [DividendEvent] {
        let now = Date()
        let cutoff = Calendar.current.date(byAdding: horizon == "3m" ? .month : .year, value: horizon == "3m" ? 3 : 1, to: now) ?? now
        return events.filter { (parseDay($0.dividendDate) ?? .distantFuture) <= cutoff }
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
                        HStack {
                            Text(ev.symbol).fontWeight(.bold)
                            if ev.status == "estimated" {
                                Label("est.", systemImage: "clock").font(.caption2).foregroundStyle(.orange)
                            } else {
                                Image(systemName: "checkmark.seal.fill").font(.caption2).foregroundStyle(.green)
                            }
                            Spacer()
                            VStack(alignment: .trailing) {
                                Text("Ex \(ev.exDividendDate)").font(.caption2).foregroundStyle(.secondary)
                                Text("Pay \(ev.dividendDate)").font(.caption2).foregroundStyle(.secondary)
                            }
                            Text(Fmt.currency(ev.amount, code: currency)).fontWeight(.bold).foregroundStyle(.green)
                                .frame(width: 90, alignment: .trailing)
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
        let cutoff = Calendar.current.date(byAdding: .year, value: -1, to: Date())!
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
                        Text("\(idx + 1)").font(.caption2.bold()).foregroundStyle(.secondary).frame(width: 18, alignment: .trailing)
                        StockIcon(symbol: row.symbol, size: 18)
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text(row.symbol).fontWeight(.bold)
                                Text("· \(row.count) \(row.count == 1 ? "pay" : "pays")").font(.caption2).foregroundStyle(.secondary)
                            }
                            GeometryReader { g in
                                ZStack(alignment: .leading) {
                                    Capsule().fill(.quaternary)
                                    Capsule().fill(.green).frame(width: g.size.width * min(1, row.pct / 100))
                                }
                            }.frame(height: 6)
                        }
                        VStack(alignment: .trailing, spacing: 1) {
                            Text(Fmt.currency(row.gross, code: currency)).font(.caption.bold()).foregroundStyle(.green)
                            Text(String(format: "%.1f%% of top", row.pct)).font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }.buttonStyle(.plain)
            }
        }
    }
}

// MARK: - By account (trailing 12M — mirrors income/ByAccount.tsx)

struct ByAccountCard: View {
    let dividends: [Dividend]
    let currency: String

    private struct Row: Identifiable { let account: String; let gross12m: Double; var id: String { account } }
    private var rows: [Row] {
        let cutoff = Calendar.current.date(byAdding: .year, value: -1, to: Date())!
        var byAcc: [String: Double] = [:]
        for d in dividends {
            guard let dt = parseDay(d.date), dt >= cutoff else { continue }
            byAcc[d.account.isEmpty ? "—" : d.account, default: 0] += d.amountDisplay
        }
        return byAcc.map { Row(account: $0.key, gross12m: $0.value) }.sorted { $0.gross12m > $1.gross12m }
    }

    var body: some View {
        let data = rows
        let total = data.reduce(0) { $0 + $1.gross12m }
        ISection(title: "By Account") {
            if data.isEmpty { Text("No dividends in the last year.").foregroundStyle(.secondary) }
            ForEach(data) { acc in
                let pct = total > 0 ? acc.gross12m / total * 100 : 0
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(acc.account).fontWeight(.bold).lineLimit(1)
                        Spacer()
                        Text(String(format: "%.1f%%", pct)).font(.caption2).foregroundStyle(.secondary)
                        Text(Fmt.currency(acc.gross12m, code: currency)).font(.caption.bold()).foregroundStyle(.green)
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

    var body: some View {
        let data = rows
        ISection(title: "Annual Dividends") {
            if data.isEmpty {
                Text("No dividends.").foregroundStyle(.secondary)
            } else {
                Chart(data) { row in
                    BarMark(x: .value("Year", row.year), y: .value("Amount", row.amount))
                        .foregroundStyle(.green)
                        .annotation(position: .top) {
                            if let y = row.yoy {
                                Text("\(y > 0 ? "+" : "")\(String(format: "%.0f", y))%")
                                    .font(.caption2.bold()).foregroundStyle(y >= 0 ? .green : .red)
                            }
                        }
                }
                .chartHoverTooltip(data.map(\.year)) { i in
                    var rows = [ChartTooltipRow(color: .green, label: "Dividends",
                                               value: Fmt.currency(data[i].amount, code: currency))]
                    if let y = data[i].yoy {
                        rows.append(ChartTooltipRow(label: "YoY", value: "\(y > 0 ? "+" : "")\(String(format: "%.1f", y))%"))
                    }
                    return ChartTooltipContent(title: data[i].year, rows: rows)
                }
                .frame(height: 260)
            }
        }
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
            TextField("Search symbol or account…", text: $search).textFieldStyle(.roundedBorder).frame(width: 220))) {
            if rows.isEmpty {
                Text("No dividend transactions.").foregroundStyle(.secondary)
            } else {
                Table(rows, sortOrder: $sortOrder) {
                    TableColumn("Date", value: \.date) { Text($0.date).foregroundStyle(.secondary) }
                    TableColumn("Symbol", value: \.symbol) { Text($0.symbol).fontWeight(.medium) }
                    TableColumn("Account", value: \.account) { Text($0.account).font(.caption).foregroundStyle(.secondary) }
                    TableColumn("Gross", value: \.gross) { Text(Fmt.currency($0.gross, code: currency)).monospacedDigit().foregroundStyle(.green) }
                    TableColumn("Tax", value: \.tax) { Text($0.tax > 0 ? Fmt.currency($0.tax, code: currency) : "—").monospacedDigit().foregroundStyle(.red) }
                    TableColumn("Net", value: \.net) { Text(Fmt.currency($0.net, code: currency)).fontWeight(.bold).monospacedDigit() }
                }
                .frame(minHeight: 320)
            }
        }
    }
}
