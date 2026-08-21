import SwiftUI
import Charts

/// Shared card chrome: brand-accented icon + uppercase tracked title over a
/// hairline divider, on the standard card surface (see Theme.swift).
private struct Card<Content: View>: View {
    let title: String
    var icon: String? = nil
    /// Optional trailing accessory rendered at the right of the header row.
    var accessory: AnyView? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 7) {
                if let icon {
                    Image(systemName: icon).font(.caption.weight(.semibold)).foregroundStyle(Theme.brand)
                }
                SectionLabel(title: title)
                Spacer(minLength: 0)
                if let accessory { accessory }
            }
            Divider().opacity(0.6)
            content
            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }
}

// MARK: - Portfolio hero (mirrors Dashboard.tsx PortfolioHeroCard)

enum HeroPeriod: String, CaseIterable, Identifiable {
    case day = "1D", wtd = "WTD", mtd = "MTD", ytd = "YTD", y1 = "1Y"
    var id: String { rawValue }
}

struct PortfolioHeroCard: View {
    let metrics: Metrics?
    let currency: String
    let longHistory: [PerformancePoint]
    var intradayHistory: [PerformancePoint] = []
    var wtdHistory: [PerformancePoint] = []
    var isLoading: Bool = false
    @State private var period: HeroPeriod = .day
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var compact: Bool { hSize == .compact }
    #else
    private var compact: Bool { false }
    #endif

    private var dayGL: Double? { metrics?.dayChangeDisplay }
    private var dayGLPct: Double? { metrics?.dayChangePercent }

    private var periodView: (series: [Double], pct: Double?, abs: Double?) {
        if period == .day {
            let series = intradayHistory.map { $0.value }
            return (series, dayGLPct, dayGL)
        }
        let historyToUse = period == .wtd ? wtdHistory : longHistory
        let rows = historyToUse.compactMap { p -> (Date, Double)? in
            guard let d = p.parsedDate else { return nil }; return (d, p.value)
        }
        guard !rows.isEmpty else { return ([], nil, nil) }
        let cut = Self.cutoff(period)
        let beforeIdx = rows.firstIndex { $0.0 >= cut }
        let sliced = rows.filter { $0.0 >= cut }
        let anchor = (beforeIdx ?? 0) > 0 ? rows[(beforeIdx ?? 0) - 1] : (sliced.first ?? rows[0])
        let tail = sliced.isEmpty ? Array(rows.suffix(1)) : sliced
        let series = [anchor.1] + tail.map { $0.1 }
        let start = anchor.1, end = tail.last?.1 ?? anchor.1
        guard start != 0 else { return (series, nil, nil) }
        let absV = end - start
        return (series, absV / start * 100, absV)
    }

    var body: some View {
        let positive = (dayGL ?? 0) >= 0
        return VStack(alignment: .leading, spacing: 14) {
            // Stacked on phone; side-by-side (with graceful fallback) when wide.
            if compact {
                VStack(alignment: .leading, spacing: 12) { valueBlock; pillCluster }
            } else {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top) { valueBlock; Spacer(); pillCluster }
                    VStack(alignment: .leading, spacing: 12) { valueBlock; pillCluster }
                }
            }
            VStack(alignment: .leading, spacing: 8) {
                if compact {
                    VStack(alignment: .trailing, spacing: 8) {
                        periodControls
                        if period != .day, let pct = periodView.pct, let absV = periodView.abs {
                            periodReturnText(pct: pct, absV: absV)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                } else {
                    HStack(alignment: .center) {
                        periodControls
                        Spacer()
                        if period != .day, let pct = periodView.pct, let absV = periodView.abs {
                            periodReturnText(pct: pct, absV: absV)
                        }
                    }
                }
                if periodView.series.count > 1 {
                    let up = (periodView.pct ?? 0) >= 0
                    let domain = chartDomain(periodView.series)
                    Chart(Array(periodView.series.enumerated()), id: \.offset) { i, v in
                        AreaMark(
                            x: .value("i", i),
                            yStart: .value("Min", domain.lowerBound),
                            yEnd: .value("v", v)
                        )
                        .foregroundStyle(
                            LinearGradient(
                                colors: [
                                    (up ? Color.up : Color.down).opacity(0.32),
                                    (up ? Color.up : Color.down).opacity(0.0)
                                ],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        LineMark(x: .value("i", i), y: .value("v", v))
                            .foregroundStyle(up ? Color.up : Color.down)
                            .lineStyle(StrokeStyle(lineWidth: 2))
                    }
                    .chartYScale(domain: domain)
                    .chartXAxis(.hidden).chartYAxis(.hidden).frame(height: 50)
                }
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            // Faint directional wash: green when the day is up, red when down.
            LinearGradient(
                colors: [(positive ? Color.up : .down).opacity(0.08), .clear],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
            .clipShape(RoundedRectangle(cornerRadius: Theme.heroRadius, style: .continuous))
        )
        .card(.hero)
    }

    private var periodControls: some View {
        HStack(spacing: 2) {
            ForEach(HeroPeriod.allCases) { p in
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { period = p }
                } label: {
                    Text(p.rawValue)
                        .font(.system(size: 11, weight: .bold))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 4.5)
                        .background(
                            period == p ? Color.brand : Color.clear,
                            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
                        )
                        .foregroundStyle(period == p ? Color.white : Color.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(2.5)
        .background(Color.cardBorder.opacity(0.25), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private func periodReturnText(pct: Double, absV: Double) -> some View {
        HStack(spacing: 4) {
            Text("\(pct >= 0 ? "+" : "")\(String(format: "%.2f%%", pct))")
                .font(.system(size: 13, weight: .bold))
                .monospacedDigit()
                .foregroundStyle(Fmt.tint(for: pct))
            Text("(\(pct >= 0 ? "+" : "")\(Fmt.currency(absV, code: currency)))")
                .font(.system(size: 11, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(.secondary)
            Text(period.rawValue)
                .font(.system(size: 10, weight: .bold))
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
        }
        .lineLimit(1)
        .minimumScaleFactor(0.8)
    }

    private var valueBlock: some View {
        let positive = (dayGL ?? 0) >= 0
        return VStack(alignment: .leading, spacing: compact ? 4 : 6) {
            HStack(spacing: 6) {
                Image(systemName: "wallet.pass")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.sectionText)
                SectionLabel(title: "Total Portfolio Value")
                if isLoading {
                    ProgressView().controlSize(.mini).padding(.leading, 2)
                }
            }
            // The day-change sits beside the value when wide, below it on phone
            // (so the big value never gets squeezed/truncated).
            if compact {
                valueText
                deltaView(positive)
            } else {
                HStack(alignment: .firstTextBaseline, spacing: 12) { valueText; deltaView(positive) }
            }
        }
    }

    private var valueText: some View {
        Text(Fmt.currency(metrics?.marketValue, code: currency))
            .font(.system(size: compact ? 34 : 42, weight: .black, design: .rounded))
            .monospacedDigit()
            .minimumScaleFactor(0.5).lineLimit(1)
    }

    @ViewBuilder private func deltaView(_ positive: Bool) -> some View {
        if let g = dayGL {
            HStack(spacing: 5) {
                Image(systemName: positive ? "arrow.up.right" : "arrow.down.right")
                    .font(.system(size: 13, weight: .bold))
                Text("\(g >= 0 ? "+" : "")\(Fmt.currency(g, code: currency))")
                    .font(.system(size: 16, weight: .semibold))
                    .monospacedDigit()
                if let p = dayGLPct {
                    Text("\(p >= 0 ? "+" : "")\(String(format: "%.2f%%", p))")
                        .font(.system(size: 12, weight: .bold))
                        .monospacedDigit()
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2.5)
                        .background((positive ? Color.up : Color.down).opacity(0.12), in: Capsule())
                        .overlay(Capsule().strokeBorder((positive ? Color.up : Color.down).opacity(0.25), lineWidth: 0.8))
                }
                Text("today").font(.caption).foregroundStyle(.secondary)
            }
            .foregroundStyle(positive ? Color.up : Color.down)
            .lineLimit(1).minimumScaleFactor(0.7)
        }
    }

    private var pillCluster: some View {
        HStack(spacing: 0) {
            statPill("Total TWR", metrics?.cumulativeTWR, sub: nil)
            if let a = metrics?.annualizedTWR {
                Rectangle().fill(Color.cardBorder.opacity(0.5)).frame(width: 1, height: 32)
                statPill("Ann. TWR", a, sub: "p.a.")
            }
            if let irr = metrics?.portfolioMWR {
                Rectangle().fill(Color.cardBorder.opacity(0.5)).frame(width: 1, height: 32)
                statPill("IRR (MWR)", irr, sub: "p.a.")
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 4)
        .frame(maxWidth: compact ? .infinity : nil)
        .background(Color.cardBorder.opacity(0.15), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Color.cardBorder.opacity(0.4), lineWidth: 0.8))
    }

    private func statPill(_ label: String, _ value: Double?, sub: String?) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            SectionLabel(title: label)
                .minimumScaleFactor(0.7)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(Fmt.percent(value, includeSign: true))
                    .font(.system(size: 18, weight: .bold))
                    .monospacedDigit()
                    .foregroundStyle(Fmt.tint(for: value))
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                if let sub {
                    Text(sub)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: compact ? .infinity : nil, alignment: .leading)
        .padding(.horizontal, compact ? 10 : 16)
    }

    private static func cutoff(_ period: HeroPeriod) -> Date {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let startOfDay = cal.startOfDay(for: Date())
        switch period {
        case .day: return startOfDay
        case .wtd:
            let dow = (cal.component(.weekday, from: startOfDay) + 5) % 7
            return cal.date(byAdding: .day, value: -dow, to: startOfDay) ?? startOfDay
        case .mtd:
            return cal.date(from: cal.dateComponents([.year, .month], from: startOfDay)) ?? startOfDay
        case .ytd:
            return cal.date(from: cal.dateComponents([.year], from: startOfDay)) ?? startOfDay
        case .y1:
            return cal.date(byAdding: .year, value: -1, to: startOfDay) ?? startOfDay
        }
    }
}


// MARK: - Today strip (mirrors dashboard/TodayStrip.tsx)

struct TodayStripCard: View {
    let holdings: [Holding]
    let currency: String
    let portfolioDayPct: Double?
    let indices: [IndexQuote]
    let onSelectSymbol: (String) -> Void

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var compact: Bool { hSize == .compact }
    #else
    private var compact: Bool { false }
    #endif

    private struct Mover { let symbol: String; let pct: Double; let contribution: Double }

    private func isCash(_ s: String) -> Bool {
        let u = s.uppercased(); return u == "$CASH" || u == "CASH" || u.hasPrefix("CASH (")
    }

    private var movers: (gainers: [Mover], losers: [Mover]) {
        var bySymbol: [String: (mv: Double, pct: Double)] = [:]
        for h in holdings where !isCash(h.symbol) {
            guard let pct = h.dayChangePct else { continue }
            let mv = h.marketValue(currency: currency) ?? 0
            if var cur = bySymbol[h.symbol] { cur.mv += mv; bySymbol[h.symbol] = cur }
            else { bySymbol[h.symbol] = (mv, pct) }
        }
        let rows = bySymbol.map { Mover(symbol: $0.key, pct: $0.value.pct, contribution: $0.value.mv * $0.value.pct / 100) }
        let sorted = rows.sorted { $0.contribution > $1.contribution }
        return (Array(sorted.filter { $0.contribution > 0 }.prefix(3)),
                Array(sorted.filter { $0.contribution < 0 }.suffix(3).reversed()))
    }

    var body: some View {
        Card(title: "Today", icon: "sun.max") {
            let m = movers
            if compact {
                VStack(alignment: .leading, spacing: 16) { marketCol; moverCol("Top Gainers", m.gainers, true); moverCol("Top Losers", m.losers, false) }
            } else {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 24) { marketCol; moverCol("Top Gainers", m.gainers, true); moverCol("Top Losers", m.losers, false) }
                    VStack(alignment: .leading, spacing: 16) { marketCol; moverCol("Top Gainers", m.gainers, true); moverCol("Top Losers", m.losers, false) }
                }
            }
        }
    }

    private var marketCol: some View {
        let top = indices.sorted { abs($0.changesPercentage ?? 0) > abs($1.changesPercentage ?? 0) }.prefix(3)
        return VStack(alignment: .leading, spacing: 6) {
            Label("Market Today", systemImage: "globe").font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            if let you = portfolioDayPct {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text("\(you >= 0 ? "+" : "")\(String(format: "%.2f%%", you))")
                        .font(.title3.bold()).foregroundStyle(Fmt.tint(for: you))
                    Text("you").font(.caption2).foregroundStyle(.secondary)
                }
            }
            ForEach(Array(top)) { idx in
                let pct = idx.changesPercentage ?? 0
                let delta = portfolioDayPct.map { $0 - pct }   // portfolio vs index
                HStack(spacing: 6) {
                    Text(idx.name ?? "Index").lineLimit(1)
                    Spacer(minLength: 6)
                    Text("\(pct >= 0 ? "+" : "")\(String(format: "%.2f%%", pct))").fontWeight(.bold)
                        .foregroundStyle(Fmt.tint(for: idx.change))
                    if let d = delta {
                        Text("(\(d >= 0 ? "+" : "")\(String(format: "%.2f", d)))")
                            .font(.caption2).monospacedDigit()
                            .foregroundStyle((d >= 0 ? Color.up : Color.down).opacity(0.7))
                            .frame(width: 48, alignment: .trailing)
                    }
                }.font(.caption)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func moverCol(_ title: String, _ rows: [Mover], _ positive: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: positive ? "chart.line.uptrend.xyaxis" : "chart.line.downtrend.xyaxis")
                .font(.caption2).textCase(.uppercase).foregroundStyle(positive ? Color.up : Color.down)
            if rows.isEmpty {
                Text("No movers.").font(.caption).foregroundStyle(.secondary)
            } else {
                ForEach(rows, id: \.symbol) { r in
                    Button { onSelectSymbol(r.symbol) } label: {
                        HStack(spacing: 6) {
                            StockIcon(symbol: r.symbol, size: 16)
                            Text(r.symbol).fontWeight(.bold).lineLimit(1)
                            Spacer(minLength: 6)
                            Text("\(r.pct >= 0 ? "+" : "")\(String(format: "%.2f%%", r.pct))")
                                .fontWeight(.bold).foregroundStyle(positive ? Color.up : Color.down)
                            Text(compactChange(r.contribution))
                                .font(.caption2).monospacedDigit()
                                .foregroundStyle((positive ? Color.up : Color.down).opacity(0.8))
                                .lineLimit(1).minimumScaleFactor(0.7)
                                .frame(width: 66, alignment: .trailing)
                        }
                        .font(.caption)
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .contentShape(Rectangle())
                        .rowHover()
                    }.buttonStyle(.plain)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Today's $ change shown compactly (e.g. "+$1.2K"), mirroring the web column.
    private func compactChange(_ v: Double) -> String {
        let sign = v >= 0 ? "+" : "-"
        let a = abs(v)
        let num: String
        if a >= 1_000_000 { num = String(format: "%.1fM", a / 1_000_000) }
        else if a >= 1_000 { num = String(format: "%.1fK", a / 1_000) }
        else { num = String(format: "%.0f", a) }
        return "\(sign)\(Fmt.symbol(currency))\(num)"
    }
}

// MARK: - Upcoming dividends + earnings (mirrors dashboard/DashboardEvents.tsx)

/// A dividend payment or an earnings report, normalized onto one timeline.
enum TimelineEvent: Identifiable {
    case dividend(DividendEvent)
    case earnings(EarningsEvent)

    var id: String {
        switch self {
        case .dividend(let d): return "div|\(d.id)"
        case .earnings(let e): return "earn|\(e.id)"
        }
    }
    var symbol: String {
        switch self {
        case .dividend(let d): return d.symbol
        case .earnings(let e): return e.symbol
        }
    }
    /// Company name, shown under the ticker (GOOG vs GOOGL both read "Alphabet Inc.").
    var name: String? {
        switch self {
        case .dividend(let d): return d.name
        case .earnings(let e): return e.name
        }
    }
    var date: String {
        switch self {
        case .dividend(let d): return String(d.dividendDate.prefix(10))
        case .earnings(let e): return String(e.earningsDate.prefix(10))
        }
    }
    var status: String {
        switch self {
        case .dividend(let d): return d.status
        case .earnings(let e): return e.status
        }
    }
    /// The exchange zone `date` belongs to — every day count uses it, not the device's.
    var marketTimezone: String? {
        switch self {
        case .dividend(let d): return d.marketTimezone
        case .earnings(let e): return e.marketTimezone
        }
    }
}

struct UpcomingEventsCard: View {
    let dividends: [DividendEvent]
    var earnings: [EarningsEvent] = []
    let currency: String
    var windowDays = 14
    var onSelectSymbol: (String) -> Void = { _ in }
    @State private var showConfirmed = false
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var compact: Bool { hSize == .compact }
    #else
    private var compact: Bool { false }
    #endif

    private var confirmedCount: Int { dividends.filter { $0.status == "confirmed" }.count }

    /// Mirrors `REPORTED_LOOKBACK_DAYS` in `server/calendar_events.py`.
    private static let reportedWindowDays = 5

    /// "today" / "3d" / "Jul 29" for an event date, counted in the *market's*
    /// local time: in Bangkok the device rolls into tomorrow while New York is
    /// still mid-afternoon, which would show a US report happening today as "1d ago".
    /// The short day label is the fallback past the relative-day window — the full
    /// ISO string is too wide for a phone row.
    private func relativeDay(_ iso: String, _ timeZone: String?) -> String {
        guard let days = MarketTime.dayDiff(iso, timeZone: timeZone) else { return "" }
        switch days {
        case 0: return "today"
        case 1: return "tomorrow"
        case ..<0: return "\(-days)d ago"
        case 1..<7: return "\(days)d"
        default: return MarketTime.shortDay(iso)
        }
    }

    private var upcoming: [TimelineEvent] {
        let all = dividends.map { TimelineEvent.dividend($0) } + earnings.map { TimelineEvent.earnings($0) }
        return all.filter {
            // The window is measured on each event's own exchange clock, so the
            // row shown as "today" is the one the market calls today.
            guard let days = MarketTime.dayDiff($0.date, timeZone: $0.marketTimezone) else { return false }
            // A quarter that has just been reported is the one row that looks
            // backwards: it stays a few days so a report resolves into its result
            // here instead of vanishing the morning after. The backend already
            // filters to that window; this only stops an older row slipping in.
            if case .earnings(let e) = $0, e.isReported {
                return days <= 0 && days >= -Self.reportedWindowDays
            }
            return days >= 0 && days <= windowDays
        }
        .sorted { $0.date == $1.date ? $0.symbol < $1.symbol : $0.date < $1.date }
        .prefix(8).map { $0 }
    }

    private var confirmedButton: some View {
        Button { showConfirmed = true } label: {
            Text("Confirmed (\(confirmedCount)) →")
                .font(.system(size: 11, weight: .semibold)).textCase(.uppercase).tracking(0.5)
                .foregroundStyle(confirmedCount == 0 ? Color.secondary : Theme.brand)
        }
        .buttonStyle(.plain).disabled(confirmedCount == 0)
        .help("View all confirmed dividends")
    }

    var body: some View {
        Card(title: "Upcoming Events", icon: "calendar", accessory: AnyView(confirmedButton)) {
            if upcoming.isEmpty {
                EmptyHint(text: "No dividend or earnings events in the next \(windowDays) days.", systemImage: "calendar")
            } else {
                ForEach(upcoming) { ev in
                    Button { onSelectSymbol(ev.symbol) } label: {
                        HStack(alignment: .center, spacing: 10) {
                            // Sized to the full height of the two-line row beside it.
                            StockIcon(symbol: ev.symbol, size: 34)
                            // Ticker + type marker on top, company name beneath, so a
                            // narrow (phone) row never has to wrap the marker's text.
                            VStack(alignment: .leading, spacing: 1) {
                                HStack(spacing: 6) {
                                    // fixedSize: the ticker is the row's identity, so it
                                    // must never be the thing that truncates.
                                    Text(ev.symbol).font(.callout.weight(.bold)).lineLimit(1).fixedSize()
                                    badge(ev)
                                }
                                if let name = ev.name, !name.isEmpty {
                                    Text(name).font(.caption2).foregroundStyle(.secondary)
                                        .lineLimit(1).truncationMode(.tail)
                                }
                            }
                            Spacer(minLength: 6)
                            Text(relativeDay(ev.date, ev.marketTimezone)).font(.caption).textCase(.uppercase)
                                .foregroundStyle(isSoon(ev) ? Color.up : .secondary)
                                .lineLimit(1).fixedSize()
                            trailing(ev)
                                .lineLimit(1).minimumScaleFactor(0.7)
                                .frame(minWidth: 80, alignment: .trailing)
                        }
                        .padding(.horizontal, 6).padding(.vertical, 4)
                        .contentShape(Rectangle())
                        .rowHover()
                    }.buttonStyle(.plain)
                }
            }
        }
        .sheet(isPresented: $showConfirmed) {
            ConfirmedDividendsSheet(events: dividends, currency: currency, onSelectSymbol: onSelectSymbol)
        }
    }

    /// Event-type marker: earnings are violet, dividends keep the est./confirmed pair.
    /// On a phone the word "earnings" is dropped — the icon and the trailing
    /// "… EPS" already say it, and the width belongs to the ticker. "reported"
    /// is kept even when compact: it is what distinguishes a result from a date.
    @ViewBuilder private func badge(_ ev: TimelineEvent) -> some View {
        switch ev {
        case .earnings(let e):
            let estimated = e.status == "estimated"
            let text: String = {
                if e.isReported { return "reported" }
                if compact { return estimated ? "est." : "" }
                return estimated ? "earnings est." : "earnings"
            }()
            // fixedSize keeps the label on one line when the row is tight.
            HStack(spacing: 3) {
                Image(systemName: "chart.bar.fill")
                if !text.isEmpty { Text(text) }
            }
            .font(.caption2).foregroundStyle(Theme.earnings)
            .lineLimit(1).fixedSize()
        case .dividend(let d):
            if d.status == "estimated" {
                Label("est.", systemImage: "clock").font(.caption2).foregroundStyle(.orange)
            } else {
                Image(systemName: "checkmark.seal.fill").font(.caption2).foregroundStyle(Color.up)
            }
        }
    }

    /// Dividends show the cash amount; a scheduled report shows the consensus EPS
    /// estimate, a reported one shows what was actually printed and by how much
    /// it beat or missed.
    @ViewBuilder private func trailing(_ ev: TimelineEvent) -> some View {
        switch ev {
        case .dividend(let d):
            Text(Fmt.currency(d.amount, code: currency)).font(.callout.weight(.bold)).foregroundStyle(Color.up)
        case .earnings(let e) where e.isReported:
            if let actual = e.epsActual {
                // A beat is green and a miss red, matching every other
                // gain/loss figure in the app.
                let tint: Color = e.surprisePct.map { $0 >= 0 ? Color.up : Color.down } ?? Theme.earnings
                VStack(alignment: .trailing, spacing: 0) {
                    Text(String(format: "%.2f EPS", actual))
                        .font(.callout.weight(.bold)).foregroundStyle(tint)
                    if let surprise = e.surprisePct {
                        Text(String(format: "%+.1f%%", surprise))
                            .font(.caption2.weight(.semibold)).foregroundStyle(tint)
                    }
                }
            } else {
                // Reported, but Yahoo has not attached the figures yet.
                Text("Reported").font(.callout.weight(.bold)).foregroundStyle(.secondary)
            }
        case .earnings(let e):
            Text(e.epsEstimate.map { String(format: "%.2f EPS", $0) } ?? "Report")
                .font(.callout.weight(.bold)).foregroundStyle(Theme.earnings)
        }
    }

    private func isSoon(_ ev: TimelineEvent) -> Bool {
        guard let days = MarketTime.dayDiff(ev.date, timeZone: ev.marketTimezone) else { return false }
        return days == 0 || days == 1
    }
}

// MARK: - Actionable insights (mirrors dashboard/DashboardInsights.tsx)

struct ActionableInsightsCard: View {
    let holdings: [Holding]
    let currency: String
    var targets: [String: [String: Double]] = [:]
    @State private var scope: InsightScope?

    private var computed: (summaries: [InsightSummary], details: InsightDetails) {
        computeInsights(holdings: holdings, currency: currency, targets: targets)
    }

    var body: some View {
        let (summaries, details) = computed
        let hasAny = !summaries.isEmpty
        return Card(title: "Actionable Insights", icon: "lightbulb",
                    accessory: AnyView(Text("\(summaries.count) item\(summaries.count == 1 ? "" : "s")")
                        .font(.system(size: 11, weight: .semibold)).textCase(.uppercase).foregroundStyle(.secondary))) {
            if !hasAny {
                VStack(spacing: 4) {
                    Image(systemName: "sparkles").font(.title3).foregroundStyle(.tertiary)
                    Text("Nothing to flag today.").font(.callout).foregroundStyle(.secondary)
                    Text("No ripening lots, drift breaches, or new value buys.")
                        .font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity).padding(.vertical, 12)
            } else {
                ForEach(summaries) { ins in
                    Button { scope = .kind(ins.kind) } label: { insightRow(ins) }.buttonStyle(.plain)
                }
            }
        }
        // Clicking the card body (off-row) opens the combined view, like the web.
        .contentShape(Rectangle())
        .onTapGesture { if hasAny { scope = .all } }
        .sheet(item: $scope) { s in
            InsightsDetailSheet(scope: s, details: details, summaries: summaries, currency: currency)
        }
    }

    private func insightRow(_ ins: InsightSummary) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: ins.icon).font(.caption).foregroundStyle(ins.tone.color)
                .frame(width: 28, height: 28).background(ins.tone.color.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 1) {
                Text(ins.title).font(.callout.weight(.semibold))
                if let sub = ins.sub { Text(sub).font(.caption2).foregroundStyle(.secondary) }
            }
            Spacer(minLength: 4)
            Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 6).padding(.vertical, 5)
        .contentShape(Rectangle())
        .rowHover()
    }
}

// MARK: - Portfolio composition (mirrors PortfolioDonut.tsx — By Holding + By Account)

enum CompositionMetric: String, CaseIterable, Identifiable {
    case value, dayChange, totalGain
    var id: String { rawValue }
    var label: String {
        switch self { case .value: return "Total Value"; case .dayChange: return "Day's Change"; case .totalGain: return "Unrealized Gain" }
    }
}

struct DonutSlice: Identifiable {
    let name: String
    let value: Double
    let dayChange: Double
    let unrealizedGain: Double
    let costBasis: Double
    let color: Color
    var id: String { name }
}

/// Palette matching the web PortfolioDonut COLORS (last = slate for "Other").
private let donutPalette: [Color] = [
    Color(hex: 0x0097b2), Color(hex: 0x0ea5e9), Color(hex: 0x3b82f6), Color(hex: 0x6366f1),
    Color(hex: 0x8b5cf6), Color(hex: 0xd946ef), Color(hex: 0xec4899), Color(hex: 0xf43f5e),
    Color(hex: 0xf59e0b), Color(hex: 0x10b981), Color(hex: 0x14b8a6),
]
private let otherColor = Color(hex: 0x94a3b8)

/// Group holdings into donut slices by a key, with a 2% "Other" bucket.
private func donutSlices(_ holdings: [Holding], currency: String, by keyFor: (Holding) -> String) -> [DonutSlice] {
    struct Acc { var value = 0.0; var day = 0.0; var gain = 0.0; var basis = 0.0 }
    var grouped: [String: Acc] = [:]
    let mapping = ["GOOGL": "GOOG"]
    for h in holdings {
        let val = h.marketValue(currency: currency) ?? 0
        guard val > 0 else { continue }
        var key = keyFor(h)
        if let mapped = mapping[key] { key = mapped }
        var a = grouped[key] ?? Acc()
        a.value += val
        a.day += h.currencyValue("Day Change", currency: currency) ?? 0
        let gain = h.currencyValue("Unreal. Gain", currency: currency) ?? 0
        a.gain += gain
        a.basis += h.currencyValue("Cost Basis", currency: currency) ?? (val - gain)
        grouped[key] = a
    }
    let sorted = grouped.map { (name: $0.key, a: $0.value) }.sorted { $0.a.value > $1.a.value }
    let total = sorted.reduce(0) { $0 + $1.a.value }
    var slices: [DonutSlice] = []
    var other = Acc()
    var otherCount = 0
    for (i, item) in sorted.enumerated() {
        if total > 0 && item.a.value / total >= 0.02 {
            slices.append(DonutSlice(name: item.name, value: item.a.value, dayChange: item.a.day,
                                     unrealizedGain: item.a.gain, costBasis: item.a.basis,
                                     color: donutPalette[slices.count % donutPalette.count]))
        } else {
            other.value += item.a.value; other.day += item.a.day; other.gain += item.a.gain; other.basis += item.a.basis; otherCount += 1
        }
        _ = i
    }
    if otherCount > 0 {
        slices.append(DonutSlice(name: "Other", value: other.value, dayChange: other.day,
                                 unrealizedGain: other.gain, costBasis: other.basis, color: otherColor))
    }
    return slices
}

struct PortfolioCompositionCard: View {
    let holdings: [Holding]
    let currency: String

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    private var compact: Bool { hSize == .compact }
    #else
    private var compact: Bool { false }
    #endif

    private var byHolding: [DonutSlice] { donutSlices(holdings, currency: currency) { $0.symbol } }
    private var byAccount: [DonutSlice] { donutSlices(holdings, currency: currency) { $0.account ?? "Unknown" } }

    /// The donuts grow with the card so a wide window doesn't leave a gulf
    /// between them; clamped so they stay readable narrow and sane when huge.
    private static let minSide: CGFloat = 300
    private static let maxSide: CGFloat = 560
    private static let columnGap: CGFloat = 24
    @State private var contentWidth: CGFloat = 0

    private var sideBySide: Bool {
        !compact && contentWidth >= 2 * Self.minSide + Self.columnGap
    }

    private var side: CGFloat {
        guard contentWidth > 0 else { return compact ? Self.minSide : 360 }
        let available = sideBySide ? (contentWidth - Self.columnGap) / 2 : contentWidth
        return min(max(available, Self.minSide), Self.maxSide)
    }

    var body: some View {
        Card(title: "Portfolio Composition", icon: "chart.pie") {
            Group {
                if sideBySide {
                    HStack(alignment: .top, spacing: Self.columnGap) {
                        SingleDonut(title: "By Holding", slices: byHolding, currency: currency, side: side)
                        SingleDonut(title: "By Account", slices: byAccount, currency: currency, side: side, forceAllLabels: true)
                    }
                } else {
                    VStack(spacing: 32) {
                        SingleDonut(title: "By Holding", slices: byHolding, currency: currency, side: side)
                        SingleDonut(title: "By Account", slices: byAccount, currency: currency, side: side, forceAllLabels: true)
                    }
                }
            }
            .frame(maxWidth: .infinity)
            // Must be the *offered* width. Measuring the laid-out width latched
            // the donuts at `maxSide` after a landscape trip: the oversized
            // donuts grew the card, the card re-measured its own overflow, and
            // portrait never came back. See `readingContainerWidth`.
            .readingContainerWidth { contentWidth = $0 }
        }
    }
}

private struct SingleDonut: View {
    let title: String
    let slices: [DonutSlice]
    let currency: String
    var side: CGFloat = 360
    var forceAllLabels = false
    @State private var metric: CompositionMetric = .value
    @State private var selectedValue: Double?

    /// Type and icons grow with the ring, but slower, so a big donut doesn't
    /// read as a blown-up screenshot.
    private var scale: CGFloat { min(max(side / 360, 1), 1.35) }

    private var totalValue: Double { slices.reduce(0) { $0 + $1.value } }
    private var totalDay: Double { slices.reduce(0) { $0 + $1.dayChange } }
    private var totalGain: Double { slices.reduce(0) { $0 + $1.unrealizedGain } }
    private var totalBasis: Double { let b = slices.reduce(0) { $0 + $1.costBasis }; return b > 0 ? b : totalValue - totalGain }

    /// The slice under the current angle selection.
    private var active: DonutSlice? {
        guard let v = selectedValue else { return nil }
        var cum = 0.0
        for s in slices { cum += s.value; if v <= cum { return s } }
        return slices.last
    }

    /// Slices with their mid-angle fraction (0…1, clockwise from top) for ring labels.
    private var labeled: [(slice: DonutSlice, mid: Double)] {
        var out: [(DonutSlice, Double)] = []; var cum = 0.0
        for s in slices {
            let frac = totalValue > 0 ? s.value / totalValue : 0
            let mid = (cum + s.value / 2) / max(totalValue, 1)
            if (forceAllLabels || frac >= 0.03) && s.name != "Other" { out.append((s, mid)) }
            cum += s.value
        }
        return out
    }

    var body: some View {
        VStack(spacing: 6) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                .frame(maxWidth: .infinity, alignment: .leading)
            if slices.isEmpty {
                ContentUnavailableView("No active holdings", systemImage: "chart.pie").frame(height: side)
            } else {
                ZStack {
                    chart
                    ringLabels
                    centerOverlay
                }
                // `side` comes from a measurement, so it lags a rotation by a
                // layout pass. Cap the width rather than pinning it: a pinned
                // width would overflow the container for that one pass, and a
                // vertical ScrollView adopts an over-wide child as its own
                // width and then re-proposes it to its content — which makes
                // the stale measurement measure itself, and portrait never
                // comes back. Height may lag freely; nothing latches on it.
                .frame(height: side)
                .frame(maxWidth: side)
                .frame(maxWidth: .infinity)   // center within the column
            }
        }
    }

    private var chart: some View {
        Chart(slices) { s in
            // Ring kept compact so the logo/name labels sit outside it (like the web).
            SectorMark(angle: .value("Value", s.value), innerRadius: .ratio(0.52), outerRadius: .ratio(0.72), angularInset: 2)
                .foregroundStyle(by: .value("Name", s.name))
                .opacity(active == nil || active?.id == s.id ? 1 : 0.3)
        }
        .chartForegroundStyleScale(domain: slices.map(\.name), range: slices.map(\.color))
        .chartLegend(.hidden)
        .chartAngleSelection(value: $selectedValue)
    }

    private var ringLabels: some View {
        GeometryReader { geo in
            let cx = geo.size.width / 2, cy = geo.size.height / 2
            // Place labels just outside the ring's outer edge.
            let r = min(geo.size.width, geo.size.height) / 2 * 0.87
            ForEach(labeled, id: \.slice.id) { item in
                let theta = item.mid * 2 * .pi
                HStack(spacing: 3) {
                    StockIcon(symbol: item.slice.name, size: 18 * scale)
                    Text(item.slice.name)
                        .font(.system(size: 10 * scale, weight: .bold)).foregroundStyle(.primary).lineLimit(1)
                }
                .padding(.horizontal, 4).padding(.vertical, 2)
                .background(.background, in: Capsule())
                .overlay(Capsule().strokeBorder(.quaternary, lineWidth: 0.5))
                .fixedSize()
                .position(x: cx + r * sin(theta), y: cy - r * cos(theta))
            }
        }
        .allowsHitTesting(false)
    }

    @ViewBuilder private var centerOverlay: some View {
        // Constrain content to the inner hole so long values (e.g. THB) scale to fit.
        GeometryReader { geo in
            let holeWidth = min(geo.size.width, geo.size.height) * 0.30
            VStack(spacing: 2) {
                if let a = active {
                    StockIcon(symbol: a.name, size: 27 * scale)
                    Text(a.name).font(.system(size: 13 * scale, weight: .medium)).foregroundStyle(.secondary).lineLimit(1)
                    Text(value(of: a)).font(.system(size: 18 * scale, weight: .bold)).foregroundStyle(tint(of: a))
                        .lineLimit(1).minimumScaleFactor(0.4)
                    Text(subtitle(of: a)).font(.system(size: 14 * scale, weight: .bold)).foregroundStyle(subtitleTint(of: a))
                        .lineLimit(1).minimumScaleFactor(0.4)
                } else {
                    PopoverMenu(minWidth: 180) {
                        ForEach(CompositionMetric.allCases) { m in
                            MenuToggleRow(title: m.label, isOn: m == metric, dismissOnTap: true) { metric = m }
                        }
                    } label: {
                        HStack(spacing: 3) {
                            Text(metric.label).font(.system(size: 11 * scale, weight: .semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                            Image(systemName: "chevron.down").font(.system(size: 9 * scale)).foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(.background.tertiary, in: Capsule())
                        .overlay(Capsule().strokeBorder(.white.opacity(0.06), lineWidth: 1))
                    }.fixedSize()
                    Text(value(of: nil)).font(.system(size: 18 * scale, weight: .bold)).foregroundStyle(tint(of: nil))
                        .lineLimit(1).minimumScaleFactor(0.4)
                    Text(subtitle(of: nil)).font(.system(size: 14 * scale, weight: .bold)).foregroundStyle(subtitleTint(of: nil))
                        .lineLimit(1).minimumScaleFactor(0.4)
                }
            }
            .frame(width: holeWidth)
            .frame(width: geo.size.width, height: geo.size.height)   // center within the donut
            .allowsHitTesting(active == nil)
        }
    }

    // value/subtitle for the active slice (or the total when `s` is nil).
    private func value(of s: DonutSlice?) -> String {
        let v: Double = { switch metric {
            case .value: return s?.value ?? totalValue
            case .dayChange: return s?.dayChange ?? totalDay
            case .totalGain: return s?.unrealizedGain ?? totalGain } }()
        if metric == .value { return Fmt.currency(v, code: currency) }
        return "\(v >= 0 ? "+" : "-")\(Fmt.currency(abs(v), code: currency))"
    }
    private func tint(of s: DonutSlice?) -> Color {
        switch metric {
        case .value: return .primary
        case .dayChange: return Fmt.tint(for: s?.dayChange ?? totalDay)
        case .totalGain: return Fmt.tint(for: s?.unrealizedGain ?? totalGain)
        }
    }
    private func subtitle(of s: DonutSlice?) -> String {
        switch metric {
        case .value:
            if let s { return String(format: "%.1f%%", totalValue > 0 ? s.value / totalValue * 100 : 0) }
            return "\(totalDay >= 0 ? "+" : "-")\(Fmt.currency(abs(totalDay), code: currency))"
        case .dayChange:
            let day = s?.dayChange ?? totalDay; let base = (s?.value ?? totalValue) - day
            let pct = base != 0 ? day / base * 100 : 0
            return String(format: "%@%.2f%%", pct >= 0 ? "+" : "-", abs(pct))
        case .totalGain:
            let gain = s?.unrealizedGain ?? totalGain; let basis = s?.costBasis ?? totalBasis
            let pct = basis != 0 ? gain / basis * 100 : 0
            return String(format: "%@%.2f%%", pct >= 0 ? "+" : "-", abs(pct))
        }
    }
    private func subtitleTint(of s: DonutSlice?) -> Color {
        if metric == .value, s != nil { return .secondary }
        return tint(of: s) == .primary ? Fmt.tint(for: s?.dayChange ?? totalDay) : tint(of: s)
    }
}

// MARK: - Portfolio health

struct PortfolioHealthCard: View {
    let health: PortfolioHealth
    @State private var showAnalysis = false
    var body: some View {
        Card(title: "Portfolio Health", icon: "waveform.path.ecg",
             accessory: AnyView(Image(systemName: "info.circle").font(.caption).foregroundStyle(.secondary))) {
            HStack(spacing: 24) {
                // Left side: Circular Progress
                VStack(spacing: 8) {
                    ZStack {
                        Circle()
                            .stroke(color(health.overallScore).opacity(0.15), lineWidth: 8)
                        Circle()
                            .trim(from: 0, to: CGFloat(health.overallScore / 100.0))
                            .stroke(color(health.overallScore), style: StrokeStyle(lineWidth: 8, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                        
                        Text(String(format: "%g", health.overallScore))
                            .font(.title2.bold())
                            .foregroundStyle(color(health.overallScore))
                    }
                    .frame(width: 80, height: 80)
                    
                    Text(health.rating)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(color(health.overallScore))
                }
                .padding(.leading, 8)
                
                // Right side: Metrics
                VStack(spacing: 16) {
                    ForEach([
                        ("chart.pie", "Diversification", health.components.diversification),
                        ("arrow.up.forward", "Efficiency", health.components.efficiency),
                        ("checkmark.shield", "Stability", health.components.stability),
                    ], id: \.1) { icon, name, comp in
                        HStack(spacing: 12) {
                            Image(systemName: icon)
                                .font(.body.weight(.medium))
                            Text(name.uppercased())
                                .font(.caption.weight(.bold))
                                .tracking(0.5)
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                            Spacer()
                            Text(String(format: "%g", comp.score))
                                .font(.subheadline.weight(.bold))
                                .foregroundStyle(color(comp.score))
                        }
                    }
                }
            }
            .padding(.vertical, 8)
        }
        .contentShape(Rectangle())
        .onTapGesture { showAnalysis = true }
        .sheet(isPresented: $showAnalysis) { HealthAnalysisSheet(health: health) }
    }
    private func color(_ s: Double) -> Color { s < 40 ? .red : (s < 70 ? .orange : .green) }
}

// MARK: - Risk metrics

struct RiskMetricsCard: View {
    let risk: RiskMetrics?
    @State private var selectedMetric: String?

    /// (explanation key, label, formatted value, tint) for each tile, in web order.
    private var items: [(key: String, label: String, value: String, tint: Color)] {
        [
            ("Sharpe Ratio", "Sharpe Ratio", Fmt.number(risk?.sharpe), (risk?.sharpe ?? 0) > 1 ? .up : .primary),
            ("Sortino Ratio", "Sortino Ratio", Fmt.number(risk?.sortino), (risk?.sortino ?? 0) > 1 ? .up : .primary),
            ("Volatility", "Volatility", Fmt.percent(risk?.volatilityAnn), .primary),
            ("Max Drawdown", "Max Drawdown", Fmt.percent(risk?.maxDrawdown), .down),
            ("Beta", "Beta", Fmt.number(risk?.beta), (risk?.beta ?? 0) > 1.2 ? .orange : .primary),
            ("Alpha", "Alpha", Fmt.percent(risk?.alpha, includeSign: true),
             (risk?.alpha ?? 0) > 0 ? .up : ((risk?.alpha ?? 0) < 0 ? .down : .primary)),
        ]
    }

    var body: some View {
        Card(title: "Risk Analytics", icon: "shield.lefthalf.filled") {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 3), spacing: 10) {
                ForEach(items, id: \.key) { item in
                    Button { selectedMetric = item.key } label: { cell(item.label, item.value, item.tint) }
                        .buttonStyle(.plain)
                }
            }
        }
        .sheet(item: Binding(get: { selectedMetric.map(MetricID.init) },
                             set: { selectedMetric = $0?.id })) { m in
            MetricExplanationSheet(metricKey: m.id)
        }
    }

    private func cell(_ t: String, _ v: String, _ tint: Color) -> some View {
        VStack(spacing: 4) {
            Text(t).font(.system(size: 10, weight: .bold)).textCase(.uppercase).tracking(0.5)
                .foregroundStyle(.secondary).lineLimit(1)
            Text(v).font(.title3.weight(.bold)).monospacedDigit().foregroundStyle(tint)
                .lineLimit(1).minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 12).padding(.horizontal, 6)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.white.opacity(0.05), lineWidth: 1))
        .contentShape(Rectangle())
        .rowHover()
    }
}

/// Identifiable wrapper so a metric key can drive `.sheet(item:)`.
private struct MetricID: Identifiable { let id: String }

// MARK: - Attribution

/// Currency with no fraction digits (matches the web attribution cards).
private func attrCurrency(_ v: Double, _ code: String) -> String {
    let f = NumberFormatter(); f.numberStyle = .currency; f.currencyCode = code
    f.currencySymbol = Fmt.symbol(code)
    f.maximumFractionDigits = 0; f.minimumFractionDigits = 0
    return f.string(from: NSNumber(value: v)) ?? "—"
}
/// `contribution` is a fraction from the backend — render as a 1-decimal percent.
private func attrPct(_ c: Double) -> String { String(format: "%.1f%%", c * 100) }
private func firstSymbol(_ s: String) -> String {
    s.split(separator: ",").first.map { $0.trimmingCharacters(in: .whitespaces) } ?? s
}

struct SectorAttributionCard: View {
    let attribution: Attribution
    let currency: String
    var body: some View {
        Card(title: "Sector Contribution", icon: "square.stack.3d.up") {
            let sectors = attribution.sectors.sorted { $0.gain > $1.gain }
            if sectors.isEmpty {
                EmptyHint(text: "No sector data available", systemImage: "square.stack.3d.up")
            } else {
                ForEach(sectors) { s in row(s) }
            }
        }
    }
    private func row(_ s: Attribution.SectorContribution) -> some View {
        let tone: Color = s.gain >= 0 ? .up : .down
        return VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(s.sector).font(.caption.weight(.semibold)).foregroundStyle(.primary.opacity(0.85)).lineLimit(1)
                Spacer()
                Text("\(attrCurrency(s.gain, currency)) (\(attrPct(s.contribution)))")
                    .font(.caption.bold()).foregroundStyle(tone).monospacedDigit()
                    .lineLimit(1).minimumScaleFactor(0.7)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(.quaternary)
                    Capsule().fill(tone).frame(width: geo.size.width * min(1, abs(s.contribution)))
                }
            }
            .frame(height: 6)
        }
        .padding(.vertical, 3)
    }
}

struct TopContributorsCard: View {
    let attribution: Attribution
    let currency: String
    var accounts: [String]?
    var showClosed = false
    var onSelectSymbol: (String) -> Void = { _ in }
    @State private var showAll = false

    var body: some View {
        Card(title: "Top Contributors", icon: "trophy") {
            let stocks = attribution.stocks.sorted { $0.gain > $1.gain }
            if stocks.isEmpty {
                EmptyHint(text: "No contributor data available", systemImage: "trophy")
            } else {
                ForEach(Array(stocks.prefix(10))) { s in
                    ContributorRow(stock: s, currency: currency, onSelectSymbol: onSelectSymbol)
                }
                Divider().padding(.top, 4)
                Button { showAll = true } label: {
                    Text("View All Contributors").font(.system(size: 11, weight: .bold)).textCase(.uppercase).tracking(1.5)
                        .frame(maxWidth: .infinity).padding(.vertical, 6).foregroundStyle(.secondary)
                }.buttonStyle(.plain)
            }
        }
        .sheet(isPresented: $showAll) {
            AllContributorsSheet(currency: currency, accounts: accounts, showClosed: showClosed,
                                 initial: attribution.stocks, onSelectSymbol: onSelectSymbol)
        }
    }
}

/// A single contributor row (shared by the card and the View-All sheet).
private struct ContributorRow: View {
    let stock: Attribution.StockContribution
    let currency: String
    var large = false
    var onSelectSymbol: (String) -> Void

    var body: some View {
        let tone: Color = stock.gain >= 0 ? .up : .down
        Button {
            // Only open detail for a single (non-aggregated) symbol, like the web.
            if !stock.symbol.contains(",") { onSelectSymbol(firstSymbol(stock.symbol)) }
        } label: {
            HStack(spacing: large ? 12 : 10) {
                StockIcon(symbol: firstSymbol(stock.symbol), size: large ? 34 : 20)
                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 4) {
                        Text(stock.symbol).font((large ? Font.body : Font.callout).weight(.bold)).lineLimit(1)
                        if stock.value > 0 {
                            Text("HELD").font(.system(size: 9, weight: .bold))
                                .padding(.horizontal, 4).padding(.vertical, 1)
                                .background(Color.up.opacity(0.12), in: Capsule()).foregroundStyle(Color.up)
                        }
                    }
                    Text(stock.name).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                        .frame(maxWidth: large ? .infinity : 130, alignment: .leading)
                }
                Spacer(minLength: 8)
                VStack(alignment: .trailing, spacing: 1) {
                    HStack(spacing: 3) {
                        Image(systemName: stock.gain >= 0 ? "arrow.up.right" : "arrow.down.right").font(.system(size: 10))
                        Text("\(attrCurrency(stock.gain, currency)) (\(attrPct(stock.contribution)))")
                            .font((large ? Font.callout : Font.caption).weight(.bold)).monospacedDigit()
                            .lineLimit(1).minimumScaleFactor(0.7)
                    }.foregroundStyle(tone)
                    Text(stock.sector).font(.system(size: 10)).foregroundStyle(Color(hex: 0x06b6d4)).textCase(.uppercase).lineLimit(1)
                }
            }
            .padding(.vertical, large ? 8 : 4).padding(.horizontal, large ? 12 : 4)
            .contentShape(Rectangle())
            .background(large ? AnyShapeStyle(.background.tertiary) : AnyShapeStyle(.clear),
                        in: RoundedRectangle(cornerRadius: 12))
            .rowHover()
        }.buttonStyle(.plain)
    }
}

/// Searchable grid of every contributor (fetches the full list with show_all).
private struct AllContributorsSheet: View {
    @Environment(\.dismiss) private var dismiss
    let currency: String
    let accounts: [String]?
    let showClosed: Bool
    var onSelectSymbol: (String) -> Void
    @State private var all: [Attribution.StockContribution]
    @State private var search = ""
    @State private var isLoading = false

    init(currency: String, accounts: [String]?, showClosed: Bool,
         initial: [Attribution.StockContribution], onSelectSymbol: @escaping (String) -> Void) {
        self.currency = currency; self.accounts = accounts; self.showClosed = showClosed
        self.onSelectSymbol = onSelectSymbol
        _all = State(initialValue: initial)
    }

    private var filtered: [Attribution.StockContribution] {
        let q = search.lowercased()
        return all.filter {
            q.isEmpty || $0.symbol.lowercased().contains(q) || $0.name.lowercased().contains(q) || $0.sector.lowercased().contains(q)
        }.sorted { $0.gain > $1.gain }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 1) {
                    Text("All Contributors").font(.title2.bold())
                    Text("Impact of individual holdings on performance")
                        .font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
                }
                Spacer()
                Button { dismiss() } label: { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.plain).font(.title2).foregroundStyle(.secondary)
            }
            .padding(20)
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search symbols, names, or sectors…", text: $search).textFieldStyle(.plain)
            }
            .padding(10).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
            .padding(.horizontal, 20)
            Divider().padding(.top, 12)
            ScrollView {
                if isLoading {
                    ProgressView("Calculating full impact…").frame(maxWidth: .infinity).padding(40)
                } else if filtered.isEmpty {
                    ContentUnavailableView("No matching holdings", systemImage: "magnifyingglass").padding(40)
                } else {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 320), spacing: 12)], spacing: 12) {
                        ForEach(filtered) { s in
                            ContributorRow(stock: s, currency: currency, large: true, onSelectSymbol: onSelectSymbol)
                        }
                    }
                    .padding(20)
                }
            }
        }
        #if os(macOS)
        .frame(width: 800, height: 600)
        #else
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        #endif
        .task { await loadFull() }
    }

    private func loadFull() async {
        guard all.count <= 10 else { return }   // already top-10; fetch the rest
        isLoading = true; defer { isLoading = false }
        var q = [URLQueryItem(name: "currency", value: currency),
                 URLQueryItem(name: "show_all", value: "true"),
                 URLQueryItem(name: "show_closed", value: showClosed ? "true" : "false")]
        accounts?.forEach { q.append(URLQueryItem(name: "accounts", value: $0)) }
        if let resp: Attribution = try? await APIClient.shared.get("/attribution", query: q) { all = resp.stocks }
    }
}

// MARK: - Allocation donut (reusable)

struct AllocationSlice: Identifiable {
    let id = UUID()
    let label: String
    let value: Double
}

/// Computes market-value allocation slices for a holding key (e.g. "Sector").
func allocationSlices(_ holdings: [Holding], key: String, currency: String) -> [AllocationSlice] {
    var totals: [String: Double] = [:]
    for h in holdings {
        let bucket = h.string(key) ?? "Unknown"
        let mv = h.marketValue(currency: currency) ?? 0
        guard mv > 0 else { continue }
        totals[bucket, default: 0] += mv
    }
    return totals.map { AllocationSlice(label: $0.key, value: $0.value) }.sorted { $0.value > $1.value }
}

struct AllocationDonutCard: View {
    let title: String
    let slices: [AllocationSlice]
    var icon: String = "chart.pie"
    var showLegend = true
    private var total: Double { slices.reduce(0) { $0 + $1.value } }

    var body: some View {
        Card(title: title, icon: icon) {
            if slices.isEmpty {
                EmptyHint(text: "No data", systemImage: "chart.pie")
            } else {
                Chart(slices) { slice in
                    SectorMark(angle: .value("Value", slice.value),
                               innerRadius: .ratio(0.6), angularInset: 1.5)
                        .foregroundStyle(by: .value("Label", slice.label))
                        .cornerRadius(3)
                }
                .frame(height: 220)
                if showLegend {
                    ForEach(slices.prefix(8)) { s in
                        HStack {
                            Text(s.label).lineLimit(1)
                            Spacer()
                            Text(total > 0 ? String(format: "%.1f%%", s.value / total * 100) : "—")
                                .monospacedDigit().foregroundStyle(.secondary)
                        }
                        .font(.caption)
                    }
                }
            }
        }
    }
}

// MARK: - Projected income

struct ProjectedIncomeCard: View {
    let income: [ProjectedIncome]
    var body: some View {
        Card(title: "Projected Dividend Income (12 mo)", icon: "chart.bar") {
            if income.isEmpty {
                EmptyHint(text: "No projected income.", systemImage: "chart.bar")
            } else {
                Chart(income) { item in
                    BarMark(x: .value("Month", item.month), y: .value("Income", item.value))
                        .foregroundStyle(.tint)
                }
                .chartHoverTooltip(income.map(\.month)) { i in
                    ChartTooltipContent(title: income[i].month,
                                        rows: [ChartTooltipRow(color: .accentColor, label: "Income",
                                                               value: Fmt.number(income[i].value, fractionDigits: 0))])
                }
                .frame(height: 200)
            }
        }
    }
}

// MARK: - Dividend calendar

struct DividendCalendarCard: View {
    @EnvironmentObject private var appState: AppState
    let events: [DividendEvent]
    let currency: String
    var body: some View {
        Card(title: "Upcoming Dividends", icon: "calendar") {
            if events.isEmpty {
                EmptyHint(text: "No upcoming dividends.", systemImage: "calendar")
            } else {
                ForEach(events.sorted { $0.exDividendDate < $1.exDividendDate }.prefix(10)) { ev in
                    VStack(spacing: 4) {
                        HStack {
                            Button {
                                appState.openStock(ev.symbol)
                            } label: {
                                Text(ev.symbol).fontWeight(.bold).foregroundStyle(.indigo)
                            }
                            .buttonStyle(.plain)
                            .frame(width: 70, alignment: .leading)
                            VStack(alignment: .leading) {
                                Text("Ex: \(ev.exDividendDate)").font(.caption)
                                Text("Pay: \(ev.dividendDate)").font(.caption2).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if ev.status == "estimated" {
                                Text("est.").font(.caption2).foregroundStyle(.orange)
                            }
                            Text(Fmt.currency(ev.amount, code: currency)).monospacedDigit()
                        }
                        Divider()
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }
}
