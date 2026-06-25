import SwiftUI
import Charts

// MARK: - Model

@MainActor
final class StockChartModel: ObservableObject {
    let symbol: String
    private let api: APIClient

    @Published var points: [ChartPoint] = []
    @Published var isLoading = false

    private var transactions: [Transaction] = []
    private var dividends: [Dividend] = []
    private var earnings: [EarningsDate] = []
    private var capitalGains: [CapitalGain] = []
    private var loadedTx = false, loadedDiv = false, loadedEarn = false, loadedCG = false

    init(symbol: String, api: APIClient = .shared) { self.symbol = symbol; self.api = api }

    enum EventKind: String { case buy, sell, dividend, earnings
        var color: Color { switch self {
            case .buy: return Color(hex: 0x16a34a); case .sell: return Color(hex: 0xdc2626)
            case .dividend: return Color(hex: 0xd97706); case .earnings: return Color(hex: 0x9333ea) } }
        var letter: String { switch self { case .buy: return "B"; case .sell: return "S"; case .dividend: return "D"; case .earnings: return "E" } }
    }
    struct ChartEvent: Identifiable { let id = UUID(); let kind: EventKind; let y: Double; var label: String; var gain: Double?; var gainPct: Double? }
    struct ChartPoint: Identifiable {
        let id = UUID(); let date: Date; var value: Double; var returnPct: Double; let volume: Double
        var sma50: Double?; var sma200: Double?; var bench: [String: Double]; var events: [ChartEvent] = []
    }

    // MARK: History

    private func fetchPeriod(_ p: String) -> String {
        switch p {
        case "1d": return "5d"; case "5d": return "1mo"; case "1m": return "3mo"; case "3m": return "6mo"
        case "6m": return "1y"; case "ytd": return "2y"; case "1y": return "5y"; case "3y": return "5y"
        case "5y": return "10y"; case "10y": return "max"; default: return "max"   // incl. custom → full history
        }
    }
    private func interval(_ p: String) -> String {
        switch p { case "1d": return "2m"; case "5d": return "15m"; default: return "1d" }
    }

    func loadHistory(period: String, fxRate: Double, benchmarks: [String], customFrom: Date? = nil, customTo: Date? = nil) async {
        isLoading = true; defer { isLoading = false }
        var query = [URLQueryItem(name: "period", value: fetchPeriod(period)),
                     URLQueryItem(name: "interval", value: interval(period))]
        query += benchmarks.map { URLQueryItem(name: "benchmarks", value: $0) }
        let raw: [StockHistoryPoint] = (try? await api.get("/stock_history/\(symbol)", query: query)) ?? []
        points = Self.process(raw, period: period, fxRate: fxRate, customFrom: customFrom, customTo: customTo)
    }

    static func process(_ raw: [StockHistoryPoint], period: String, fxRate: Double,
                        customFrom: Date? = nil, customTo: Date? = nil) -> [ChartPoint] {
        let pts: [(date: Date, value: Double, volume: Double, ret: Double, bench: [String: Double])] = raw.compactMap {
            guard let d = $0.parsedDate else { return nil }
            return (d, $0.value * fxRate, $0.volume, $0.returnPct ?? 0, $0.benchmarks)
        }
        guard !pts.isEmpty else { return [] }
        let sma50 = sma(pts.map(\.value), 50), sma200 = sma(pts.map(\.value), 200)
        let cut = cutoff(period, lastDate: pts.last!.date)
        let lowerTs = customFrom.map { Calendar.current.startOfDay(for: $0).timeIntervalSince1970 }
        let upperTs = customTo.map { Calendar.current.startOfDay(for: $0).timeIntervalSince1970 + 86400 }
        var out: [ChartPoint] = []
        for (i, p) in pts.enumerated() {
            let t = p.date.timeIntervalSince1970
            if period == "custom" {
                if let lo = lowerTs, t < lo { continue }
                if let hi = upperTs, t > hi { continue }
            } else if period == "1d" {
                var cal = Calendar(identifier: .gregorian)
                if let tz = TimeZone(identifier: "America/New_York") {
                    cal.timeZone = tz
                    let comps = cal.dateComponents([.year, .month, .day], from: pts.last!.date)
                    if let start = cal.date(from: DateComponents(year: comps.year, month: comps.month, day: comps.day, hour: 9, minute: 30))?.timeIntervalSince1970,
                       let end = cal.date(from: DateComponents(year: comps.year, month: comps.month, day: comps.day, hour: 16, minute: 0))?.timeIntervalSince1970 {
                        // Skip if it's off-hour today (t is outside 9:30-16:00 NY time).
                        if t < start || t > end { continue }
                    }
                }
            } else if period != "all" && period != "max" && t < cut { continue }
            out.append(ChartPoint(date: p.date, value: p.value, returnPct: p.ret, volume: p.volume,
                                  sma50: sma50[i], sma200: sma200[i], bench: p.bench))
        }
        guard let base = out.first?.value else { return out }
        let benchBase = out.first?.bench ?? [:]
        for i in out.indices {
            if base > 0 { out[i].returnPct = (out[i].value / base - 1) * 100 }
            var nb: [String: Double] = [:]
            for (k, bt) in out[i].bench where benchBase[k] != nil {
                let b0 = benchBase[k]!
                nb[k] = ((1 + bt / 100) / (1 + b0 / 100) - 1) * 100
            }
            out[i].bench = nb
        }
        return out
    }

    static func sma(_ values: [Double], _ period: Int) -> [Double?] {
        guard values.count >= period else { return Array(repeating: nil, count: values.count) }
        var out = [Double?](repeating: nil, count: values.count); var sum = 0.0
        for i in values.indices {
            sum += values[i]
            if i >= period { sum -= values[i - period] }
            if i >= period - 1 { out[i] = sum / Double(period) }
        }
        return out
    }

    static func cutoff(_ period: String, lastDate: Date) -> TimeInterval {
        let now = Date().timeIntervalSince1970, day = 86400.0
        switch period {
        case "1d": return Calendar.current.startOfDay(for: lastDate).timeIntervalSince1970
        case "5d": return now - 5 * day
        case "1m": return now - 30 * day
        case "3m": return now - 90 * day
        case "6m": return now - 180 * day
        case "ytd":
            let c = Calendar(identifier: .gregorian)
            return (c.date(from: c.dateComponents([.year], from: Date())) ?? Date()).timeIntervalSince1970
        case "1y": return now - 365 * day
        case "3y": return now - 3 * 365 * day
        case "5y": return now - 5 * 365 * day
        case "10y": return now - 10 * 365 * day
        default: return 0
        }
    }

    // MARK: Overlays

    private static let dayFmt: DateFormatter = {
        let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC"); f.dateFormat = "yyyy-MM-dd"; return f
    }()
    private func ts(_ s: String) -> TimeInterval? { Self.dayFmt.date(from: String(s.prefix(10)))?.timeIntervalSince1970 }

    func ensureOverlayData(showBuys: Bool, showSells: Bool, showDividends: Bool, showEarnings: Bool,
                           currency: String, accounts: [String]?) async {
        if (showBuys || showSells || showDividends) && !loadedTx {
            loadedTx = true
            var q = [URLQueryItem]()
            accounts?.forEach { q.append(URLQueryItem(name: "accounts", value: $0)) }
            transactions = (try? await api.get("/transactions", query: q)) ?? []
        }
        if showSells && !loadedCG {
            loadedCG = true
            var q = [URLQueryItem(name: "currency", value: currency)]
            accounts?.forEach { q.append(URLQueryItem(name: "accounts", value: $0)) }
            capitalGains = (try? await api.get("/capital_gains", query: q)) ?? []
        }
        if showDividends && !loadedDiv {
            loadedDiv = true
            var q = [URLQueryItem(name: "currency", value: currency)]
            accounts?.forEach { q.append(URLQueryItem(name: "accounts", value: $0)) }
            dividends = (try? await api.get("/dividends", query: q)) ?? []
        }
        if showEarnings && !loadedEarn {
            loadedEarn = true
            earnings = (try? await api.get("/earnings_dates/\(symbol)")) ?? []
        }
    }

    func applyEvents(showBuys: Bool, showSells: Bool, showDividends: Bool, showEarnings: Bool,
                     currency: String, fxRate: Double) {
        for i in points.indices { points[i].events = [] }
        guard !points.isEmpty else { return }
        let pts = points
        let day = 86400.0
        let lower = pts.first!.date.timeIntervalSince1970 - 4 * day
        let upper = pts.last!.date.timeIntervalSince1970 + 4 * day
        func snap(_ t: TimeInterval) -> Int? {
            if t < lower || t > upper { return nil }
            var best = 0; var bestDiff = abs(pts[0].date.timeIntervalSince1970 - t)
            for i in 1..<pts.count {
                let d = abs(pts[i].date.timeIntervalSince1970 - t)
                if d < bestDiff { bestDiff = d; best = i }
            }
            return best
        }

        // Split adjustment: divide trade price/qty by ratio of splits AFTER the trade.
        let splits: [(ts: TimeInterval, ratio: Double)] = transactions.compactMap { t in
            guard t.symbol == symbol else { return nil }
            let ty = t.type.lowercased()
            guard (ty == "split" || ty == "stock split"), let r = t.splitRatio, r > 0, let tt = ts(t.date) else { return nil }
            return (tt, r)
        }
        func splitFactorAfter(_ tradeTs: TimeInterval) -> Double {
            splits.reduce(1.0) { $1.ts > tradeTs ? $0 * $1.ratio : $0 }
        }
        func fmtQty(_ q: Double) -> String { q == q.rounded() ? String(Int(q)) : String(format: "%g", (q * 10000).rounded() / 10000) }

        var cgByTx: [Int: CapitalGain] = [:]
        if showSells { for cg in capitalGains { if let id = cg.originalTxId { cgByTx[id] = cg } } }

        // Signed split-adjusted share moves → shares held at a date (for dividend yield).
        var shareMoves: [(ts: TimeInterval, q: Double)] = []
        for t in transactions where t.symbol == symbol {
            let ty = t.type.lowercased()
            let isBuy = ty == "buy" || ty == "buy to cover", isSell = ty == "sell" || ty == "short sell"
            guard (isBuy || isSell), let tt = ts(t.date) else { continue }
            let qty = t.quantity * splitFactorAfter(tt)
            shareMoves.append((tt, isBuy ? qty : -qty))
        }
        func sharesHeldAt(_ t: TimeInterval) -> Double { shareMoves.reduce(0) { $1.ts <= t ? $0 + $1.q : $0 } }

        var agg: [String: ChartEvent] = [:]
        var sellCost: [String: Double] = [:]

        if showBuys || showSells {
            for t in transactions where t.symbol == symbol {
                let ty = t.type.lowercased()
                let isBuy = ty == "buy" || ty == "buy to cover", isSell = ty == "sell" || ty == "short sell"
                guard (isBuy && showBuys) || (isSell && showSells), let tt = ts(t.date), let idx = snap(tt) else { continue }
                let kind: EventKind = isBuy ? .buy : .sell
                let factor = splitFactorAfter(tt)
                let qty = t.quantity * factor
                let priceLocal = t.pricePerShare / factor
                let price = priceLocal > 0 ? priceLocal * fxRate : pts[idx].value
                let key = "\(kind.rawValue):\(idx)"
                var gain: Double?; var cost = 0.0
                if isSell, let id = t.id, let cg = cgByTx[id] { cost = cg.costBasisDisplay; gain = cg.realizedGainDisplay }
                let segment = "\(fmtQty(qty)) @ \(Fmt.currency(price, code: currency))"
                if var e = agg[key] {
                    e.label += ", \(segment)"
                    if let g = gain { e.gain = (e.gain ?? 0) + g }
                    agg[key] = e
                } else {
                    agg[key] = ChartEvent(kind: kind, y: price, label: "\(isBuy ? "Buy" : "Sell") \(segment)", gain: gain)
                }
                if isSell {
                    sellCost[key, default: 0] += cost
                    if var e = agg[key] {
                        let tc = sellCost[key] ?? 0
                        e.gainPct = (tc > 0 && e.gain != nil) ? (e.gain! / tc) * 100 : nil
                        agg[key] = e
                    }
                }
            }
        }

        if showDividends {
            for d in dividends where d.symbol == symbol {
                guard let dt = ts(d.date), let idx = snap(dt) else { continue }
                let key = "dividend:\(idx)"
                let prev = agg[key]?.gain ?? 0   // reuse gain slot to accumulate total amount
                let total = prev + d.amountDisplay
                let mv = sharesHeldAt(dt) * pts[idx].value
                let yieldPct = mv > 0 ? total / mv * 100 : nil
                var label = "Dividend \(Fmt.currency(total, code: currency))"
                if let y = yieldPct { label += String(format: " · %.2f%% yield", y) }
                agg[key] = ChartEvent(kind: .dividend, y: pts[idx].value, label: label, gain: total)
            }
        }

        if showEarnings {
            for e in earnings {
                guard let et = ts(e.date), let idx = snap(et) else { continue }
                var parts = ["Earnings"]
                if let a = e.epsActual { parts.append(String(format: "EPS %.2f", a)) }
                else if let est = e.epsEstimate { parts.append(String(format: "Est. EPS %.2f", est)) }
                if let s = e.surprisePct { parts.append(String(format: "(%@%.1f%%)", s >= 0 ? "+" : "", s)) }
                agg["earnings:\(idx)"] = ChartEvent(kind: .earnings, y: pts[idx].value, label: parts.joined(separator: " "))
            }
        }

        for (key, ev) in agg {
            guard let idx = Int(key.split(separator: ":").last ?? "") else { continue }
            var e = ev
            if e.kind == .dividend { e = ChartEvent(kind: .dividend, y: ev.y, label: ev.label) }  // clear amount-in-gain slot
            points[idx].events.append(e)
        }
    }
}

// MARK: - View

struct StockPriceChartView: View {
    let symbol: String
    let currency: String
    var avgCost: Double?
    var fxRate: Double = 1
    var accounts: [String]?
    var hidePrice = false

    @StateObject private var model: StockChartModel
    @State private var view: ChartViewMode = .price
    @State private var period = "1y"
    @State private var showSMA50 = false
    @State private var showSMA200 = false
    @State private var showBuys = false
    @State private var showSells = false
    @State private var showDividends = false
    @State private var showEarnings = false
    @State private var selectedBenchmarks: [String] = []
    @State private var customFrom = Calendar.current.date(byAdding: .year, value: -1, to: Date()) ?? Date()
    @State private var customTo = Date()

    init(symbol: String, currency: String, avgCost: Double? = nil, fxRate: Double = 1,
         accounts: [String]? = nil, hidePrice: Bool = false) {
        self.symbol = symbol; self.currency = currency; self.avgCost = avgCost
        self.fxRate = fxRate; self.accounts = accounts; self.hidePrice = hidePrice
        _model = StateObject(wrappedValue: StockChartModel(symbol: symbol))
    }

    enum ChartViewMode { case price, return_ }
    private struct Benchmark { let name: String; let key: String; let color: Color }
    private let benchmarks = [
        Benchmark(name: "S&P 500", key: "^GSPC", color: Color(hex: 0xf59e0b)),
        Benchmark(name: "NASDAQ", key: "^IXIC", color: Color(hex: 0x8b5cf6)),
        Benchmark(name: "Dow Jones", key: "^DJI", color: Color(hex: 0x0ea5e9)),
    ]
    private let periods: [(String, String)] = [
        ("1D", "1d"), ("5D", "5d"), ("1M", "1m"), ("3M", "3m"), ("6M", "6m"), ("YTD", "ytd"),
        ("1Y", "1y"), ("3Y", "3y"), ("5Y", "5y"), ("10Y", "10y"), ("All", "all"), ("Custom", "custom"),
    ]

    private var pts: [StockChartModel.ChartPoint] { model.points }
    private var intraday: Bool { period == "1d" || period == "5d" }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            headerRow
            periodRow
            if period == "custom" { customDateRow }
            if view == .price { overlayRow } else { benchmarkRow }
            chartArea
        }
        .task { await reloadHistory() }
        .onChange(of: period) { _, _ in Task { await reloadHistory() } }
        .onChange(of: selectedBenchmarks) { _, _ in Task { await reloadHistory() } }
        .onChange(of: customFrom) { _, _ in Task { await reloadHistory() } }
        .onChange(of: customTo) { _, _ in Task { await reloadHistory() } }
        .onChange(of: [showBuys, showSells, showDividends, showEarnings]) { _, _ in Task { await refreshEvents() } }
    }

    private func reloadHistory() async {
        let names = view == .return_ ? selectedBenchmarks : []
        await model.loadHistory(period: period, fxRate: fxRate, benchmarks: names,
                                customFrom: period == "custom" ? customFrom : nil,
                                customTo: period == "custom" ? customTo : nil)
        await refreshEvents()
    }

    private var customDateRow: some View {
        HStack(spacing: 8) {
            Text("FROM").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
            DatePicker("", selection: $customFrom, in: ...customTo, displayedComponents: .date)
                .labelsHidden().datePickerStyle(.compact).fixedSize().gregorianCalendar()
            Text("TO").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
            DatePicker("", selection: $customTo, in: customFrom...Date(), displayedComponents: .date)
                .labelsHidden().datePickerStyle(.compact).fixedSize().gregorianCalendar()
            Spacer()
        }
    }
    private func refreshEvents() async {
        await model.ensureOverlayData(showBuys: showBuys, showSells: showSells, showDividends: showDividends,
                                      showEarnings: showEarnings, currency: currency, accounts: accounts)
        model.applyEvents(showBuys: showBuys, showSells: showSells, showDividends: showDividends,
                          showEarnings: showEarnings, currency: currency, fxRate: fxRate)
    }

    // MARK: Header

    private var stats: (change: Double, pct: Double)? {
        guard let first = pts.first, let last = pts.last, pts.count >= 2 else { return nil }
        let change = last.value - first.value
        return (change, first.value != 0 ? change / first.value * 100 : 0)
    }

    private var headerRow: some View {
        HStack(alignment: .center) {
            if let s = stats {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    if !hidePrice, let last = pts.last {
                        Text(Fmt.currency(last.value, code: currency)).font(.title.bold())
                    }
                    Text("\(Fmt.currency(s.change, code: currency)) (\(String(format: "%.2f%%", s.pct)))")
                        .font(.callout.weight(.medium)).foregroundStyle(s.change >= 0 ? .green : .red)
                }
            } else { Color.clear.frame(height: 28) }
            Spacer()
            if view == .price {
                HStack(spacing: 4) {
                    toggleChip("MA50", showSMA50, Color(hex: 0xf97316)) { showSMA50.toggle() }
                    toggleChip("MA200", showSMA200, Color(hex: 0x9333ea)) { showSMA200.toggle() }
                }
            }
            Picker("", selection: $view) {
                Text("Price").tag(ChartViewMode.price)
                Text("Return %").tag(ChartViewMode.return_)
            }
            .pickerStyle(.segmented).fixedSize()
            .onChange(of: view) { _, _ in Task { await reloadHistory() } }
        }
    }

    private var periodRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(periods, id: \.1) { label, value in
                    Button { period = value } label: {
                        Text(label).font(.caption.weight(.semibold))
                            .padding(.horizontal, 10).padding(.vertical, 4)
                            .background(period == value ? Color.accentColor : Color.gray.opacity(0.15), in: Capsule())
                            .foregroundStyle(period == value ? .white : .secondary)
                    }.buttonStyle(.plain)
                }
            }
        }
    }

    private var overlayRow: some View {
        HStack(spacing: 8) {
            Text("OVERLAYS").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
            // Scrolls so the pills keep their natural width instead of being
            // squeezed (which wrapped "Dividends"/"Earnings" onto two lines).
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    pill("Buys", showBuys, StockChartModel.EventKind.buy.color) { showBuys.toggle() }
                    pill("Sells", showSells, StockChartModel.EventKind.sell.color) { showSells.toggle() }
                    pill("Dividends", showDividends, StockChartModel.EventKind.dividend.color) { showDividends.toggle() }
                    pill("Earnings", showEarnings, StockChartModel.EventKind.earnings.color) { showEarnings.toggle() }
                }
            }
        }
    }

    private var benchmarkRow: some View {
        HStack(spacing: 8) {
            Text("BENCHMARKS").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(benchmarks, id: \.key) { b in
                        pill(b.name, selectedBenchmarks.contains(b.name), b.color) {
                            if let i = selectedBenchmarks.firstIndex(of: b.name) { selectedBenchmarks.remove(at: i) }
                            else { selectedBenchmarks.append(b.name) }
                        }
                    }
                }
            }
        }
    }

    private func toggleChip(_ label: String, _ on: Bool, _ color: Color, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label).font(.system(size: 11, weight: .bold))
                .lineLimit(1).fixedSize(horizontal: true, vertical: false)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(on ? color : Color.gray.opacity(0.15), in: RoundedRectangle(cornerRadius: 6))
                .foregroundStyle(on ? .white : .secondary)
        }.buttonStyle(.plain)
    }
    private func pill(_ label: String, _ on: Bool, _ color: Color, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Circle().fill(on ? .white : color).frame(width: 7, height: 7)
                Text(label).font(.system(size: 11, weight: .bold))
                    .lineLimit(1).fixedSize(horizontal: true, vertical: false)
            }
            .padding(.horizontal, 9).padding(.vertical, 4)
            .background(on ? color : Color.gray.opacity(0.12), in: Capsule())
            .foregroundStyle(on ? .white : .secondary)
        }.buttonStyle(.plain)
    }

    // MARK: Chart

    private var yDomain: ClosedRange<Double> {
        var vals: [Double] = []
        if view == .price {
            vals = pts.map(\.value)
            if showSMA50 { vals += pts.compactMap(\.sma50) }
            if showSMA200 { vals += pts.compactMap(\.sma200) }
            if let a = avgCost, a > 0 { vals.append(a) }
        } else {
            vals = pts.map(\.returnPct) + [0]
            for b in benchmarks where selectedBenchmarks.contains(b.name) {
                vals += pts.compactMap { $0.bench[b.key] }
            }
        }
        return chartDomain(vals)
    }

    private var gradientOffset: Double {
        let rs = pts.map(\.returnPct)
        guard let mx = rs.max(), let mn = rs.min() else { return 0 }
        if mx <= 0 { return 0 }; if mn >= 0 { return 1 }
        return mx / (mx - mn)
    }

    private struct EventMark: Identifiable { let id = UUID(); let date: Date; let y: Double; let kind: StockChartModel.EventKind }
    private var eventMarks: [EventMark] {
        guard view == .price else { return [] }
        return pts.flatMap { p in p.events.map { EventMark(date: p.date, y: $0.y, kind: $0.kind) } }
    }

    @ViewBuilder private var chartArea: some View {
        ZStack {
            if model.isLoading && pts.isEmpty {
                ProgressView().frame(height: 400)
            } else if pts.isEmpty {
                ContentUnavailableView("No data available", systemImage: "chart.xyaxis.line").frame(height: 400)
            } else {
                chart.frame(height: 400)
            }
        }
    }

    private var chart: some View {
        let lo = yDomain.lowerBound, hi = yDomain.upperBound
        let volMax = max(pts.map(\.volume).max() ?? 1, 1)
        let off = gradientOffset
        return Chart {
            // Volume bars anchored at the domain floor.
            ForEach(pts) { p in
                BarMark(x: .value("Date", p.date),
                        yStart: .value("v0", lo),
                        yEnd: .value("v1", lo + (p.volume / volMax) * (hi - lo) * 0.18),
                        width: intraday ? .automatic : .fixed(3))
                    .foregroundStyle(.gray.opacity(0.15))
            }

            if view == .price {
                ForEach(pts) { p in
                    AreaMark(x: .value("Date", p.date), y: .value("Price", p.value))
                        .foregroundStyle(.linearGradient(colors: [Color(hex: 0x2563eb).opacity(0.3), .clear], startPoint: .top, endPoint: .bottom))
                    LineMark(x: .value("Date", p.date), y: .value("Price", p.value))
                        .foregroundStyle(Color(hex: 0x2563eb)).lineStyle(.init(lineWidth: 2)).interpolationMethod(.monotone)
                }
                if showSMA50 {
                    ForEach(pts.filter { $0.sma50 != nil }) { p in
                        LineMark(x: .value("Date", p.date), y: .value("SMA50", p.sma50!), series: .value("s", "sma50"))
                            .foregroundStyle(Color(hex: 0xf97316)).lineStyle(.init(lineWidth: 1.5))
                    }
                }
                if showSMA200 {
                    ForEach(pts.filter { $0.sma200 != nil }) { p in
                        LineMark(x: .value("Date", p.date), y: .value("SMA200", p.sma200!), series: .value("s", "sma200"))
                            .foregroundStyle(Color(hex: 0x9333ea)).lineStyle(.init(lineWidth: 1.5))
                    }
                }
                if let a = avgCost, a > 0 {
                    RuleMark(y: .value("Avg Cost", a))
                        .foregroundStyle(Color(hex: 0x64748b)).lineStyle(.init(lineWidth: 1.5, dash: [5, 5]))
                        .annotation(position: .top, alignment: .trailing) {
                            Text("AVG COST: \(Fmt.currency(a, code: currency))")
                                .font(.system(size: 11, weight: .bold)).foregroundStyle(Color(hex: 0x64748b))
                        }
                }
                ForEach(eventMarks) { m in
                    PointMark(x: .value("Date", m.date), y: .value("y", m.y))
                        .symbolSize(150).foregroundStyle(m.kind.color)
                        .annotation(position: .overlay) {
                            Text(m.kind.letter).font(.system(size: 10, weight: .bold)).foregroundStyle(.white)
                        }
                }
            } else {
                ForEach(pts) { p in
                    AreaMark(x: .value("Date", p.date), y: .value("Return", p.returnPct))
                        .foregroundStyle(.linearGradient(stops: [
                            .init(color: Color(hex: 0x10b981).opacity(0.15), location: 0),
                            .init(color: Color(hex: 0x10b981).opacity(0.15), location: off),
                            .init(color: Color(hex: 0xef4444).opacity(0.15), location: off),
                            .init(color: Color(hex: 0xef4444).opacity(0.15), location: 1),
                        ], startPoint: .top, endPoint: .bottom))
                    LineMark(x: .value("Date", p.date), y: .value("Return", p.returnPct))
                        .foregroundStyle(.linearGradient(stops: [
                            .init(color: Color(hex: 0x10b981), location: 0),
                            .init(color: Color(hex: 0x10b981), location: off),
                            .init(color: Color(hex: 0xef4444), location: off),
                            .init(color: Color(hex: 0xef4444), location: 1),
                        ], startPoint: .top, endPoint: .bottom))
                        .lineStyle(.init(lineWidth: 2)).interpolationMethod(.monotone)
                }
                ForEach(benchmarks.filter { selectedBenchmarks.contains($0.name) }, id: \.key) { b in
                    ForEach(pts.filter { $0.bench[b.key] != nil }) { p in
                        LineMark(x: .value("Date", p.date), y: .value(b.name, p.bench[b.key]!), series: .value("b", b.key))
                            .foregroundStyle(b.color).lineStyle(.init(lineWidth: 1.5))
                    }
                }
                RuleMark(y: .value("Zero", 0)).foregroundStyle(.secondary.opacity(0.5)).lineStyle(.init(lineWidth: 1, dash: [3, 3]))
            }
        }
        .chartYScale(domain: yDomain)
        .chartYAxis {
            AxisMarks { v in
                AxisGridLine()
                AxisValueLabel {
                    if let d = v.as(Double.self) {
                        Text(view == .return_ ? String(format: "%.1f%%", d)
                                              : Fmt.number(d, fractionDigits: d < 10 ? 2 : 0))
                    }
                }
            }
        }
        .chartXAxis {
            AxisMarks { value in
                AxisGridLine()
                AxisValueLabel {
                    if let d = value.as(Date.self) { Text(xLabel(d)) }
                }
            }
        }
        .chartHoverTooltip(pts.map(\.date)) { i in tooltip(pts[i]) }
    }

    private func xLabel(_ d: Date) -> String {
        let f = DateFormatter(); f.timeZone = TimeZone(identifier: "America/New_York")
        f.dateFormat = intraday ? "h:mm a" : "MMM d"
        return f.string(from: d)
    }

    private func tooltip(_ p: StockChartModel.ChartPoint) -> ChartTooltipContent {
        let tf = DateFormatter(); tf.timeZone = TimeZone(identifier: "America/New_York")
        tf.dateFormat = intraday ? "EEE, MMM d h:mm a" : "EEE, MMM d, yyyy"
        var rows: [ChartTooltipRow] = []
        rows.append(ChartTooltipRow(color: Color(hex: 0x2563eb), label: symbol,
                                    value: view == .price ? Fmt.currency(p.value, code: currency)
                                                          : String(format: "%.2f%%", p.returnPct)))
        rows.append(ChartTooltipRow(label: "Volume", value: formatVolume(p.volume)))
        if view == .price, showSMA50, let s = p.sma50 { rows.append(ChartTooltipRow(color: Color(hex: 0xf97316), label: "SMA 50", value: Fmt.currency(s, code: currency))) }
        if view == .price, showSMA200, let s = p.sma200 { rows.append(ChartTooltipRow(color: Color(hex: 0x9333ea), label: "SMA 200", value: Fmt.currency(s, code: currency))) }
        if view == .price {
            for e in p.events {
                var v = e.label
                if let g = e.gain { v = "\(g >= 0 ? "+" : "−")\(Fmt.currency(abs(g), code: currency))" + (e.gainPct.map { String(format: " (%@%.2f%%)", $0 >= 0 ? "+" : "", $0) } ?? "") }
                rows.append(ChartTooltipRow(color: e.kind.color, label: "\(e.kind.letter) \(e.label)", value: e.gain != nil ? v : ""))
            }
        } else {
            for b in benchmarks where selectedBenchmarks.contains(b.name) {
                if let v = p.bench[b.key] { rows.append(ChartTooltipRow(color: b.color, label: b.name, value: String(format: "%.2f%%", v))) }
            }
        }
        return ChartTooltipContent(title: tf.string(from: p.date), rows: rows)
    }

    private func formatVolume(_ v: Double) -> String {
        if v >= 1e9 { return String(format: "%.2fB", v / 1e9) }
        if v >= 1e6 { return String(format: "%.2fM", v / 1e6) }
        if v >= 1e3 { return String(format: "%.2fK", v / 1e3) }
        return String(Int(v))
    }
}
