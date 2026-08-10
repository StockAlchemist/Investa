import SwiftUI
import Charts

// MARK: - Column model (mirrors web HoldingsTable COLUMN_DEFINITIONS)

/// header → backing data key (resolved with currency suffix at read time).
private let columnKey: [String: String] = [
    "Account": "Account", "Symbol": "Symbol", "Sector": "Sector", "Industry": "Industry",
    "Quantity": "Quantity", "Day Chg": "Day Change", "Day Chg %": "Day Change %",
    "Avg Cost": "Avg Cost", "Price": "Price", "Cost Basis": "Cost Basis", "Mkt Val": "Market Value",
    "% of Total": "pct_of_total", "Unreal. G/L": "Unreal. Gain", "Unreal. G/L %": "Unreal. Gain %",
    "Real. G/L": "Realized Gain", "Divs": "Dividends", "Fees": "Commissions", "Total G/L": "Total Gain",
    "Total Ret %": "Total Return %", "IRR (%)": "IRR (%)", "Total Buy Cost": "Total Buy Cost",
    "Yield (Cost) %": "Div. Yield (Cost) %", "Yield (Mkt) %": "Div. Yield (Current) %",
    "FX G/L %": "FX Gain/Loss %", "Est. Income": "Est. Ann. Income", "1M Trend": "sparkline_1m",
    "Tags": "Tags", "Contribution %": "Contribution %", "AI Score": "ai_score", "Intrinsic Value": "intrinsic_value",
]

private let columnPickerGroups: [(label: String, cols: [String])] = [
    ("Core", ["Symbol", "Account", "Quantity", "Price", "Mkt Val", "% of Total", "1M Trend"]),
    ("Daily", ["Day Chg", "Day Chg %"]),
    ("Returns", ["Unreal. G/L", "Unreal. G/L %", "Real. G/L", "Total G/L", "Total Ret %", "IRR (%)"]),
    ("Cost", ["Avg Cost", "Cost Basis", "Total Buy Cost"]),
    ("Income", ["Divs", "Est. Income", "Yield (Cost) %", "Yield (Mkt) %"]),
    ("Details", ["Sector", "Industry", "FX G/L %", "Fees", "Tags", "Contribution %", "AI Score", "Intrinsic Value"]),
]

private let defaultVisibleColumns = columnPickerGroups.flatMap { $0.cols }

private let leftAlignedHeaders: Set<String> = ["Symbol", "Account", "Sector", "Industry", "Tags"]
private let glHeaders: Set<String> = ["Day Chg", "Day Chg %", "Unreal. G/L", "Unreal. G/L %", "Real. G/L", "Total G/L", "Total Ret %", "FX G/L %", "IRR (%)"]
private let heatmapHeaders: Set<String> = ["Day Chg %", "Unreal. G/L %", "Total Ret %"]
private let sumHeaders = ["Quantity", "Mkt Val", "Cost Basis", "Day Chg", "Unreal. G/L", "Real. G/L", "Divs", "Fees", "Total G/L", "Total Buy Cost", "Est. Income", "Contribution %", "% of Total"]

private func columnWidth(_ h: String) -> CGFloat {
    switch h {
    case "Symbol": return 172
    case "1M Trend": return 110
    case "Tags": return 170
    case "Industry": return 150
    case "Sector": return 130
    case "Intrinsic Value": return 150
    case "Account", "Total Buy Cost", "% of Total", "Contribution %", "Mkt Val", "Unreal. G/L": return 112
    case "Cost Basis", "Real. G/L", "Total G/L", "Est. Income", "Yield (Cost) %", "Yield (Mkt) %", "Unreal. G/L %", "Total Ret %", "Day Chg": return 100
    case "Price", "Avg Cost", "Day Chg %", "Divs", "FX G/L %": return 90
    case "Quantity", "IRR (%)": return 80
    case "AI Score": return 72
    default: return 96
    }
}

// MARK: - Grouping maps (mirror web)

private let groupingOptions: [(key: String, label: String)] = [
    ("Market", "Market"), ("Currency", "Currency"), ("Sector", "Sector"),
    ("Industry", "Industry"), ("quoteType", "Investment Type"), ("Country", "Country"),
]
private let investmentTypeMap = ["EQUITY": "Stocks", "ETF": "ETFs", "CASH": "Cash", "MUTUALFUND": "Mutual Funds"]
private let currencyNameMap = ["USD": "US Dollar", "THB": "Thai Baht", "EUR": "Euro", "GBP": "British Pound",
                               "SGD": "Singapore Dollar", "JPY": "Japanese Yen", "HKD": "Hong Kong Dollar"]

private func normalizeMarketName(_ market: String) -> String {
    if market.isEmpty { return "Unknown" }
    let m = market.uppercased()
    if m.contains("NASDAQ") || m == "NMS" || m == "NGM" || m == "NCM" { return "NASDAQ" }
    if m == "NYQ" || m == "NYSE" || m.contains("NEW YORK") { return "NYSE" }
    if m == "ASE" || m == "AMEX" { return "AMEX" }
    if m == "PCX" || m == "ARCA" || m.contains("ARCA") { return "NYSE Arca" }
    return market
}

// MARK: - Resolved row

/// A holding flattened to header→value, supporting per-symbol aggregation.
private struct HRow: Identifiable {
    var id: String
    var symbol: String
    var account: String
    var num: [String: Double] = [:]   // numeric headers present
    var text: [String: String] = [:]  // Account / Symbol / Sector / Industry
    var tags: [String] = []
    /// ~22 daily closes — the "1M Trend" column and the trend line inside an
    /// expanded holding card.
    var trend1m: [Double] = []
    var lots: [JSONValue] = []
    var meta: [String: String] = [:]  // grouping fields
    var price: Double { num["Price"] ?? 0 }
    var intrinsic: Double? { num["Intrinsic Value"] }
    var mos: Double?
    var aiScore: Double? { num["AI Score"] }
}

// MARK: - View

struct HoldingsTableView: View {
    let holdings: [Holding]
    let currency: String

    @Environment(\.horizontalSizeClass) var hSizeClass
    @State private var visibleColumns = defaultVisibleColumns
    @State private var sortKey = "Mkt Val"
    @State private var sortAsc = false
    @State private var search = ""
    @State private var selectedAccounts: Set<String> = []
    @State private var groupBy: String?
    @State private var expandedGroups: Set<String> = []
    @State private var expandedLots: Set<String> = []
    /// Holding cards start collapsed to a single summary line — the full metric
    /// grid is ~20 fields, so one expanded card fills a phone screen. Tracked as
    /// the *expanded* set (empty = all collapsed) and persisted, so the cards a
    /// user opens stay open across launches.
    @AppStorage("investa.holdings.expandedCards") private var expandedCardsRaw = ""
    @State private var visibleRows = 10
    @State private var showColumns = false
    @State private var detail: SymbolID?
    @State private var tagEdit: TagEdit?

    /// Identifies a holding whose tags are being edited (applies across all of
    /// its accounts).
    struct TagEdit: Identifiable { let id = UUID(); let symbol: String; let accounts: [String]; let tags: [String] }

    private let rowHeight: CGFloat = 46
    // Lot rows need a fixed height too: the pinned Symbol column and the
    // scrollable columns render lots independently, so a content-sized height
    // (italic date text vs. monospaced numbers) diverges and drifts the two
    // panels out of vertical alignment. A shared fixed height keeps them locked.
    private let lotRowHeight: CGFloat = 26

    private let numericHeaders: Set<String> = ["Quantity", "Day Chg", "Day Chg %", "Avg Cost", "Price", "Cost Basis", "Mkt Val", "% of Total", "Unreal. G/L", "Unreal. G/L %", "Real. G/L", "Divs", "Fees", "Total G/L", "Total Ret %", "IRR (%)", "Total Buy Cost", "Yield (Cost) %", "Yield (Mkt) %", "FX G/L %", "Est. Income", "Contribution %", "AI Score", "Intrinsic Value"]

    // MARK: Value resolution

    private func resolve(_ h: Holding, _ key: String) -> JSONValue? {
        if let v = h.raw[key] { return v }
        if let v = h.raw["\(key) (\(currency))"] { return v }
        if let k = h.raw.keys.first(where: { $0.hasPrefix(key) }) { return h.raw[k] }
        return nil
    }

    private var totalMarketValue: Double {
        holdings.reduce(0) { $0 + ($1.marketValue(currency: currency) ?? 0) }
    }

    // MARK: Build resolved rows

    private func makeRow(_ h: Holding) -> HRow {
        var r = HRow(id: h.id, symbol: h.symbol, account: h.account ?? "")
        for header in numericHeaders {
            guard let key = columnKey[header] else { continue }
            if let d = resolve(h, key)?.doubleValue { r.num[header] = d }
        }
        let mv = h.marketValue(currency: currency) ?? 0
        if r.num["% of Total"] == nil { r.num["% of Total"] = totalMarketValue > 0 ? mv / totalMarketValue * 100 : 0 }
        r.text["Symbol"] = h.symbol
        r.text["Account"] = h.account ?? ""
        r.text["Sector"] = h.string("Sector") ?? ""
        r.text["Industry"] = h.string("Industry") ?? ""
        r.tags = h.raw["Tags"]?.arrayValue?.compactMap { $0.stringValue } ?? []
        r.trend1m = h.raw["sparkline_1m"]?.arrayValue?.compactMap { $0.doubleValue } ?? []
        r.lots = h.raw["lots"]?.arrayValue ?? []
        r.mos = h.double("margin_of_safety")
        r.meta["exchange"] = h.string("fullExchangeName") ?? h.string("exchange") ?? h.string("Market") ?? "Unknown"
        r.meta["currency"] = h.string("Local Currency") ?? "Unknown"
        r.meta["quoteType"] = h.string("quoteType") ?? "Other"
        r.meta["geography"] = h.string("geography") ?? h.string("Country") ?? "Unknown"
        r.meta["Country"] = r.meta["geography"]
        r.meta["Sector"] = h.string("Sector") ?? "Other"
        r.meta["Industry"] = h.string("Industry") ?? "Other"
        return r
    }

    /// A cash balance below a cent is rounding residue left by FX conversion and
    /// fee postings — a row reading "-0.0002" is noise, not a position. Compared
    /// on magnitude so a genuine negative (margin) balance still gets a row.
    private static let cashDustThreshold = 0.01

    private func isCashDust(_ h: Holding) -> Bool {
        let s = h.symbol.uppercased()
        guard s == "$CASH" || s == "CASH" || s.hasPrefix("CASH (") else { return false }
        // Already in the display currency, so the threshold reads as a cent of
        // whatever the user is looking at.
        return abs(h.marketValue(currency: currency) ?? 0) <= Self.cashDustThreshold
    }

    /// Filtered + (optionally) symbol-aggregated rows.
    private var baseRows: [HRow] {
        let filtered = holdings.filter {
            (search.isEmpty || $0.symbol.lowercased().contains(search.lowercased()))
            && (selectedAccounts.isEmpty || selectedAccounts.contains($0.account ?? ""))
            && !isCashDust($0)
        }
        let rows = filtered.map(makeRow)
        if visibleColumns.contains("Account") { return rows }

        // Aggregate by symbol.
        var order: [String] = []
        var bySymbol: [String: HRow] = [:]
        for r in rows {
            if var cur = bySymbol[r.symbol] {
                for k in sumHeaders { if let v = r.num[k] { cur.num[k, default: 0] += v } }
                cur.lots += r.lots
                cur.tags = Array(Set(cur.tags + r.tags))
                cur.id = r.symbol
                bySymbol[r.symbol] = cur
            } else {
                var c = r; c.id = r.symbol; bySymbol[r.symbol] = c; order.append(r.symbol)
            }
        }
        return order.compactMap { bySymbol[$0] }.map(recompute)
    }

    private func recompute(_ row: HRow) -> HRow {
        var r = row
        let qty = r.num["Quantity"] ?? 0, mv = r.num["Mkt Val"] ?? 0, cost = r.num["Cost Basis"] ?? 0
        let day = r.num["Day Chg"] ?? 0, unreal = r.num["Unreal. G/L"] ?? 0
        let est = r.num["Est. Income"] ?? 0, total = r.num["Total G/L"] ?? 0
        if qty != 0 { r.num["Price"] = mv / qty; r.num["Avg Cost"] = cost / qty }
        r.num["Day Chg %"] = (mv - day != 0) ? day / (mv - day) * 100 : 0
        let eps = 0.0001
        let buy = r.num["Total Buy Cost"] ?? 0
        let denom = abs(buy) > eps ? buy : cost
        if abs(denom) > eps {
            r.num["Unreal. G/L %"] = unreal / denom * 100
            r.num["Yield (Cost) %"] = est / denom * 100
            r.num["Total Ret %"] = total / denom * 100
        }
        if mv != 0 { r.num["Yield (Mkt) %"] = est / mv * 100 }
        return r
    }

    private func numValue(_ r: HRow, _ header: String) -> Double? { r.num[header] }
    private func textValue(_ r: HRow, _ header: String) -> String? { r.text[header] }

    private func sortedRows(_ rows: [HRow]) -> [HRow] {
        rows.sorted { a, b in
            if numericHeaders.contains(sortKey) {
                let va = a.num[sortKey], vb = b.num[sortKey]
                if va == nil { return false }; if vb == nil { return true }
                return sortAsc ? va! < vb! : va! > vb!
            } else {
                let va = a.text[sortKey] ?? "", vb = b.text[sortKey] ?? ""
                return sortAsc ? va < vb : va > vb
            }
        }
    }

    // MARK: Grouping

    private struct HGroup: Identifiable { let key: String; var rows: [HRow]; var agg: [String: Double]; var id: String { key } }

    private func groupKey(_ r: HRow) -> String {
        switch groupBy {
        case "Market": return normalizeMarketName(r.meta["exchange"] ?? "Unknown")
        case "quoteType": let t = r.meta["quoteType"] ?? "Other"; return investmentTypeMap[t] ?? t
        case "Country": return r.meta["geography"] ?? "Unknown"
        case "Currency": let c = r.meta["currency"] ?? "Unknown"; return currencyNameMap[c] ?? c
        case "Sector": return r.meta["Sector"] ?? "Other"
        case "Industry": return r.meta["Industry"] ?? "Other"
        default: return "Other"
        }
    }

    private var groups: [HGroup] {
        var map: [String: HGroup] = [:]
        var order: [String] = []
        for r in baseRows {
            let k = groupKey(r)
            if map[k] == nil { map[k] = HGroup(key: k, rows: [], agg: [:]); order.append(k) }
            map[k]!.rows.append(r)
            for f in ["Mkt Val", "Day Chg", "Cost Basis", "Unreal. G/L", "Real. G/L", "Divs", "Fees", "Total G/L", "Total Buy Cost"] {
                map[k]!.agg[f, default: 0] += r.num[f] ?? 0
            }
        }
        return order.compactMap { map[$0] }.map { g in
            var gg = g
            let mv = gg.agg["Mkt Val"] ?? 0, day = gg.agg["Day Chg"] ?? 0
            if mv != 0, mv - day != 0 { gg.agg["Day Chg %"] = day / (mv - day) * 100 }
            let denom = abs(gg.agg["Total Buy Cost"] ?? 0) > 0.0001 ? (gg.agg["Total Buy Cost"] ?? 0) : (gg.agg["Cost Basis"] ?? 0)
            if abs(denom) > 0.0001 {
                gg.agg["Unreal. G/L %"] = (gg.agg["Unreal. G/L"] ?? 0) / denom * 100
                gg.agg["Total Ret %"] = (gg.agg["Total G/L"] ?? 0) / denom * 100
            }
            gg.rows = sortedRows(gg.rows)
            return gg
        }.sorted { ($0.agg["Mkt Val"] ?? 0) > ($1.agg["Mkt Val"] ?? 0) }
    }

    private var totalWidth: CGFloat { visibleColumns.reduce(0) { $0 + columnWidth($1) } }

    // MARK: Body

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            toolbar
            if holdings.isEmpty {
                EmptyHint(text: "No holdings found.", systemImage: "tray").frame(height: 160)
            } else {
                #if os(iOS)
                if hSizeClass == .compact {
                    if let gb = groupBy, !gb.isEmpty { iosGroupedBody } else { iosFlatBody }
                } else {
                    macBody
                }
                #else
                macBody
                #endif
                
                if groupBy == nil { pagination }
            }
        }
        .padding(16)
        .overlay(alignment: .top) { Rectangle().fill(Color(hex: 0x6366f1).opacity(0.8)).frame(height: 2) }
        .card(.standard)
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: currency) }
        .sheet(item: $tagEdit) { edit in
            TagEditorSheet(edit: edit) {
                NotificationCenter.default.post(name: .refreshRequested, object: nil)
            }
        }
        .onChange(of: groupBy) { _, _ in expandedGroups = Set(groups.map(\.key)) }
    }

    private var macBody: some View {
        // `.top` alignment is essential: the scrollable side is a horizontal
        // ScrollView that expands to fill the card's height, while the pinned
        // first column is sized to its content. Without `.top` the HStack
        // centers the shorter pinned column, offsetting it from the rows beside it.
        HStack(alignment: .top, spacing: 0) {
            if let firstCol = visibleColumns.first {
                let pinnedCols = [firstCol]
                let scrollCols = Array(visibleColumns.dropFirst())
                let fw = columnWidth(firstCol)
                let scrollWidth = scrollCols.reduce(0) { $0 + columnWidth($1) }

                // Pinned first column — renders only the first column's cells.
                VStack(alignment: .leading, spacing: 0) {
                    headerRow(columns: pinnedCols)
                    Divider()
                    if let gb = groupBy, !gb.isEmpty { groupedBody(columns: pinnedCols, width: fw, part: .label) } else { flatBody(columns: pinnedCols) }
                }
                .frame(width: fw, alignment: .leading)
                .background(.background.secondary)
                .zIndex(1)

                Divider()

                // Scrollable remaining columns — renders only the non-pinned cells.
                ScrollView(.horizontal, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 0) {
                        headerRow(columns: scrollCols)
                        Divider()
                        if let gb = groupBy, !gb.isEmpty { groupedBody(columns: scrollCols, width: scrollWidth, part: .stats) } else { flatBody(columns: scrollCols) }
                    }
                    .frame(width: scrollWidth, alignment: .leading)
                }
                .clipped()
            } else {
                ScrollView(.horizontal, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 0) {
                        headerRow
                        Divider()
                        if let gb = groupBy, !gb.isEmpty { groupedBody } else { flatBody }
                    }
                    .frame(width: totalWidth, alignment: .leading)
                }
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "tablecells").font(.caption.weight(.semibold)).foregroundStyle(Color(hex: 0x6366f1))
            #if !os(iOS)
            Text("Holdings").font(.caption.weight(.semibold)).tracking(0.8).textCase(.uppercase).foregroundStyle(.secondary)
            #endif
            Text(groupBy != nil ? "\(baseRows.count) items · \(groups.count) groups" : "\(baseRows.count)")
                .font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
                .padding(.horizontal, 8).padding(.vertical, 2)
                .background(.background.tertiary, in: Capsule())
            Spacer()
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").font(.caption).foregroundStyle(.secondary)
                TextField("Search symbol…", text: $search).textFieldStyle(.plain).frame(width: 160)
                if !search.isEmpty { Button { search = "" } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).foregroundStyle(.secondary) }
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
        }
    }

    // MARK: Toolbar

    private var toolbar: some View {
        #if os(iOS)
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                groupMenu
                if !uniqueAccounts.isEmpty { accountMenu }
                columnsButton
                if hSizeClass == .compact {
                    toolButton("Details", "chevron.up.chevron.down", active: !expandedCards.isEmpty) {
                        withAnimation(.easeInOut(duration: 0.18)) { toggleAllCards() }
                    }
                }
                toolButton("Lots", "square.stack.3d.up", active: !expandedLots.isEmpty) { toggleAllLots() }
                toolButton("Export", "square.and.arrow.down", active: false) { exportCSV() }
            }
        }
        #else
        HStack(spacing: 8) {
            groupMenu
            if !uniqueAccounts.isEmpty { accountMenu }
            columnsButton
            toolButton("Lots", "square.stack.3d.up", active: !expandedLots.isEmpty) { toggleAllLots() }
            Spacer()
            toolButton("Export", "square.and.arrow.down", active: false) { exportCSV() }
        }
        #endif
    }

    private var groupMenu: some View {
        PopoverMenu(minWidth: 200) {
            MenuToggleRow(title: "Do not group", isOn: groupBy == nil, dismissOnTap: true) { groupBy = nil }
            MenuDivider()
            ForEach(groupingOptions, id: \.key) { opt in
                MenuToggleRow(title: opt.label, isOn: groupBy == opt.key, dismissOnTap: true) { groupBy = opt.key }
            }
        } label: {
            toolLabel("line.3.horizontal.decrease", groupBy.map { k in "By \(groupingOptions.first { $0.key == k }?.label ?? k)" } ?? "Group", active: groupBy != nil)
        }.fixedSize()
    }

    private var accountMenu: some View {
        PopoverMenu(minWidth: 200) {
            MenuRow(title: "Clear Filter") { selectedAccounts = [] }
            MenuDivider()
            ForEach(uniqueAccounts, id: \.self) { acc in
                MenuToggleRow(title: acc, isOn: selectedAccounts.contains(acc)) { toggleAccount(acc) }
            }
        } label: {
            toolLabel("person.crop.circle", selectedAccounts.isEmpty ? "Account" : "Account (\(selectedAccounts.count))", active: !selectedAccounts.isEmpty)
        }.fixedSize()
    }

    private var columnsButton: some View {
        Button { showColumns.toggle() } label: {
            HStack(spacing: 5) {
                Image(systemName: "slider.horizontal.3").font(.caption)
                Text("Columns").font(.subheadline.weight(.medium))
                Text("\(visibleColumns.count)").font(.system(size: 11, weight: .bold))
                    .padding(.horizontal, 5).padding(.vertical, 1).background(Theme.brand.opacity(0.15), in: Capsule()).foregroundStyle(Theme.brand)
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(.background.secondary, in: Capsule())
            .overlay(Capsule().strokeBorder(.quaternary, lineWidth: 0.5))
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showColumns) { columnsPanel }
    }

    private var columnsPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Visible Columns").font(.caption.bold())
                Spacer()
                Button("Reset") { visibleColumns = defaultVisibleColumns }.font(.caption2.weight(.semibold)).buttonStyle(.plain).foregroundStyle(Theme.brand)
            }
            .padding(10).background(.background.tertiary)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(columnPickerGroups, id: \.label) { group in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(group.label).font(.system(size: 10, weight: .bold)).tracking(1.5).textCase(.uppercase).foregroundStyle(.secondary)
                            LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)], spacing: 3) {
                                ForEach(group.cols, id: \.self) { col in
                                    let on = visibleColumns.contains(col)
                                    Button { toggleColumn(col) } label: {
                                        HStack(spacing: 6) {
                                            Image(systemName: on ? "checkmark.square.fill" : "square")
                                                .foregroundStyle(on ? Theme.brand : .secondary).font(.caption)
                                            Text(col).font(.caption).foregroundStyle(on ? Theme.brand : .primary).lineLimit(1)
                                            Spacer(minLength: 0)
                                        }
                                        .padding(.horizontal, 6).padding(.vertical, 3)
                                        .background(on ? Theme.brand.opacity(0.10) : .clear, in: RoundedRectangle(cornerRadius: 6))
                                        .contentShape(Rectangle())
                                    }.buttonStyle(.plain)
                                }
                            }
                        }
                    }
                }.padding(10)
            }
        }
        #if os(iOS)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        #else
        .frame(width: 320, height: 420)
        #endif
    }

    private func toolButton(_ title: String, _ icon: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) { toolLabel(icon, title, active: active) }.buttonStyle(.plain)
    }

    private func toolLabel(_ icon: String, _ title: String, active: Bool) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon).font(.caption)
            Text(title).font(.subheadline.weight(.medium))
        }
        .foregroundStyle(active ? .white : .primary)
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(active ? AnyShapeStyle(Theme.brand) : AnyShapeStyle(.background.secondary), in: Capsule())
        .overlay(Capsule().strokeBorder(active ? AnyShapeStyle(Color.clear) : AnyShapeStyle(.quaternary), lineWidth: 0.5))
    }

    // MARK: Header row

    private func columnIcon(_ h: String) -> String {
        switch h {
        case "Symbol": return "textformat"
        case "Account": return "building.columns"
        case "Sector": return "chart.pie.fill"
        case "Industry": return "building.2"
        case "Quantity": return "number"
        case "Price": return "tag"
        case "Avg Cost", "Cost Basis", "Total Buy Cost": return "cart"
        case "Mkt Val": return "dollarsign.circle"
        case "Day Chg", "Day Chg %": return "clock"
        case "% of Total", "Contribution %": return "chart.pie"
        case "Unreal. G/L", "Unreal. G/L %": return "chart.line.uptrend.xyaxis"
        case "Real. G/L", "Total G/L", "Total Ret %": return "dollarsign.square"
        case "IRR (%)": return "arrow.up.right"
        case "Divs", "Est. Income", "Yield (Cost) %", "Yield (Mkt) %": return "banknote"
        case "Fees": return "minus.circle"
        case "FX G/L %": return "arrow.left.arrow.right"
        case "1M Trend": return "chart.xyaxis.line"
        case "Tags": return "tag"
        case "AI Score": return "sparkles"
        case "Intrinsic Value": return "target"
        default: return "info.circle"
        }
    }

    private func shortColumnName(_ h: String) -> String {
        switch h {
        case "Symbol": return "Sym"
        case "1M Trend": return "1M"
        case "Quantity": return "Qty"
        case "Avg Cost": return "Avg Cost"
        case "Cost Basis": return "Basis"
        case "Total Buy Cost": return "Buy Cost"
        case "Mkt Val": return "Value"
        case "Day Chg": return "Day"
        case "Day Chg %": return "Day %"
        case "% of Total": return "% Tot"
        case "Contribution %": return "% Con"
        case "Unreal. G/L": return "U. G/L"
        case "Unreal. G/L %": return "U. G/L%"
        case "Real. G/L": return "R. G/L"
        case "Total G/L": return "Tot G/L"
        case "Total Ret %": return "Ret %"
        case "IRR (%)": return "IRR"
        case "Divs": return "Div"
        case "Est. Income": return "Inc"
        case "Yield (Cost) %": return "Yld(C)"
        case "Yield (Mkt) %": return "Yld(M)"
        case "Fees": return "Fee"
        case "FX G/L %": return "FX %"
        case "Tags": return "Tag"
        case "AI Score": return "AI"
        case "Intrinsic Value": return "IV"
        default: return h
        }
    }

    private var headerRow: some View { headerRow(columns: visibleColumns) }

    private func headerRow(columns: [String]) -> some View {
        HStack(spacing: 0) {
            ForEach(columns, id: \.self) { h in
                Button { sort(h) } label: {
                    HStack(spacing: 3) {
                        if leftAlignedHeaders.contains(h) {
                            #if os(iOS)
                            Text(shortColumnName(h))
                            #else
                            Text(h)
                            #endif
                            sortArrow(h); Spacer(minLength: 0)
                        } else {
                            Spacer(minLength: 0)
                            #if os(iOS)
                            Text(shortColumnName(h))
                            #else
                            Text(h)
                            #endif
                            sortArrow(h)
                        }
                    }
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    // Pad inside the frame (matching `cell`) so each header is exactly
                    // columnWidth wide and stays aligned with the data columns below.
                    .padding(.horizontal, 8)
                    .frame(width: columnWidth(h), alignment: leftAlignedHeaders.contains(h) ? .leading : .trailing)
                    .padding(.vertical, 8)
                    .contentShape(Rectangle())
                }.buttonStyle(.plain)
            }
        }
        .background(.background.secondary)
    }

    @ViewBuilder private func sortArrow(_ h: String) -> some View {
        if sortKey == h { Image(systemName: sortAsc ? "arrow.up" : "arrow.down").font(.system(size: 9, weight: .bold)).foregroundStyle(Theme.brand) }
    }

    // MARK: Bodies

    private var flatBody: some View { flatBody(columns: visibleColumns) }

    private func flatBody(columns: [String]) -> some View {
        let rows = sortedRows(baseRows)
        let shown = Array(rows.prefix(visibleRows))
        // Plain VStack, not LazyVStack: this body is nested inside the horizontal
        // ScrollView of the split layout, where a LazyVStack mis-detects row
        // visibility against the outer vertical scroll and leaves expanded lot
        // rows blank (space reserved, content never rendered). The row count is
        // already bounded by pagination, so eager rendering is cheap.
        return VStack(spacing: 0) {
            ForEach(shown) { r in rowAndLots(r, columns: columns) }
        }
    }

    /// Which part of a group header to render. When the table splits into a
    /// pinned first column + scrollable remainder, the chevron/name/badge belong
    /// to the pinned side (`.label`) and the aggregate stats to the scroll side
    /// (`.stats`); a non-split table renders everything (`.full`).
    private enum GroupHeaderPart { case full, label, stats }

    @ViewBuilder private var groupedBody: some View { groupedBody(columns: visibleColumns, width: totalWidth) }

    @ViewBuilder private func groupedBody(columns: [String], width: CGFloat, part: GroupHeaderPart = .full) -> some View {
        // Plain VStack, not LazyVStack — see the note in `flatBody`: a lazy stack
        // nested in the split layout's horizontal ScrollView leaves expanded lot
        // rows blank.
        VStack(spacing: 0) {
            ForEach(groups) { g in
                groupHeaderRow(g, columns: columns, width: width, part: part)
                if expandedGroups.contains(g.key) {
                    ForEach(g.rows) { r in rowAndLots(r, columns: columns) }
                }
            }
        }
    }

    @ViewBuilder private func rowAndLots(_ r: HRow) -> some View { rowAndLots(r, columns: visibleColumns) }

    @ViewBuilder private func rowAndLots(_ r: HRow, columns: [String]) -> some View {
        dataRow(r, columns: columns)
        Divider().opacity(0.4)
        if expandedLots.contains(r.symbol), !r.lots.isEmpty {
            ForEach(Array(r.lots.enumerated()), id: \.offset) { _, lot in lotRow(r, lot, columns: columns) }
        }
    }

    private func groupHeaderRow(_ g: HGroup) -> some View { groupHeaderRow(g, columns: visibleColumns, width: totalWidth) }

    private func groupHeaderRow(_ g: HGroup, columns: [String], width: CGFloat, part: GroupHeaderPart = .full) -> some View {
        Button { toggleGroup(g.key) } label: {
            HStack(spacing: 8) {
                // Chevron + name + count live on the pinned side (or the full row).
                if part != .stats {
                    Image(systemName: expandedGroups.contains(g.key) ? "chevron.down" : "chevron.right").font(.caption2).foregroundStyle(.secondary)
                    Text(g.key).fontWeight(.semibold)
                    Text("\(g.rows.count)").font(.caption2).foregroundStyle(.secondary)
                        .padding(.horizontal, 6).padding(.vertical, 1).background(.background.tertiary, in: Capsule())
                }
                // Aggregate stats live on the scroll side (or the full row).
                if part != .label {
                    Spacer()
                    HStack(spacing: 18) {
                        if columns.contains("Mkt Val") { groupStat("Mkt", g.agg["Mkt Val"], .primary) }
                        if columns.contains("Day Chg") { groupStat("Day", g.agg["Day Chg"], glColor(g.agg["Day Chg"])) }
                        if columns.contains("Day Chg %") { groupStat(nil, g.agg["Day Chg %"], glColor(g.agg["Day Chg %"]), pct: true) }
                        if columns.contains("Unreal. G/L") { groupStat("Unreal", g.agg["Unreal. G/L"], glColor(g.agg["Unreal. G/L"])) }
                    }.padding(.trailing, 8)
                }
            }
            .font(.subheadline)
            .padding(.horizontal, 8).padding(.vertical, 9)
            .frame(width: width, alignment: .leading)
            .background(.background.secondary.opacity(0.6))
            .contentShape(Rectangle())
        }.buttonStyle(.plain)
    }

    private func groupStat(_ label: String?, _ value: Double?, _ color: Color, pct: Bool = false) -> some View {
        HStack(spacing: 4) {
            if let label { Text("\(label):").font(.caption2).foregroundStyle(.secondary) }
            Text(pct ? pctString(value) : Fmt.currency(value, code: currency)).font(.caption.weight(.medium)).monospacedDigit().foregroundStyle(color)
        }
    }

    // MARK: - iOS Compact Views

    private var iosFlatBody: some View {
        let rows = sortedRows(baseRows)
        let shown = Array(rows.prefix(visibleRows))
        return LazyVStack(spacing: 12) {
            ForEach(shown) { r in iosHoldingRow(r) }
        }
    }
    
    private var iosGroupedBody: some View {
        LazyVStack(spacing: 12) {
            ForEach(groups) { g in
                iosGroupHeaderRow(g)
                if expandedGroups.contains(g.key) {
                    ForEach(g.rows) { r in iosHoldingRow(r) }
                }
            }
        }
    }

    private func iosGroupHeaderRow(_ g: HGroup) -> some View {
        Button { toggleGroup(g.key) } label: {
            HStack(spacing: 8) {
                Image(systemName: expandedGroups.contains(g.key) ? "chevron.down" : "chevron.right").font(.caption).foregroundStyle(.secondary)
                Text(g.key).font(.subheadline.weight(.semibold))
                Text("\(g.rows.count)").font(.caption2).foregroundStyle(.secondary)
                    .padding(.horizontal, 6).padding(.vertical, 1).background(.background.tertiary, in: Capsule())
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    FittedMoney(value: g.agg["Mkt Val"], code: currency)
                        .font(.caption.weight(.bold)).monospacedDigit().lineLimit(1)
                    Text(pctString(g.agg["Day Chg %"])).font(.caption2).monospacedDigit().foregroundStyle(glColor(g.agg["Day Chg %"]))
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(.background.secondary.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
            .contentShape(Rectangle())
        }.buttonStyle(.plain)
    }

    private func iosHoldingRow(_ r: HRow) -> some View {
        VStack(spacing: 12) {
            // Even without the sparkline, a phone row can't always fit eight
            // digits of THB beside a symbol and a chevron, so it gives up the
            // cents first and the magnitude only as a last resort. Every rung
            // shows a whole number — the row used to clip ("฿15,347,04…").
            ViewThatFits(in: .horizontal) {
                iosHoldingRowHeader(r, money: .exact)
                iosHoldingRowHeader(r, money: .whole)
                iosHoldingRowHeader(r, money: .compact)
            }

            if isCardExpanded(r) {
                iosSparkline(r)
                iosExtraColumns(r)
            }

            if isCardExpanded(r), !r.lots.isEmpty {
                Button { toggleLot(r.symbol) } label: {
                    HStack {
                        Text("\(r.lots.count) Lots")
                        Spacer()
                        Image(systemName: expandedLots.contains(r.symbol) ? "chevron.up" : "chevron.down")
                    }
                    .font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                    .padding(.top, 4)
                }.buttonStyle(.plain)
            }
            if isCardExpanded(r), expandedLots.contains(r.symbol), !r.lots.isEmpty {
                Divider()
                ForEach(Array(r.lots.enumerated()), id: \.offset) { _, lot in iosLotRow(r, lot) }
            }
        }
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
        .onTapGesture { detail = SymbolID(id: r.symbol) }
    }

    /// One rung of the ladder above. Nothing in here may report a width smaller
    /// than it wants, or every rung would "fit" and the first would always win.
    private func iosHoldingRowHeader(_ r: HRow, money: MoneyForm) -> some View {
        HStack(alignment: .center, spacing: 8) {
            StockIcon(symbol: r.symbol, size: 45)
            // Single-line and served first: without this the symbol and share
            // count are the most compressible things in the row, so they wrap
            // ("GOO/G") as soon as anything else needs the width.
            VStack(alignment: .leading, spacing: 4) {
                Text(r.symbol).font(.headline.weight(.bold))
                    .lineLimit(1).minimumScaleFactor(0.7)
                Text("\(Fmt.number(r.num["Quantity"])) shs")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1).minimumScaleFactor(0.8)
            }
            // Deliberately uncapped: a `maxWidth` here would let this block
            // report less than it wants, every rung of the ladder would "fit",
            // and the amount would go back to being clipped.
            .layoutPriority(1)

            Spacer(minLength: 6)

            VStack(alignment: .trailing, spacing: 4) {
                Text(money.string(r.num["Mkt Val"], code: currency))
                    .font(.subheadline.weight(.bold))
                    .monospacedDigit()
                    .lineLimit(1)

                if let dayChgPct = r.num["Day Chg %"] {
                    HStack(spacing: 5) {
                        // What today moved in money, not just in percent. It
                        // follows the same rung as the value above it, so a row
                        // that has to shorten one shortens both. Tinted by its
                        // own sign: the percentage is the stock's move in its
                        // local currency, while this is the position in the
                        // display currency, so under a moving FX rate the two
                        // legitimately disagree.
                        if let dayChg = r.num["Day Chg"] {
                            Text(money.signedString(dayChg, code: currency))
                                .font(.system(size: 11, weight: .semibold))
                                .monospacedDigit()
                                .foregroundStyle(glColor(dayChg))
                                .lineLimit(1)
                        }
                        Text(pctString(dayChgPct))
                            .font(.system(size: 11, weight: .bold))
                            .padding(.horizontal, 4).padding(.vertical, 2)
                            .background(glColor(dayChgPct).opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                            .foregroundStyle(glColor(dayChgPct))
                    }
                }
            }
            .layoutPriority(1)

            // Its own button rather than a tap on the card, which already
            // opens the stock detail sheet. Kept narrow — the header row is
            // width-bound on a phone — but tall, so the tap target stays
            // comfortable without costing horizontal space.
            Button { withAnimation(.easeInOut(duration: 0.18)) { toggleCard(r) } } label: {
                Image(systemName: isCardExpanded(r) ? "chevron.up" : "chevron.down")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 20, height: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(isCardExpanded(r) ? "Hide \(r.symbol) details" : "Show \(r.symbol) details")
        }
    }

    /// The one-month trend, shown inside the expanded card. It used to be seven
    /// days in the collapsed header, where 44pt was all it could have; with the
    /// card's full width a month of closes is legible and says more.
    @ViewBuilder private func iosSparkline(_ r: HRow) -> some View {
        if r.trend1m.count > 1 {
            let up = (r.trend1m.last ?? 0) >= (r.trend1m.first ?? 0)
            Divider()
            VStack(alignment: .leading, spacing: 4) {
                Text("1M Trend")
                    .font(.system(size: 9, weight: .bold)).foregroundStyle(.tertiary).textCase(.uppercase)
                Chart(Array(r.trend1m.enumerated()), id: \.offset) { i, v in
                    AreaMark(x: .value("i", i),
                             yStart: .value("Min", chartDomain(r.trend1m).lowerBound),
                             yEnd: .value("v", v))
                        .foregroundStyle((up ? Color.up : .down).opacity(0.15))
                    LineMark(x: .value("i", i), y: .value("v", v))
                        .foregroundStyle(up ? Color.up : Color.down)
                        .lineStyle(StrokeStyle(lineWidth: 1.5))
                }
                .chartYScale(domain: chartDomain(r.trend1m))
                .chartXAxis(.hidden).chartYAxis(.hidden)
                .frame(height: 44)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 4)
        }
    }

    @ViewBuilder private func iosExtraColumns(_ r: HRow) -> some View {
        let excluded: Set<String> = ["Symbol", "Quantity", "1M Trend", "Mkt Val", "Day Chg %"]
        let extras = visibleColumns.filter { !excluded.contains($0) }
        
        if !extras.isEmpty {
            Divider()
            LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)], spacing: 10) {
                ForEach(extras, id: \.self) { h in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(shortColumnName(h)).font(.system(size: 9, weight: .bold)).foregroundStyle(.tertiary).textCase(.uppercase).lineLimit(1)
                        iosExtraCell(h, r)
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    @ViewBuilder private func iosExtraCell(_ h: String, _ r: HRow) -> some View {
        switch h {
        case "% of Total", "Contribution %": progressCell(r.num[h])
        case "AI Score": aiScoreCell(r.num["AI Score"])
        case "Intrinsic Value":
            intrinsicCell(r).font(.caption2.weight(.medium))
        case "Tags": tagsCellEditable(r)
        case "Account", "Sector", "Industry":
            Text(textValue(r, h).flatMap { $0.isEmpty ? nil : $0 } ?? "—")
                .font(.caption2)
                .foregroundStyle(h == "Account" ? .primary : .secondary).lineLimit(1)
        default:
            Text(format(r.num[h], h))
                .font(.caption2.weight(.medium)).monospacedDigit()
                .foregroundStyle(cellColor(h, r.num[h]))
                .lineLimit(1)
        }
    }

    private func iosLotRow(_ r: HRow, _ lot: JSONValue) -> some View {
        let qty = lot["Quantity"]?.doubleValue ?? 0
        let cost = lot["Cost Basis"]?.doubleValue
        let date = lot["Date"]?.stringValue?.prefix(10) ?? ""
        func mkt() -> Double? { lot["Market Value"]?.doubleValue ?? ((r.num["Price"] ?? 0) > 0 ? (r.num["Price"]!) * qty : nil) }
        let g = lot["Unreal. Gain"]?.doubleValue ?? ((mkt() ?? 0) - (cost ?? 0))
        
        return HStack {
            Text("↳ \(date)").font(.caption2).foregroundStyle(.tertiary)
            Spacer()
            Text("\(Fmt.number(qty)) shs").font(.caption2).foregroundStyle(.secondary)
            Spacer()
            Text(format(g, "Unreal. G/L")).font(.caption2).monospacedDigit().foregroundStyle(glColor(g))
        }
    }

    // MARK: Data row

    private func dataRow(_ r: HRow) -> some View { dataRow(r, columns: visibleColumns) }

    private func dataRow(_ r: HRow, columns: [String]) -> some View {
        HStack(spacing: 0) { ForEach(columns, id: \.self) { h in cell(h, r) } }
    }

    private func cell(_ h: String, _ r: HRow) -> some View {
        let align: Alignment = leftAlignedHeaders.contains(h) ? .leading : .trailing
        // Fixed height + clip keeps each row uniform and stops the sparkline's
        // area fill from bleeding into the row below.
        return cellContent(h, r)
            .padding(.horizontal, 8)
            .frame(width: columnWidth(h), height: rowHeight, alignment: align)
            .background(heatmapHeaders.contains(h) ? heatmapColor(r.num[h]) : .clear)
            .clipped()
    }

    @ViewBuilder private func cellContent(_ h: String, _ r: HRow) -> some View {
        switch h {
        case "Symbol": symbolCell(r)
        case "1M Trend": sparklineCell(r)
        case "% of Total", "Contribution %": progressCell(r.num[h])
        case "AI Score": aiScoreCell(r.num["AI Score"])
        case "Intrinsic Value": intrinsicCell(r)
        case "Tags": tagsCellEditable(r)
        case "Account", "Sector", "Industry":
            Text(textValue(r, h).flatMap { $0.isEmpty ? nil : $0 } ?? "—")
                .font(.footnote)
                .foregroundStyle(h == "Account" ? .primary : .secondary).lineLimit(1)
        default:
            Text(format(r.num[h], h))
                .font(.footnote).monospacedDigit()
                .foregroundStyle(cellColor(h, r.num[h])).lineLimit(1)
        }
    }

    private func symbolCell(_ r: HRow) -> some View {
        HStack(spacing: 6) {
            StockIcon(symbol: r.symbol, size: 15)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Button { detail = SymbolID(id: r.symbol) } label: { Text(r.symbol).font(.footnote.weight(.bold)).lineLimit(1).fixedSize() }.buttonStyle(.plain)
                    if !r.lots.isEmpty {
                        Button { toggleLot(r.symbol) } label: {
                            Image(systemName: expandedLots.contains(r.symbol) ? "chevron.down" : "chevron.right")
                                .font(.system(size: 10)).foregroundStyle(expandedLots.contains(r.symbol) ? Theme.brand : .secondary)
                        }.buttonStyle(.plain)
                    }
                }
                if !r.lots.isEmpty {
                    HStack(spacing: 2) {
                        Image(systemName: "square.stack.3d.up").font(.system(size: 9))
                        Text("\(r.lots.count) Lots").font(.system(size: 10))
                    }.foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder private func sparklineCell(_ r: HRow) -> some View {
        if r.trend1m.count > 1 {
            let up = (r.trend1m.last ?? 0) >= (r.trend1m.first ?? 0)
            Chart(Array(r.trend1m.enumerated()), id: \.offset) { i, v in
                AreaMark(x: .value("i", i), y: .value("v", v))
                    .foregroundStyle((up ? Color.up : Color.down).opacity(0.18))
                LineMark(x: .value("i", i), y: .value("v", v)).foregroundStyle(up ? Color.up : Color.down)
            }
            .chartYScale(domain: chartDomain(r.trend1m)).chartXAxis(.hidden).chartYAxis(.hidden)
            .frame(height: 28).clipped()
        } else {
            Text("no data").font(.system(size: 10)).foregroundStyle(.tertiary)
        }
    }

    private func progressCell(_ v: Double?) -> some View {
        let val = v ?? 0
        return ZStack(alignment: .leading) {
            GeometryReader { geo in
                Rectangle().fill((val < 0 ? Color.down : Theme.brand).opacity(0.22))
                    .frame(width: geo.size.width * min(1, abs(val) / 100))
            }
            Text(pctString(v)).font(.caption.weight(.medium)).monospacedDigit()
                .foregroundStyle(val < 0 ? Color.down : .primary)
                .frame(maxWidth: .infinity, alignment: .trailing).padding(.horizontal, 5)
        }
        .frame(height: 22).clipShape(RoundedRectangle(cornerRadius: 4))
    }

    @ViewBuilder private func aiScoreCell(_ v: Double?) -> some View {
        if let v, v > 0 {
            Text(String(format: "%.1f", v)).font(.system(size: 11, weight: .bold)).foregroundStyle(.white)
                .padding(.horizontal, 5).padding(.vertical, 2)
                .background(v >= 8 ? Color.up : (v >= 6 ? .orange : Color.down), in: RoundedRectangle(cornerRadius: 4))
        } else { Text("—").foregroundStyle(.tertiary) }
    }

    @ViewBuilder private func intrinsicCell(_ r: HRow) -> some View {
        if let iv = r.intrinsic, iv > 0 {
            let tone: Color = iv > r.price ? .up : (iv < r.price ? .down : .primary)
            
            #if os(iOS)
            let align: HorizontalAlignment = hSizeClass == .compact ? .leading : .trailing
            #else
            let align: HorizontalAlignment = .trailing
            #endif
            
            VStack(alignment: align, spacing: 3) {
                HStack(spacing: 4) {
                    Text(Fmt.currency(iv, code: currency))
                        .font(.footnote)
                        .foregroundStyle(tone)
                    if let m = r.mos { Text("(\(String(format: "%.1f", abs(m)))%)").font(.system(size: 10)).foregroundStyle(.secondary) }
                }
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.8)

                if let m = r.mos {
                    GeometryReader { geo in
                        let half = geo.size.width / 2
                        ZStack(alignment: .leading) {
                            Capsule().fill(.quaternary)
                            Capsule().fill(m > 0 ? Color.up : Color.down)
                                .frame(width: half * min(abs(m), 100) / 100)
                                .offset(x: m > 0 ? half : half - half * min(abs(m), 100) / 100)
                        }
                    }.frame(width: 64, height: 3)
                }
            }
        } else { Text("—").foregroundStyle(.tertiary) }
    }

    /// Editable Tags cell — tapping opens the tag editor for this holding
    /// (applied across every account the symbol is held in).
    private func tagsCellEditable(_ r: HRow) -> some View {
        Button {
            let accounts = Array(Set(holdings.filter { $0.symbol == r.symbol }
                .compactMap { $0.account }.filter { !$0.isEmpty }))
            tagEdit = TagEdit(symbol: r.symbol,
                              accounts: accounts.isEmpty ? [r.account] : accounts,
                              tags: r.tags)
        } label: {
            HStack(spacing: 4) {
                tagsCell(r.tags)
                Image(systemName: "pencil").font(.system(size: 9)).foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder private func tagsCell(_ tags: [String]) -> some View {
        if tags.isEmpty { Text("—").foregroundStyle(.tertiary) }
        else {
            HStack(spacing: 3) {
                ForEach(tags.prefix(3), id: \.self) { t in
                    Text(t.uppercased()).font(.system(size: 10, weight: .bold)).tracking(0.5)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(badgeColor(t).opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(badgeColor(t))
                }
            }
        }
    }

    // MARK: Lot row

    private func lotRow(_ r: HRow, _ lot: JSONValue) -> some View { lotRow(r, lot, columns: visibleColumns) }

    private func lotRow(_ r: HRow, _ lot: JSONValue, columns: [String]) -> some View {
        HStack(spacing: 0) {
            ForEach(columns, id: \.self) { h in
                Group {
                    if h == "Symbol" {
                        HStack(spacing: 3) {
                            Text("↳").font(.system(size: 10)).foregroundStyle(.tertiary)
                            Text("Lot: \(lot["Date"]?.stringValue?.prefix(10) ?? "")").italic().lineLimit(1).fixedSize()
                        }
                    } else {
                        Text(lotCell(lot, h, r.price)).monospacedDigit().lineLimit(1)
                    }
                }
                // Pad inside the frame (matching `cell`) so each lot cell is exactly
                // columnWidth wide and stays aligned with the header/data columns.
                // A fixed height (matching `cell`) keeps the pinned Symbol column
                // and the scrollable columns aligned — they lay out lots separately.
                .padding(.horizontal, 8)
                .frame(width: columnWidth(h), height: lotRowHeight, alignment: leftAlignedHeaders.contains(h) ? .leading : .trailing)
                .clipped()
                .foregroundStyle(.secondary)
            }
        }
        .font(.caption)
        .background(.background.secondary.opacity(0.35))
    }

    private func lotCell(_ lot: JSONValue, _ h: String, _ price: Double) -> String {
        let qty = lot["Quantity"]?.doubleValue ?? 0
        let cost = lot["Cost Basis"]?.doubleValue
        func mkt() -> Double? { lot["Market Value"]?.doubleValue ?? (price > 0 ? price * qty : nil) }
        switch h {
        case "Quantity": return format(qty, h)
        case "Cost Basis", "Total Buy Cost": return format(cost, h)
        case "Mkt Val": return format(mkt(), h)
        case "Unreal. G/L", "Total G/L":
            let g = lot["Unreal. Gain"]?.doubleValue ?? ((mkt() ?? 0) - (cost ?? 0)); return format(g, "Unreal. G/L")
        case "Unreal. G/L %", "Total Ret %":
            if let p = lot["Unreal. Gain %"]?.doubleValue { return format(p, h) }
            if let c = cost, c != 0 { return format(((mkt() ?? 0) - c) / c * 100, h) }; return "—"
        case "Price", "Avg Cost": return qty != 0 ? format((cost ?? 0) / qty, h) : "—"
        default: return ""
        }
    }

    // MARK: Pagination

    @ViewBuilder private var pagination: some View {
        let count = sortedRows(baseRows).count
        if count > 10 {
            HStack(spacing: 12) {
                Spacer()
                if visibleRows > 10 {
                    Button("Show Less") { visibleRows = max(10, min(visibleRows, count) - 20) }.controlSize(.small)
                }
                if visibleRows < count {
                    Button("Show More") { visibleRows = min(count, visibleRows + 20) }
                        .buttonStyle(.borderedProminent).tint(Theme.brand).controlSize(.small)
                    Button("Show All") { visibleRows = count }.controlSize(.small)
                }
                Spacer()
            }.padding(.top, 4)
        }
    }

    // MARK: Formatting & color

    // Mirrors the web formatValue: currency only for headers containing one of
    // Price/Value/Cost/Gain/Div/Balance; everything else is a plain number.
    private func format(_ v: Double?, _ header: String) -> String {
        guard let v else { return "—" }
        if header.contains("%") || header.contains("Yield") || header == "Weight" { return pctString(v) }
        if header.contains("Price") || header.contains("Value") || header.contains("Cost")
            || header.contains("Gain") || header.contains("Div") || header.contains("Balance") { return Fmt.currency(v, code: currency) }
        if header == "Quantity" { return Fmt.number(v, fractionDigits: 4) }
        return Fmt.number(v, fractionDigits: 2)
    }

    private func pctString(_ v: Double?) -> String {
        guard let v else { return "—" }
        if v.isInfinite { return v > 0 ? "∞" : "-∞" }
        return String(format: "%.2f%%", v)
    }

    private func cellColor(_ header: String, _ v: Double?) -> Color {
        guard let v else { return .secondary.opacity(0.4) }
        if glHeaders.contains(header) {
            if abs(v) < 0.001 { return .secondary.opacity(0.4) }
            return v > 0 ? .up : .down
        }
        if abs(v) < 0.0001 { return .secondary.opacity(0.4) }
        return .secondary
    }

    private func glColor(_ v: Double?) -> Color {
        guard let v, abs(v) >= 0.001 else { return .secondary }
        return v > 0 ? .up : .down
    }

    private func heatmapColor(_ v: Double?) -> Color {
        guard let v, abs(v) >= 0.01 else { return .clear }
        let intensity = min(abs(v) / 20, 1)
        let op: Double = intensity < 0.2 ? 0.08 : (intensity < 0.4 ? 0.15 : (intensity < 0.7 ? 0.24 : 0.34))
        return (v > 0 ? Color.up : Color.down).opacity(op)
    }

    private func badgeColor(_ s: String) -> Color {
        let palette = [Color(hex: 0x6366f1), Color(hex: 0x06b6d4), Color(hex: 0x10b981), Color(hex: 0xf59e0b),
                       Color(hex: 0xec4899), Color(hex: 0x8b5cf6), Color(hex: 0xf97316)]
        return palette[abs(s.hashValue) % palette.count]
    }

    // MARK: Actions

    private var uniqueAccounts: [String] { Array(Set(holdings.compactMap { $0.account })).sorted() }
    private func sort(_ h: String) { if sortKey == h { sortAsc.toggle() } else { sortKey = h; sortAsc = false } }
    private func toggleColumn(_ h: String) { if let i = visibleColumns.firstIndex(of: h) { visibleColumns.remove(at: i) } else { visibleColumns.append(h) } }
    private func toggleAccount(_ a: String) { if selectedAccounts.contains(a) { selectedAccounts.remove(a) } else { selectedAccounts.insert(a) } }
    private func toggleGroup(_ k: String) { if expandedGroups.contains(k) { expandedGroups.remove(k) } else { expandedGroups.insert(k) } }
    private func toggleLot(_ s: String) { if expandedLots.contains(s) { expandedLots.remove(s) } else { expandedLots.insert(s) } }
    private func toggleAllLots() {
        if !expandedLots.isEmpty { expandedLots = [] }
        else { expandedLots = Set(baseRows.filter { !$0.lots.isEmpty }.map(\.symbol)) }
    }

    // MARK: Card expansion

    /// Row ids of the cards showing their full metric grid.
    private var expandedCards: Set<String> {
        get { expandedCardsRaw.isEmpty ? [] : Set(expandedCardsRaw.components(separatedBy: "\n")) }
        nonmutating set { expandedCardsRaw = newValue.sorted().joined(separator: "\n") }
    }

    private func isCardExpanded(_ r: HRow) -> Bool { expandedCards.contains(r.id) }

    private func toggleCard(_ r: HRow) {
        var next = expandedCards
        if next.contains(r.id) { next.remove(r.id) } else { next.insert(r.id) }
        expandedCards = next
    }

    private func toggleAllCards() {
        if !expandedCards.isEmpty { expandedCards = [] }
        else { expandedCards = Set(baseRows.map(\.id)) }
    }

    private func exportCSV() {
        let cols = visibleColumns.filter { $0 != "1M Trend" }
        var csv = cols.joined(separator: ",") + "\n"
        for r in sortedRows(baseRows) {
            let line = cols.map { h -> String in
                if numericHeaders.contains(h) { return r.num[h].map { String($0) } ?? "" }
                if h == "Tags" { return "\"\(r.tags.joined(separator: ", "))\"" }
                return "\"\(r.text[h] ?? "")\""
            }.joined(separator: ",")
            csv += line + "\n"
        }
        exportText(csv, filename: "holdings.csv")
    }
}

// MARK: - Tag editor

/// Edits a holding's custom tags (comma-separated) and writes them to every
/// account the symbol is held in via `/holdings/update_tags` (mirrors the web).
private struct TagEditorSheet: View {
    @Environment(\.dismiss) private var dismiss
    let edit: HoldingsTableView.TagEdit
    let onSaved: () -> Void

    @State private var text = ""
    @State private var isSaving = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Edit Tags").font(.title2.bold()).padding(20)
            Divider()
            Form {
                Section {
                    LabeledContent("Symbol") { Text(edit.symbol).foregroundStyle(.secondary) }
                    TextField("Tags (comma-separated)", text: $text)
                        #if os(iOS)
                        .textInputAutocapitalization(.never)
                        #endif
                } footer: {
                    Text("Comma-separated, e.g. Core, Speculative, Dividend")
                }
                if let error { Text(error).foregroundStyle(.red).font(.callout) }
            }
            .formStyle(.grouped)
            Divider()
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { save() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isSaving)
            }
            .padding(16)
        }
        #if os(macOS)
        .frame(width: 420, height: 320)
        #endif
        .onAppear { text = edit.tags.joined(separator: ", ") }
    }

    private func save() {
        isSaving = true; error = nil
        let tags = text.trimmingCharacters(in: .whitespaces)
        struct Req: Encodable { let account: String; let symbol: String; let tags: String }
        Task {
            do {
                for account in edit.accounts {
                    let _: StatusResponse = try await APIClient.shared.send(
                        method: "POST", path: "/holdings/update_tags",
                        body: Req(account: account, symbol: edit.symbol, tags: tags))
                }
                onSaved()
                dismiss()
            } catch {
                isSaving = false
                self.error = "Failed to update tags. Please try again."
            }
        }
    }
}

/// How much of an amount to spell out when the row it sits in is too narrow for
/// all of it: `฿15,347,047.68`, `฿15,347,048`, `฿15.35M`.
enum MoneyForm {
    case exact, whole, compact

    func string(_ value: Double?, code: String) -> String {
        guard let value else { return "—" }
        switch self {
        case .exact: return Fmt.currency(value, code: code)
        case .whole: return Fmt.symbol(code) + Fmt.number(value, fractionDigits: 0)
        case .compact: return Fmt.compact(value, code: code)
        }
    }

    /// The same amount with an explicit sign, for a figure that is a change
    /// rather than a level: `+฿52,300`, `-฿1,204`. The sign leads the currency
    /// symbol, which is where a reader looks for it.
    func signedString(_ value: Double?, code: String) -> String {
        guard let value else { return "—" }
        return (value < 0 ? "-" : "+") + string(abs(value), code: code)
    }
}

/// An amount that gets shorter rather than clipped when its container is tight.
///
/// `minimumScaleFactor` is no help here: a `Text` only scales against a width it
/// was *proposed*, and a text squeezed by a stack is placed into the leftover
/// instead — so it truncates at full size, which is what the holdings rows used
/// to do. `ViewThatFits` decides from the same proposal the stack would squeeze
/// into, so what's shown is always a whole number.
struct FittedMoney: View {
    let value: Double?
    let code: String

    var body: some View {
        ViewThatFits(in: .horizontal) {
            Text(MoneyForm.exact.string(value, code: code))
            Text(MoneyForm.whole.string(value, code: code))
            Text(MoneyForm.compact.string(value, code: code))
        }
    }
}
