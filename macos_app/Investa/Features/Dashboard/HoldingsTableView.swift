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
    "FX G/L %": "FX Gain/Loss %", "Est. Income": "Est. Ann. Income", "7d Trend": "sparkline_7d",
    "Tags": "Tags", "Contribution %": "Contribution %", "AI Score": "ai_score", "Intrinsic Value": "intrinsic_value",
]

private let defaultVisibleColumns = ["Symbol", "7d Trend", "Quantity", "% of Total", "Price", "Mkt Val", "Day Chg", "Day Chg %", "Unreal. G/L"]

private let columnPickerGroups: [(label: String, cols: [String])] = [
    ("Core", ["Symbol", "Account", "Quantity", "Price", "Mkt Val", "% of Total", "7d Trend"]),
    ("Daily", ["Day Chg", "Day Chg %"]),
    ("Returns", ["Unreal. G/L", "Unreal. G/L %", "Real. G/L", "Total G/L", "Total Ret %", "IRR (%)"]),
    ("Cost", ["Avg Cost", "Cost Basis", "Total Buy Cost"]),
    ("Income", ["Divs", "Est. Income", "Yield (Cost) %", "Yield (Mkt) %"]),
    ("Details", ["Sector", "Industry", "FX G/L %", "Fees", "Tags", "Contribution %", "AI Score", "Intrinsic Value"]),
]

private let leftAlignedHeaders: Set<String> = ["Symbol", "Account", "Sector", "Industry", "Tags"]
private let glHeaders: Set<String> = ["Day Chg", "Day Chg %", "Unreal. G/L", "Unreal. G/L %", "Real. G/L", "Total G/L", "Total Ret %", "FX G/L %", "IRR (%)"]
private let heatmapHeaders: Set<String> = ["Day Chg %", "Unreal. G/L %", "Total Ret %"]
private let sumHeaders = ["Quantity", "Mkt Val", "Cost Basis", "Day Chg", "Unreal. G/L", "Real. G/L", "Divs", "Fees", "Total G/L", "Total Buy Cost", "Est. Income", "Contribution %", "% of Total"]

private func columnWidth(_ h: String) -> CGFloat {
    switch h {
    case "Symbol": return 172
    case "7d Trend": return 110
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
    var sparkline: [Double] = []
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

    @State private var visibleColumns = defaultVisibleColumns
    @State private var sortKey = "Mkt Val"
    @State private var sortAsc = false
    @State private var search = ""
    @State private var selectedAccounts: Set<String> = []
    @State private var groupBy: String?
    @State private var expandedGroups: Set<String> = []
    @State private var expandedLots: Set<String> = []
    @State private var visibleRows = 10
    @State private var showColumns = false
    @State private var detail: SymbolID?

    private let rowHeight: CGFloat = 46

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
        r.sparkline = h.raw["sparkline_7d"]?.arrayValue?.compactMap { $0.doubleValue } ?? []
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

    /// Filtered + (optionally) symbol-aggregated rows.
    private var baseRows: [HRow] {
        let filtered = holdings.filter {
            (search.isEmpty || $0.symbol.lowercased().contains(search.lowercased()))
            && (selectedAccounts.isEmpty || selectedAccounts.contains($0.account ?? ""))
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
                ScrollView(.horizontal, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 0) {
                        headerRow
                        Divider()
                        if let gb = groupBy, !gb.isEmpty { groupedBody } else { flatBody }
                    }
                    .frame(width: totalWidth, alignment: .leading)
                }
                if groupBy == nil { pagination }
            }
        }
        .padding(16)
        .overlay(alignment: .top) { Rectangle().fill(Color(hex: 0x6366f1).opacity(0.8)).frame(height: 2) }
        .card(.standard)
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: currency) }
        .onChange(of: groupBy) { _, _ in expandedGroups = Set(groups.map(\.key)) }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "tablecells").font(.caption.weight(.semibold)).foregroundStyle(Color(hex: 0x6366f1))
            #if !os(iOS)
            Text("Holdings").font(.caption.weight(.semibold)).tracking(0.8).textCase(.uppercase).foregroundStyle(.secondary)
            #endif
            Text(groupBy != nil ? "\(baseRows.count) items · \(groups.count) groups" : "\(baseRows.count)")
                .font(.system(size: 10, weight: .bold)).foregroundStyle(.secondary)
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
        Menu {
            Button { groupBy = nil } label: { Label("Do not group", systemImage: groupBy == nil ? "checkmark" : "") }
            Divider()
            ForEach(groupingOptions, id: \.key) { opt in
                Button { groupBy = opt.key } label: { Label(opt.label, systemImage: groupBy == opt.key ? "checkmark" : "") }
            }
        } label: {
            toolLabel("line.3.horizontal.decrease", groupBy.map { k in "By \(groupingOptions.first { $0.key == k }?.label ?? k)" } ?? "Group", active: groupBy != nil)
        }.borderlessMenu().fixedSize()
    }

    private var accountMenu: some View {
        Menu {
            Button("Clear Filter") { selectedAccounts = [] }
            Divider()
            ForEach(uniqueAccounts, id: \.self) { acc in
                Button { toggleAccount(acc) } label: { Label(acc, systemImage: selectedAccounts.contains(acc) ? "checkmark" : "") }
            }
        } label: {
            toolLabel("person.crop.circle", selectedAccounts.isEmpty ? "Account" : "Account (\(selectedAccounts.count))", active: !selectedAccounts.isEmpty)
        }.borderlessMenu().fixedSize()
    }

    private var columnsButton: some View {
        Button { showColumns.toggle() } label: {
            HStack(spacing: 5) {
                Image(systemName: "slider.horizontal.3").font(.caption)
                #if !os(iOS)
                Text("Columns").font(.subheadline.weight(.medium))
                #endif
                Text("\(visibleColumns.count)").font(.system(size: 10, weight: .bold))
                    .padding(.horizontal, 5).background(Theme.brand.opacity(0.15), in: Capsule()).foregroundStyle(Theme.brand)
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showColumns, arrowEdge: .bottom) { columnsPanel }
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
                            Text(group.label).font(.system(size: 9, weight: .bold)).tracking(1.5).textCase(.uppercase).foregroundStyle(.secondary)
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
        HStack(spacing: 5) {
            Image(systemName: icon).font(.caption)
            #if !os(iOS)
            Text(title).font(.subheadline.weight(.medium))
            #endif
        }
        .foregroundStyle(active ? .white : .primary)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(active ? AnyShapeStyle(Theme.brand) : AnyShapeStyle(.background.tertiary), in: RoundedRectangle(cornerRadius: 8))
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
        case "7d Trend": return "chart.xyaxis.line"
        case "Tags": return "tag"
        case "AI Score": return "sparkles"
        case "Intrinsic Value": return "target"
        default: return "info.circle"
        }
    }

    private func shortColumnName(_ h: String) -> String {
        switch h {
        case "Symbol": return "Sym"
        case "7d Trend": return "7d"
        case "Quantity": return "Qty"
        case "Avg Cost": return "Avg Cost"
        case "Cost Basis", "Total Buy Cost": return "Cost"
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

    private var headerRow: some View {
        HStack(spacing: 0) {
            ForEach(visibleColumns, id: \.self) { h in
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
        if sortKey == h { Image(systemName: sortAsc ? "arrow.up" : "arrow.down").font(.system(size: 8, weight: .bold)).foregroundStyle(Theme.brand) }
    }

    // MARK: Bodies

    private var flatBody: some View {
        let rows = sortedRows(baseRows)
        let shown = Array(rows.prefix(visibleRows))
        return LazyVStack(spacing: 0) {
            ForEach(shown) { r in rowAndLots(r) }
        }
    }

    @ViewBuilder private var groupedBody: some View {
        LazyVStack(spacing: 0) {
            ForEach(groups) { g in
                groupHeaderRow(g)
                if expandedGroups.contains(g.key) {
                    ForEach(g.rows) { r in rowAndLots(r) }
                }
            }
        }
    }

    @ViewBuilder private func rowAndLots(_ r: HRow) -> some View {
        dataRow(r)
        Divider().opacity(0.4)
        if expandedLots.contains(r.symbol), !r.lots.isEmpty {
            ForEach(Array(r.lots.enumerated()), id: \.offset) { _, lot in lotRow(r, lot) }
        }
    }

    private func groupHeaderRow(_ g: HGroup) -> some View {
        Button { toggleGroup(g.key) } label: {
            HStack(spacing: 8) {
                Image(systemName: expandedGroups.contains(g.key) ? "chevron.down" : "chevron.right").font(.caption2).foregroundStyle(.secondary)
                Text(g.key).fontWeight(.semibold)
                Text("\(g.rows.count)").font(.caption2).foregroundStyle(.secondary)
                    .padding(.horizontal, 6).padding(.vertical, 1).background(.background.tertiary, in: Capsule())
                Spacer()
                HStack(spacing: 18) {
                    if visibleColumns.contains("Mkt Val") { groupStat("Mkt", g.agg["Mkt Val"], .primary) }
                    if visibleColumns.contains("Day Chg") { groupStat("Day", g.agg["Day Chg"], glColor(g.agg["Day Chg"])) }
                    if visibleColumns.contains("Day Chg %") { groupStat(nil, g.agg["Day Chg %"], glColor(g.agg["Day Chg %"]), pct: true) }
                    if visibleColumns.contains("Unreal. G/L") { groupStat("Unreal", g.agg["Unreal. G/L"], glColor(g.agg["Unreal. G/L"])) }
                }.padding(.trailing, 8)
            }
            .font(.callout)
            .padding(.horizontal, 8).padding(.vertical, 9)
            .frame(width: totalWidth, alignment: .leading)
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

    // MARK: Data row

    private func dataRow(_ r: HRow) -> some View {
        HStack(spacing: 0) { ForEach(visibleColumns, id: \.self) { h in cell(h, r) } }
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
        case "7d Trend": sparklineCell(r)
        case "% of Total", "Contribution %": progressCell(r.num[h])
        case "AI Score": aiScoreCell(r.num["AI Score"])
        case "Intrinsic Value": intrinsicCell(r)
        case "Tags": tagsCell(r.tags)
        case "Account", "Sector", "Industry":
            Text(textValue(r, h).flatMap { $0.isEmpty ? nil : $0 } ?? "—")
                .foregroundStyle(h == "Account" ? .primary : .secondary).lineLimit(1)
        default:
            Text(format(r.num[h], h)).monospacedDigit().foregroundStyle(cellColor(h, r.num[h])).lineLimit(1)
        }
    }

    private func symbolCell(_ r: HRow) -> some View {
        HStack(spacing: 6) {
            StockIcon(symbol: r.symbol, size: 15)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Button { detail = SymbolID(id: r.symbol) } label: { Text(r.symbol).fontWeight(.bold).lineLimit(1).fixedSize() }.buttonStyle(.plain)
                    if !r.lots.isEmpty {
                        Button { toggleLot(r.symbol) } label: {
                            Image(systemName: expandedLots.contains(r.symbol) ? "chevron.down" : "chevron.right")
                                .font(.system(size: 9)).foregroundStyle(expandedLots.contains(r.symbol) ? Theme.brand : .secondary)
                        }.buttonStyle(.plain)
                    }
                }
                if !r.lots.isEmpty {
                    HStack(spacing: 2) {
                        Image(systemName: "square.stack.3d.up").font(.system(size: 8))
                        Text("\(r.lots.count) Lots").font(.system(size: 9))
                    }.foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder private func sparklineCell(_ r: HRow) -> some View {
        if r.sparkline.count > 1 {
            let up = (r.sparkline.last ?? 0) >= (r.sparkline.first ?? 0)
            Chart(Array(r.sparkline.enumerated()), id: \.offset) { i, v in
                AreaMark(x: .value("i", i), y: .value("v", v))
                    .foregroundStyle((up ? Color.up : Color.down).opacity(0.18))
                LineMark(x: .value("i", i), y: .value("v", v)).foregroundStyle(up ? Color.up : Color.down)
            }
            .chartYScale(domain: chartDomain(r.sparkline)).chartXAxis(.hidden).chartYAxis(.hidden)
            .frame(height: 28).clipped()
        } else {
            Text("no data").font(.system(size: 9)).foregroundStyle(.tertiary)
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
            Text(String(format: "%.1f", v)).font(.system(size: 10, weight: .bold)).foregroundStyle(.white)
                .padding(.horizontal, 5).padding(.vertical, 2)
                .background(v >= 8 ? Color.up : (v >= 6 ? .orange : Color.down), in: RoundedRectangle(cornerRadius: 4))
        } else { Text("—").foregroundStyle(.tertiary) }
    }

    @ViewBuilder private func intrinsicCell(_ r: HRow) -> some View {
        if let iv = r.intrinsic, iv > 0 {
            let tone: Color = iv > r.price ? .up : (iv < r.price ? .down : .primary)
            VStack(alignment: .trailing, spacing: 3) {
                HStack(spacing: 4) {
                    Text(Fmt.currency(iv, code: currency)).foregroundStyle(tone)
                    if let m = r.mos { Text("(\(String(format: "%.1f", abs(m)))%)").font(.system(size: 9)).foregroundStyle(.secondary) }
                }.monospacedDigit()
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

    @ViewBuilder private func tagsCell(_ tags: [String]) -> some View {
        if tags.isEmpty { Text("—").foregroundStyle(.tertiary) }
        else {
            HStack(spacing: 3) {
                ForEach(tags.prefix(3), id: \.self) { t in
                    Text(t.uppercased()).font(.system(size: 9, weight: .bold)).tracking(0.5)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(badgeColor(t).opacity(0.18), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(badgeColor(t))
                }
            }
        }
    }

    // MARK: Lot row

    private func lotRow(_ r: HRow, _ lot: JSONValue) -> some View {
        HStack(spacing: 0) {
            ForEach(visibleColumns, id: \.self) { h in
                Group {
                    if h == "Symbol" {
                        HStack(spacing: 3) {
                            Text("↳").font(.system(size: 9)).foregroundStyle(.tertiary)
                            Text("Lot: \(lot["Date"]?.stringValue?.prefix(10) ?? "")").italic().lineLimit(1).fixedSize()
                        }
                    } else {
                        Text(lotCell(lot, h, r.price)).monospacedDigit().lineLimit(1)
                    }
                }
                // Pad inside the frame (matching `cell`) so each lot cell is exactly
                // columnWidth wide and stays aligned with the header/data columns.
                .padding(.horizontal, 8)
                .frame(width: columnWidth(h), alignment: leftAlignedHeaders.contains(h) ? .leading : .trailing)
                .padding(.vertical, 4)
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

    private func exportCSV() {
        let cols = visibleColumns.filter { $0 != "7d Trend" }
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
