import SwiftUI
import Charts

/// Identifiable wrapper so a bare symbol string can drive `.sheet(item:)`.
struct SymbolID: Identifiable, Hashable { let id: String }

private enum DetailTab: String, CaseIterable, Identifiable {
    case overview = "Overview", chart = "Chart", analysis = "Analysis"
    case financials = "Financials", ratios = "Ratios & Trends", valuation = "Valuation"
    case holdings = "Holdings", news = "News"
    var id: String { rawValue }
    var icon: String {
        switch self {
        case .overview: return "rectangle.3.group"; case .chart: return "chart.xyaxis.line"
        case .analysis: return "sparkles"; case .financials: return "doc.text"; case .ratios: return "chart.bar"
        case .valuation: return "dollarsign"; case .holdings: return "chart.pie"; case .news: return "newspaper"
        }
    }
}

/// Detailed stock view presented as a sheet — mirrors the web StockDetailModal
/// (Overview / Chart / Analysis / Financials / Ratios / Valuation / News, plus
/// Holdings for ETFs).
struct StockDetailView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel: StockDetailViewModel
    @State private var tab: DetailTab = .overview
    @State private var finType = "income"
    @State private var detail: SymbolID?

    init(symbol: String, currency: String = "USD") {
        _viewModel = StateObject(wrappedValue: StockDetailViewModel(symbol: symbol, currency: currency))
    }

    private var f: Fundamentals? { viewModel.fundamentals }
    private var nativeCur: String { f?.currency ?? "USD" }
    private var cur: String { viewModel.currency }

    private var visibleTabs: [DetailTab] {
        var tabs: [DetailTab] = [.overview, .chart, .analysis]
        if !(f?.isETF ?? false) { tabs += [.financials, .ratios] }
        tabs.append(.valuation)
        if f?.isETF ?? false { tabs.append(.holdings) }
        tabs.append(.news)
        return tabs
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            tabBar
            Divider()
            ScrollView {
                Group {
                    switch tab {
                    case .overview: overviewTab
                    case .chart: chartTab
                    case .analysis: analysisTab
                    case .financials: financialsTab
                    case .ratios: ratiosTab
                    case .valuation: valuationTab
                    case .holdings: holdingsTab
                    case .news: newsTab
                    }
                }
                .padding(20)
            }
        }
        .frame(width: 860, height: 720)
        .task { await viewModel.loadAll() }
        .onChange(of: tab) { _, t in
            Task {
                switch t {
                case .analysis: if viewModel.analysis == nil { await viewModel.loadAnalysis() }
                case .financials, .ratios: await viewModel.loadFinancials()
                case .news: await viewModel.loadNews()
                default: break
                }
            }
        }
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: cur) }
    }

    // MARK: - Header + tabs

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            StockIcon(symbol: viewModel.symbol, size: 52)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Text(f?.shortName ?? viewModel.symbol).font(.title.bold()).lineLimit(1)
                    Text(viewModel.symbol).font(.caption.monospaced())
                        .padding(.horizontal, 6).padding(.vertical, 2).background(.quaternary, in: Capsule())
                }
                if f?.sector != nil || f?.industry != nil {
                    HStack(spacing: 4) {
                        if let s = f?.sector { Text(s).foregroundStyle(.indigo) }
                        if f?.sector != nil && f?.industry != nil { Text("•").foregroundStyle(.secondary) }
                        if let i = f?.industry { Text(i).foregroundStyle(.secondary) }
                    }.font(.caption)
                }
            }
            Spacer()
            VStack(alignment: .trailing) {
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                if let p = f?.price { Text(Fmt.currency(p, code: nativeCur)).font(.title2.bold()).foregroundStyle(.indigo) }
            }
            Button { dismiss() } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary).font(.title2)
        }
        .padding(20)
    }

    private var tabBar: some View {
        HStack(spacing: 18) {
            ForEach(visibleTabs) { t in
                Button { tab = t } label: {
                    VStack(spacing: 4) {
                        Label(t.rawValue, systemImage: t.icon)
                            .font(.callout.weight(tab == t ? .bold : .regular))
                            .foregroundStyle(tab == t ? Color.indigo : .secondary)
                        Rectangle().fill(tab == t ? Color.indigo : .clear).frame(height: 2)
                    }
                }.buttonStyle(.plain)
            }
            Spacer()
        }
        .padding(.horizontal, 20)
    }

    // MARK: - Overview

    @ViewBuilder private var overviewTab: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let pos = viewModel.userPosition { positionSection(pos) }
            // Intrinsic value cards
            if let iv = viewModel.intrinsic {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 12)], spacing: 12) {
                    if let dcf = iv.models?.dcf?.intrinsicValue {
                        ivCard("DCF Intrinsic Value", dcf, upside: upside(dcf, iv.currentPrice), range: iv.models?.dcf?.mc, tint: .green)
                    }
                    if let g = iv.models?.graham?.intrinsicValue {
                        ivCard("Graham Intrinsic Value", g, upside: upside(g, iv.currentPrice), range: iv.models?.graham?.mc, tint: .orange)
                    }
                }
            }
            // Market overview stats
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
                statCard("Market Cap", Fmt.number(f?.marketCap, fractionDigits: 0))
                statCard("P/E (TTM)", Fmt.number(f?.trailingPE))
                statCard("Dividend Yield", Fmt.percent(f?.dividendYield))
                statCard("52W High", Fmt.currency(f?.high52, code: nativeCur))
                statCard("52W Low", Fmt.currency(f?.low52, code: nativeCur))
                if !(f?.isETF ?? false) { statCard("Beta", Fmt.number(f?.beta)) }
                if let e = f?.expenseRatio { statCard("Expense Ratio", Fmt.percent(e)) }
            }
            if let summary = f?.summary, !summary.isEmpty {
                card("Business Summary") { Text(summary).font(.callout).foregroundStyle(.secondary) }
            }
        }
    }

    private func positionSection(_ pos: Holding) -> some View {
        card("Your Position") {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
                statCard("Quantity", Fmt.number(pos.quantity))
                statCard("Avg Cost", Fmt.currency(pos.currencyValue("Avg Cost", currency: cur), code: cur))
                statCard("Market Value", Fmt.currency(pos.marketValue(currency: cur), code: cur))
                statCard("Unrealized G/L", Fmt.currency(pos.currencyValue("Unreal. Gain", currency: cur), code: cur),
                         sub: Fmt.percent(pos.unrealizedGainPct), tint: Fmt.tint(for: pos.currencyValue("Unreal. Gain", currency: cur)))
                statCard("Total Return", Fmt.currency(pos.currencyValue("Total Gain", currency: cur), code: cur),
                         sub: Fmt.percent(pos.totalReturnPct), tint: Fmt.tint(for: pos.currencyValue("Total Gain", currency: cur)))
                statCard("IRR %", Fmt.percent(pos.irrPct), tint: Fmt.tint(for: pos.irrPct))
            }
        }
    }

    private func upside(_ iv: Double, _ price: Double?) -> Double? {
        guard let price, price != 0 else { return nil }
        return (iv / price - 1) * 100
    }

    private func ivCard(_ title: String, _ value: Double, upside: Double?, range: IntrinsicValueResponse.MC?, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            HStack(alignment: .firstTextBaseline) {
                Text(Fmt.currency(value, code: nativeCur)).font(.title3.bold()).foregroundStyle(tint)
                if let u = upside { Text(Fmt.percent(u)).font(.caption.bold()).foregroundStyle(Fmt.tint(for: u)) }
            }
            if let r = range, let bear = r.bear, let bull = r.bull {
                Text("Range: \(Fmt.currency(bear, code: nativeCur)) – \(Fmt.currency(bull, code: nativeCur))")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Chart

    @ViewBuilder private var chartTab: some View {
        StockPriceChartView(
            symbol: viewModel.symbol,
            currency: cur,
            avgCost: viewModel.userPosition?.double("Avg Cost"),
            fxRate: viewModel.userPosition?.double("fx_rate") ?? 1,
            accounts: appState.accountsQuery,
            hidePrice: true
        )
    }

    // MARK: - Analysis

    @ViewBuilder private var analysisTab: some View {
        if viewModel.isLoadingAnalysis {
            ProgressView("Generating analysis…").frame(maxWidth: .infinity).padding(40)
        } else if let a = viewModel.analysis, a.scorecard != nil || a.summary != nil {
            VStack(alignment: .leading, spacing: 16) {
                card("AI Fundamental Review", trailing: AnyView(
                    Button { Task { await viewModel.loadAnalysis(force: true) } } label: { Label("Regenerate", systemImage: "arrow.clockwise") }.font(.caption2))) {
                    if let s = a.summary { Text(Self.md(s)).font(.callout).foregroundStyle(.secondary) }
                }
                let topics: [(String, String, Double?, String?, Color)] = [
                    ("Moat & Edge", "shield", a.scorecard?.moat, a.analysis?.moat, .blue),
                    ("Financial Strength", "bolt", a.scorecard?.financialStrength, a.analysis?.financialStrength, .orange),
                    ("Predictability", "target", a.scorecard?.predictability, a.analysis?.predictability, .green),
                    ("Growth Perspective", "chart.line.uptrend.xyaxis", a.scorecard?.growth, a.analysis?.growthPerspective, .purple),
                ]
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 12)], spacing: 12) {
                    ForEach(topics, id: \.0) { t in
                        VStack(spacing: 4) {
                            Text(t.0).font(.caption2).foregroundStyle(.secondary).multilineTextAlignment(.center)
                            Text("\(Fmt.number(t.2, fractionDigits: 0))").font(.system(size: 30, weight: .black)).foregroundStyle(t.4)
                            + Text("/10").font(.caption).foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity).padding(.vertical, 8)
                        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
                    }
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 12)], spacing: 12) {
                    ForEach(topics, id: \.0) { t in
                        card(t.0) {
                            Text(Self.md(t.3 ?? "No analysis available.")).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                if let sentiment = a.sentiment { sentimentCard(sentiment) }
                if !a.catalysts.isEmpty { catalystsCard(a.catalysts) }
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "sparkles").font(.largeTitle).foregroundStyle(.purple.opacity(0.4))
                Text("No analysis data available.").foregroundStyle(.secondary)
                Button("Generate Analysis") { Task { await viewModel.loadAnalysis(force: true) } }.buttonStyle(.borderedProminent)
            }.frame(maxWidth: .infinity).padding(40)
        }
    }

    private func sentimentCard(_ s: Double) -> some View {
        let tone: Color = s >= 70 ? .green : (s >= 40 ? .orange : .red)
        let label = s >= 70 ? "Bullish" : (s >= 40 ? "Neutral" : "Bearish")
        return card("Market Sentiment", trailing: AnyView(
            Text(label).font(.caption.bold()).padding(.horizontal, 8).padding(.vertical, 2).background(tone.opacity(0.2), in: Capsule()).foregroundStyle(tone))) {
            ProgressView(value: max(0, min(s, 100)), total: 100).tint(tone)
            Text("\(Int(s))/100").font(.caption).foregroundStyle(.secondary)
        }
    }

    private func catalystsCard(_ catalysts: [StockAnalysis.Catalyst]) -> some View {
        card("Catalysts") {
            ForEach(catalysts) { c in
                HStack(alignment: .top) {
                    Image(systemName: "bolt.fill").foregroundStyle(.indigo).font(.caption)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(c.event).font(.callout.weight(.medium))
                        Text("\(c.date) · \(c.impact)").font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer()
                }.padding(.vertical, 2)
            }
        }
    }

    // MARK: - Financials

    @ViewBuilder private var financialsTab: some View {
        if viewModel.isLoadingFinancials {
            ProgressView().frame(maxWidth: .infinity).padding(40)
        } else if let fin = viewModel.financials {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Statement", selection: $finType) {
                    Text("Income").tag("income"); Text("Balance").tag("balance")
                    Text("Cash Flow").tag("cash"); Text("Equity").tag("equity")
                }.pickerStyle(.segmented)
                let stmt = statement(for: finType, fin)
                if let stmt, !stmt.index.isEmpty { statementTable(stmt) }
                else { ContentUnavailableView("No data for this statement", systemImage: "doc").frame(height: 200) }
            }
        } else {
            ContentUnavailableView("No financials", systemImage: "doc").frame(height: 200)
        }
    }

    private func statement(for type: String, _ f: FinancialsResponse) -> FinancialStatement? {
        switch type {
        case "balance": return f.balanceSheet; case "cash": return f.cashflow
        case "equity": return f.shareholdersEquity; default: return f.financials
        }
    }

    private func statementTable(_ s: FinancialStatement) -> some View {
        ScrollView(.horizontal, showsIndicators: true) {
            Grid(alignment: .trailing, horizontalSpacing: 16, verticalSpacing: 4) {
                GridRow {
                    Text("").gridColumnAlignment(.leading)
                    Text("Trend")
                    ForEach(Array(s.columns.prefix(6).enumerated()), id: \.offset) { _, c in Text(String(c.prefix(4))) }
                }.font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                Divider()
                ForEach(Array(s.index.enumerated()), id: \.offset) { i, label in
                    GridRow {
                        Text(label).font(.caption).lineLimit(1).gridColumnAlignment(.leading)
                        sparkline(i < s.data.count ? s.data[i].compactMap { $0 } : [])
                        ForEach(0..<min(6, s.columns.count), id: \.self) { j in
                            let v = (i < s.data.count && j < s.data[i].count) ? s.data[i][j] : nil
                            Text(v.map { compact($0) } ?? "—").font(.caption).monospacedDigit()
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder private func sparkline(_ data: [Double]) -> some View {
        if data.count > 1 {
            Chart(Array(data.reversed().enumerated()), id: \.offset) { i, v in
                LineMark(x: .value("i", i), y: .value("v", v)).foregroundStyle(.indigo)
            }
            .chartXAxis(.hidden).chartYAxis(.hidden).chartYScale(domain: chartDomain(data)).frame(width: 56, height: 18)
        } else { Text("—").foregroundStyle(.secondary) }
    }

    // MARK: - Ratios

    @ViewBuilder private var ratiosTab: some View {
        if viewModel.isLoadingFinancials {
            ProgressView().frame(maxWidth: .infinity).padding(40)
        } else if let r = viewModel.ratios, !r.historical.isEmpty {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 16)], spacing: 16) {
                ratioChart("Return on Equity", "Return on Equity (ROE) (%)", r.historical, .green, suffix: "%")
                ratioChart("Gross Margin", "Gross Profit Margin (%)", r.historical, .cyan, suffix: "%")
                ratioChart("Net Margin", "Net Profit Margin (%)", r.historical, .purple, suffix: "%")
                ratioChart("Asset Turnover", "Asset Turnover", r.historical, .orange, suffix: "")
            }
        } else {
            ContentUnavailableView("No historical ratio data", systemImage: "chart.bar").frame(height: 200)
        }
    }

    private func ratioChart(_ title: String, _ key: String, _ rows: [[String: JSONValue]], _ color: Color, suffix: String) -> some View {
        let points: [(String, Double)] = rows.compactMap { row in
            guard let period = row["Period"]?.stringValue, let v = row[key]?.doubleValue else { return nil }
            return (period, v)
        }
        return card(title) {
            if points.isEmpty {
                Text("No data.").foregroundStyle(.secondary)
            } else {
                Chart(points, id: \.0) { p in
                    LineMark(x: .value("Period", p.0), y: .value(title, p.1)).foregroundStyle(color).interpolationMethod(.monotone)
                    PointMark(x: .value("Period", p.0), y: .value(title, p.1)).foregroundStyle(color)
                }
                .chartYScale(domain: chartDomain(points.map(\.1)))
                .chartHoverTooltip(points.map(\.0)) { i in
                    ChartTooltipContent(title: points[i].0,
                                        rows: [ChartTooltipRow(color: color, label: title,
                                                               value: String(format: "%.2f", points[i].1) + suffix)])
                }
                .frame(height: 160)
            }
        }
    }

    // MARK: - Valuation

    @ViewBuilder private var valuationTab: some View {
        if let iv = viewModel.intrinsic {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 12) {
                    valBox("Average Intrinsic Value", Fmt.currency(iv.averageIntrinsicValue, code: nativeCur), .indigo)
                    valBox("Current Price", Fmt.currency(iv.currentPrice, code: nativeCur), .primary)
                    valBox("Margin of Safety", Fmt.percent(iv.marginOfSafetyPct), Fmt.tint(for: iv.marginOfSafetyPct))
                }
                if let note = iv.valuationNote, !note.isEmpty {
                    HStack(alignment: .top) {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                        Text(note).font(.callout).italic().foregroundStyle(.orange)
                    }.padding(12).background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 280), spacing: 16)], spacing: 16) {
                    modelCard("Discounted Cash Flow", iv.models?.dcf, .green)
                    modelCard("Graham Formula", iv.models?.graham, .orange)
                }
            }
        } else {
            ContentUnavailableView("No valuation data", systemImage: "dollarsign").frame(height: 200)
        }
    }

    private func valBox(_ label: String, _ value: String, _ tint: Color) -> some View {
        VStack(spacing: 4) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase).multilineTextAlignment(.center)
            Text(value).font(.title.bold()).foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity).padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func modelCard(_ title: String, _ model: IntrinsicValueResponse.Model?, _ tint: Color) -> some View {
        card(title) {
            if let m = model {
                HStack {
                    Text(m.model ?? title).font(.subheadline.weight(.medium))
                    Spacer()
                    if let iv = m.intrinsicValue {
                        Text(Fmt.currency(iv, code: nativeCur)).font(.callout.bold())
                            .padding(.horizontal, 8).padding(.vertical, 2).background(tint.opacity(0.2), in: Capsule()).foregroundStyle(tint)
                    }
                }
                if let err = m.error, !err.isEmpty { Text(err).font(.caption).foregroundStyle(.red) }
                if let mc = m.mc, let bear = mc.bear, let bull = mc.bull {
                    Text("Bear \(Fmt.currency(bear, code: nativeCur)) · Base \(Fmt.currency(mc.base, code: nativeCur)) · Bull \(Fmt.currency(bull, code: nativeCur))")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            } else { Text("Not available.").foregroundStyle(.secondary) }
        }
    }

    // MARK: - Holdings (ETF)

    @ViewBuilder private var holdingsTab: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let top = f?.etfTopHoldings, !top.isEmpty {
                card("Top Holdings") {
                    ForEach(Array(top.enumerated()), id: \.offset) { _, h in
                        HStack {
                            StockIcon(symbol: h.symbol, size: 18)
                            Text(h.symbol).fontWeight(.medium)
                            Text(h.name).foregroundStyle(.secondary).lineLimit(1)
                            Spacer()
                            Text(String(format: "%.2f%%", h.percent)).monospacedDigit()
                        }.font(.callout).padding(.vertical, 2)
                    }
                }
            }
            if let sectors = f?.etfSectorWeightings, !sectors.isEmpty {
                card("Sector Weightings") {
                    ForEach(sectors, id: \.0) { s in
                        HStack { Text(s.0); Spacer(); Text(String(format: "%.1f%%", s.1 * (s.1 <= 1 ? 100 : 1))).monospacedDigit() }
                            .font(.callout).padding(.vertical, 1)
                    }
                }
            }
        }
    }

    // MARK: - News

    @ViewBuilder private var newsTab: some View {
        if viewModel.isLoadingNews {
            ProgressView().frame(maxWidth: .infinity).padding(40)
        } else if viewModel.news.isEmpty {
            ContentUnavailableView("No recent news", systemImage: "newspaper").frame(height: 200)
        } else {
            VStack(spacing: 10) {
                ForEach(viewModel.news) { item in
                    Button { if let u = URL(string: item.url) { openURL(u) } } label: {
                        HStack(alignment: .top, spacing: 10) {
                            if let t = item.thumbnail, let u = URL(string: t) {
                                AsyncImage(url: u) { $0.resizable().aspectRatio(contentMode: .fill) } placeholder: { Color.gray.opacity(0.15) }
                                    .frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                            VStack(alignment: .leading, spacing: 3) {
                                Text(item.title).font(.callout.weight(.semibold)).multilineTextAlignment(.leading).lineLimit(2)
                                Text(item.provider).font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                            }
                            Spacer()
                        }
                        .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
                    }.buttonStyle(.plain)
                }
            }
        }
    }

    // MARK: - Helpers

    private func card<C: View>(_ title: String, trailing: AnyView? = nil, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack { Text(title).font(.headline); Spacer(); if let trailing { trailing } }
            content()
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func statCard(_ label: String, _ value: String, sub: String? = nil, tint: Color = .primary) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(value).font(.title3.weight(.semibold)).foregroundStyle(tint).lineLimit(1).minimumScaleFactor(0.6)
            if let sub { Text(sub).font(.caption2).foregroundStyle(tint) }
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
    }

    private func compact(_ v: Double) -> String {
        let a = abs(v)
        if a >= 1_000_000_000 { return String(format: "%.2fB", v / 1_000_000_000) }
        if a >= 1_000_000 { return String(format: "%.2fM", v / 1_000_000) }
        if a >= 1_000 { return String(format: "%.1fK", v / 1_000) }
        return Fmt.number(v, fractionDigits: 0)
    }

    static let dateFmt: DateFormatter = { let f = DateFormatter(); f.dateStyle = .medium; return f }()

    static func md(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(s)
    }
}
