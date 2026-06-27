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
        case .overview: return "square.grid.2x2"; case .chart: return "chart.line.uptrend.xyaxis"
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
    @Environment(\.horizontalSizeClass) var hSizeClass
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel: StockDetailViewModel
    @State private var tab: DetailTab = .overview
    @State private var finType = "income"
    @State private var detail: SymbolID?
    @State private var showGrahamExplanation = false

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
        .macMinSize(width: 860, height: 720)
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
        Group {
            #if os(iOS)
            if hSizeClass == .compact {
                compactHeader
            } else {
                regularHeader
            }
            #else
            regularHeader
            #endif
        }
        .padding(20)
    }

    private var regularHeader: some View {
        HStack(alignment: .center, spacing: 16) {
            ZStack {
                LinearGradient(colors: [.indigo, .purple], startPoint: .topLeading, endPoint: .bottomTrailing)
                StockIcon(symbol: viewModel.symbol, size: 48)
                    .padding(8)
                    .background(.white)
            }
            .frame(width: 64, height: 64)
            .clipShape(RoundedRectangle(cornerRadius: 16))

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(f?.shortName ?? viewModel.symbol)
                        .font(.system(size: 32, weight: .black, design: .default))
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                    
                    Text(viewModel.symbol)
                        .font(.system(size: 13, weight: .bold, design: .monospaced))
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                        .foregroundStyle(.secondary)
                }
                if f?.sector != nil || f?.industry != nil {
                    HStack(spacing: 6) {
                        if let s = f?.sector { Text(s).font(.subheadline.weight(.semibold)).foregroundStyle(.indigo).lineLimit(1) }
                        if f?.sector != nil && f?.industry != nil { Text("•").foregroundStyle(.secondary) }
                        if let i = f?.industry { Text(i).font(.subheadline).foregroundStyle(.secondary).lineLimit(1) }
                    }
                }
            }
            
            Spacer(minLength: 16)
            
            VStack(alignment: .trailing, spacing: 4) {
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                if let p = f?.price {
                    Text(Fmt.currency(p, code: nativeCur))
                        .font(.system(size: 32, weight: .black, design: .default))
                        .foregroundStyle(.indigo)
                }
            }
            
            Button { dismiss() } label: { Image(systemName: "xmark") }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .font(.system(size: 23, weight: .bold))
                .padding(8)
                .background(.background.secondary, in: Circle())
                .padding(.leading, 8)
        }
    }

    private var compactHeader: some View {
        VStack(spacing: 16) {
            HStack(alignment: .top, spacing: 12) {
                ZStack {
                    LinearGradient(colors: [.indigo, .purple], startPoint: .topLeading, endPoint: .bottomTrailing)
                    StockIcon(symbol: viewModel.symbol, size: 45)
                        .padding(6)
                        .background(.white)
                }
                .frame(width: 56, height: 56)
                .clipShape(RoundedRectangle(cornerRadius: 14))

                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(f?.shortName ?? viewModel.symbol)
                            .font(.system(size: 25, weight: .black, design: .default))
                            .lineLimit(2)
                            .minimumScaleFactor(0.8)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    HStack(spacing: 6) {
                        Text(viewModel.symbol)
                            .font(.system(size: 13, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                            .foregroundStyle(.secondary)
                        if viewModel.isLoading { ProgressView().controlSize(.small) }
                    }
                }
                
                Spacer(minLength: 8)
                
                Button { dismiss() } label: { Image(systemName: "xmark") }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .font(.system(size: 18, weight: .bold))
                    .padding(8)
                    .background(.background.secondary, in: Circle())
            }
            
            HStack(alignment: .bottom) {
                if let p = f?.price {
                    Text(Fmt.currency(p, code: nativeCur))
                        .font(.system(size: 41, weight: .black, design: .default))
                        .foregroundStyle(.indigo)
                        .minimumScaleFactor(0.8)
                        .lineLimit(1)
                }
                Spacer()
                if f?.sector != nil || f?.industry != nil {
                    VStack(alignment: .trailing, spacing: 2) {
                        if let s = f?.sector { Text(s).font(.caption.weight(.semibold)).foregroundStyle(.indigo).lineLimit(1) }
                        if let i = f?.industry { Text(i).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
                    }
                }
            }
        }
    }

    private var tabBar: some View {
        #if os(iOS)
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 20) {
                ForEach(visibleTabs) { t in
                    Button { tab = t } label: {
                        VStack(spacing: 6) {
                            Image(systemName: t.icon)
                                .font(.system(size: 23, weight: tab == t ? .semibold : .regular))
                            Text(t.rawValue)
                                .font(.caption.weight(tab == t ? .bold : .medium))
                                .fixedSize()
                        }
                        .padding(.bottom, 8)
                        .foregroundStyle(tab == t ? Color.indigo : .secondary)
                        .overlay(alignment: .bottom) {
                            if tab == t { Rectangle().fill(Color.indigo).frame(height: 2) }
                        }
                    }.buttonStyle(.plain)
                }
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.top, 8)
        }
        #else
        HStack(spacing: 24) {
            ForEach(visibleTabs) { t in
                Button { tab = t } label: {
                    VStack(spacing: 6) {
                        Image(systemName: t.icon)
                            .font(.system(size: 23, weight: tab == t ? .semibold : .regular))
                        Text(t.rawValue)
                            .font(.caption.weight(tab == t ? .bold : .medium))
                            .fixedSize()
                    }
                    .padding(.bottom, 8)
                    .foregroundStyle(tab == t ? Color.indigo : .secondary)
                    .overlay(alignment: .bottom) {
                        if tab == t { Rectangle().fill(Color.indigo).frame(height: 2) }
                    }
                }.buttonStyle(.plain)
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
        #endif
    }

    // MARK: - Overview

    @ViewBuilder private var overviewTab: some View {
        VStack(alignment: .leading, spacing: 24) {
            if let pos = viewModel.userPosition {
                positionSection(pos)
            }
            
            marketOverviewHeader
            intrinsicValueSection
            marketStatsSection
            businessSummarySection
        }
    }

    @ViewBuilder private var marketOverviewHeader: some View {
        HStack {
            Label("Market Overview", systemImage: "square.grid.2x2").font(.headline)
            Spacer()
            Button { Task { await viewModel.loadAll() } } label: {
                Label("Refresh Data", systemImage: "arrow.clockwise")
            }
            .font(.caption2.weight(.bold)).foregroundStyle(.cyan)
            .buttonStyle(.plain)
        }
    }

    @ViewBuilder private var intrinsicValueSection: some View {
        if let iv = viewModel.intrinsic {
            // Web app uses 2 columns for Intrinsic Value on md+
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: hSizeClass == .regular ? 2 : 1), spacing: 12) {
                if let dcf = iv.models?.dcf?.intrinsicValue {
                    ivCard("DCF Intrinsic Value", dcf, upside: upside(dcf, iv.currentPrice), range: iv.models?.dcf?.mc, tint: .green, icon: "chart.line.uptrend.xyaxis")
                }
                if let g = iv.models?.graham?.intrinsicValue {
                    ivCard("Graham Intrinsic Value", g, upside: upside(g, iv.currentPrice), range: iv.models?.graham?.mc, tint: .orange, icon: "scalemass")
                }
            }
        }
    }

    @ViewBuilder private var marketStatsSection: some View {
        // Web app uses 3 columns for Market Stats on md+
        let cols = hSizeClass == .regular ? 3 : 2
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: cols), spacing: 12) {
            statCard("Market Cap", Fmt.compact(f?.marketCap ?? 0, code: nativeCur), icon: "globe", iconTint: .indigo)
            statCard("P/E Ratio (TTM)", Fmt.number(f?.trailingPE, fractionDigits: 2), icon: "chart.line.uptrend.xyaxis", iconTint: .green)
            statCard("Dividend Yield", Fmt.percent(f?.dividendYield), icon: "dollarsign", iconTint: .orange)
            statCard("52W High", Fmt.currency(f?.high52, code: nativeCur), icon: "chart.line.uptrend.xyaxis", iconTint: .blue)
            statCard("52W Low", Fmt.currency(f?.low52, code: nativeCur), icon: "chart.line.downtrend.xyaxis", iconTint: .pink)
            if !(f?.isETF ?? false) { statCard("Beta", Fmt.number(f?.beta, fractionDigits: 2), icon: "bolt.heart", iconTint: .purple) }
            if let e = f?.expenseRatio { statCard("Expense Ratio", Fmt.percent(e), icon: "receipt", iconTint: .orange) }
        }
    }

    @ViewBuilder private var businessSummarySection: some View {
        if let summary = f?.summary, !summary.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Label("Business Summary", systemImage: "building.2").font(.headline)
                Text(summary).font(.subheadline).foregroundStyle(.secondary)
                    .lineSpacing(4)
            }
            .padding(20).frame(maxWidth: .infinity, alignment: .leading)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
        }
    }


    private func positionSection(_ pos: Holding) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Your Position", systemImage: "wallet.pass").font(.headline)
                Spacer()
                Text("AGGREGATED").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
                    .padding(.horizontal, 6).padding(.vertical, 2).background(.background.tertiary, in: RoundedRectangle(cornerRadius: 6))
            }
            
            let cols = hSizeClass == .regular ? 3 : 2
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: cols), spacing: 12) {
                statCard("Quantity", Fmt.number(pos.quantity), icon: "number", iconTint: .indigo)
                statCard("Avg Cost", Fmt.currency(pos.currencyValue("Avg Cost", currency: cur), code: cur), icon: "tag", iconTint: .secondary)
                statCard("Market Value", Fmt.currency(pos.marketValue(currency: cur), code: cur), icon: "chart.pie", iconTint: .indigo)
                
                let urGain = pos.currencyValue("Unreal. Gain", currency: cur)
                statCard("Unrealized G/L", Fmt.currency(urGain, code: cur),
                         sub: pos.unrealizedGainPct == .infinity ? "∞" : Fmt.percent(pos.unrealizedGainPct),
                         icon: "bolt.heart",
                         iconTint: (urGain ?? 0) >= 0 ? .green : .red,
                         subTint: (urGain ?? 0) >= 0 ? .green : .red,
                         bgTint: ((urGain ?? 0) >= 0 ? Color.green : Color.red).opacity(0.1))
                
                let tGain = pos.currencyValue("Total Gain", currency: cur)
                statCard("Total Return", Fmt.currency(tGain, code: cur),
                         sub: pos.totalReturnPct == .infinity ? "∞" : Fmt.percent(pos.totalReturnPct),
                         icon: "chart.line.uptrend.xyaxis",
                         iconTint: (tGain ?? 0) >= 0 ? .green : .red,
                         subTint: (tGain ?? 0) >= 0 ? .green : .red,
                         bgTint: ((tGain ?? 0) >= 0 ? Color.green : Color.red).opacity(0.1))
                
                statCard("IRR %", pos.irrPct == .infinity ? "∞" : Fmt.percent(pos.irrPct),
                         icon: "chart.xyaxis.line",
                         iconTint: (pos.irrPct ?? 0) >= 0 ? .green : .red,
                         bgTint: ((pos.irrPct ?? 0) >= 0 ? Color.green : Color.red).opacity(0.1))
            }
            Divider()
        }
    }

    private func upside(_ iv: Double, _ price: Double?) -> Double? {
        guard let price, price != 0 else { return nil }
        return (iv / price - 1) * 100
    }

    private func ivCard(_ title: String, _ value: Double, upside: Double?, range: IntrinsicValueResponse.MC?, tint: Color, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: icon).foregroundStyle(tint)
                Text(title).font(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
            }
            HStack(alignment: .bottom) {
                Text(Fmt.currency(value, code: nativeCur)).font(.title3.weight(.bold)).foregroundStyle(.primary)
                Spacer()
                if let u = upside { Text(Fmt.percent(u)).font(.caption2.weight(.bold)).foregroundStyle(Fmt.tint(for: u)) }
            }
            if let r = range, let bear = r.bear, let bull = r.bull {
                Text("Range: \(Fmt.currency(bear, code: nativeCur)) – \(Fmt.currency(bull, code: nativeCur))")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
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
                VStack(alignment: .leading, spacing: 12) {
                    HStack(alignment: .top, spacing: 16) {
                        Image(systemName: "sparkles")
                            .font(.system(size: 27))
                            .foregroundStyle(.white)
                            .frame(width: 48, height: 48)
                            .background(Color.purple, in: RoundedRectangle(cornerRadius: 12))
                        
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("AI Fundamental Review").font(.title3.bold())
                                Spacer()
                                Button { Task { await viewModel.loadAnalysis(force: true) } } label: { 
                                    Label("Regenerate", systemImage: "arrow.clockwise") 
                                }
                                .font(.caption2.weight(.bold)).foregroundStyle(.purple)
                                .buttonStyle(.plain)
                            }
                            if let s = a.summary { Text(Self.md(s)).font(.subheadline).foregroundStyle(.secondary) }
                        }
                    }
                }
                .padding(24).frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.purple.opacity(0.1), in: RoundedRectangle(cornerRadius: 24))
                
                let topics: [(String, String, Double?, String?, Color)] = [
                    ("Moat & Edge", "shield", a.scorecard?.moat, a.analysis?.moat, .blue),
                    ("Financial Strength", "bolt.fill", a.scorecard?.financialStrength, a.analysis?.financialStrength, .orange),
                    ("Predictability", "target", a.scorecard?.predictability, a.analysis?.predictability, .green),
                    ("Growth Perspective", "chart.line.uptrend.xyaxis", a.scorecard?.growth, a.analysis?.growthPerspective, .purple),
                ]
                
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 16)], spacing: 16) {
                    ForEach(topics, id: \.0) { t in
                        VStack(spacing: 8) {
                            Text(t.0).font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary).textCase(.uppercase)
                            Text("\(Fmt.number(t.2, fractionDigits: 0))").font(.system(size: 36, weight: .black)).foregroundStyle(t.4)
                            + Text("/10").font(.callout).foregroundStyle(.secondary).baselineOffset(8)
                        }
                        .frame(maxWidth: .infinity).padding(.vertical, 16)
                        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
                    }
                }
                
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 16)], spacing: 16) {
                    ForEach(topics, id: \.0) { t in
                        VStack(alignment: .leading, spacing: 12) {
                            HStack(spacing: 12) {
                                Image(systemName: t.1)
                                    .font(.system(size: 18))
                                    .foregroundStyle(t.4)
                                    .frame(width: 36, height: 36)
                                    .background(t.4.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                                Text(t.0).font(.headline)
                            }
                            Text(Self.md(t.3 ?? "No analysis available.")).font(.subheadline).foregroundStyle(.secondary)
                        }
                        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
                        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 20))
                    }
                }
                
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 16)], spacing: 16) {
                    if let sentiment = a.sentiment { sentimentCard(sentiment) }
                    if !a.catalysts.isEmpty { catalystsCard(a.catalysts) }
                }
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
        return VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundStyle(.indigo).frame(width: 32, height: 32)
                    .background(Color.indigo.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                Text("Market Sentiment").font(.headline)
                Spacer()
                Text(label).font(.caption.bold()).padding(.horizontal, 8).padding(.vertical, 4).background(tone.opacity(0.2), in: Capsule()).foregroundStyle(tone)
            }
            VStack(spacing: 8) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.secondary.opacity(0.2)).frame(height: 12)
                        Capsule().fill(tone).frame(width: max(0, min(geo.size.width * CGFloat(s / 100.0), geo.size.width)), height: 12)
                    }
                }.frame(height: 12).padding(.vertical, 8)
                HStack {
                    Text("Extreme Fear").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary).textCase(.uppercase)
                    Spacer()
                    Text("\(Int(s))%").font(.title3.weight(.bold)).foregroundStyle(.primary)
                    Spacer()
                    Text("Extreme Greed").font(.system(size: 11, weight: .bold)).foregroundStyle(.secondary).textCase(.uppercase)
                }
            }
            Text("Current market vibe based on news flow, analyst ratings, and social trends.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: .infinity).padding(.top, 8)
        }
        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 20))
    }

    private func catalystsCard(_ catalysts: [StockAnalysis.Catalyst]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "calendar")
                    .foregroundStyle(.orange).frame(width: 32, height: 32)
                    .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                Text("Upcoming Catalysts").font(.headline)
            }
            VStack(alignment: .leading, spacing: 12) {
                ForEach(Array(catalysts.enumerated()), id: \.offset) { i, c in
                    HStack(alignment: .top, spacing: 12) {
                        VStack(spacing: 0) {
                            Circle().fill(c.impact == "High" ? Color.red : (c.impact == "Medium" ? .orange : .blue)).frame(width: 8, height: 8).padding(.top, 4)
                            if i < catalysts.count - 1 {
                                Rectangle().fill(Color.secondary.opacity(0.3)).frame(width: 1).padding(.top, 4)
                            }
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(alignment: .top) {
                                Text(c.event).font(.subheadline.weight(.semibold))
                                Spacer()
                                Text(c.impact).font(.system(size: 10, weight: .bold)).textCase(.uppercase).foregroundStyle(.secondary)
                                    .padding(.horizontal, 4).padding(.vertical, 2).overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Color.secondary.opacity(0.3)))
                            }
                            Text(c.date).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                        }.padding(.bottom, i < catalysts.count - 1 ? 12 : 0)
                    }
                }
            }
        }
        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 20))
    }

    // MARK: - Financials

    @ViewBuilder private var financialsTab: some View {
        if viewModel.isLoadingFinancials {
            ProgressView().frame(maxWidth: .infinity).padding(40)
        } else if let fin = viewModel.financials {
            VStack(alignment: .leading, spacing: 16) {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        let tabs = [("income", "Income", "receipt"), ("balance", "Balance", "scalemass"), ("cash", "Cash Flow", "wallet.pass"), ("equity", "Equity", "person.2")]
                        ForEach(tabs, id: \.0) { t in
                            Button { finType = t.0 } label: {
                                HStack(spacing: 6) {
                                    Image(systemName: t.2).font(.system(size: 16))
                                    Text(t.1)
                                }
                                .font(.caption.weight(.bold))
                                .padding(.horizontal, 16).padding(.vertical, 8)
                                .foregroundStyle(finType == t.0 ? Color.white : .secondary)
                                .background(finType == t.0 ? Color.indigo : Color.secondary.opacity(0.15), in: Capsule())
                            }.buttonStyle(.plain)
                        }
                    }
                }
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
            Grid(alignment: .trailing, horizontalSpacing: 24, verticalSpacing: 12) {
                GridRow {
                    Text("Metric").gridColumnAlignment(.leading)
                    Text("Trend").gridColumnAlignment(.center)
                    ForEach(Array(s.columns.prefix(6).enumerated()), id: \.offset) { _, c in Text(String(c.prefix(4))) }
                }
                .font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                
                Divider()
                
                ForEach(Array(s.index.enumerated()), id: \.offset) { i, label in
                    GridRow {
                        Text(label).font(.subheadline.weight(.semibold)).lineLimit(1).gridColumnAlignment(.leading)
                        sparkline(i < s.data.count ? s.data[i].compactMap { $0 } : [])
                        ForEach(0..<min(6, s.columns.count), id: \.self) { j in
                            let v = (i < s.data.count && j < s.data[i].count) ? s.data[i][j] : nil
                            Text(v.map { compact($0) } ?? "—")
                                .font(.subheadline).monospacedDigit()
                                .foregroundStyle((v ?? 0) < 0 ? .red : .primary)
                        }
                    }
                    Divider()
                }
            }
            .padding(20)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.quaternary, lineWidth: 1))
            .padding(1)
        }
    }

    @ViewBuilder private func sparkline(_ data: [Double]) -> some View {
        if data.count > 1 {
            let timeFirst = data.last ?? 0
            let timeLast = data.first ?? 0
            let color: Color = timeLast >= timeFirst ? .green : .red
            
            Chart(Array(data.reversed().enumerated()), id: \.offset) { i, v in
                LineMark(x: .value("i", i), y: .value("v", v))
                    .foregroundStyle(color)
                    .interpolationMethod(.monotone)
                AreaMark(x: .value("i", i), y: .value("v", v))
                    .foregroundStyle(LinearGradient(colors: [color.opacity(0.3), color.opacity(0.0)], startPoint: .top, endPoint: .bottom))
                    .interpolationMethod(.monotone)
            }
            .chartXAxis(.hidden).chartYAxis(.hidden).chartYScale(domain: chartDomain(data))
            .frame(width: 64, height: 24)
            // Confine the gradient area fill to the cell; otherwise it bleeds into
            // neighbouring rows and the columns merge into one continuous band.
            .clipped()
        } else { Text("—").foregroundStyle(.secondary).frame(width: 64, alignment: .center) }
    }

    // MARK: - Ratios

    @ViewBuilder private var ratiosTab: some View {
        if let h = viewModel.ratios?.historical, !h.isEmpty {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 16)], spacing: 16) {
                ratioChart("Return on Equity", h, "Return on Equity (ROE) (%)", Color(red: 16/255, green: 185/255, blue: 129/255), isPercent: true)
                ratioChart("Gross Margin", h, "Gross Profit Margin (%)", Color(red: 6/255, green: 182/255, blue: 212/255), isPercent: true)
                ratioChart("Net Margin", h, "Net Profit Margin (%)", Color(red: 139/255, green: 92/255, blue: 246/255), isPercent: true)
                ratioChart("Asset Turnover", h, "Asset Turnover", Color(red: 245/255, green: 158/255, blue: 11/255), isPercent: false)
            }
        } else if viewModel.isLoadingFinancials {
            ProgressView().frame(maxWidth: .infinity).padding(40)
        } else {
            ContentUnavailableView("No ratio data", systemImage: "chart.line.uptrend.xyaxis").frame(height: 200)
        }
    }

    private func ratioChart(_ title: String, _ data: [[String: JSONValue]], _ key: String, _ color: Color, isPercent: Bool) -> some View {
        let valid = data.filter { $0[key]?.doubleValue != nil }.reversed()
        return card(title) {
            Chart {
                ForEach(Array(valid.enumerated()), id: \.offset) { _, item in
                    if let val = item[key]?.doubleValue, let dateStr = item["Period"]?.stringValue {
                        let displayDate = String(dateStr.prefix(4))
                        LineMark(x: .value("Year", displayDate), y: .value(title, val))
                            .foregroundStyle(color).interpolationMethod(.monotone)
                        AreaMark(x: .value("Year", displayDate), y: .value(title, val))
                            .foregroundStyle(LinearGradient(colors: [color.opacity(0.3), color.opacity(0.0)], startPoint: .top, endPoint: .bottom))
                            .interpolationMethod(.monotone)
                        PointMark(x: .value("Year", displayDate), y: .value(title, val))
                            .foregroundStyle(color)
                    }
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(isPercent ? Fmt.percent(v) : Fmt.number(v, fractionDigits: 2))
                        }
                    }
                }
            }
            .frame(height: 200)
        }
    }

    // MARK: - Valuation

    @ViewBuilder private var valuationTab: some View {
        VStack(spacing: 24) {
            if let iv = viewModel.intrinsic {
                valuationSummaryCards(iv)

                if let note = iv.valuationNote {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange).font(.title3)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Model Discrepancy Note").font(.caption.weight(.bold)).foregroundStyle(.orange).textCase(.uppercase)
                            Text(note).font(.subheadline.italic()).foregroundStyle(.orange)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
                }
                
                if let models = iv.models {
                    if let dcf = models.dcf {
                        dcfCard("Discounted Cash Flow", "chart.line.uptrend.xyaxis", .green, dcf, modelKey: "dcf", iv: iv)
                    }
                    if let g = models.graham {
                        grahamCard("Graham Formula", "scalemass", .orange, g, modelKey: "graham", iv: iv)
                    }
                }
                
                if (f?.isETF ?? false) && (iv.models?.dcf == nil && iv.models?.graham == nil) {
                    card("Why standard models aren't shown?") {
                        Text("Traditional valuation methods like Discounted Cash Flow (DCF) and Graham's Formula rely on free cash flow and earnings growth, which are company-specific metrics. For ETFs, which are baskets of many securities, these metrics cannot be reliably aggregated or projected. Therefore, intrinsic value modeling is not applicable.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
            } else if viewModel.isLoadingFinancials {
                ProgressView().frame(maxWidth: .infinity).padding(40)
            } else {
                ContentUnavailableView("Valuation unavailable", systemImage: "dollarsign.circle").frame(height: 200)
            }
        }
    }

    /// The three valuation summary cards (intrinsic value / current price /
    /// margin of safety). Side-by-side on regular widths; stacked on compact
    /// (iPhone) so the large figures aren't squeezed into a third of the screen
    /// and wrapped character-by-character.
    @ViewBuilder private func valuationSummaryCards(_ iv: IntrinsicValueResponse) -> some View {
        let mos = iv.marginOfSafetyPct ?? 0
        let intrinsic = valuationCard(label: "Average Intrinsic Value",
                                      value: Fmt.currency(iv.averageIntrinsicValue, code: nativeCur),
                                      valueColor: .indigo) {
            if let r = iv.range {
                Text("Range: \(Fmt.currency(r.bear, code: nativeCur)) - \(Fmt.currency(r.bull, code: nativeCur))")
                    .font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        let current = valuationCard(label: "Current Price",
                                    value: Fmt.currency(iv.currentPrice, code: nativeCur),
                                    valueColor: .primary) { EmptyView() }
        let safety = valuationCard(label: "Margin of Safety",
                                   value: Fmt.percent(mos),
                                   valueColor: mos >= 0 ? .green : .red,
                                   tint: mos >= 0 ? Color.green.opacity(0.1) : Color.red.opacity(0.1)) { EmptyView() }

        if hSizeClass == .compact {
            VStack(spacing: 12) { intrinsic; current; safety }
        } else {
            HStack(spacing: 16) { intrinsic; current; safety }
        }
    }

    private func valuationCard<Sub: View>(label: String, value: String, valueColor: Color,
                                          tint: Color? = nil,
                                          @ViewBuilder sub: () -> Sub) -> some View {
        VStack(spacing: 8) {
            Text(label).font(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
                .multilineTextAlignment(.center)
            Text(value).font(.system(size: 32, weight: .bold)).foregroundStyle(valueColor)
                .lineLimit(1).minimumScaleFactor(0.5)   // shrink instead of wrapping
            sub()
        }
        .frame(maxWidth: .infinity)
        .padding(hSizeClass == .compact ? 16 : 24)
        .background {
            if let tint {
                RoundedRectangle(cornerRadius: 16).fill(tint)
            } else {
                RoundedRectangle(cornerRadius: 16).fill(.background.secondary)
            }
        }
    }

    private func paramRow(_ label: String, _ val: String, _ isNote: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label).font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
            Text(val).font(isNote ? .caption : .subheadline.weight(.semibold))
                .foregroundStyle(isNote ? Color.secondary : .primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func mcGrid(_ mc: IntrinsicValueResponse.MC?, type: String) -> some View {
        VStack(alignment: .center, spacing: 12) {
            Text("Probabilistic Scenarios (Monte Carlo)").font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
            HStack(spacing: 8) {
                VStack(spacing: 6) {
                    Text("Bear (10th)").font(.caption2.weight(.bold)).foregroundStyle(.red).textCase(.uppercase)
                    Text(Fmt.currency(mc?.bear, code: nativeCur)).font(.subheadline.bold())
                }
                .frame(maxWidth: .infinity).padding(8)
                .background(Color.red.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
                
                let midColor: Color = type == "dcf" ? .indigo : .orange
                VStack(spacing: 6) {
                    Text("Median (50th)").font(.caption2.weight(.bold)).foregroundStyle(midColor).textCase(.uppercase)
                    Text(Fmt.currency(mc?.base, code: nativeCur)).font(.subheadline.bold())
                }
                .frame(maxWidth: .infinity).padding(8)
                .background(midColor.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
                
                VStack(spacing: 6) {
                    Text("Bull (90th)").font(.caption2.weight(.bold)).foregroundStyle(.green).textCase(.uppercase)
                    Text(Fmt.currency(mc?.bull, code: nativeCur)).font(.subheadline.bold())
                }
                .frame(maxWidth: .infinity).padding(8)
                .background(Color.green.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func dcfCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                HStack(spacing: 8) {
                    Image(systemName: icon).foregroundStyle(color)
                    Text(title).font(.headline)
                }
                Spacer()
                if m.error == nil {
                    Text(Fmt.currency(m.intrinsicValue, code: nativeCur))
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(color)
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(color.opacity(0.2), in: Capsule())
                }
            }
            
            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if let p = m.parameters {
                    let columns = hSizeClass == .compact 
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)] 
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
                        if let v = p["discount_rate"]?.doubleValue { paramRow("Discount Rate (WACC)", Fmt.percent(v)) }
                        if let v = p["growth_rate"]?.doubleValue { paramRow("Growth Rate", Fmt.percent(v)) }
                        if let v = p["applied_growth"]?.doubleValue { paramRow("Applied Growth", Fmt.percent(v)) }
                        if let v = p["terminal_growth_rate"]?.doubleValue { paramRow("Terminal Growth", Fmt.percent(v)) }
                        if let v = p["projection_years"]?.doubleValue { paramRow("Projection Years", "\(Int(v))") }
                        if let v = p["base_fcf"]?.doubleValue { paramRow("Base FCF", Fmt.compact(v, code: nativeCur)) }
                        if let v = p["fcf_margin"]?.doubleValue { paramRow("Est. FCF Margin", Fmt.percent(v)) }
                    }
                    if let n = p["note"]?.stringValue { 
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Note").font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                            Text(n).font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(.top, 8)
                    }
                }
                
                if let hist = m.mc?.histogram, !hist.isEmpty {
                    VStack(spacing: 16) {
                        mcGrid(m.mc, type: modelKey)
                        histogramChart(hist, mc: m.mc, currentPrice: iv.currentPrice)
                    }
                    .padding(.top, 24)
                }
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
    }

    private func grahamMathBlock(_ p: [String: JSONValue]?) -> some View {
        let y = p?["bond_yield_proxy"]?.doubleValue ?? 4.5
        return Button {
            showGrahamExplanation = true
        } label: {
            VStack(spacing: 16) {
                // Formula
                HStack(spacing: 8) {
                    Text("V").fontWeight(.bold)
                    Text("=").opacity(0.5)
                    Text("EPS").fontWeight(.bold)
                    Text("×").opacity(0.5)
                    Text("8.5 + 2G").fontWeight(.bold)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(.secondary.opacity(0.2), in: RoundedRectangle(cornerRadius: 6))
                    Text("×").opacity(0.5)
                    Text("4.4").fontWeight(.bold)
                    Text("/").opacity(0.5)
                    Text("Y").fontWeight(.bold)
                }
                .font(.system(.body, design: .monospaced))
                .lineLimit(1)
                .minimumScaleFactor(0.5)
                .padding()
                .frame(maxWidth: .infinity)
                .background(.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))

                // Legend
                VStack(alignment: .leading, spacing: 8) {
                    grahamLegend("V", "Intrinsic Value")
                    grahamLegend("EPS", "Trailing 12-Month Earnings")
                    grahamLegend("8.5 + 2G", "Growth Multiplier")
                    grahamLegend("4.4", "Historic Corporate Bond Yield")
                    grahamLegend("Y", "Current Yield (\(Fmt.number(y, fractionDigits: 1))%)")
                }
                .padding(.horizontal, 4)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showGrahamExplanation) {
            grahamExplanationView(y: y)
        }
    }

    private func grahamExplanationView(y: Double) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Graham's Intrinsic Value Formula")
                    .font(.headline)
                Text("This is Benjamin Graham's revised formula for calculating the intrinsic value of a stock, adapted for modern markets.")
                    .font(.subheadline)
                    .fixedSize(horizontal: false, vertical: true)
                
                VStack(alignment: .leading, spacing: 12) {
                    explanationRow("V", "Intrinsic Value", "The estimated true value of the stock.")
                    explanationRow("EPS", "Earnings Per Share", "Trailing 12-month earnings per share.")
                    explanationRow("8.5", "Base P/E", "The price-to-earnings ratio of a no-growth company.")
                    explanationRow("2G", "Growth Multiplier", "G is the expected long-term earnings growth rate. Graham multiplied it by 2.")
                    explanationRow("4.4", "Historic Yield", "The historic average yield of high-grade corporate bonds.")
                    explanationRow("Y", "Current Yield", "The current yield of AAA-rated corporate bonds (\(Fmt.number(y, fractionDigits: 1))%).")
                }
                .font(.caption)
            }
            .padding(24)
        }
        .frame(width: 320)
    }

    private func explanationRow(_ symbol: String, _ title: String, _ desc: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(symbol).fontWeight(.bold)
                Text("-").opacity(0.5)
                Text(title).fontWeight(.semibold)
            }
            Text(desc).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func grahamLegend(_ symbol: String, _ desc: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(symbol)
                .font(.caption.weight(.bold))
                .frame(width: 70, alignment: .trailing)
            Text(desc)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func grahamCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                HStack(spacing: 8) {
                    Image(systemName: icon).foregroundStyle(color)
                    Text(title).font(.headline)
                }
                Spacer()
                if m.error == nil {
                    Text(Fmt.currency(m.intrinsicValue, code: nativeCur))
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(color)
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(color.opacity(0.2), in: Capsule())
                }
            }
            
            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if let p = m.parameters {
                    let columns = hSizeClass == .compact 
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)] 
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
                        if let v = p["eps"]?.doubleValue { paramRow("Trailing EPS", Fmt.number(v, fractionDigits: 2)) }
                        if let v = p["growth_rate_pct"]?.doubleValue { paramRow("Growth Rate (G)", "\(Fmt.number(v, fractionDigits: 2))%") }
                        if let v = p["applied_growth_pct"]?.doubleValue { paramRow("Applied Growth", "\(Fmt.number(v, fractionDigits: 2))%") }
                        if let v = p["bond_yield_proxy"]?.doubleValue { paramRow("Bond Yield (Y)", "\(Fmt.number(v, fractionDigits: 2))%") }
                    }
                    if let n = p["note"]?.stringValue { 
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Note").font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                            Text(n).font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(.top, 8)
                    }
                    grahamMathBlock(p)
                        .padding(.top, 16)
                }
                
                if let hist = m.mc?.histogram, !hist.isEmpty {
                    VStack(spacing: 16) {
                        mcGrid(m.mc, type: modelKey)
                        histogramChart(hist, mc: m.mc, currentPrice: iv.currentPrice)
                    }
                    .padding(.top, 24)
                }
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
    }

    @ViewBuilder
    private func histogramChart(_ hist: [IntrinsicValueResponse.HistogramPoint], mc: IntrinsicValueResponse.MC?, currentPrice: Double?) -> some View {
        let validHist = hist.filter { $0.price != nil && $0.count != nil }
        let minPrice = validHist.first?.price ?? 0
        let maxPrice = validHist.last?.price ?? 1
        let range = maxPrice - minPrice > 0 ? maxPrice - minPrice : 1
        
        let bearPct = max(0, min(1, ((mc?.bear ?? minPrice) - minPrice) / range))
        let bullPct = max(0, min(1, ((mc?.bull ?? maxPrice) - minPrice) / range))
        
        let grad = LinearGradient(
            stops: [
                .init(color: .red, location: 0),
                .init(color: .red, location: bearPct),
                .init(color: .cyan, location: bearPct),
                .init(color: .cyan, location: bullPct),
                .init(color: .green, location: bullPct),
                .init(color: .green, location: 1)
            ],
            startPoint: .leading, endPoint: .trailing
        )
        
        Chart {
            ForEach(validHist, id: \.price) { h in
                if let price = h.price, let count = h.count {
                    AreaMark(
                        x: .value("Price", price),
                        y: .value("Count", count)
                    )
                    .foregroundStyle(grad.opacity(0.4))
                }
            }
            if let c = currentPrice {
                RuleMark(x: .value("Current Price", c))
                    .foregroundStyle(.primary)
                    .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 5]))
                    .annotation(position: .top) {
                        Text("Current").font(.system(size: 11, weight: .bold))
                    }
            }
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { v in
                if let val = v.as(Double.self) {
                    AxisValueLabel { Text(Fmt.compact(val, code: nativeCur)) }
                    AxisGridLine()
                    AxisTick()
                }
            }
        }
        .chartYAxis(.hidden)
        .frame(height: 150)
        .padding(.top, 16)
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
        if !f!.etfTopHoldings.isEmpty {
            card("Top 10 Holdings") {
                VStack(spacing: 8) {
                    HStack {
                        Text("Symbol").font(.caption.weight(.bold)).foregroundStyle(.secondary)
                        Spacer()
                        Text("% Assets").font(.caption.weight(.bold)).foregroundStyle(.secondary)
                    }
                    Divider()
                    ForEach(f!.etfTopHoldings, id: \.symbol) { h in
                        HStack {
                            Text(h.symbol).font(.headline)
                            Spacer()
                            Text(Fmt.percent(h.percent)).font(.subheadline.bold())
                        }
                        Divider()
                    }
                }
            }
        }
        if !f!.etfSectorWeightings.isEmpty {
            card("Sector Allocation") {
                Chart(f!.etfSectorWeightings, id: \.0) { s in
                    SectorMark(
                        angle: .value("Weight", s.1),
                        innerRadius: .ratio(0.6),
                        angularInset: 1.5
                    )
                    .foregroundStyle(by: .value("Sector", s.0))
                }
                .frame(height: 250)
            }
        }
        if f!.etfTopHoldings.isEmpty && f!.etfSectorWeightings.isEmpty {
            ContentUnavailableView("No holdings data", systemImage: "briefcase").frame(height: 200)
        }
    }

    // MARK: - News

    @ViewBuilder private var newsTab: some View {
        if viewModel.isLoadingNews {
            ProgressView().frame(maxWidth: .infinity).padding(40)
        } else if viewModel.news.isEmpty {
            ContentUnavailableView("No recent news", systemImage: "newspaper").frame(height: 200)
        } else {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 16)], spacing: 16) {
                ForEach(viewModel.news) { item in
                    Button { if let u = URL(string: item.url) { openURL(u) } } label: {
                        VStack(alignment: .leading, spacing: 0) {
                            if let t = item.thumbnail, let u = URL(string: t) {
                                AsyncImage(url: u) { $0.resizable().aspectRatio(contentMode: .fill) } placeholder: { Color.gray.opacity(0.15) }
                                    .frame(height: 160).clipped()
                            } else {
                                ZStack {
                                    Rectangle().fill(.quaternary)
                                    Image(systemName: "newspaper").font(.largeTitle).foregroundStyle(.tertiary)
                                }
                                .frame(height: 160).clipped()
                            }
                            VStack(alignment: .leading, spacing: 12) {
                                HStack(spacing: 8) {
                                    Text(item.provider).font(.system(size: 11, weight: .bold)).textCase(.uppercase).foregroundStyle(.indigo)
                                        .padding(.horizontal, 8).padding(.vertical, 4).background(Color.indigo.opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
                                    Text(item.pubDate).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                                }
                                Text(item.title).font(.headline).foregroundStyle(.primary).lineLimit(3)
                            }.padding(20)
                            Spacer(minLength: 0)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
                        .clipShape(RoundedRectangle(cornerRadius: 16))
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

    private func statCard(_ label: String, _ value: String, sub: String? = nil, icon: String? = nil, iconTint: Color = .primary, subTint: Color? = nil, bgTint: Color? = nil) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                if let icon {
                    Image(systemName: icon).foregroundStyle(iconTint).font(.system(size: 16))
                }
                Text(label).font(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
            }
            HStack(alignment: .bottom) {
                Text(value).font(.title3.weight(.bold)).foregroundStyle(icon == nil ? iconTint : .primary).lineLimit(1).minimumScaleFactor(0.6)
                Spacer()
                if let sub { Text(sub).font(.caption2.weight(.bold)).foregroundStyle(subTint ?? iconTint) }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding(16)
        .background(bgTint ?? Color.gray.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
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
