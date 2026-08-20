import SwiftUI
import Charts

/// Identifiable wrapper so a bare symbol string can drive `.sheet(item:)`.
struct SymbolID: Identifiable, Hashable { let id: String }

private enum DetailTab: String, CaseIterable, Identifiable {
    case overview = "Overview", position = "Position & Lots", chart = "Chart", analysis = "Analysis"
    case financials = "Financials", ratios = "Ratios & Trends", valuation = "Valuation"
    case holdings = "Holdings", news = "News"
    var id: String { rawValue }
    var icon: String {
        switch self {
        case .overview: return "square.grid.2x2"
        case .position: return "briefcase"
        case .chart: return "chart.line.uptrend.xyaxis"
        case .analysis: return "sparkles"
        case .financials: return "doc.text"
        case .ratios: return "chart.bar"
        case .valuation: return "dollarsign"
        case .holdings: return "chart.pie"
        case .news: return "newspaper"
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
    @Environment(\.colorScheme) private var colorScheme
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel: StockDetailViewModel
    @State private var tab: DetailTab = .overview
    @State private var finType = "income"
    /// Charted line items by colour slot, so removing one never repaints the
    /// others. Empty means "no explicit choice yet".
    @State private var chartSlots: [String?] = []
    @State private var showAllMetrics = false
    /// nil follows the period's own default; a tap pins a range.
    @State private var chartRange: StatementRange?
    @State private var detail: SymbolID?
    @State private var showGrahamExplanation = false
    @State private var summaryExpanded = false

    init(symbol: String, currency: String = "USD") {
        _viewModel = StateObject(wrappedValue: StockDetailViewModel(symbol: symbol, currency: currency))
    }

    private var f: Fundamentals? { viewModel.fundamentals }
    private var nativeCur: String { f?.currency ?? "USD" }
    private var cur: String { viewModel.currency }

    private var visibleTabs: [DetailTab] {
        var tabs: [DetailTab] = [.overview, .position, .chart, .analysis]
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
                    case .position: positionTab
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
        .macSheetSize(width: 860, height: 720)
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
            
            upcomingEventsSection
            marketOverviewHeader
            intrinsicValueSection
            marketStatsSection
            StockKeyMetricsView(
                metrics: f?.keyMetrics ?? [:],
                beta: f?.beta,
                averageVolume: f?.double("averageVolume")
            )
            businessSummarySection
        }
    }

    /// Just-reported quarter / next earnings report / next dividend, when the
    /// backend could derive any of them.
    @ViewBuilder private var upcomingEventsSection: some View {
        let earnings = f?.upcomingEarnings
        let reported = f?.recentEarnings
        let dividend = f?.upcomingDividend
        if earnings != nil || dividend != nil || reported != nil {
            VStack(alignment: .leading, spacing: 12) {
                Label("Upcoming Events", systemImage: "calendar").font(.headline)
                // One panel of full-width rows, not a grid of cards: there are
                // one to three of these, and as cards the odd one out left half
                // the section blank.
                VStack(spacing: 0) {
                    if let r = reported {
                        eventRow(
                            "Latest Earnings", icon: "chart.bar.fill", tint: Theme.earnings,
                            date: r.date, dateEnd: nil, status: r.status,
                            timeZone: r.marketTimezone,
                            detail: Self.reportedDetail(r),
                            detailTint: r.surprisePct.map { $0 >= 0 ? Color.up : Color.down })
                    }
                    if let e = earnings {
                        if reported != nil { Divider().opacity(0.5) }
                        eventRow(
                            "Next Earnings", icon: "chart.bar.fill", tint: Theme.earnings,
                            date: e.date, dateEnd: e.dateEnd, status: e.status,
                            timeZone: e.marketTimezone,
                            detail: e.epsEstimate.map { est in
                                let base = "Est. EPS \(String(format: "%.2f", est))"
                                guard let ago = e.epsYearAgo else { return base }
                                return base + " vs \(String(format: "%.2f", ago)) a year ago"
                            })
                    }
                    if let d = dividend {
                        if reported != nil || earnings != nil { Divider().opacity(0.5) }
                        eventRow(
                            "Next Dividend", icon: "dollarsign.circle.fill", tint: Color.up,
                            date: d.date, dateEnd: nil, status: d.status,
                            timeZone: d.marketTimezone,
                            detail: [
                                d.amountPerShare.map { "\(Fmt.currency($0, code: nativeCur)) / share" },
                                d.exDate.map { "ex-div \(Self.eventDate($0))" },
                            ].compactMap { $0 }.joined(separator: " · "))
                    }
                }
                .background(Color.gray.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    /// "Jul 30, 2026" — the date form used by the Upcoming Events cards. The value
    /// is a calendar day on an exchange, so it is not re-localized (see `MarketTime`).
    private static func eventDate(_ iso: String) -> String { MarketTime.formatted(iso) }

    /// "today" / "in 8 days" / "3 days ago", counted in the market's local time:
    /// on a device whose calendar has already rolled into tomorrow (Bangkok while
    /// New York is mid-afternoon) the device clock puts the count a day out.
    private static func relativeEventDay(_ iso: String, _ timeZone: String?) -> String? {
        guard let days = MarketTime.dayDiff(iso, timeZone: timeZone) else { return nil }
        switch days {
        case 0: return "today"
        case 1: return "tomorrow"
        case ..<0: return "\(-days) day\(days == -1 ? "" : "s") ago"
        default: return "in \(days) days"
        }
    }

    /// "EPS 2.10 vs 1.95 expected · +7.7%", or the fact of the report alone while
    /// Yahoo has yet to attach the figures.
    private static func reportedDetail(_ r: UpcomingEarnings) -> String {
        guard let actual = r.epsActual else { return "Figures not published yet" }
        var parts = [String(format: "EPS %.2f", actual)]
        if let estimate = r.epsEstimate { parts.append(String(format: "vs %.2f expected", estimate)) }
        if let surprise = r.surprisePct { parts.append(String(format: "%+.1f%%", surprise)) }
        return parts.joined(separator: " · ")
    }

    /// One event as a full-width row: label and badge, the date, and the figures
    /// pushed to the trailing edge. On a phone there is no room for all three on
    /// one line, so it stacks — the same switch the web makes at its `sm` width.
    private func eventRow(_ label: String, icon: String, tint: Color, date: String,
                          dateEnd: String?, status: String, timeZone: String?,
                          detail: String?, detailTint: Color? = nil) -> some View {
        // "reported" is the backward-looking status: a quarter already printed.
        let reported = status == "reported"
        let confirmed = status == "confirmed"
        let badgeText = reported ? "reported" : (confirmed ? "confirmed" : "est.")
        let badgeTint: Color = reported ? Theme.earnings : (confirmed ? Color.up : .orange)

        let heading = HStack(spacing: 6) {
            Image(systemName: icon).foregroundStyle(tint).font(.system(size: 14))
            Text(label).font(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
            Text(badgeText)
                .font(.system(size: 9, weight: .bold)).textCase(.uppercase)
                .padding(.horizontal, 4).padding(.vertical, 1)
                .background(badgeTint.opacity(0.12), in: RoundedRectangle(cornerRadius: 4))
                .foregroundStyle(badgeTint)
        }
        .lineLimit(1).fixedSize(horizontal: true, vertical: false)

        let when = HStack(spacing: 4) {
            Text(Self.eventDate(date) + (dateEnd.map { " – " + Self.eventDate($0) } ?? ""))
                .font(.callout.weight(.bold))
            if let rel = Self.relativeEventDay(date, timeZone) {
                Text("· \(rel)").font(.caption).foregroundStyle(.secondary)
            }
        }
        .lineLimit(1).minimumScaleFactor(0.8)

        let figures = Group {
            if let detail, !detail.isEmpty {
                Text(detail).font(.caption2).foregroundStyle(detailTint ?? .secondary)
                    .lineLimit(1).minimumScaleFactor(0.7)
            }
        }

        return Group {
            if hSizeClass == .regular {
                HStack(spacing: 10) {
                    heading
                    when
                    Spacer(minLength: 12)
                    figures
                }
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) { heading; Spacer(minLength: 0) }
                    when
                    figures
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12).padding(.vertical, 8)
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
            // Column count follows how many models actually returned a value, so
            // a third card never lands alone on a half-empty row.
            let present = [iv.models?.dcf?.intrinsicValue,
                           iv.models?.graham?.intrinsicValue,
                           iv.models?.epv?.intrinsicValue].compactMap { $0 }.count
            let cols = hSizeClass == .regular ? min(max(present, 1), 3) : 1
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: cols), spacing: 12) {
                if let dcf = iv.models?.dcf?.intrinsicValue {
                    ivCard("DCF Intrinsic Value", dcf, upside: upside(dcf, iv.currentPrice), range: iv.models?.dcf?.mc, tint: .green, icon: "chart.line.uptrend.xyaxis")
                }
                if let g = iv.models?.graham?.intrinsicValue {
                    ivCard("Graham Intrinsic Value", g, upside: upside(g, iv.currentPrice), range: iv.models?.graham?.mc, tint: .orange, icon: "scalemass")
                }
                // Earnings Power Value: the business valued with no growth at
                // all. Shown beside the others as a floor, not blended in.
                if let epv = iv.models?.epv?.intrinsicValue {
                    ivCard("Earnings Power (no growth)", epv, upside: upside(epv, iv.currentPrice), range: nil, tint: .cyan, icon: "anchor")
                }
            }
        }
    }

    /// The headline three. Everything that used to sit here as well — P/E,
    /// dividend yield, beta — now reads in Key Metrics below, beside the figures
    /// it should be compared against.
    @ViewBuilder private var marketStatsSection: some View {
        let cols = hSizeClass == .regular ? 3 : 1
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: cols), spacing: 12) {
            statCard("Market Cap", Fmt.compact(f?.marketCap ?? 0, code: nativeCur), icon: "globe", iconTint: .indigo)
            fiftyTwoWeekCard
            if let e = f?.expenseRatio {
                statCard("Expense Ratio", Fmt.percent(e), icon: "receipt", iconTint: .orange)
            } else {
                // A fund has no dividend yield of its own worth leading with; a
                // company does, and it is the third thing a reader looks for
                // after size and range.
                statCard("Dividend Yield", Fmt.percent(f?.dividendYield), icon: "dollarsign", iconTint: .orange)
            }
        }
    }

    /// Where the price sits inside its own 52-week range.
    ///
    /// Replaces the separate "52W High" and "52W Low" cards: the two numbers
    /// only ever meant anything relative to today's price, and side by side they
    /// made the reader do the arithmetic.
    @ViewBuilder private var fiftyTwoWeekCard: some View {
        let low = f?.low52
        let high = f?.high52
        let usable = (low != nil && high != nil && high! > low!)
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "arrow.left.and.right").foregroundStyle(.blue).font(.system(size: 16))
                Text("52-Week Range").font(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
            }
            if usable, let low, let high {
                HStack {
                    Text(Fmt.currency(low, code: nativeCur)).font(.callout.weight(.bold))
                    Spacer()
                    Text(Fmt.currency(high, code: nativeCur)).font(.callout.weight(.bold))
                }
                .lineLimit(1).minimumScaleFactor(0.7)
                GeometryReader { geo in
                    // Clamped: an intraday print can sit a hair outside a range
                    // Yahoo has yet to update, and a marker off the end of its
                    // own track reads as a bug.
                    let t = min(max(((f?.price ?? low) - low) / (high - low), 0), 1)
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(LinearGradient(colors: [Color.down.opacity(0.4), .orange.opacity(0.4), Color.up.opacity(0.5)],
                                                 startPoint: .leading, endPoint: .trailing))
                            .frame(height: 4)
                        Capsule().fill(Color.primary)
                            .frame(width: 3, height: 12)
                            .offset(x: max(0, min(geo.size.width - 3, geo.size.width * t - 1.5)))
                    }
                    .frame(height: 12)
                }
                .frame(height: 12)
            } else {
                Text("-").font(.title3.weight(.bold))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading).padding(16)
        .background(Color.gray.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder private var businessSummarySection: some View {
        if let summary = f?.summary, !summary.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Label("Business Summary", systemImage: "building.2").font(.headline)
                // Clamped by default: these run to a dozen lines and pushed
                // everything measurable off the screen.
                Text(summary).font(.subheadline).foregroundStyle(.secondary)
                    .lineSpacing(4)
                    .lineLimit(summaryExpanded ? nil : 4)
                // Only offered when there is something behind the clamp — a
                // toggle that does nothing is worse than no toggle.
                if summary.count > 320 {
                    Button(summaryExpanded ? "Show less" : "Read more") {
                        withAnimation(.easeInOut(duration: 0.2)) { summaryExpanded.toggle() }
                    }
                    .buttonStyle(.plain)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.indigo)
                }
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
                         sub: pos.unrealizedGainPct == .infinity ? "∞" : Fmt.percent(pos.unrealizedGainPct, includeSign: true),
                         icon: "bolt.heart",
                         iconTint: (urGain ?? 0) >= 0 ? .green : .red,
                         subTint: (urGain ?? 0) >= 0 ? .green : .red,
                         bgTint: ((urGain ?? 0) >= 0 ? Color.green : Color.red).opacity(0.1))
                
                let tGain = pos.currencyValue("Total Gain", currency: cur)
                statCard("Total Return", Fmt.currency(tGain, code: cur),
                         sub: pos.totalReturnPct == .infinity ? "∞" : Fmt.percent(pos.totalReturnPct, includeSign: true),
                         icon: "chart.line.uptrend.xyaxis",
                         iconTint: (tGain ?? 0) >= 0 ? .green : .red,
                         subTint: (tGain ?? 0) >= 0 ? .green : .red,
                         bgTint: ((tGain ?? 0) >= 0 ? Color.green : Color.red).opacity(0.1))
                
                statCard("IRR %", pos.irrPct == .infinity ? "∞" : Fmt.percent(pos.irrPct, includeSign: true),
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
                if let u = upside { Text(Fmt.percent(u, includeSign: true)).font(.caption2.weight(.bold)).foregroundStyle(Fmt.tint(for: u)) }
            }
            if let r = range, let bear = r.bear, let bull = r.bull {
                Text("Range: \(Fmt.currency(bear, code: nativeCur)) – \(Fmt.currency(bull, code: nativeCur))")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading).padding(16)
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
            hidePrice: true,
            exchange: viewModel.fundamentals?.exchange
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
        VStack(alignment: .leading, spacing: 16) {
            // The statement picker and the period switch stay put through
            // loading and empty states: "quarterly has nothing for this
            // company" is only actionable if Annual is still one tap away.
            HStack(alignment: .center, spacing: 12) {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        let tabs = [("income", "Income", "receipt"), ("balance", "Balance", "scalemass"), ("cash", "Cash Flow", "wallet.pass"), ("equity", "Equity", "person.2")]
                        ForEach(tabs, id: \.0) { t in
                            Button { finType = t.0; chartSlots = []; showAllMetrics = false } label: {
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
                Picker("", selection: Binding(
                    get: { viewModel.financialsPeriod },
                    set: { p in
                        chartSlots = []
                        showAllMetrics = false
                        chartRange = nil
                        Task { await viewModel.loadFinancials(period: p) }
                    }
                )) {
                    ForEach(StatementPeriod.allCases) { p in Text(p.title).tag(p) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 190)
            }

            if viewModel.isLoadingFinancials {
                ProgressView().frame(maxWidth: .infinity).padding(40)
            } else if let stmt = viewModel.financials.flatMap({ statement(for: finType, $0) }), !stmt.index.isEmpty {
                financialsBody(stmt)
            } else {
                VStack(spacing: 10) {
                    ContentUnavailableView(
                        "No \(viewModel.financialsPeriod.title.lowercased()) data for this statement",
                        systemImage: "doc"
                    )
                    if viewModel.financialsPeriod == .quarterly {
                        Button("Show annual instead") {
                            Task { await viewModel.loadFinancials(period: .annual) }
                        }
                        .buttonStyle(.plain)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(Color.indigo)
                    }
                }
                .frame(height: 200)
            }
        }
    }

    /// The ranked rows of one statement, the chart built from the picked ones,
    /// and the full table underneath.
    @ViewBuilder private func financialsBody(_ stmt: FinancialStatement) -> some View {
        let period = viewModel.financialsPeriod
        let ranked = rankedRows(stmt)
        let chartable = ranked.filter(\.isChartable)
        let slots = effectiveSlots(chartable)
        let colors = StatementChartConfig.colors(colorScheme)

        // Newest-first from the API; charts read left-to-right in time.
        let range = chartRange ?? period.defaultRange
        let order = Array(stmt.columns.enumerated()).reversed().suffix(range.periods(period))
        let periods = order.map(\.element)
        let series: [StatementSeries] = slots.enumerated().compactMap { slot, label in
            guard let label, let row = chartable.first(where: { $0.label == label }) else { return nil }
            return StatementSeries(
                slot: slot,
                label: label,
                color: colors[slot % colors.count],
                values: order.map { row.values.indices.contains($0.offset) ? row.values[$0.offset] : nil }
            )
        }

        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center) {
                Text("\(period.title) trend")
                    .font(.caption.weight(.bold)).textCase(.uppercase).foregroundStyle(.secondary)
                Text("\(periods.count) \(period == .quarterly ? "quarters" : "years")")
                    .font(.caption2).foregroundStyle(.tertiary)
                Spacer()
                Picker("", selection: Binding(
                    get: { range }, set: { chartRange = $0 }
                )) {
                    ForEach(StatementRange.allCases) { r in Text(r.rawValue).tag(r) }
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 170)
            }

            metricChips(chartable, slots: slots, colors: colors)

            if series.isEmpty {
                Text("Pick a line item above to chart it.")
                    .font(.callout).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 160)
            } else {
                // One y-axis per chart, always: series whose magnitudes are too
                // far apart to share a scale get their own chart.
                ForEach(Array(groupBySharedScale(series, maxAbs: { $0.maxAbs }).enumerated()), id: \.offset) { _, group in
                    StatementChartView(periods: periods, series: group, periodType: period)
                }
                if let primary = series.first {
                    StatementChangeStrip(series: primary, periods: periods, periodType: period)
                }
            }
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))

        Text(period == .quarterly
             ? "Quarterly figures are built from the company\u{2019}s own 10-Q filings, differenced out of the year-to-date numbers where that is all it tags. Tap any row to chart it."
             : "Annual statements are extended with SEC-filed history where the company files one. Tap any row to chart it.")
            .font(.caption2).foregroundStyle(.tertiary)

        statementTable(stmt, slots: slots, colors: colors)
    }

    /// Line items in importance order, the way the web app ranks them.
    private func rankedRows(_ s: FinancialStatement) -> [StatementRow] {
        let ranking = StockDetailView.rankingConfig[finType] ?? []
        return s.index.enumerated()
            .map { StatementRow(label: $0.element, values: $0.offset < s.data.count ? s.data[$0.offset] : []) }
            .enumerated()
            .sorted { a, b in
                let ia = ranking.firstIndex(of: a.element.label)
                let ib = ranking.firstIndex(of: b.element.label)
                switch (ia, ib) {
                case let (x?, y?): return x < y
                case (_?, nil): return true
                case (nil, _?): return false
                // Stable: equal keys keep the order the statement arrived in.
                default: return a.offset < b.offset
                }
            }
            .map(\.element)
    }

    /// The user's picks, or the statement's opening set when they have not
    /// picked yet.
    private func effectiveSlots(_ chartable: [StatementRow]) -> [String?] {
        if !chartSlots.isEmpty { return chartSlots }
        var defaults = pickDefaultMetrics(
            StatementChartConfig.defaultMetrics[finType] ?? [], chartable
        ).map { Optional($0) }
        if defaults.isEmpty, let first = chartable.first { defaults = [first.label] }
        return defaults
    }

    private func toggleMetric(_ label: String, _ slots: [String?]) {
        chartSlots = toggleStatementSlot(slots, label)
    }

    @ViewBuilder private func metricChips(
        _ chartable: [StatementRow],
        slots: [String?],
        colors: [Color]
    ) -> some View {
        let ranking = StockDetailView.rankingConfig[finType] ?? []
        let key = chartable.filter { ranking.contains($0.label) }
        let shown = (showAllMetrics || key.isEmpty) ? chartable : key
        let full = slots.compactMap { $0 }.count >= StatementChartConfig.maxSeries && !slots.contains(where: { $0 == nil })

        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(shown) { row in
                    let slot = slots.firstIndex(of: row.label)
                    let selected = slot != nil
                    Button { toggleMetric(row.label, slots) } label: {
                        HStack(spacing: 5) {
                            Circle()
                                .fill(selected ? colors[(slot ?? 0) % colors.count] : Color.secondary.opacity(0.3))
                                .frame(width: 7, height: 7)
                            Text(row.label).font(.caption)
                        }
                        .foregroundStyle(selected ? Color.primary : .secondary)
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(selected ? AnyShapeStyle(.background) : AnyShapeStyle(Color.clear), in: Capsule())
                        .overlay(
                            Capsule().strokeBorder(
                                selected ? AnyShapeStyle(.quaternary) : AnyShapeStyle(Color.clear),
                                lineWidth: 1
                            )
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(!selected && full)
                    .opacity(!selected && full ? 0.4 : 1)
                    .help(!selected && full ? "Deselect one first — \(StatementChartConfig.maxSeries) is the limit" : row.label)
                }
                if !key.isEmpty, chartable.count > key.count {
                    Button(showAllMetrics ? "Show key items" : "+\(chartable.count - key.count) more") {
                        showAllMetrics.toggle()
                    }
                    .buttonStyle(.plain).font(.caption.weight(.bold)).foregroundStyle(Color.indigo)
                }
            }
        }
    }

    /// Line-item importance per statement — the same order the web app ranks
    /// by, so the two clients open on the same rows.
    static let rankingConfig: [String: [String]] = [
        "income": [
            "Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Expense",
            "Operating Income", "EBITDA", "EBIT", "Pretax Income", "Tax Provision",
            "Net Income Common Stockholders", "Net Income", "Normalized Income",
            "Basic EPS", "Diluted EPS",
        ],
        "balance": [
            "Total Assets", "Current Assets", "Cash And Cash Equivalents", "Receivables",
            "Inventory", "Total Liabilities Net Minority Interest", "Current Liabilities",
            "Total Debt", "Net Debt", "Total Equity Gross Minority Interest",
            "Stockholders Equity", "Common Stock Equity", "Retained Earnings",
            "Working Capital", "Invested Capital", "Tangible Book Value",
        ],
        "cash": [
            "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
            "Capital Expenditure", "Free Cash Flow", "End Cash Position", "Net Income",
        ],
        "equity": [
            "Total Equity Gross Minority Interest", "Stockholders Equity",
            "Common Stock Equity", "Retained Earnings", "Capital Stock", "Common Stock",
        ],
    ]

    private func statement(for type: String, _ f: FinancialsResponse) -> FinancialStatement? {
        switch type {
        case "balance": return f.balanceSheet; case "cash": return f.cashflow
        case "equity": return f.shareholdersEquity; default: return f.financials
        }
    }

    /// Annual statements carry ~19 filed years, so every period is shown and the
    /// table scrolls. The year alone would not identify an annual column: filed
    /// period ends are the company's own 52/53-week dates, and two of them can
    /// fall in one calendar year (Advance Auto Parts closed fiscal years on
    /// 2022-01-01 and 2022-12-31), so the end date sits under the year. A
    /// quarter is headed by its month for the same reason.
    private func statementTable(_ s: FinancialStatement, slots: [String?], colors: [Color]) -> some View {
        let rows = rankedRows(s)
        return ScrollView(.horizontal, showsIndicators: true) {
            Grid(alignment: .trailing, horizontalSpacing: 24, verticalSpacing: 12) {
                GridRow {
                    Text("Metric").gridColumnAlignment(.leading)
                    Text("Trend").gridColumnAlignment(.center)
                    ForEach(Array(s.columns.enumerated()), id: \.offset) { _, c in
                        VStack(alignment: .trailing, spacing: 1) {
                            Text(statementPeriodLabel(c, viewModel.financialsPeriod))
                            Text(MarketTime.shortDay(c))
                                .font(.system(size: 9, weight: .regular))
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
                .font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)

                Divider()

                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    let slot = slots.firstIndex(of: row.label)
                    GridRow {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(slot.map { colors[$0 % colors.count] } ?? .clear)
                                .frame(width: 7, height: 7)
                            Text(row.label).font(.subheadline.weight(.semibold)).lineLimit(1)
                        }
                        .gridColumnAlignment(.leading)
                        sparkline(row.values.compactMap { $0 })
                        ForEach(Array(row.values.enumerated()), id: \.offset) { _, v in
                            Text(v.map { compact($0) } ?? "—")
                                .font(.subheadline).monospacedDigit()
                                .foregroundStyle((v ?? 0) < 0 ? .red : .primary)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        guard row.isChartable else { return }
                        toggleMetric(row.label, slots)
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
        let history = viewModel.ratios?.historical ?? []
        let period = viewModel.ratiosPeriod

        VStack(alignment: .leading, spacing: 24) {
            // Chrome first, so the switch survives an empty quarterly answer.
            HStack(alignment: .center) {
                Text(period == .quarterly
                     ? "Measured on the trailing twelve months at each quarter end — the same ratios the annual view reports, sampled four times as often."
                     : "Measured on each filed fiscal year.")
                    .font(.caption2).foregroundStyle(.tertiary).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 12)
                Picker("", selection: Binding(
                    get: { period },
                    set: { p in Task { await viewModel.loadRatios(period: p) } }
                )) {
                    ForEach(StatementPeriod.allCases) { p in Text(p.title).tag(p) }
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 190)
            }

            if viewModel.isLoadingRatios {
                ProgressView().frame(maxWidth: .infinity).padding(40)
            } else if history.isEmpty && viewModel.trackRecord == nil {
                ContentUnavailableView("No ratio data", systemImage: "chart.line.uptrend.xyaxis").frame(height: 200)
            } else {
                if let record = viewModel.trackRecord { trackRecordPanel(record) }
                if !history.isEmpty {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 16)], spacing: 16) {
                        ratioChart("Return on Equity", history, "Return on Equity (ROE) (%)", Color(red: 16/255, green: 185/255, blue: 129/255), isPercent: true, periodType: period)
                        ratioChart("Gross Margin", history, "Gross Profit Margin (%)", Color(red: 6/255, green: 182/255, blue: 212/255), isPercent: true, periodType: period)
                        ratioChart("Net Margin", history, "Net Profit Margin (%)", Color(red: 139/255, green: 92/255, blue: 246/255), isPercent: true, periodType: period)
                        ratioChart("Asset Turnover", history, "Asset Turnover", Color(red: 245/255, green: 158/255, blue: 11/255), isPercent: false, periodType: period)
                        ratioChart("Return on Invested Capital", history, "Return on Invested Capital (ROIC) (%)", Color(red: 236/255, green: 72/255, blue: 153/255), isPercent: true, periodType: period)
                        ratioChart("Free Cash Flow Margin", history, "Free Cash Flow Margin (%)", Color(red: 20/255, green: 184/255, blue: 166/255), isPercent: true, periodType: period)
                        // A falling line is the owner's slice growing. Over
                        // nineteen years it is the clearest picture of whether
                        // management returned capital or issued it away.
                        ratioChart("Diluted Shares Outstanding", history, "Diluted Shares Outstanding", Color(red: 100/255, green: 116/255, blue: 139/255), isPercent: false, isCount: true, periodType: period)
                    }
                }
            }
        }
    }

    /// The measured quality record — the metrics the Buffett ranking scores on.
    ///
    /// Nothing here is coloured good or bad: a median ROE of 13% is excellent
    /// for a bank and mediocre for a software company. The only absolute
    /// thresholds this system holds are the hard gates, and those appear as the
    /// exclusion reasons they are.
    @ViewBuilder private func trackRecordPanel(_ record: TrackRecord) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Label("Track Record", systemImage: "checkmark.shield")
                        .font(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                    Text(trackRecordSpan(record))
                        .font(.caption2).foregroundStyle(.tertiary)
                }
                Spacer()
                if let rank = record.rank?.rank {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("#\(rank)").font(.title2.weight(.bold)).monospacedDigit()
                        Text("Buffett rank").font(.system(size: 9)).foregroundStyle(.tertiary).textCase(.uppercase)
                    }
                }
            }

            if !record.gateFailures.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text("Not eligible for the ranking: "
                         + record.gateFailures.map { $0.replacingOccurrences(of: "_", with: " ") }
                            .joined(separator: ", "))
                        .font(.caption).foregroundStyle(.orange)
                }
                .padding(10)
                .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
            }

            if let bands = record.valuationBands, !bands.isEmpty {
                valuationBands(bands)
            }

            if let stress = record.stress, stress.contains(where: { $0.covered }) {
                stressResponse(stress)
            }

            if let revisions = record.revisions, revisions.count > 0 {
                revisionHistory(revisions)
            }

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 260), spacing: 16)], alignment: .leading, spacing: 16) {
                ForEach(record.groups) { group in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(group.title)
                                .font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                            Spacer()
                            if let score = record.rank?.pillars?[group.key] ?? nil {
                                Text(String(format: "%.0f", score))
                                    .font(.caption.weight(.semibold)).monospacedDigit()
                                    .foregroundStyle(.secondary)
                            }
                        }
                        ForEach(group.items) { item in
                            HStack(alignment: .firstTextBaseline) {
                                Text(item.label).font(.subheadline).foregroundStyle(.secondary)
                                Spacer(minLength: 12)
                                Text(item.display ?? (item.note != nil ? "n/a" : "—"))
                                    .font(.subheadline.weight(.medium)).monospacedDigit()
                                    .foregroundStyle(item.display == nil ? .tertiary : .primary)
                                    .help(item.note ?? "")
                            }
                        }
                    }
                    .padding(14)
                    .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
                }
            }
        }
        .padding(20)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.quaternary, lineWidth: 1))
    }

    /// Today's multiples against the company's own fifteen-year record.
    ///
    /// The bar is presentation only — every number a reader acts on is printed
    /// beside it. Cross-sectional cheapness is what the ranking scores; this
    /// answers the different question of whether *this* business is dear
    /// relative to how it has been priced before.
    @ViewBuilder private func valuationBands(_ bands: [TrackRecordBand]) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Against its own history")
                .font(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
            ForEach(bands) { band in
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(band.label).font(.subheadline).foregroundStyle(.secondary)
                        Spacer()
                        Text(band.display).font(.subheadline.weight(.semibold)).monospacedDigit()
                        Text("vs \(band.medianDisplay) median")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                    GeometryReader { geo in
                        let span = max(band.high - band.low, 1e-9)
                        let x = { (v: Double) in
                            CGFloat(min(max((v - band.low) / span, 0), 1)) * geo.size.width
                        }
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.secondary.opacity(0.15)).frame(height: 8)
                            Capsule().fill(Color.indigo.opacity(0.25))
                                .frame(width: max(2, x(band.p75) - x(band.p25)), height: 8)
                                .offset(x: x(band.p25))
                            Rectangle().fill(Color.secondary.opacity(0.6))
                                .frame(width: 1, height: 8)
                                .offset(x: x(band.median))
                            RoundedRectangle(cornerRadius: 1).fill(Color.indigo)
                                .frame(width: 3, height: 12)
                                .offset(x: max(0, x(band.current) - 1.5))
                        }
                        .frame(height: 12)
                    }
                    .frame(height: 12)
                    Text("\(band.summary) (\(band.observations) years)")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
        }
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    /// What each downturn in the filed history did to the business.
    ///
    /// Evidence about one company and never a score: only the companies filing
    /// in 2008 have a reading for it, and the ones that did not survive are not
    /// in the fact store to be compared against.
    @ViewBuilder private func stressResponse(_ windows: [TrackRecordStress]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("In a downturn")
                .font(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
            ForEach(windows) { window in
                // Three metrics beside the window label want ~460pt. An iPhone
                // has half that, and a row with nowhere to give does not
                // truncate — SwiftUI compresses every Text to one character per
                // line. So measure, and stack the metrics under the label when
                // they cannot sit beside it.
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .firstTextBaseline, spacing: 12) {
                        stressLabel(window).frame(width: 150, alignment: .leading)
                        HStack(spacing: 16) { stressMetrics(window) }
                        Spacer(minLength: 0)
                    }
                    VStack(alignment: .leading, spacing: 3) {
                        stressLabel(window)
                        VStack(alignment: .leading, spacing: 3) { stressMetrics(window) }
                            .padding(.leading, 10)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func stressLabel(_ window: TrackRecordStress) -> some View {
        Text(window.label).font(.subheadline).foregroundStyle(.secondary)
    }

    @ViewBuilder private func stressMetrics(_ window: TrackRecordStress) -> some View {
        if window.covered {
            ForEach(window.items) { item in
                // One Text, not an HStack of three: a metric too wide for the
                // line then wraps at its spaces instead of being squeezed a
                // character at a time.
                (Text(item.label + " ").font(.subheadline).foregroundStyle(.secondary)
                 + Text(item.display).font(.subheadline.weight(.medium)).monospacedDigit()
                    .foregroundStyle(item.changePct < 0 ? Color.red : Color.green)
                 + Text(" (\(item.recoveryDisplay ?? "no fall"))")
                    .font(.caption2).foregroundStyle(.tertiary))
                .fixedSize(horizontal: false, vertical: true)
            }
        } else {
            // Not the same claim as "did not fall".
            Text("not filing then").font(.subheadline.italic()).foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Numbers the company changed after first reporting them.
    ///
    /// Presented as history, not as an accusation: most revisions are an
    /// accounting standard adopted retrospectively or a discontinued operation
    /// reclassifying years of revenue at once.
    @ViewBuilder private func revisionHistory(_ revisions: TrackRecordRevisions) -> some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 8) {
                Text("Later filings changed these. Usually a retrospectively adopted accounting standard or a reclassification — the size and the gap are what matter.")
                    .font(.caption2).foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                ForEach(revisions.items) { item in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(item.label).font(.subheadline).foregroundStyle(.secondary)
                        Text(String(item.periodEnd.prefix(4)))
                            .font(.caption).monospacedDigit().foregroundStyle(.tertiary)
                        Spacer(minLength: 8)
                        Text(item.display).font(.subheadline).monospacedDigit()
                        Text(item.changeDisplay)
                            .font(.subheadline.weight(.medium)).monospacedDigit()
                            .foregroundStyle(item.changePct < 0 ? .red : .green)
                            .frame(width: 72, alignment: .trailing)
                        Text("\(item.firstFiled.prefix(4)) → \(item.restatedFiled.prefix(4))")
                            .font(.caption2).monospacedDigit().foregroundStyle(.tertiary)
                    }
                }
                if revisions.count > revisions.items.count {
                    Text("Showing the \(revisions.items.count) largest of \(revisions.count).")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
            .padding(.top, 8)
        } label: {
            Label(
                "\(revisions.count) figure\(revisions.count == 1 ? "" : "s") revised after first reporting",
                systemImage: "clock.arrow.circlepath"
            )
            .font(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
        }
        .padding(14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func trackRecordSpan(_ record: TrackRecord) -> String {
        var parts = ["\(record.periodCount) years of SEC filings"]
        if let first = record.firstPeriod, let last = record.latestPeriod {
            parts[0] += " (\(first.prefix(4))–\(last.prefix(4)))"
        }
        parts.append("measured over the last \(record.windowYears)")
        if record.model != "generic" { parts.append("\(record.model) model") }
        return parts.joined(separator: " · ")
    }

    /// The ratio history runs as far back as the filings do, so the x-axis plots
    /// the period end itself rather than a year string: two fiscal years can end
    /// in the same calendar year, and as categories they would collapse onto one
    /// point. Dates are read at UTC midnight so a period end never slides a day.
    private func ratioChart(
        _ title: String,
        _ data: [[String: JSONValue]],
        _ key: String,
        _ color: Color,
        isPercent: Bool,
        isCount: Bool = false,
        periodType: StatementPeriod = .annual
    ) -> some View {
        // Parsed once rather than inside the chart body: the hover tooltip needs
        // the same x-values the marks are drawn at to find the nearest period.
        let points: [(period: Date, iso: String, value: Double)] = data.reversed().compactMap { item in
            guard let val = item[key]?.doubleValue,
                  let dateStr = item["Period"]?.stringValue,
                  let period = MarketTime.calendarDay(dateStr) else { return nil }
            return (period, dateStr, val)
        }
        // Sixty quarterly points would be a row of touching dots; the line
        // carries the shape on its own once they stop being distinguishable.
        let showPoints = points.count <= StatementChartConfig.barToLineThreshold
        return card(title) {
            Chart {
                ForEach(Array(points.enumerated()), id: \.offset) { _, p in
                    LineMark(x: .value("Period", p.period), y: .value(title, p.value))
                        .foregroundStyle(color).interpolationMethod(.monotone)
                    AreaMark(x: .value("Period", p.period), y: .value(title, p.value))
                        .foregroundStyle(LinearGradient(colors: [color.opacity(0.3), color.opacity(0.0)], startPoint: .top, endPoint: .bottom))
                        .interpolationMethod(.monotone)
                    if showPoints {
                        PointMark(x: .value("Period", p.period), y: .value(title, p.value))
                            .foregroundStyle(color)
                    }
                }
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 5)) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let date = value.as(Date.self) {
                            // Four quarters a year would all read "2026".
                            Text(periodType == .quarterly
                                 ? MarketTime.monthYear(date)
                                 : MarketTime.year(date))
                        }
                    }
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            // A raw share count is unreadable on an axis;
                            // 15.00B is the same number said usefully.
                            Text(ratioValueLabel(v, isPercent: isPercent, isCount: isCount))
                        }
                    }
                }
            }
            .frame(height: 200)
            // Five axis ticks across nineteen filed years leave most periods
            // unlabelled; the tooltip is how a particular year is read off.
            .chartHoverTooltip(points.map(\.period)) { i in
                guard points.indices.contains(i) else { return nil }
                let p = points[i]
                return ChartTooltipContent(
                    title: MarketTime.formatted(p.iso),
                    rows: [ChartTooltipRow(
                        color: color,
                        label: title,
                        value: ratioValueLabel(p.value, isPercent: isPercent, isCount: isCount)
                    )]
                )
            }
        }
    }

    /// One ratio rendered the same way on the axis and in the tooltip.
    private func ratioValueLabel(_ v: Double, isPercent: Bool, isCount: Bool) -> String {
        if isCount { return compact(v) }
        return isPercent ? Fmt.percent(v) : Fmt.number(v, fractionDigits: 2)
    }

    // MARK: - Valuation

    @ViewBuilder private var valuationTab: some View {
        VStack(spacing: 24) {
            if let iv = viewModel.intrinsic {
                valuationSummaryCards(iv)

                if let note = iv.valuationNote {
                    // A refusal is information, not a warning — tint it neutrally
                    // so it doesn't read as an alarm about the company.
                    let tint: Color = iv.isRefusal ? .secondary : .orange
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: iv.isRefusal ? "info.circle.fill" : "exclamationmark.triangle.fill")
                            .foregroundStyle(tint).font(.title3)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(valuationNoteTitle(iv)).font(.caption.weight(.bold)).foregroundStyle(tint).textCase(.uppercase)
                            Text(note).font(.subheadline.italic()).foregroundStyle(tint)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background((iv.isRefusal ? Color.secondary : Color.orange).opacity(0.1),
                                in: RoundedRectangle(cornerRadius: 12))
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
        // The backend now declines to value companies whose fundamentals can't
        // support one. Distinguish "no answer" from "an answer of zero".
        let hasValue = iv.averageIntrinsicValue != nil
        let intrinsic = valuationCard(label: iv.status == .nav ? "Net Asset Value" : "Blended Intrinsic Value",
                                      value: hasValue ? Fmt.currency(iv.averageIntrinsicValue, code: nativeCur) : "Not valued",
                                      valueColor: hasValue ? .indigo : .secondary) {
            if hasValue, let r = iv.range {
                Text("Range: \(Fmt.currency(r.bear, code: nativeCur)) - \(Fmt.currency(r.bull, code: nativeCur))")
                    .font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            if hasValue, let floor = iv.earningsPowerFloor {
                Text("No-growth floor: \(Fmt.currency(floor, code: nativeCur))")
                    .font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        let current = valuationCard(label: "Current Price",
                                    value: Fmt.currency(iv.currentPrice, code: nativeCur),
                                    valueColor: .primary) { EmptyView() }
        let safety = valuationCard(label: "Margin of Safety",
                                   value: hasValue ? Fmt.percent(mos, includeSign: true) : "—",
                                   valueColor: hasValue ? (mos >= 0 ? .green : .red) : .secondary,
                                   tint: hasValue ? (mos >= 0 ? Color.green.opacity(0.1) : Color.red.opacity(0.1)) : nil) { EmptyView() }

        if hSizeClass == .compact {
            VStack(spacing: 12) { intrinsic; current; safety }
        } else {
            HStack(spacing: 16) { intrinsic; current; safety }
        }
    }

    /// Headline for the valuation note, keyed to why the backend produced it.
    private func valuationNoteTitle(_ iv: IntrinsicValueResponse) -> String {
        switch iv.status {
        case .noModel:        return "Cannot be valued"
        case .ineligible:     return "Not eligible for valuation"
        case .clamped:        return "Output outside credible range"
        case .lowConfidence:  return "Models disagree"
        default:              return "Valuation note"
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

    // MARK: - Position & Lots Tab

    @ViewBuilder
    private var positionTab: some View {
        if let pos = viewModel.positionData, pos.hasPosition {
            VStack(alignment: .leading, spacing: 20) {
                // 1. Overview KPIs
                if let summary = pos.summary, let ret = pos.returns {
                    let unreal = ret.unrealizedGain
                    let unrealPct = ret.unrealizedGainPct
                    let totalG = ret.totalGain
                    let totalRetPct = ret.totalReturnPct

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Position Overview").font(.headline)
                            Spacer()
                            if let w = summary.portfolioWeightPct, w > 0 {
                                Text("\(Fmt.number(w, fractionDigits: 2))% of Portfolio")
                                    .font(.caption2.weight(.bold))
                                    .padding(.horizontal, 8).padding(.vertical, 4)
                                    .background(Color.indigo.opacity(0.12), in: Capsule())
                                    .foregroundStyle(Color.indigo)
                            }
                        }

                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
                            statCard("Shares Held", Fmt.number(summary.quantity, fractionDigits: 4), icon: "number", iconTint: .indigo)
                            statCard("Avg Cost", Fmt.currency(summary.avgCostPrice, currency: cur), icon: "tag", iconTint: .secondary)
                            statCard("Market Value", Fmt.currency(summary.marketValue, currency: cur), icon: "chart.pie", iconTint: .indigo)
                            statCard("Cost Basis", Fmt.currency(summary.costBasis, currency: cur), icon: "scalemass", iconTint: .secondary)
                            statCard(
                                "Unrealized G/L",
                                Fmt.currency(unreal, currency: cur),
                                sub: "\(unrealPct >= 0 ? "+" : "")\(Fmt.percent(unrealPct))",
                                icon: "chart.line.uptrend.xyaxis",
                                iconTint: unreal >= 0 ? .green : .red,
                                subTint: unrealPct >= 0 ? .green : .red,
                                bgTint: (unreal >= 0 ? Color.green : Color.red).opacity(0.08)
                            )
                            statCard(
                                "Total Return",
                                Fmt.currency(totalG, currency: cur),
                                sub: "\(totalRetPct >= 0 ? "+" : "")\(Fmt.percent(totalRetPct))",
                                icon: "arrow.up.right.circle",
                                iconTint: totalG >= 0 ? .green : .red,
                                subTint: totalRetPct >= 0 ? .green : .red,
                                bgTint: (totalG >= 0 ? Color.green : Color.red).opacity(0.08)
                            )
                        }

                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
                            statCard("IRR (Annualized)", ret.irrPct != nil ? "\(ret.irrPct! >= 0 ? "+" : "")\(Fmt.percent(ret.irrPct!))" : "—", icon: "percent", iconTint: (ret.irrPct ?? 0) >= 0 ? .green : .red)
                            statCard("Yield on Cost", ret.yieldOnCostPct != nil ? Fmt.percent(ret.yieldOnCostPct!) : "—", sub: ret.marketYieldPct != nil ? "Mkt: \(Fmt.percent(ret.marketYieldPct!))" : nil, icon: "dollarsign.circle", iconTint: .orange)
                            statCard("Lifetime Dividends", Fmt.currency(ret.lifetimeDividends, currency: cur), icon: "banknote", iconTint: .orange)
                            statCard("Realized G/L", Fmt.currency(ret.realizedGain, currency: cur), icon: "checkmark.circle", iconTint: ret.realizedGain >= 0 ? .green : .red)
                        }
                    }

                    // 2. Position Performance History Chart (Value & Return %)
                    PositionHistoryChartView(symbol: viewModel.symbol, currency: cur, viewModel: viewModel)

                    // 3. Return Attribution Breakdown
                    card("Return Attribution Breakdown") {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 10)], spacing: 10) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Capital Appreciation").font(.caption2).foregroundStyle(.secondary)
                                Text(Fmt.currency(unreal + ret.realizedGain, currency: cur)).font(.subheadline.bold())
                                Text("Unreal: \(Fmt.currency(unreal, currency: cur)) · Real: \(Fmt.currency(ret.realizedGain, currency: cur))").font(.caption2).foregroundStyle(.secondary)
                            }
                            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Dividend Income").font(.caption2).foregroundStyle(.secondary)
                                Text("+\(Fmt.currency(ret.lifetimeDividends, currency: cur))").font(.subheadline.bold()).foregroundStyle(.green)
                                Text("YoC: \(ret.yieldOnCostPct != nil ? Fmt.percent(ret.yieldOnCostPct!) : "—")").font(.caption2).foregroundStyle(.secondary)
                            }
                            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Currency (FX) Impact").font(.caption2).foregroundStyle(.secondary)
                                Text("\(ret.fxGainLoss >= 0 ? "+" : "")\(Fmt.currency(ret.fxGainLoss, currency: cur))").font(.subheadline.bold()).foregroundStyle(ret.fxGainLoss >= 0 ? .green : .red)
                                Text("\(ret.fxGainLossPct >= 0 ? "+" : "")\(Fmt.percent(ret.fxGainLossPct)) on cost").font(.caption2).foregroundStyle(.secondary)
                            }
                            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Fees & Taxes Friction").font(.caption2).foregroundStyle(.secondary)
                                Text("-\(Fmt.currency(ret.commissions + ret.withholdingTaxes, currency: cur))").font(.subheadline.bold()).foregroundStyle(.red)
                                Text("Fees: \(Fmt.currency(ret.commissions, currency: cur)) · Tax: \(Fmt.currency(ret.withholdingTaxes, currency: cur))").font(.caption2).foregroundStyle(.secondary)
                            }
                            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                        }
                    }
                }

                // 3. Open FIFO Lots
                if !pos.openLots.isEmpty {
                    card("Open FIFO Tax Lots (\(pos.openLots.count))") {
                        VStack(spacing: 8) {
                            ForEach(pos.openLots) { lot in
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        HStack(spacing: 6) {
                                            Text(lot.date).font(.subheadline.weight(.semibold))
                                            Text(lot.account).font(.caption2).foregroundStyle(.secondary)
                                            Text(lot.taxTerm == "long_term" ? "Long-Term" : "Short-Term")
                                                .font(.system(size: 9, weight: .bold))
                                                .padding(.horizontal, 6).padding(.vertical, 2)
                                                .background(lot.taxTerm == "long_term" ? Color.green.opacity(0.15) : Color.blue.opacity(0.15), in: Capsule())
                                                .foregroundStyle(lot.taxTerm == "long_term" ? Color.green : Color.blue)
                                        }
                                        Text("\(Fmt.number(lot.quantity, fractionDigits: 4)) shares @ \(Fmt.currency(lot.costPerShareLocal, currency: pos.localCurrency))")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    VStack(alignment: .trailing, spacing: 2) {
                                        Text(Fmt.currency(lot.marketValueDisplay, currency: cur)).font(.subheadline.weight(.semibold))
                                        Text("\(lot.unrealizedGainDisplay >= 0 ? "+" : "")\(Fmt.currency(lot.unrealizedGainDisplay, currency: cur)) (\(Fmt.percent(lot.unrealizedGainPct)))")
                                            .font(.caption.weight(.medium))
                                            .foregroundStyle(lot.unrealizedGainDisplay >= 0 ? .green : .red)
                                    }
                                }
                                .padding(10)
                                .background(Color.gray.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                }

                // 4. Closed Trades
                if !pos.closedTrades.isEmpty {
                    card("Closed Trades & Realized Sells (\(pos.closedTrades.count))") {
                        VStack(spacing: 8) {
                            ForEach(pos.closedTrades) { trade in
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        HStack(spacing: 6) {
                                            Text(trade.sellDate).font(.subheadline.weight(.semibold))
                                            Text(trade.account).font(.caption2).foregroundStyle(.secondary)
                                        }
                                        Text("Sold \(Fmt.number(trade.quantitySold, fractionDigits: 4)) shares @ \(Fmt.currency(trade.salePrice, currency: pos.localCurrency))")
                                            .font(.caption).foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    VStack(alignment: .trailing, spacing: 2) {
                                        Text("Proceeds: \(Fmt.currency(trade.proceedsDisplay, currency: cur))").font(.caption.weight(.medium))
                                        Text("Gain: \(trade.realizedGainDisplay >= 0 ? "+" : "")\(Fmt.currency(trade.realizedGainDisplay, currency: cur))")
                                            .font(.subheadline.weight(.bold))
                                            .foregroundStyle(trade.realizedGainDisplay >= 0 ? .green : .red)
                                    }
                                }
                                .padding(10)
                                .background(Color.gray.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                }
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "briefcase").font(.system(size: 36)).foregroundStyle(.secondary)
                Text("No Position in \(viewModel.symbol)").font(.headline)
                Text("You currently have no recorded transactions for this symbol.")
                    .font(.subheadline).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity).padding(40)
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
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading).padding(16)
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
