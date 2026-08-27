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
    // Two labels on this screen are built by concatenating `Text` values, which
    // only compose while they stay `Text` — so they resolve their `AppFont` by
    // hand instead of going through the `.appFont(_:)` view modifier.
    @Environment(\.appFontScale) private var fontScale
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
    @State private var summaryExpanded = false
    @State private var ratiosCategory = "All"
    @State private var ratiosRange: StatementRange = .fiveYears

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
        #if os(iOS)
        // The whole page — header, tab strip and content — shares one vertical
        // ScrollView, and a ScrollView takes its width from its widest
        // descendant and re-proposes that width to everything inside it. So a
        // single card that *demands* more room than the screen has does not
        // merely overflow itself: it shifts the entire page off the right edge,
        // and takes every horizontal scroller's trailing end somewhere no
        // gesture can reach.
        //
        // The cap comes from a GeometryReader placed *outside* the ScrollView,
        // so it reports the room the page actually has and can never be
        // inflated by the content it is capping — unlike a probe inside, which
        // would measure the overflow it was meant to prevent and latch. A card
        // that still over-reaches now clips at the screen edge on its own
        // instead of moving everything else.
        GeometryReader { geo in
            ScrollView {
                VStack(spacing: 0) {
                    header
                    tabBar
                    Divider()
                    // Above the tab content, not inside one tab: a broken price
                    // history affects the chart, the returns and every figure
                    // derived from them.
                    if let flag = viewModel.dataQuality {
                        DataQualityBanner(flag: flag)
                            .padding(.horizontal, 20)
                            .padding(.top, 16)
                    }
                    Group {
                        switch tab {
                        case .overview: overviewTab
                        case .position: positionTab
                        case .chart: chartTab
                        case .analysis: analysisTab
                        case .financials: financialsTab
                        case .ratios: ratiosTab
                        case .valuation: StockValuationTabView(viewModel: viewModel)
                        case .holdings: holdingsTab
                        case .news: newsTab
                        }
                    }
                    .padding(20)
                }
                .frame(maxWidth: geo.size.width)
            }
        }
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
        #else
        VStack(spacing: 0) {
            header
            tabBar
            Divider()
            ScrollView {
                if let flag = viewModel.dataQuality {
                    DataQualityBanner(flag: flag)
                        .padding(.horizontal, 20)
                        .padding(.top, 16)
                }
                Group {
                    switch tab {
                    case .overview: overviewTab
                    case .position: positionTab
                    case .chart: chartTab
                    case .analysis: analysisTab
                    case .financials: financialsTab
                    case .ratios: ratiosTab
                    case .valuation: StockValuationTabView(viewModel: viewModel)
                    case .holdings: holdingsTab
                    case .news: newsTab
                    }
                }
                .padding(20)
            }
        }
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
        #endif
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
        VStack(alignment: .leading, spacing: 12) {
            Button {
                appState.closeStock()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.left")
                        .appFont(.system(size: 13, weight: .bold))
                    if let previous = appState.stockHistory.last {
                        Text("Back to \(previous)")
                            .appFont(.system(size: 13, weight: .semibold))
                    } else {
                        Text("Back")
                            .appFont(.system(size: 13, weight: .semibold))
                    }
                }
                .foregroundStyle(.primary)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.cardBorder.opacity(0.2), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)

            HStack(alignment: .center, spacing: 16) {
                ZStack {
                    LinearGradient(colors: [Color.brandIndigo, Color.brandPurple], startPoint: .topLeading, endPoint: .bottomTrailing)
                    StockIcon(symbol: viewModel.symbol, size: 48)
                        .padding(8)
                        .background(.white)
                }
                .frame(width: 64, height: 64)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(f?.shortName ?? viewModel.symbol)
                            .appFont(.system(size: 32, weight: .black, design: .default))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                        
                        Text(viewModel.symbol)
                            .appFont(.system(size: 13, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.cardBorder.opacity(0.25), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                            .foregroundStyle(.secondary)
                    }
                    if f?.sector != nil || f?.industry != nil {
                        HStack(spacing: 6) {
                            if let s = f?.sector { Text(s).appFont(.subheadline.weight(.semibold)).foregroundStyle(Color.brandIndigo).lineLimit(1) }
                            if f?.sector != nil && f?.industry != nil { Text("•").foregroundStyle(.secondary) }
                            if let i = f?.industry { Text(i).appFont(.subheadline).foregroundStyle(.secondary).lineLimit(1) }
                        }
                    }
                }
                
                Spacer(minLength: 16)
                
                VStack(alignment: .trailing, spacing: 4) {
                    if viewModel.isLoading { ProgressView().controlSize(.small) }
                    if let p = f?.price {
                        Text(Fmt.currency(p, code: nativeCur))
                            .appFont(.system(size: 32, weight: .black, design: .default))
                            .foregroundStyle(Color.brandIndigo)
                    }
                }
            }
        }
    }

    private var compactHeader: some View {
        VStack(alignment: .leading, spacing: 14) {
            Button {
                appState.closeStock()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left")
                        .appFont(.system(size: 12, weight: .bold))
                    if let previous = appState.stockHistory.last {
                        Text("Back to \(previous)")
                            .appFont(.system(size: 12, weight: .semibold))
                    } else {
                        Text("Back")
                            .appFont(.system(size: 12, weight: .semibold))
                    }
                }
                .foregroundStyle(.primary)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(Color.cardBorder.opacity(0.2), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)

            HStack(alignment: .top, spacing: 12) {
                ZStack {
                    LinearGradient(colors: [Color.brandIndigo, Color.brandPurple], startPoint: .topLeading, endPoint: .bottomTrailing)
                    StockIcon(symbol: viewModel.symbol, size: 45)
                        .padding(6)
                        .background(.white)
                }
                .frame(width: 56, height: 56)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(f?.shortName ?? viewModel.symbol)
                            .appFont(.system(size: 25, weight: .black, design: .default))
                            .lineLimit(2)
                            .minimumScaleFactor(0.8)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    HStack(spacing: 6) {
                        Text(viewModel.symbol)
                            .appFont(.system(size: 13, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.cardBorder.opacity(0.25), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                            .foregroundStyle(.secondary)
                        if viewModel.isLoading { ProgressView().controlSize(.small) }
                    }
                }
            }
            
            HStack(alignment: .bottom) {
                if let p = f?.price {
                    Text(Fmt.currency(p, code: nativeCur))
                        .appFont(.system(size: 41, weight: .black, design: .default))
                        .foregroundStyle(Color.brandIndigo)
                        .minimumScaleFactor(0.8)
                        .lineLimit(1)
                }
                Spacer()
                if f?.sector != nil || f?.industry != nil {
                    VStack(alignment: .trailing, spacing: 2) {
                        if let s = f?.sector { Text(s).appFont(.caption.weight(.semibold)).foregroundStyle(Color.brandIndigo).lineLimit(1) }
                        if let i = f?.industry { Text(i).appFont(.caption).foregroundStyle(.secondary).lineLimit(1) }
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
                                .appFont(.system(size: 23, weight: tab == t ? .semibold : .regular))
                            Text(t.rawValue)
                                .appFont(.caption.weight(tab == t ? .bold : .medium))
                                .fixedSize()
                        }
                        .padding(.bottom, 8)
                        .foregroundStyle(tab == t ? Color.brandIndigo : .secondary)
                        .overlay(alignment: .bottom) {
                            if tab == t { Rectangle().fill(Color.brandIndigo).frame(height: 2) }
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
                            .appFont(.system(size: 23, weight: tab == t ? .semibold : .regular))
                        Text(t.rawValue)
                            .appFont(.caption.weight(tab == t ? .bold : .medium))
                            .fixedSize()
                    }
                    .padding(.bottom, 8)
                    .foregroundStyle(tab == t ? Color.brandIndigo : .secondary)
                    .overlay(alignment: .bottom) {
                        if tab == t { Rectangle().fill(Color.brandIndigo).frame(height: 2) }
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
            aiScorecardSection
            marketOverviewHeader
            intrinsicValueSection
            marketStatsSection
            StockKeyMetricsView(
                metrics: f?.keyMetrics ?? [:],
                beta: f?.beta,
                averageVolume: f?.double("averageVolume"),
                viewModel: viewModel
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
                Label("Upcoming Events", systemImage: "calendar").appFont(.headline)
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
            Image(systemName: icon).foregroundStyle(tint).appFont(.system(size: 14))
            Text(label).appFont(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
            Text(badgeText)
                .appFont(.system(size: 9, weight: .bold)).textCase(.uppercase)
                .padding(.horizontal, 4).padding(.vertical, 1)
                .background(badgeTint.opacity(0.12), in: RoundedRectangle(cornerRadius: 4))
                .foregroundStyle(badgeTint)
        }
        .lineLimit(1).fixedSize(horizontal: true, vertical: false)

        let when = HStack(spacing: 4) {
            Text(Self.eventDate(date) + (dateEnd.map { " – " + Self.eventDate($0) } ?? ""))
                .appFont(.callout.weight(.bold))
            if let rel = Self.relativeEventDay(date, timeZone) {
                Text("· \(rel)").appFont(.caption).foregroundStyle(.secondary)
            }
        }
        .lineLimit(1).minimumScaleFactor(0.8)

        let figures = Group {
            if let detail, !detail.isEmpty {
                Text(detail).appFont(.caption2).foregroundStyle(detailTint ?? .secondary)
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

    private func getPillarTier(id: String, score: Double) -> String {
        switch id {
        case "moat":
            return score >= 9 ? "Wide Moat" : score >= 7.5 ? "Solid Moat" : score >= 5.5 ? "Narrow Moat" : "No Moat"
        case "strength":
            return score >= 9 ? "Fortress" : score >= 7.5 ? "Healthy" : score >= 5.5 ? "Adequate" : "Constrained"
        case "predictability":
            return score >= 9 ? "High Visibility" : score >= 7.5 ? "Predictable" : score >= 5.5 ? "Moderate" : "Volatile"
        case "growth":
            return score >= 9 ? "High Growth" : score >= 7.5 ? "Solid Growth" : score >= 5.5 ? "Moderate" : "Sluggish"
        default:
            return score >= 8 ? "Strong" : "Moderate"
        }
    }

    @ViewBuilder private var aiScorecardSection: some View {
        if let sc = viewModel.analysis?.scorecard {
            let topics: [(id: String, name: String, icon: String, score: Double?, tint: Color)] = [
                ("moat", "Moat & Edge", "shield.fill", sc.moat, .blue),
                ("strength", "Financial Strength", "bolt.fill", sc.financialStrength, .orange),
                ("predictability", "Predictability", "target", sc.predictability, .green),
                ("growth", "Growth Pace", "chart.line.uptrend.xyaxis", sc.growth, .purple),
            ]
            
            let validScores = topics.compactMap(\.score)
            let compositeScore = validScores.isEmpty ? nil : validScores.reduce(0, +) / Double(validScores.count)
            let compositeTier: String? = compositeScore.map { s in
                s >= 8.5 ? "Exceptional" : s >= 7.0 ? "Strong" : s >= 5.5 ? "Moderate" : "Weak"
            }
            
            let compositeTierColor: Color = {
                guard let cs = compositeScore else { return .secondary }
                if cs >= 8.5 { return .green }
                if cs >= 7.0 { return .indigo }
                if cs >= 5.5 { return .orange }
                return .red
            }()
            
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .appFont(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 26, height: 26)
                        .background(
                            LinearGradient(colors: [.purple, .indigo], startPoint: .topLeading, endPoint: .bottomTrailing),
                            in: RoundedRectangle(cornerRadius: 7)
                        )
                    
                    Text("AI Fundamental Health")
                        .appFont(.subheadline.weight(.bold))
                        .lineLimit(1)
                    
                    Spacer(minLength: 4)
                    
                    Text("Gemini AI")
                        .appFont(.system(size: 9, weight: .bold))
                        .foregroundStyle(.purple)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.purple.opacity(0.12), in: Capsule())
                }
                
                if let cs = compositeScore {
                    HStack {
                        Text("Composite Score:")
                            .appFont(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                        
                        Spacer()
                        
                        HStack(spacing: 3) {
                            Text(String(format: "%.1f", cs))
                                .appFont(.system(size: 13, weight: .black))
                                .foregroundStyle(compositeTierColor)
                            Text("/10")
                                .appFont(.system(size: 10, weight: .bold))
                                .foregroundStyle(.secondary.opacity(0.7))
                            if let ct = compositeTier {
                                Text("· \(ct)")
                                    .appFont(.system(size: 11, weight: .bold))
                                    .foregroundStyle(compositeTierColor)
                            }
                        }
                    }
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(compositeTierColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(compositeTierColor.opacity(0.25), lineWidth: 0.5))
                }
                
                let cols = hSizeClass == .regular ? 4 : 2
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: cols), spacing: 10) {
                    ForEach(topics, id: \.id) { t in
                        VStack(alignment: .leading, spacing: 8) {
                            // Top Row: Icon on left, Tier Badge on right
                            HStack(alignment: .center) {
                                Image(systemName: t.icon)
                                    .appFont(.system(size: 10, weight: .semibold))
                                    .foregroundStyle(t.tint)
                                    .frame(width: 22, height: 22)
                                    .background(t.tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                                
                                Spacer(minLength: 2)
                                
                                if let s = t.score {
                                    Text(getPillarTier(id: t.id, score: s))
                                        .appFont(.system(size: 8.5, weight: .bold))
                                        .foregroundStyle(t.tint)
                                        .padding(.horizontal, 5).padding(.vertical, 2)
                                        .background(t.tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 5))
                                        .overlay(RoundedRectangle(cornerRadius: 5).strokeBorder(t.tint.opacity(0.2), lineWidth: 0.5))
                                        .fixedSize(horizontal: true, vertical: false)
                                }
                            }
                            
                            // Middle Row: Full Pillar Title (Never Truncated)
                            Text(t.name)
                                .appFont(.system(size: 12, weight: .semibold))
                                .foregroundStyle(.primary)
                                .lineLimit(1)
                                .padding(.top, 2)
                            
                            // Bottom Row: Score and Progress Bar
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(alignment: .lastTextBaseline, spacing: 2) {
                                    if let s = t.score {
                                        let formatted = s.truncatingRemainder(dividingBy: 1) == 0 ? String(format: "%.0f", s) : String(format: "%.1f", s)
                                        Text(formatted).appFont(.system(size: 24, weight: .black)).foregroundStyle(t.tint)
                                        Text("/10").appFont(.system(size: 11, weight: .medium)).foregroundStyle(.secondary.opacity(0.7))
                                    } else {
                                        Text("—").appFont(.title3.weight(.bold)).foregroundStyle(.secondary)
                                    }
                                }
                                
                                GeometryReader { geo in
                                    let progress = t.score != nil ? max(0, min(1.0, (t.score! / 10.0))) : 0.0
                                    ZStack(alignment: .leading) {
                                        Capsule().fill(Color.secondary.opacity(0.12)).frame(height: 4)
                                        Capsule().fill(t.tint).frame(width: max(3, geo.size.width * CGFloat(progress)), height: 4)
                                    }
                                }
                                .frame(height: 4)
                            }
                        }
                        .padding(12)
                        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14))
                        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.quaternary, lineWidth: 0.5))
                    }
                }
            }
            .padding(14)
            .background(Color.secondary.opacity(0.04), in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(.quaternary.opacity(0.6), lineWidth: 1))
        }
    }

    @ViewBuilder private var marketOverviewHeader: some View {
        HStack {
            HStack(spacing: 8) {
                Image(systemName: "square.grid.2x2").foregroundStyle(Color.brandIndigo)
                Text("Market Overview").appFont(.headline)
            }
            Spacer()
            Button { Task { await viewModel.loadAll() } } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.clockwise")
                    Text("Refresh")
                }
                .appFont(.caption2.weight(.bold))
                .foregroundStyle(.cyan)
            }
            .buttonStyle(.plain)
        }
    }

    @ViewBuilder private var intrinsicValueSection: some View {
        if let iv = viewModel.intrinsic {
            let rec = iv.recommendedMethod
            let bestFitVal = rec?.intrinsicValue
            let bestFitRange: IntrinsicValueResponse.MC? = {
                guard let k = rec?.methodKey else { return nil }
                if k == "dcf" { return iv.models?.dcf?.mc }
                if k == "graham" { return iv.models?.graham?.mc }
                if k == "ddm" { return iv.models?.ddm?.mc }
                return nil
            }()
            let blendedVal = iv.averageIntrinsicValue
            let blendedRange: IntrinsicValueResponse.MC? = {
                guard let r = iv.range else { return nil }
                return .init(bear: r.bear, base: blendedVal, bull: r.bull, histogram: nil)
            }()

            let cols = hSizeClass == .regular ? 2 : 1
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: cols), spacing: 12) {
                if let bestFitVal, bestFitVal > 0 {
                    ivCard("Best-Fit: \(rec?.name ?? "Valuation Method")", bestFitVal, upside: upside(bestFitVal, iv.currentPrice), range: bestFitRange, tint: .indigo, icon: "sparkles")
                }
                if let blendedVal, blendedVal > 0 {
                    let title = iv.valuationStatus == "nav" ? "Net Asset Value (NAV)" : "Blended Intrinsic Value"
                    ivCard(title, blendedVal, upside: upside(blendedVal, iv.currentPrice), range: blendedRange, tint: .indigo, icon: "scalemass")
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
                Image(systemName: "arrow.left.and.right").foregroundStyle(.blue).appFont(.system(size: 16))
                Text("52-Week Range").appFont(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
            }
            if usable, let low, let high {
                HStack {
                    Text(Fmt.currency(low, code: nativeCur)).appFont(.callout.weight(.bold))
                    Spacer()
                    Text(Fmt.currency(high, code: nativeCur)).appFont(.callout.weight(.bold))
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
                Text("-").appFont(.title3.weight(.bold))
            }
        }
        .gridTile()
        .padding(16)
        .background(Color.gray.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder private var businessSummarySection: some View {
        if let summary = f?.summary, !summary.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Label("Business Summary", systemImage: "building.2").appFont(.headline)
                // Clamped by default: these run to a dozen lines and pushed
                // everything measurable off the screen.
                Text(summary).appFont(.subheadline).foregroundStyle(.secondary)
                    .lineSpacing(4)
                    .lineLimit(summaryExpanded ? nil : 4)
                // Only offered when there is something behind the clamp — a
                // toggle that does nothing is worse than no toggle.
                if summary.count > 320 {
                    Button(summaryExpanded ? "Show less" : "Read more") {
                        withAnimation(.easeInOut(duration: 0.2)) { summaryExpanded.toggle() }
                    }
                    .buttonStyle(.plain)
                    .appFont(.caption.weight(.bold))
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
                Label("Your Position", systemImage: "wallet.pass").appFont(.headline)
                Spacer()
                Text("AGGREGATED").appFont(.system(size: 11, weight: .bold)).foregroundStyle(.secondary)
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
                Text(title).appFont(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
            }
            HStack(alignment: .bottom) {
                Text(Fmt.currency(value, code: nativeCur)).appFont(.title3.weight(.bold)).foregroundStyle(.primary)
                Spacer()
                if let u = upside { Text(Fmt.percent(u, includeSign: true)).appFont(.caption2.weight(.bold)).foregroundStyle(Fmt.tint(for: u)) }
            }
            if let r = range, let bear = r.bear, let bull = r.bull {
                Text("Range: \(Fmt.currency(bear, code: nativeCur)) – \(Fmt.currency(bull, code: nativeCur))")
                    .appFont(.system(size: 11)).foregroundStyle(.secondary)
            }
        }
        .gridTile()
        .padding(16)
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
                            .appFont(.system(size: 27))
                            .foregroundStyle(.white)
                            .frame(width: 48, height: 48)
                            .background(Color.purple, in: RoundedRectangle(cornerRadius: 12))
                        
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("AI Fundamental Review").appFont(.title3.bold())
                                Spacer()
                                Button { Task { await viewModel.loadAnalysis(force: true) } } label: { 
                                    Label("Regenerate", systemImage: "arrow.clockwise") 
                                }
                                .appFont(.caption2.weight(.bold)).foregroundStyle(.purple)
                                .buttonStyle(.plain)
                            }
                            if let s = a.summary { Text(Self.md(s)).appFont(.subheadline).foregroundStyle(.secondary) }
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
                            Text(t.0).appFont(.system(size: 11, weight: .bold)).foregroundStyle(.secondary).textCase(.uppercase)
                            Text("\(Fmt.number(t.2, fractionDigits: 0))")
                                .font(AppFont.system(size: 36, weight: .black).resolved(scale: fontScale))
                                .foregroundStyle(t.4)
                            + Text("/10")
                                .font(AppFont.callout.resolved(scale: fontScale))
                                .foregroundStyle(.secondary).baselineOffset(8)
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
                                    .appFont(.system(size: 18))
                                    .foregroundStyle(t.4)
                                    .frame(width: 36, height: 36)
                                    .background(t.4.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                                Text(t.0).appFont(.headline)
                            }
                            Text(Self.md(t.3 ?? "No analysis available.")).appFont(.subheadline).foregroundStyle(.secondary)
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
                Image(systemName: "sparkles").appFont(.largeTitle).foregroundStyle(.purple.opacity(0.4))
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
                Text("Market Sentiment").appFont(.headline)
                Spacer()
                Text(label).appFont(.caption.bold()).padding(.horizontal, 8).padding(.vertical, 4).background(tone.opacity(0.2), in: Capsule()).foregroundStyle(tone)
            }
            VStack(spacing: 8) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.secondary.opacity(0.2)).frame(height: 12)
                        Capsule().fill(tone).frame(width: max(0, min(geo.size.width * CGFloat(s / 100.0), geo.size.width)), height: 12)
                    }
                }.frame(height: 12).padding(.vertical, 8)
                HStack {
                    Text("Extreme Fear").appFont(.system(size: 11, weight: .bold)).foregroundStyle(.secondary).textCase(.uppercase)
                    Spacer()
                    Text("\(Int(s))%").appFont(.title3.weight(.bold)).foregroundStyle(.primary)
                    Spacer()
                    Text("Extreme Greed").appFont(.system(size: 11, weight: .bold)).foregroundStyle(.secondary).textCase(.uppercase)
                }
            }
            Text("Current market vibe based on news flow, analyst ratings, and social trends.")
                .appFont(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: .infinity).padding(.top, 8)
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
                Text("Upcoming Catalysts").appFont(.headline)
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
                                Text(c.event).appFont(.subheadline.weight(.semibold))
                                Spacer()
                                Text(c.impact).appFont(.system(size: 10, weight: .bold)).textCase(.uppercase).foregroundStyle(.secondary)
                                    .padding(.horizontal, 4).padding(.vertical, 2).overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Color.secondary.opacity(0.3)))
                            }
                            Text(MarketTime.formatted(c.date)).appFont(.caption2.weight(.medium)).foregroundStyle(.secondary)
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
            StatementTypeBar(
                statement: finType,
                period: viewModel.financialsPeriod,
                onSelectStatement: { type in
                    finType = type
                    chartSlots = []
                    showAllMetrics = false
                },
                onSelectPeriod: { p in
                    chartSlots = []
                    showAllMetrics = false
                    chartRange = nil
                    Task { await viewModel.loadFinancials(period: p) }
                }
            )

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
                        .appFont(.caption.weight(.bold))
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
            StatementTrendHeader(
                period: period,
                periodCount: periods.count,
                range: Binding(get: { range }, set: { chartRange = $0 })
            )

            // Stays inside the card's padding. Bleeding this row to the card
            // edge with negative padding is the obvious polish and it is a
            // trap: the over-wide child inflates the page's vertical
            // ScrollView, which re-proposes the inflated width to everything
            // in it, and the whole screen shifts off the right edge — header,
            // tabs and this card's own scrollers, whose trailing ends then sit
            // where no amount of scrolling reaches them. See `readingContainerWidth`.
            metricChips(chartable, slots: slots, colors: colors)

            if series.isEmpty {
                Text("Pick a line item above to chart it.")
                    .appFont(.callout).foregroundStyle(.secondary)
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
            .appFont(.caption2).foregroundStyle(.tertiary)

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
                            Text(row.label).appFont(.caption)
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
                    .buttonStyle(.plain).appFont(.caption.weight(.bold)).foregroundStyle(Color.indigo)
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
                                .appFont(.system(size: 9, weight: .regular))
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
                .appFont(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)

                Divider()

                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    let slot = slots.firstIndex(of: row.label)
                    GridRow {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(slot.map { colors[$0 % colors.count] } ?? .clear)
                                .frame(width: 7, height: 7)
                            Text(row.label).appFont(.subheadline.weight(.semibold)).lineLimit(1)
                        }
                        .gridColumnAlignment(.leading)
                        sparkline(row.values.compactMap { $0 })
                        ForEach(Array(row.values.enumerated()), id: \.offset) { _, v in
                            Text(v.map { compact($0) } ?? "—")
                                .appFont(.subheadline).monospacedDigit()
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
        let period = viewModel.ratiosPeriod
        let rawHistory = viewModel.ratios?.historical ?? []
        let rangeLimit = ratiosRange.periods(period)
        let history = Array(rawHistory.prefix(rangeLimit))

        let categories = ["All", "Valuation", "Profitability", "Balance Sheet", "Earnings & Sales"]
        let activeDefs = ratiosCategory == "All"
            ? StockKeyMetricsView.chartDefs
            : StockKeyMetricsView.chartDefs.filter { $0.group == ratiosCategory }

        VStack(alignment: .leading, spacing: 20) {
            if viewModel.isLoadingRatios {
                ProgressView().frame(maxWidth: .infinity).padding(40)
            } else if history.isEmpty && viewModel.trackRecord == nil {
                ContentUnavailableView("No ratio data", systemImage: "chart.line.uptrend.xyaxis").frame(height: 200)
            } else {
                if let record = viewModel.trackRecord { trackRecordPanel(record) }

                if !history.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        // Category filter pills + Period/Range Pickers Toolbar
                        ViewThatFits(in: .horizontal) {
                            HStack(alignment: .center) {
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 6) {
                                        ForEach(categories, id: \.self) { cat in
                                            Button {
                                                ratiosCategory = cat
                                            } label: {
                                                Text(cat)
                                                    .appFont(.caption.weight(.semibold))
                                                    .padding(.horizontal, 10).padding(.vertical, 5)
                                                    .background(ratiosCategory == cat ? Color.accentColor : Color.gray.opacity(0.12), in: Capsule())
                                                    .foregroundStyle(ratiosCategory == cat ? Color.white : Color.secondary)
                                            }
                                            .buttonStyle(.plain)
                                        }
                                    }
                                }

                                Spacer(minLength: 12)

                                HStack(spacing: 8) {
                                    Picker("", selection: $ratiosRange) {
                                        ForEach(StatementRange.allCases) { r in
                                            Text(r.rawValue == "MAX" ? "ALL" : r.rawValue).tag(r)
                                        }
                                    }
                                    .pickerStyle(.segmented).labelsHidden().frame(width: 140)

                                    Picker("", selection: Binding(
                                        get: { period },
                                        set: { p in Task { await viewModel.loadRatios(period: p) } }
                                    )) {
                                        ForEach(StatementPeriod.allCases) { p in Text(p.title).tag(p) }
                                    }
                                    .pickerStyle(.segmented).labelsHidden().frame(width: 170)
                                }
                            }

                            VStack(alignment: .leading, spacing: 10) {
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 6) {
                                        ForEach(categories, id: \.self) { cat in
                                            Button {
                                                ratiosCategory = cat
                                            } label: {
                                                Text(cat)
                                                    .appFont(.caption.weight(.semibold))
                                                    .padding(.horizontal, 10).padding(.vertical, 5)
                                                    .background(ratiosCategory == cat ? Color.accentColor : Color.gray.opacity(0.12), in: Capsule())
                                                    .foregroundStyle(ratiosCategory == cat ? Color.white : Color.secondary)
                                            }
                                            .buttonStyle(.plain)
                                        }
                                    }
                                }

                                HStack(spacing: 8) {
                                    Picker("", selection: $ratiosRange) {
                                        ForEach(StatementRange.allCases) { r in
                                            Text(r.rawValue == "MAX" ? "ALL" : r.rawValue).tag(r)
                                        }
                                    }
                                    .pickerStyle(.segmented).labelsHidden().frame(width: 140)

                                    Picker("", selection: Binding(
                                        get: { period },
                                        set: { p in Task { await viewModel.loadRatios(period: p) } }
                                    )) {
                                        ForEach(StatementPeriod.allCases) { p in Text(p.title).tag(p) }
                                    }
                                    .pickerStyle(.segmented).labelsHidden().frame(width: 170)
                                }
                            }
                        }

                        Text(period == .quarterly
                             ? "Measured on the trailing twelve months at each quarter end, sampled four times as often."
                             : "Measured on each filed fiscal year.")
                            .appFont(.caption2).foregroundStyle(.tertiary).fixedSize(horizontal: false, vertical: true)
                    }

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 16)], spacing: 16) {
                        ForEach(activeDefs) { def in
                            ratioChart(
                                def.title,
                                history,
                                def.dataKey,
                                def.color,
                                isPercent: def.isPercent,
                                isCount: def.isCount,
                                periodType: period
                            )
                        }
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
                        .appFont(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                    Text(trackRecordSpan(record))
                        .appFont(.caption2).foregroundStyle(.tertiary)
                }
                Spacer()
                if let rank = record.rank?.rank {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("#\(rank)").appFont(.title2.weight(.bold)).monospacedDigit()
                        Text("Buffett rank").appFont(.system(size: 9)).foregroundStyle(.tertiary).textCase(.uppercase)
                    }
                }
            }

            if !record.gateFailures.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text("Not eligible for the ranking: "
                         + record.gateFailures.map { $0.replacingOccurrences(of: "_", with: " ") }
                            .joined(separator: ", "))
                        .appFont(.caption).foregroundStyle(.orange)
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
                                .appFont(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                            Spacer()
                            if let score = record.rank?.pillars?[group.key] ?? nil {
                                Text(String(format: "%.0f", score))
                                    .appFont(.caption.weight(.semibold)).monospacedDigit()
                                    .foregroundStyle(.secondary)
                            }
                        }
                        ForEach(group.items) { item in
                            HStack(alignment: .firstTextBaseline) {
                                Text(item.label).appFont(.subheadline).foregroundStyle(.secondary)
                                Spacer(minLength: 12)
                                Text(item.display ?? (item.note != nil ? "n/a" : "—"))
                                    .appFont(.subheadline.weight(.medium)).monospacedDigit()
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
                .appFont(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
            ForEach(bands) { band in
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(band.label).appFont(.subheadline).foregroundStyle(.secondary)
                        Spacer()
                        Text(band.display).appFont(.subheadline.weight(.semibold)).monospacedDigit()
                        Text("vs \(band.medianDisplay) median")
                            .appFont(.caption).foregroundStyle(.tertiary)
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
                        .appFont(.caption2).foregroundStyle(.tertiary)
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
                .appFont(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
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
        Text(window.label).appFont(.subheadline).foregroundStyle(.secondary)
    }

    @ViewBuilder private func stressMetrics(_ window: TrackRecordStress) -> some View {
        if window.covered {
            ForEach(window.items) { item in
                // One Text, not an HStack of three: a metric too wide for the
                // line then wraps at its spaces instead of being squeezed a
                // character at a time.
                let label = AppFont.subheadline.resolved(scale: fontScale)
                let value = AppFont.subheadline.weight(.medium).resolved(scale: fontScale)
                let note = AppFont.caption2.resolved(scale: fontScale)
                (Text(item.label + " ").font(label).foregroundStyle(.secondary)
                 + Text(item.display).font(value).monospacedDigit()
                    .foregroundStyle(item.changePct < 0 ? Color.red : Color.green)
                 + Text(" (\(item.recoveryDisplay ?? "no fall"))")
                    .font(note).foregroundStyle(.tertiary))
                .fixedSize(horizontal: false, vertical: true)
            }
        } else {
            // Not the same claim as "did not fall".
            Text("not filing then").appFont(.subheadline.italic()).foregroundStyle(.tertiary)
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
                    .appFont(.caption2).foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                ForEach(revisions.items) { item in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(item.label).appFont(.subheadline).foregroundStyle(.secondary)
                        Text(String(item.periodEnd.prefix(4)))
                            .appFont(.caption).monospacedDigit().foregroundStyle(.tertiary)
                        Spacer(minLength: 8)
                        Text(item.display).appFont(.subheadline).monospacedDigit()
                        Text(item.changeDisplay)
                            .appFont(.subheadline.weight(.medium)).monospacedDigit()
                            .foregroundStyle(item.changePct < 0 ? .red : .green)
                            .frame(width: 72, alignment: .trailing)
                        Text("\(item.firstFiled.prefix(4)) → \(item.restatedFiled.prefix(4))")
                            .appFont(.caption2).monospacedDigit().foregroundStyle(.tertiary)
                    }
                }
                if revisions.count > revisions.items.count {
                    Text("Showing the \(revisions.items.count) largest of \(revisions.count).")
                        .appFont(.caption2).foregroundStyle(.tertiary)
                }
            }
            .padding(.top, 8)
        } label: {
            Label(
                "\(revisions.count) figure\(revisions.count == 1 ? "" : "s") revised after first reporting",
                systemImage: "clock.arrow.circlepath"
            )
            .appFont(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
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

    /// One ratio across the filings. The drawing is `FiledPeriodChart`, which
    /// is also what the Key Metrics cards use — the two were separate copies of
    /// the same chart with separately wrong axis rules.
    private func ratioChart(
        _ title: String,
        _ data: [[String: JSONValue]],
        _ key: String,
        _ color: Color,
        isPercent: Bool,
        isCount: Bool = false,
        periodType: StatementPeriod = .annual
    ) -> some View {
        card(title) {
            FiledPeriodChart(
                points: FiledPeriodChart.points(data, key: key),
                color: color,
                periodType: periodType,
                label: title,
                format: { ratioValueLabel($0, isPercent: isPercent, isCount: isCount) }
            )
        }
    }

    /// One ratio rendered the same way on the axis and in the tooltip.
    private func ratioValueLabel(_ v: Double, isPercent: Bool, isCount: Bool) -> String {
        if isCount { return compact(v) }
        return isPercent ? Fmt.percent(v) : Fmt.number(v, fractionDigits: 2)
    }

    // MARK: - Holdings (ETF)

    @ViewBuilder private var holdingsTab: some View {
        if !f!.etfTopHoldings.isEmpty {
            card("Top 10 Holdings") {
                VStack(spacing: 8) {
                    HStack {
                        Text("Symbol").appFont(.caption.weight(.bold)).foregroundStyle(.secondary)
                        Spacer()
                        Text("% Assets").appFont(.caption.weight(.bold)).foregroundStyle(.secondary)
                    }
                    Divider()
                    ForEach(f!.etfTopHoldings, id: \.symbol) { h in
                        Button {
                            appState.openStock(h.symbol)
                        } label: {
                            HStack {
                                Text(h.symbol).appFont(.headline).foregroundStyle(.indigo)
                                Spacer()
                                Text(Fmt.percent(h.percent)).appFont(.subheadline.bold()).foregroundStyle(.primary)
                            }
                        }
                        .buttonStyle(.plain)
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
                                    Image(systemName: "newspaper").appFont(.largeTitle).foregroundStyle(.tertiary)
                                }
                                .frame(height: 160).clipped()
                            }
                            VStack(alignment: .leading, spacing: 12) {
                                HStack(spacing: 8) {
                                    Text(item.provider).appFont(.system(size: 11, weight: .bold)).textCase(.uppercase).foregroundStyle(.indigo)
                                        .padding(.horizontal, 8).padding(.vertical, 4).background(Color.indigo.opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
                                    Text(item.pubDate).appFont(.caption2.weight(.medium)).foregroundStyle(.secondary)
                                }
                                Text(item.title).appFont(.headline).foregroundStyle(.primary).lineLimit(3)
                            }.padding(20)
                            Spacer(minLength: 0)
                        }
                        .frame(maxWidth: .infinity, alignment: .topLeading)
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
                            Text("Position Overview").appFont(.headline)
                            Spacer()
                            if let w = summary.portfolioWeightPct, w > 0 {
                                Text("\(Fmt.number(w, fractionDigits: 2))% of Portfolio")
                                    .appFont(.caption2.weight(.bold))
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
                                Text("Capital Appreciation").appFont(.caption2).foregroundStyle(.secondary)
                                Text(Fmt.currency(unreal + ret.realizedGain, currency: cur)).appFont(.subheadline.bold())
                                Text("Unreal: \(Fmt.currency(unreal, currency: cur)) · Real: \(Fmt.currency(ret.realizedGain, currency: cur))").appFont(.caption2).foregroundStyle(.secondary)
                            }
                            .gridTile()
                            .padding(12)
                            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Dividend Income").appFont(.caption2).foregroundStyle(.secondary)
                                Text("+\(Fmt.currency(ret.lifetimeDividends, currency: cur))").appFont(.subheadline.bold()).foregroundStyle(.green)
                                Text("YoC: \(ret.yieldOnCostPct != nil ? Fmt.percent(ret.yieldOnCostPct!) : "—")").appFont(.caption2).foregroundStyle(.secondary)
                            }
                            .gridTile()
                            .padding(12)
                            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Currency (FX) Impact").appFont(.caption2).foregroundStyle(.secondary)
                                Text("\(ret.fxGainLoss >= 0 ? "+" : "")\(Fmt.currency(ret.fxGainLoss, currency: cur))").appFont(.subheadline.bold()).foregroundStyle(ret.fxGainLoss >= 0 ? .green : .red)
                                Text("\(ret.fxGainLossPct >= 0 ? "+" : "")\(Fmt.percent(ret.fxGainLossPct)) on cost").appFont(.caption2).foregroundStyle(.secondary)
                            }
                            .gridTile()
                            .padding(12)
                            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

                            VStack(alignment: .leading, spacing: 4) {
                                Text("Fees & Taxes Friction").appFont(.caption2).foregroundStyle(.secondary)
                                Text("-\(Fmt.currency(ret.commissions + ret.withholdingTaxes, currency: cur))").appFont(.subheadline.bold()).foregroundStyle(.red)
                                Text("Fees: \(Fmt.currency(ret.commissions, currency: cur)) · Tax: \(Fmt.currency(ret.withholdingTaxes, currency: cur))").appFont(.caption2).foregroundStyle(.secondary)
                            }
                            .gridTile()
                            .padding(12)
                            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                        }
                    }
                }

                // 3. Open FIFO Lots
                if !pos.openLots.isEmpty {
                    card("Open FIFO Tax Lots (\(pos.openLots.count))") {
                        VStack(spacing: 8) {
                            ForEach(pos.openLots) { lot in
                                LotDetailRow(
                                    title: MarketTime.formatted(lot.date),
                                    badge: lot.taxTerm == "long_term"
                                        ? (text: "LT", tint: .green)
                                        : (text: "ST", tint: .blue),
                                    headline: Fmt.currencyWhole(lot.marketValueDisplay, code: cur),
                                    detail: "\(Fmt.shares(lot.quantity)) sh @ \(Fmt.currency(lot.costPerShareLocal, currency: pos.localCurrency))",
                                    detailValue: Fmt.currencyWhole(lot.unrealizedGainDisplay, code: cur, signed: true),
                                    detailTint: lot.unrealizedGainDisplay >= 0 ? .green : .red,
                                    footnote: lot.account,
                                    footnoteValue: Fmt.percent(lot.unrealizedGainPct, includeSign: true),
                                    footnoteTint: lot.unrealizedGainPct >= 0 ? .green : .red
                                )
                            }
                        }
                    }
                }

                // 4. Closed Trades
                if !pos.closedTrades.isEmpty {
                    card("Closed Trades & Realized Sells (\(pos.closedTrades.count))") {
                        VStack(spacing: 8) {
                            ForEach(pos.closedTrades) { trade in
                                LotDetailRow(
                                    title: MarketTime.formatted(trade.sellDate),
                                    headline: Fmt.currencyWhole(trade.proceedsDisplay, code: cur),
                                    detail: "\(Fmt.shares(trade.quantitySold)) sh @ \(Fmt.currency(trade.salePrice, currency: pos.localCurrency))",
                                    detailValue: "Gain \(Fmt.currencyWhole(trade.realizedGainDisplay, code: cur, signed: true))",
                                    detailTint: trade.realizedGainDisplay >= 0 ? .green : .red,
                                    footnote: trade.account
                                )
                            }
                        }
                    }
                }
            }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "briefcase").appFont(.system(size: 36)).foregroundStyle(.secondary)
                Text("No Position in \(viewModel.symbol)").appFont(.headline)
                Text("You currently have no recorded transactions for this symbol.")
                    .appFont(.subheadline).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity).padding(40)
        }
    }

    // MARK: - Helpers

    private func card<C: View>(_ title: String, trailing: AnyView? = nil, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack { Text(title).appFont(.headline); Spacer(); if let trailing { trailing } }
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
                    Image(systemName: icon).foregroundStyle(iconTint).appFont(.system(size: 16))
                }
                Text(label).appFont(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
            }
            // The figure and its percentage used to share one line with no way
            // out: the figure could shrink only to 0.6, and the percentage —
            // one unbreakable word with no `lineLimit` — could not shrink at
            // all. Together they demanded more than a two-column tile is
            // offered on a phone, the grid row overflowed, and a vertical
            // ScrollView adopts an over-wide child as its own width and
            // re-proposes it to everything in it. That is how two tiles pushed
            // this whole page off the right edge. `ViewThatFits` drops the
            // percentage onto its own line rather than demanding the room.
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .bottom, spacing: 6) {
                    valueText(value, icon: icon, iconTint: iconTint)
                    Spacer(minLength: 4)
                    subText(sub, subTint ?? iconTint)
                }
                VStack(alignment: .leading, spacing: 3) {
                    valueText(value, icon: icon, iconTint: iconTint)
                    subText(sub, subTint ?? iconTint)
                }
            }
        }
        .gridTile()
        .padding(16)
        .background(bgTint ?? Color.gray.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
    }

    private func valueText(_ value: String, icon: String?, iconTint: Color) -> some View {
        Text(value)
            .appFont(.title3.weight(.bold))
            .foregroundStyle(icon == nil ? iconTint : .primary)
            .lineLimit(1).minimumScaleFactor(0.6)
    }

    @ViewBuilder private func subText(_ sub: String?, _ tint: Color) -> some View {
        if let sub {
            Text(sub)
                .appFont(.caption2.weight(.bold))
                .foregroundStyle(tint)
                // A single unbreakable word demands its full width unless it is
                // allowed to shrink; without this it is a hard minimum.
                .lineLimit(1).minimumScaleFactor(0.7)
        }
    }

    private func compact(_ v: Double) -> String {
        let a = abs(v)
        if a >= 1_000_000_000 { return String(format: "%.2fB", v / 1_000_000_000) }
        if a >= 1_000_000 { return String(format: "%.2fM", v / 1_000_000) }
        if a >= 1_000 { return String(format: "%.1fK", v / 1_000) }
        return Fmt.number(v, fractionDigits: 0)
    }

    static func md(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(s)
    }
}
