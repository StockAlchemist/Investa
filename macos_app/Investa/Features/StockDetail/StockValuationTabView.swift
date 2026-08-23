import SwiftUI
import Charts

struct DistributionModelItem: Identifiable, Sendable {
    var id: String { key }
    let key: String
    let title: String
    let color: Color
    let mc: IntrinsicValueResponse.MC
    let currentPrice: Double?
    let nativeCur: String
}

struct StockValuationTabView: View {
    @ObservedObject var viewModel: StockDetailViewModel
    @Environment(\.horizontalSizeClass) private var hSizeClass
    @State private var showGrahamExplanation: Bool = false
    @State private var expandedHistogramKeys: Set<String> = []
    @State private var selectedDistributionItem: DistributionModelItem? = nil

    @State private var customOverrides: [String: [String: Double]] = [:]
    @State private var editingModelKeys: Set<String> = []
    /// Which method card(s) to render. Defaults to the backend's best-fit model
    /// so the tab opens on one card instead of a dozen.
    @State private var methodFilter: String = StockValuationTabView.bestFitSentinel

    // MARK: - Method selector

    /// Sentinels for "show the best-fit model only" and "show every model".
    static let bestFitSentinel = "__best_fit__"
    static let allMethodsSentinel = "__all__"

    enum ModelCategory: String, CaseIterable {
        case cashEarnings, multiplesGrowth, floorsRelative

        var label: String {
            switch self {
            case .cashEarnings: return "Cash Flow & Earnings"
            case .multiplesGrowth: return "Multiples & Growth"
            case .floorsRelative: return "Floors & Relative"
            }
        }
    }

    struct ModelOption: Identifiable {
        let key: String
        let label: String
        /// Abbreviation for the collapsed selector button, which shares one row
        /// with the "Valuation Method" caption and has little width on a phone.
        let short: String
        let category: ModelCategory
        var id: String { key }
    }

    /// Every model the tab can render, in display order. Mirrors `MODEL_CATALOG`
    /// in `web_app/components/stock-detail/tabs/ValuationTab.tsx`.
    static let modelCatalog: [ModelOption] = [
        ModelOption(key: "dcf", label: "Discounted Free Cash Flow (DCF)", short: "DCF", category: .cashEarnings),
        ModelOption(key: "dcfo", label: "Discounted Cash from Operations (D-CFO)", short: "D-CFO", category: .cashEarnings),
        ModelOption(key: "dni", label: "Discounted Net Income (D-NI)", short: "D-NI", category: .cashEarnings),
        ModelOption(key: "ddm", label: "Dividend Discount Model (DDM)", short: "DDM", category: .cashEarnings),
        ModelOption(key: "mean_pe", label: "Mean P/E Ratio", short: "Mean P/E", category: .multiplesGrowth),
        ModelOption(key: "peg", label: "PEG Ratio Fair Value", short: "PEG", category: .multiplesGrowth),
        ModelOption(key: "mean_pb", label: "Mean P/B Ratio", short: "Mean P/B", category: .multiplesGrowth),
        ModelOption(key: "mean_ps", label: "Mean P/S Ratio", short: "Mean P/S", category: .multiplesGrowth),
        ModelOption(key: "psg", label: "Price-to-Sales Growth (PSG)", short: "PSG", category: .multiplesGrowth),
        ModelOption(key: "graham", label: "Graham Formula", short: "Graham", category: .floorsRelative),
        ModelOption(key: "lynch", label: "Peter Lynch Fair Value", short: "Lynch", category: .floorsRelative),
        ModelOption(key: "epv", label: "Earnings Power Value (EPV Floor)", short: "EPV", category: .floorsRelative),
    ]

    private func model(_ models: IntrinsicValueResponse.Models, for key: String) -> IntrinsicValueResponse.Model? {
        switch key {
        case "dcf": return models.dcf
        case "dcfo": return models.dcfo
        case "dni": return models.dni
        case "ddm": return models.ddm
        case "mean_pe": return models.meanPe
        case "peg": return models.peg
        case "mean_pb": return models.meanPb
        case "mean_ps": return models.meanPs
        case "psg": return models.psg
        case "graham": return models.graham
        case "lynch": return models.lynch
        case "epv": return models.epv
        default: return nil
        }
    }

    /// Only the models the backend actually returned for this symbol.
    private var availableOptions: [ModelOption] {
        guard let models = viewModel.intrinsic?.models else { return [] }
        return Self.modelCatalog.filter { model(models, for: $0.key) != nil }
    }

    private var bestFitKey: String? {
        guard let key = viewModel.intrinsic?.recommendedMethod?.methodKey,
              availableOptions.contains(where: { $0.key == key }) else { return nil }
        return key
    }

    /// A stored selection goes stale when the user opens a stock without that
    /// model, so fall back to the best fit rather than rendering nothing.
    private var effectiveFilter: String {
        if methodFilter == Self.bestFitSentinel || methodFilter == Self.allMethodsSentinel { return methodFilter }
        return availableOptions.contains(where: { $0.key == methodFilter }) ? methodFilter : Self.bestFitSentinel
    }

    private func showsModel(_ key: String) -> Bool {
        switch effectiveFilter {
        case Self.allMethodsSentinel: return true
        case Self.bestFitSentinel: return bestFitKey.map { $0 == key } ?? true
        default: return effectiveFilter == key
        }
    }

    private var selectionLabel: String {
        switch effectiveFilter {
        case Self.allMethodsSentinel: return "All Methods (\(availableOptions.count))"
        case Self.bestFitSentinel:
            guard let key = bestFitKey,
                  let opt = Self.modelCatalog.first(where: { $0.key == key }) else { return "All Methods" }
            return "Best Fit — \(opt.short)"
        default:
            return Self.modelCatalog.first(where: { $0.key == effectiveFilter })?.short ?? "Valuation Method"
        }
    }

    private var f: Fundamentals? { viewModel.fundamentals }
    private var nativeCur: String { f?.currency ?? "USD" }

    private var blendedResult: BlendedValuationResult {
        StockValuationCalculator.calculateBlendedScore(
            intrinsicValue: viewModel.intrinsic,
            customOverrides: customOverrides,
            sector: f?.sector
        )
    }

    var body: some View {
        VStack(spacing: 24) {
            if let iv = viewModel.intrinsic {
                if blendedResult.hasAnyCustom {
                    customParametersAlertBar()
                }

                valuationSummaryCards(iv)

                if let note = iv.valuationNote {
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

                BlendCompositionCard(iv: iv, currencyCode: nativeCur)

                if let rec = iv.recommendedMethod, rec.name != nil, rec.methodKey != "none" {
                    recommendedMethodBanner(rec, currentPrice: iv.currentPrice)
                }

                valuationSpectrumSection(iv)

                if let models = iv.models, !availableOptions.isEmpty {
                    methodSelector()
                    valuationModelCards(iv, models: models)
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
        .sheet(item: $selectedDistributionItem) { item in
            DistributionModalView(item: item)
        }
    }

    @ViewBuilder
    private func customParametersAlertBar() -> some View {
        HStack(spacing: 12) {
            Image(systemName: "sparkles")
                .foregroundStyle(.orange)
                .font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text("Custom Parameters Active")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.orange)
                Text("Intrinsic values and composite score are recalculated in real time.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                customOverrides.removeAll()
                editingModelKeys.removeAll()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.counterclockwise")
                    Text("Reset All Defaults")
                }
                .font(.caption.weight(.bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.orange, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(14)
        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.orange.opacity(0.3), lineWidth: 1))
    }

    @ViewBuilder
    private func methodSelector() -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 12) {
                Text("Valuation Method")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)
                Spacer(minLength: 8)
                PopoverMenu(minWidth: 280) {
                    MenuToggleRow(title: bestFitKey == nil ? "Best Fit (none selected)" : bestFitLabel,
                                  isOn: effectiveFilter == Self.bestFitSentinel,
                                  dismissOnTap: true) { methodFilter = Self.bestFitSentinel }
                    MenuToggleRow(title: "All Methods (\(availableOptions.count))",
                                  isOn: effectiveFilter == Self.allMethodsSentinel,
                                  dismissOnTap: true) { methodFilter = Self.allMethodsSentinel }
                    ForEach(ModelCategory.allCases, id: \.self) { category in
                        let group = availableOptions.filter { $0.category == category }
                        if !group.isEmpty {
                            MenuDivider()
                            MenuSectionHeader(category.label)
                            ForEach(group) { option in
                                MenuToggleRow(title: option.label,
                                              isOn: effectiveFilter == option.key,
                                              dismissOnTap: true) { methodFilter = option.key }
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 6) {
                        Text(selectionLabel)
                            .font(.subheadline.weight(.semibold))
                            .lineLimit(1)
                            .truncationMode(.tail)
                        Image(systemName: "chevron.up.chevron.down").font(.caption2)
                    }
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                }
            }

            if effectiveFilter == Self.bestFitSentinel, bestFitKey != nil {
                Text("Showing the best-fit model only — switch above to compare others.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var bestFitLabel: String {
        guard let key = bestFitKey,
              let opt = Self.modelCatalog.first(where: { $0.key == key }) else { return "Best Fit" }
        return "Best Fit — \(opt.label)"
    }

    @ViewBuilder
    private func valuationModelCards(_ iv: IntrinsicValueResponse, models: IntrinsicValueResponse.Models) -> some View {
        Group {
            if showsModel("dcf"), let dcf = models.dcf {
                dcfCard("Discounted Cash Flow", "chart.line.uptrend.xyaxis", .green, dcf, modelKey: "dcf", iv: iv)
            }
            if showsModel("dcfo"), let dcfo = models.dcfo {
                dcfoCard("Discounted Cash from Operations", "dollarsign.circle", .teal, dcfo, modelKey: "dcfo", iv: iv)
            }
            if showsModel("dni"), let dni = models.dni {
                dniCard("Discounted Net Income", "building.columns", .blue, dni, modelKey: "dni", iv: iv)
            }
            if showsModel("mean_pe"), let pe = models.meanPe {
                meanPeCard("Mean P/E Ratio", "percent", .indigo, pe, modelKey: "mean_pe", iv: iv)
            }
            if showsModel("peg"), let peg = models.peg {
                pegCard("PEG Ratio Fair Value", "bolt", .yellow, peg, modelKey: "peg", iv: iv)
            }
            if showsModel("mean_pb"), let pb = models.meanPb {
                meanPbCard("Mean P/B Ratio", "book", .orange, pb, modelKey: "mean_pb", iv: iv)
            }
        }
        Group {
            if showsModel("mean_ps"), let ps = models.meanPs {
                meanPsCard("Mean P/S Ratio", "chart.line.uptrend.xyaxis", .pink, ps, modelKey: "mean_ps", iv: iv)
            }
            if showsModel("psg"), let psg = models.psg {
                psgCard("Price-to-Sales Growth (PSG)", "sparkles", .purple, psg, modelKey: "psg", iv: iv)
            }
            if showsModel("graham"), let g = models.graham {
                grahamCard("Graham Formula", "scalemass", .orange, g, modelKey: "graham", iv: iv)
            }
            if showsModel("ddm"), let ddm = models.ddm {
                ddmCard("Dividend Discount Model", "dollarsign.circle", .purple, ddm, modelKey: "ddm", iv: iv)
            }
            if showsModel("lynch"), let lynch = models.lynch {
                lynchCard("Peter Lynch Fair Value", "equal.circle", .cyan, lynch, modelKey: "lynch", iv: iv)
            }
            if showsModel("epv"), let epv = models.epv {
                epvCard("Earnings Power Value (EPV Floor)", "anchor", .blue, epv, modelKey: "epv", iv: iv)
            }
        }
    }

    /// The three valuation summary cards (intrinsic value / current price / margin of safety).
    @ViewBuilder
    private func valuationSummaryCards(_ iv: IntrinsicValueResponse) -> some View {
        let hasCustom = blendedResult.hasAnyCustom
        let activeAvg = hasCustom ? blendedResult.customAverage : iv.averageIntrinsicValue
        let activeMos = hasCustom ? (blendedResult.customMarginOfSafety ?? 0) : (iv.marginOfSafetyPct ?? 0)
        let hasValue = activeAvg != nil

        let intrinsic = valuationCard(
            label: iv.status == .nav ? "Net Asset Value" : (hasCustom ? "Custom Blended Value" : "Blended Intrinsic Value"),
            value: hasValue ? Fmt.currency(activeAvg, code: nativeCur) : "Not valued",
            valueColor: hasCustom ? .orange : (hasValue ? .indigo : .secondary),
            tint: hasCustom ? Color.orange.opacity(0.08) : nil
        ) {
            if hasCustom, let defAvg = iv.averageIntrinsicValue, let actAvg = activeAvg, defAvg > 0 {
                let diff = ((actAvg - defAvg) / defAvg) * 100
                Text("Default: \(Fmt.currency(defAvg, code: nativeCur)) (\(Fmt.percent(diff, includeSign: true)))")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(diff >= 0 ? Color.green : Color.red)
                    .multilineTextAlignment(.center)
            }
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
            if hasValue, !hasCustom, let confidence = iv.valuationConfidence {
                ConfidenceMeter(confidence: confidence)
            }
        }

        let current = valuationCard(
            label: "Current Price",
            value: Fmt.currency(iv.currentPrice, code: nativeCur),
            valueColor: .primary
        ) { EmptyView() }

        let safety = valuationCard(
            label: "Margin of Safety",
            value: hasValue ? Fmt.percent(activeMos, includeSign: true) : "—",
            valueColor: hasValue ? (activeMos >= 0 ? .green : .red) : .secondary,
            tint: hasValue ? (activeMos >= 0 ? Color.green.opacity(0.1) : Color.red.opacity(0.1)) : nil
        ) {
            if hasCustom, let origMos = iv.marginOfSafetyPct {
                Text("Default MOS: \(Fmt.percent(origMos, includeSign: true))")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
        }

        if hSizeClass == .compact {
            VStack(spacing: 12) { intrinsic; current; safety }
        } else {
            HStack(spacing: 16) { intrinsic; current; safety }
        }
    }

    private func valuationNoteTitle(_ iv: IntrinsicValueResponse) -> String {
        switch iv.status {
        case .noModel:        return "Cannot be valued"
        case .ineligible:     return "Not eligible for valuation"
        case .clamped:        return "Output outside credible range"
        case .lowConfidence:  return "Models disagree"
        default:              return "Valuation note"
        }
    }

    private func valuationCard<Sub: View>(
        label: String,
        value: String,
        valueColor: Color,
        tint: Color? = nil,
        @ViewBuilder sub: () -> Sub
    ) -> some View {
        VStack(spacing: 8) {
            Text(label).font(.caption2.weight(.medium)).foregroundStyle(.secondary).textCase(.uppercase)
                .multilineTextAlignment(.center)
            Text(value).font(.system(size: 32, weight: .bold)).foregroundStyle(valueColor)
                .lineLimit(1).minimumScaleFactor(0.5)
            sub()
        }
        .frame(maxWidth: .infinity)
        .padding(hSizeClass == .compact ? 16 : 24)
        .background {
            if let tint {
                RoundedRectangle(cornerRadius: 16).fill(tint)
            } else {
                RoundedRectangle(cornerRadius: 16).fill(Color.cardBg)
            }
        }
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.cardBorder, lineWidth: 1))
    }

    private func paramRow(_ label: String, _ val: String, _ isNote: Bool = false, isCustom: Bool = false, defVal: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                SectionLabel(title: label)
                if isCustom {
                    Circle().fill(Color.orange).frame(width: 5, height: 5)
                }
            }
            Text(val).font(isNote ? .caption : .subheadline.weight(.semibold))
                .foregroundStyle(isCustom ? Color.orange : (isNote ? Color.secondary : .primary))
            if isCustom, let defVal {
                Text("Default: \(defVal)")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Grid columns shared by every model card's parameter grid.
    private var paramColumns: [GridItem] {
        hSizeClass == .compact
            ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
            : [GridItem(.adaptive(minimum: 150), spacing: 24)]
    }

    /// Wraps a card's parameter rows in the shared card chrome. `details` stays a
    /// closure so its view tree is built inside `ValuationModelCard.body` instead
    /// of being inlined into this view's body.
    private func modelCard<Details: View>(
        _ title: String,
        _ icon: String,
        _ color: Color,
        _ m: IntrinsicValueResponse.Model,
        modelKey: String,
        iv: IntrinsicValueResponse,
        primaryBadge: String? = nil,
        spacing: CGFloat,
        @ViewBuilder details: @escaping () -> Details
    ) -> ValuationModelCard<Details> {
        ValuationModelCard(
            title: title,
            icon: icon,
            color: color,
            modelKey: modelKey,
            model: m,
            primaryBadge: primaryBadge,
            spacing: spacing,
            nativeCur: nativeCur,
            currentPrice: iv.currentPrice,
            customModelValue: blendedResult.customModelValues[modelKey],
            customOverrides: $customOverrides,
            editingModelKeys: $editingModelKeys,
            expandedHistogramKeys: $expandedHistogramKeys,
            selectedDistributionItem: $selectedDistributionItem,
            details: details
        )
    }


    private func dcfCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, primaryBadge: "Primary", spacing: 20) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let dr = overrides["discount_rate"] ?? p["discount_rate"]?.doubleValue ?? 0
                    paramRow("Discount Rate (WACC)", Fmt.percent(dr * 100.0), isCustom: overrides["discount_rate"] != nil, defVal: Fmt.percent((p["discount_rate"]?.doubleValue ?? 0) * 100.0))

                    let gr = overrides["growth_rate"] ?? p["growth_rate"]?.doubleValue ?? 0
                    paramRow("Growth Rate", Fmt.percent(gr * 100.0), isCustom: overrides["growth_rate"] != nil, defVal: Fmt.percent((p["growth_rate"]?.doubleValue ?? 0) * 100.0))

                    if let v = p["applied_growth"]?.doubleValue { paramRow("Applied Growth", Fmt.percent(v * 100.0)) }

                    let tgr = overrides["terminal_growth_rate"] ?? p["terminal_growth_rate"]?.doubleValue ?? 0
                    paramRow("Terminal Growth", Fmt.percent(tgr * 100.0), isCustom: overrides["terminal_growth_rate"] != nil, defVal: Fmt.percent((p["terminal_growth_rate"]?.doubleValue ?? 0) * 100.0))

                    let py = overrides["projection_years"] ?? p["projection_years"]?.doubleValue ?? 10
                    paramRow("Projection Years", "\(Int(py))", isCustom: overrides["projection_years"] != nil, defVal: "\(Int(p["projection_years"]?.doubleValue ?? 10))")

                    let bf = overrides["base_fcf"] ?? p["base_fcf"]?.doubleValue ?? 0
                    paramRow("Base FCF", Fmt.compact(bf, code: nativeCur), isCustom: overrides["base_fcf"] != nil, defVal: Fmt.compact(p["base_fcf"]?.doubleValue ?? 0, code: nativeCur))

                    if let v = p["fcf_margin"]?.doubleValue { paramRow("Est. FCF Margin", Fmt.percent(v * 100.0)) }
                }
                if let n = p["note"]?.stringValue {
                    VStack(alignment: .leading, spacing: 6) {
                        SectionLabel(title: "Note")
                        Text(n).font(.caption).foregroundStyle(.secondary)
                    }
                    .padding(.top, 8)
                }
                limitationCallout("Cash-generative companies with steady, predictable Free Cash Flow (Operating Cash Flow minus Capital Expenditures).",
                                  "Highly sensitive to growth and discount rate (WACC) inputs; unsuitable for cyclical, negative-FCF, or lumpy CapEx businesses.")
            }
        }
    }

    private func grahamMathBlock(_ p: [String: JSONValue]?, overrides: [String: Double]) -> some View {
        let y = overrides["bond_yield_proxy"] ?? p?["bond_yield_proxy"]?.doubleValue ?? 4.5
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
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 20) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let eps = overrides["eps"] ?? p["eps"]?.doubleValue ?? 0
                    paramRow("Trailing EPS", Fmt.number(eps, fractionDigits: 2), isCustom: overrides["eps"] != nil, defVal: Fmt.number(p["eps"]?.doubleValue, fractionDigits: 2))

                    let gr = overrides["growth_rate_pct"] ?? p["growth_rate_pct"]?.doubleValue ?? 0
                    paramRow("Growth Rate (G)", "\(Fmt.number(gr, fractionDigits: 2))%", isCustom: overrides["growth_rate_pct"] != nil, defVal: "\(Fmt.number(p["growth_rate_pct"]?.doubleValue, fractionDigits: 2))%")

                    if let v = p["applied_growth_pct"]?.doubleValue { paramRow("Applied Growth", "\(Fmt.number(v, fractionDigits: 2))%") }

                    let y = overrides["bond_yield_proxy"] ?? p["bond_yield_proxy"]?.doubleValue ?? 4.5
                    paramRow("Bond Yield (Y)", "\(Fmt.number(y, fractionDigits: 2))%", isCustom: overrides["bond_yield_proxy"] != nil, defVal: "\(Fmt.number(p["bond_yield_proxy"]?.doubleValue, fractionDigits: 2))%")
                }
                if let n = p["note"]?.stringValue {
                    VStack(alignment: .leading, spacing: 6) {
                        SectionLabel(title: "Note")
                        Text(n).font(.caption).foregroundStyle(.secondary)
                    }
                    .padding(.top, 8)
                }
                grahamMathBlock(p, overrides: overrides)
                    .padding(.top, 16)
                limitationCallout("Defensive value screening comparing EPS and moderate growth against prevailing AAA corporate bond yield opportunity cost.",
                                  "Formula multiplier is aggressive if growth inputs are elevated; relies on normalized EPS and requires stability growth caps.")
            }
        }
    }

    private func ddmCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 20) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let div = overrides["base_dividend"] ?? p["base_dividend"]?.doubleValue ?? 0
                    paramRow("Base Dividend", Fmt.currency(div, code: nativeCur), isCustom: overrides["base_dividend"] != nil, defVal: Fmt.currency(p["base_dividend"]?.doubleValue, code: nativeCur))

                    if let v = p["dividend_yield_pct"]?.doubleValue { paramRow("Dividend Yield", "\(Fmt.number(v, fractionDigits: 2))%") }

                    let gr = overrides["growth_rate"] ?? p["growth_rate"]?.doubleValue ?? 0
                    paramRow("Dividend Growth Rate", Fmt.percent(gr * 100.0), isCustom: overrides["growth_rate"] != nil, defVal: Fmt.percent((p["growth_rate"]?.doubleValue ?? 0) * 100.0))

                    let dr = overrides["discount_rate"] ?? p["discount_rate"]?.doubleValue ?? 0
                    paramRow("Cost of Equity (CAPM)", Fmt.percent(dr * 100.0), isCustom: overrides["discount_rate"] != nil, defVal: Fmt.percent((p["discount_rate"]?.doubleValue ?? 0) * 100.0))

                    if let v = p["payout_ratio"]?.doubleValue { paramRow("Payout Ratio", Fmt.percent(v * 100.0)) }

                    let tgr = overrides["terminal_growth_rate"] ?? p["terminal_growth_rate"]?.doubleValue ?? 0
                    paramRow("Terminal Growth", Fmt.percent(tgr * 100.0), isCustom: overrides["terminal_growth_rate"] != nil, defVal: Fmt.percent((p["terminal_growth_rate"]?.doubleValue ?? 0) * 100.0))
                }
                if let n = p["note"]?.stringValue {
                    VStack(alignment: .leading, spacing: 6) {
                        SectionLabel(title: "Note")
                        Text(n).font(.caption).foregroundStyle(.secondary)
                    }
                    .padding(.top, 8)
                }
                limitationCallout("Mature dividend payers and utilities with long track records of consistent dividend growth and sustainable payout ratios (<100%).",
                                  "Only reflects value returned as direct dividends; entirely unsuited for non-dividend payers and ignores share repurchases or cash retained.")
            }
        }
    }

    private func lynchCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let eps = overrides["eps"] ?? p["eps"]?.doubleValue ?? 0
                    paramRow("Trailing EPS", Fmt.number(eps, fractionDigits: 2), isCustom: overrides["eps"] != nil, defVal: Fmt.number(p["eps"]?.doubleValue, fractionDigits: 2))

                    let gr = overrides["growth_rate_pct"] ?? p["growth_rate_pct"]?.doubleValue ?? 0
                    let dy = overrides["dividend_yield_pct"] ?? p["dividend_yield_pct"]?.doubleValue ?? 0
                    paramRow("Growth + Yield", "\(Fmt.number(gr + dy, fractionDigits: 1))%", isCustom: (overrides["growth_rate_pct"] != nil || overrides["dividend_yield_pct"] != nil), defVal: "\(Fmt.number((p["growth_rate_pct"]?.doubleValue ?? 0) + (p["dividend_yield_pct"]?.doubleValue ?? 0), fractionDigits: 1))%")

                    let mult = min(max(gr + dy, 5.0), 25.0)
                    paramRow("Fair P/E (PEG=1)", "\(Fmt.number(mult, fractionDigits: 1))x")
                    paramRow("Principle", "Peter Lynch Fair Value")
                }
                limitationCallout("Fast rule-of-thumb valuation equating fair P/E multiple to expected earnings growth rate plus dividend yield (PEG=1.0 benchmark).",
                                  "Heuristic benchmark that does not account for cost of capital, multi-stage growth decay, or balance sheet solvency.")
            }
        }
    }

    private func dcfoCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let cfo = overrides["cfo_per_share"] ?? p["cfo_per_share"]?.doubleValue ?? 0
                    paramRow("Base CFO / Share", Fmt.currency(cfo, code: nativeCur), isCustom: overrides["cfo_per_share"] != nil, defVal: Fmt.currency(p["cfo_per_share"]?.doubleValue, code: nativeCur))

                    let gr = overrides["growth_rate"] ?? p["growth_rate"]?.doubleValue ?? 0
                    paramRow("CFO Growth Rate", Fmt.percent(gr * 100.0), isCustom: overrides["growth_rate"] != nil, defVal: Fmt.percent((p["growth_rate"]?.doubleValue ?? 0) * 100.0))

                    let dr = overrides["discount_rate"] ?? p["discount_rate"]?.doubleValue ?? 0
                    paramRow("Discount Rate (WACC)", Fmt.percent(dr * 100.0), isCustom: overrides["discount_rate"] != nil, defVal: Fmt.percent((p["discount_rate"]?.doubleValue ?? 0) * 100.0))

                    let tgr = overrides["terminal_growth_rate"] ?? p["terminal_growth_rate"]?.doubleValue ?? 0
                    paramRow("Terminal Growth", Fmt.percent(tgr * 100.0), isCustom: overrides["terminal_growth_rate"] != nil, defVal: Fmt.percent((p["terminal_growth_rate"]?.doubleValue ?? 0) * 100.0))
                }
                limitationCallout("Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics).",
                                  "Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital.")
            }
        }
    }

    private func dniCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let eps = overrides["base_eps"] ?? p["base_eps"]?.doubleValue ?? 0
                    paramRow("Base EPS", Fmt.number(eps, fractionDigits: 2), isCustom: overrides["base_eps"] != nil, defVal: Fmt.number(p["base_eps"]?.doubleValue, fractionDigits: 2))

                    let gr = overrides["growth_rate"] ?? p["growth_rate"]?.doubleValue ?? 0
                    paramRow("Net Income Growth", Fmt.percent(gr * 100.0), isCustom: overrides["growth_rate"] != nil, defVal: Fmt.percent((p["growth_rate"]?.doubleValue ?? 0) * 100.0))

                    let dr = overrides["discount_rate"] ?? p["discount_rate"]?.doubleValue ?? 0
                    paramRow("Cost of Equity", Fmt.percent(dr * 100.0), isCustom: overrides["discount_rate"] != nil, defVal: Fmt.percent((p["discount_rate"]?.doubleValue ?? 0) * 100.0))

                    let tgr = overrides["terminal_growth_rate"] ?? p["terminal_growth_rate"]?.doubleValue ?? 0
                    paramRow("Terminal Growth", Fmt.percent(tgr * 100.0), isCustom: overrides["terminal_growth_rate"] != nil, defVal: Fmt.percent((p["terminal_growth_rate"]?.doubleValue ?? 0) * 100.0))
                }
                limitationCallout("Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital.",
                                  "Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag.")
            }
        }
    }

    private func meanPeCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let eps = overrides["eps"] ?? p["eps"]?.doubleValue ?? 0
                    paramRow("TTM EPS", Fmt.number(eps, fractionDigits: 2), isCustom: overrides["eps"] != nil, defVal: Fmt.number(p["eps"]?.doubleValue, fractionDigits: 2))

                    let pe = overrides["applied_pe"] ?? p["applied_pe"]?.doubleValue ?? 0
                    paramRow("Mean P/E Multiple", "\(Fmt.number(pe, fractionDigits: 1))x", isCustom: overrides["applied_pe"] != nil, defVal: "\(Fmt.number(p["applied_pe"]?.doubleValue, fractionDigits: 1))x")

                    if let v = p["pe_source"]?.stringValue { paramRow("Source", v) }
                    tradedRangeRow(p, fractionDigits: 1)
                }
                limitationCallout("Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.",
                                  "Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.")
            }
        }
    }

    private func pegCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let eps = overrides["eps"] ?? p["eps"]?.doubleValue ?? 0
                    paramRow("TTM EPS", Fmt.number(eps, fractionDigits: 2), isCustom: overrides["eps"] != nil, defVal: Fmt.number(p["eps"]?.doubleValue, fractionDigits: 2))

                    let gr = overrides["growth_rate_pct"] ?? p["growth_rate_pct"]?.doubleValue ?? 0
                    paramRow("Growth Rate", "\(Fmt.number(gr, fractionDigits: 1))%", isCustom: overrides["growth_rate_pct"] != nil, defVal: "\(Fmt.number(p["growth_rate_pct"]?.doubleValue, fractionDigits: 1))%")

                    let peg = overrides["target_peg"] ?? p["target_peg"]?.doubleValue ?? 1.0
                    paramRow("Target PEG", "\(Fmt.number(peg, fractionDigits: 1))x", isCustom: overrides["target_peg"] != nil, defVal: "\(Fmt.number(p["target_peg"]?.doubleValue, fractionDigits: 1))x")

                    let mult = peg * (gr + (p["dividend_yield_pct"]?.doubleValue ?? 0))
                    paramRow("Fair P/E Multiplier", "\(Fmt.number(mult, fractionDigits: 1))x")
                }
                limitationCallout("Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple.",
                                  "Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden.")
            }
        }
    }

    private func meanPbCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let bvps = overrides["book_value_per_share"] ?? p["book_value_per_share"]?.doubleValue ?? 0
                    paramRow("Book Value / Share", Fmt.currency(bvps, code: nativeCur), isCustom: overrides["book_value_per_share"] != nil, defVal: Fmt.currency(p["book_value_per_share"]?.doubleValue, code: nativeCur))

                    let pb = overrides["applied_pb"] ?? p["applied_pb"]?.doubleValue ?? 0
                    paramRow("Applied P/B Target", "\(Fmt.number(pb, fractionDigits: 2))x", isCustom: overrides["applied_pb"] != nil, defVal: "\(Fmt.number(p["applied_pb"]?.doubleValue, fractionDigits: 2))x")

                    if let v = p["pb_source"]?.stringValue { paramRow("Benchmark Source", v) }
                    tradedRangeRow(p, fractionDigits: 2)
                }
                limitationCallout("Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.",
                                  "Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.")
            }
        }
    }

    private func meanPsCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let sps = overrides["sales_per_share"] ?? p["sales_per_share"]?.doubleValue ?? 0
                    paramRow("Sales / Share", Fmt.currency(sps, code: nativeCur), isCustom: overrides["sales_per_share"] != nil, defVal: Fmt.currency(p["sales_per_share"]?.doubleValue, code: nativeCur))

                    let ps = overrides["applied_ps"] ?? p["applied_ps"]?.doubleValue ?? 0
                    paramRow("Mean P/S Multiple", "\(Fmt.number(ps, fractionDigits: 2))x", isCustom: overrides["applied_ps"] != nil, defVal: "\(Fmt.number(p["applied_ps"]?.doubleValue, fractionDigits: 2))x")

                    if let v = p["ps_source"]?.stringValue { paramRow("Multiple Source", v) }
                    tradedRangeRow(p, fractionDigits: 2)
                }
                limitationCallout("Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.",
                                  "Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.")
            }
        }
    }

    private func psgCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let sps = overrides["sales_per_share"] ?? p["sales_per_share"]?.doubleValue ?? 0
                    paramRow("Sales / Share", Fmt.currency(sps, code: nativeCur), isCustom: overrides["sales_per_share"] != nil, defVal: Fmt.currency(p["sales_per_share"]?.doubleValue, code: nativeCur))

                    let gr = overrides["revenue_growth_pct"] ?? p["applied_growth_pct"]?.doubleValue ?? 0
                    paramRow("Revenue Growth", "\(Fmt.number(gr, fractionDigits: 1))%", isCustom: overrides["revenue_growth_pct"] != nil, defVal: "\(Fmt.number(p["applied_growth_pct"]?.doubleValue, fractionDigits: 1))%")

                    let gm = overrides["gross_margin_pct"] ?? p["gross_margin_pct"]?.doubleValue ?? 0
                    paramRow("Gross Margin", "\(Fmt.number(gm, fractionDigits: 1))%", isCustom: overrides["gross_margin_pct"] != nil, defVal: "\(Fmt.number(p["gross_margin_pct"]?.doubleValue, fractionDigits: 1))%")

                    if let v = p["fair_ps_multiplier"]?.doubleValue { paramRow("Fair P/S Multiplier", "\(Fmt.number(v, fractionDigits: 2))x") }
                }
                limitationCallout("High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality.",
                                  "Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates.")
            }
        }
    }

    private func epvCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let overrides = customOverrides[modelKey] ?? [:]

        return modelCard(title, icon, color, m, modelKey: modelKey, iv: iv, spacing: 16) {
            if let p = m.parameters {
                LazyVGrid(columns: paramColumns, spacing: 24) {
                    let ebit = overrides["normalized_ebit"] ?? p["normalized_ebit"]?.doubleValue ?? 0
                    let tr = overrides["tax_rate"] ?? p["tax_rate"]?.doubleValue ?? 0.21
                    let nopat = ebit * (1.0 - tr)
                    paramRow("Normalized NOPAT", Fmt.currency(nopat, code: nativeCur), isCustom: (overrides["normalized_ebit"] != nil || overrides["tax_rate"] != nil), defVal: Fmt.currency(p["nopat"]?.doubleValue, code: nativeCur))

                    let dr = overrides["discount_rate"] ?? p["discount_rate"]?.doubleValue ?? 0
                    paramRow("Cost of Capital", Fmt.percent(dr * 100.0), isCustom: overrides["discount_rate"] != nil, defVal: Fmt.percent((p["discount_rate"]?.doubleValue ?? 0) * 100.0))

                    let nc = overrides["net_cash"] ?? p["net_cash"]?.doubleValue ?? 0
                    paramRow("Net Cash Added", Fmt.currency(nc, code: nativeCur), isCustom: overrides["net_cash"] != nil, defVal: Fmt.currency(p["net_cash"]?.doubleValue, code: nativeCur))

                    paramRow("Growth Assumption", "0.0% (Zero Growth)")
                }
                limitationCallout("Conservative valuation of normalized sustainable operating earnings in perpetuity assuming zero future growth.",
                                  "Strictly a no-growth baseline floor; gives zero credit to value-accretive capital reinvestment or growth opportunities.")
            }
        }
    }

    private func limitationCallout(_ bestSuitedFor: String, _ keyCaveats: String, collapsible: Bool = true) -> some View {
        LimitationCallout(bestSuitedFor: bestSuitedFor, keyCaveats: keyCaveats, collapsible: collapsible)
    }

    private var bestFitBadge: some View {
        HStack(spacing: 6) {
            Image(systemName: "sparkles").foregroundStyle(.yellow)
            Text("Best-Fit Valuation Method")
                .font(.caption.weight(.bold))
                .foregroundStyle(.indigo)
                .fixedSize(horizontal: true, vertical: false)
        }
        .padding(.horizontal, 10).padding(.vertical, 4)
        .background(Color.indigo.opacity(0.15), in: Capsule())
    }

    @ViewBuilder
    private func bestFitValue(_ rec: IntrinsicValueResponse.RecommendedMethod, currentPrice: Double?) -> some View {
        if let val = rec.intrinsicValue {
            HStack(spacing: 8) {
                Text(Fmt.currency(val, code: nativeCur))
                    .font(.headline.weight(.bold))
                    .foregroundStyle(.indigo)
                if let cp = currentPrice, cp > 0 {
                    let up = ((val - cp) / cp) * 100
                    Text(Fmt.percent(up, includeSign: true))
                        .font(.caption.weight(.bold))
                        .foregroundStyle(up >= 0 ? Color.green : Color.red)
                }
            }
            .fixedSize(horizontal: true, vertical: false)
        }
    }

    private func recommendedMethodBanner(_ rec: IntrinsicValueResponse.RecommendedMethod, currentPrice: Double?) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Side by side only where the badge keeps its one-line width: on a
            // phone the value pushed "Best-Fit Valuation Method" into a three-line
            // capsule. `ViewThatFits` measures the row at that ideal width and
            // falls back to stacking the value under the badge.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) {
                    bestFitBadge
                    Spacer(minLength: 8)
                    bestFitValue(rec, currentPrice: currentPrice)
                }
                VStack(alignment: .leading, spacing: 8) {
                    bestFitBadge
                    bestFitValue(rec, currentPrice: currentPrice)
                }
            }

            Text(rec.name ?? "Valuation Method")
                .font(.title3.weight(.bold))
                .fixedSize(horizontal: false, vertical: true)
            if let r = rec.rationale {
                Text(r).font(.subheadline).foregroundStyle(.secondary)
            }

            if let suited = rec.bestSuitedFor ?? rec.whenToUse, let caveats = rec.keyCaveats ?? rec.keyLimitation {
                limitationCallout(suited, caveats, collapsible: false)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.indigo.opacity(0.05), in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.indigo.opacity(0.2), lineWidth: 1))
    }

    private func valuationSpectrumSection(_ iv: IntrinsicValueResponse) -> some View {
        ValuationSpectrumSection(
            iv: iv,
            blendedResult: blendedResult,
            customOverrides: customOverrides,
            nativeCur: nativeCur
        )
    }


    /// The interquartile range of the multiples this company has actually traded
    /// at. The median alone reads as an opinion; "usually 12.4x-19.1x over 15
    /// years" is the record it came from, and it is what tells the reader
    /// whether today's multiple is unusual or ordinary.
    @ViewBuilder
    private func tradedRangeRow(
        _ p: [String: JSONValue], fractionDigits: Int
    ) -> some View {
        if let low = p["multiple_p25"]?.doubleValue,
           let high = p["multiple_p75"]?.doubleValue,
           let n = p["multiple_observations"]?.doubleValue, n > 0 {
            paramRow(
                "Usually Traded At",
                "\(Fmt.number(low, fractionDigits: fractionDigits))x – "
                    + "\(Fmt.number(high, fractionDigits: fractionDigits))x (\(Int(n))y)"
            )
        }
    }

    private func card<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title).font(.headline)
            content()
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }
}


/// How much the backend stands behind the number, as a bar. Confidence is
/// continuous — the models' own Monte Carlo bands, how far apart they landed,
/// and how many of them there were — so it reads as a level rather than the old
/// pass/fail that looked fine at 99% disagreement and alarming at 101%.
private struct ConfidenceMeter: View {
    let confidence: Double

    private var pct: Double { min(max(confidence, 0), 1) }
    private var tint: Color { pct >= 0.66 ? .green : (pct >= 0.4 ? .orange : .red) }

    var body: some View {
        VStack(spacing: 3) {
            HStack {
                Text("Confidence").font(.caption2.weight(.semibold)).textCase(.uppercase)
                Spacer(minLength: 6)
                Text("\(Int((pct * 100).rounded()))%").font(.caption2.weight(.bold)).monospacedDigit()
            }
            .foregroundStyle(.secondary)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.secondary.opacity(0.2))
                    Capsule().fill(tint).frame(width: max(2, geo.size.width * pct))
                }
            }
            .frame(height: 5)
        }
        .frame(maxWidth: 180)
        .padding(.top, 4)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Valuation confidence \(Int((pct * 100).rounded())) percent")
    }
}

/// The composition of the blend: which models were held out and why, and the
/// floors that travel beside the estimate rather than inside it. A model can be
/// excluded because it does not describe this business (a DCF of a bank) or
/// because it prices only part of it (a DDM of a company that retains most of
/// its earnings) — in both cases the number is still worth seeing, just not
/// worth averaging in.
///
/// A separate `View` rather than more lines in the tab's `body`: a single large
/// `body` is what blew the stack on iPhone.
private struct BlendCompositionCard: View {
    let iv: IntrinsicValueResponse
    let currencyCode: String

    private var exclusions: [(String, String)] {
        (iv.blendExclusions ?? [:]).sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }

    private var floors: [(String, Double, String)] {
        var out: [(String, Double, String)] = []
        if let epv = iv.earningsPowerFloor {
            out.append(("Earnings power floor", epv, "Current earnings, no growth"))
        }
        if let ddm = iv.dividendDiscountFloor {
            out.append(("Dividend-only value", ddm, "What the dividend stream alone is worth"))
        }
        return out
    }

    private var profileLine: String? {
        switch iv.blendProfile {
        case "financial": return "Financial — valued on discounted net income, not free cash flow."
        case "reit": return "REIT — valued on cash from operations and the distribution, since net income is charged with non-cash depreciation."
        case "operating": return "Operating company — valued on discounted free cash flow."
        default: return nil
        }
    }

    var body: some View {
        if exclusions.isEmpty && floors.isEmpty && profileLine == nil {
            EmptyView()
        } else {
            VStack(alignment: .leading, spacing: 12) {
                Label("How this blend was built", systemImage: "square.stack.3d.up")
                    .font(.caption.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)

                if let profileLine {
                    Text(profileLine).font(.subheadline).foregroundStyle(.secondary)
                }

                if !floors.isEmpty {
                    HStack(spacing: 10) {
                        ForEach(floors, id: \.0) { label, value, hint in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(label).font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                                Text(Fmt.currency(value, code: currencyCode)).font(.callout.weight(.bold)).monospacedDigit()
                                Text(hint).font(.caption2).foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                            .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                        }
                    }
                }

                ForEach(exclusions, id: \.0) { key, reason in
                    HStack(alignment: .top, spacing: 6) {
                        Text(key.uppercased()).font(.caption2.weight(.black)).foregroundStyle(.primary.opacity(0.7))
                        Text("held out — \(reason)").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
        }
    }
}

/// "Best Suited For" / "Key Caveats" note for a valuation model. Collapsed to a single
/// tappable header by default so the method cards stay scannable.
private struct LimitationCallout: View {
    let bestSuitedFor: String
    let keyCaveats: String
    var collapsible: Bool = true

    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if collapsible {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "info.circle").font(.system(size: 9, weight: .bold))
                        Text("Best Suited For & Key Caveats".uppercased())
                            .font(.caption2.weight(.bold))
                            .tracking(0.5)
                        Spacer(minLength: 8)
                        Image(systemName: "chevron.down")
                            .font(.system(size: 9, weight: .bold))
                            .rotationEffect(.degrees(expanded ? 180 : 0))
                    }
                    .foregroundStyle(.secondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Best suited for and key caveats")
                .accessibilityHint(expanded ? "Collapse" : "Expand")
            }
            if expanded || !collapsible {
                limitationLine("checkmark.seal.fill", .green, "Best Suited For", bestSuitedFor)
                Divider().opacity(0.5)
                limitationLine("exclamationmark.triangle.fill", .orange, "Key Caveats", keyCaveats)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private func limitationLine(_ icon: String, _ tint: Color, _ label: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                Image(systemName: icon).font(.system(size: 9, weight: .bold)).foregroundStyle(tint)
                Text(label.uppercased())
                    .font(.caption2.weight(.bold))
                    .tracking(0.5)
                    .foregroundStyle(tint)
            }
            Text(text)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// One valuation model card: the shared chrome (header, parameter editor, Monte
/// Carlo row) wrapped around a caller-supplied parameter grid.
///
/// This is a separate `View` rather than a set of methods on
/// `StockValuationTabView` so that each card is built in its own stack frame.
/// With all twelve cards inlined into `StockValuationTabView.body`, that single
/// frame grew large enough to overflow the main thread's stack.
struct ValuationModelCard<Details: View>: View {
    let title: String
    let icon: String
    let color: Color
    let modelKey: String
    let model: IntrinsicValueResponse.Model
    let primaryBadge: String?
    let spacing: CGFloat
    let nativeCur: String
    let currentPrice: Double?
    let customModelValue: Double?
    @Binding var customOverrides: [String: [String: Double]]
    @Binding var editingModelKeys: Set<String>
    @Binding var expandedHistogramKeys: Set<String>
    @Binding var selectedDistributionItem: DistributionModelItem?
    let details: () -> Details

    @Environment(\.horizontalSizeClass) private var hSizeClass

    var body: some View {
        VStack(alignment: .leading, spacing: spacing) {
            modelCardHeader()

            if let e = model.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if editingModelKeys.contains(modelKey) {
                    modelParameterEditor()
                } else {
                    details()
                }

                probabilisticScenariosRow()
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func toggleHistogram(_ key: String) {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
            if expandedHistogramKeys.contains(key) {
                expandedHistogramKeys.remove(key)
            } else {
                expandedHistogramKeys.insert(key)
            }
        }
    }

    private func toggleEditing(_ key: String) {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            if editingModelKeys.contains(key) {
                editingModelKeys.remove(key)
            } else {
                editingModelKeys.insert(key)
            }
        }
    }

    private func scenarioPill(label: String, value: Double?, color: Color) -> some View {
        Button {
            toggleHistogram(modelKey)
        } label: {
            VStack(spacing: 4) {
                Text(label)
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(color)
                    .textCase(.uppercase)
                    .lineLimit(1)
                Text(Fmt.currency(value, code: nativeCur))
                    .font(.subheadline.weight(.bold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .padding(.horizontal, 6)
            .background(color.opacity(0.06), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(color.opacity(0.2), lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        #if os(macOS)
        .onHover { inside in
            if inside {
                NSCursor.pointingHand.push()
            } else {
                NSCursor.pop()
            }
        }
        #endif
    }

    @ViewBuilder
    private func probabilisticScenariosRow() -> some View {
        if let mc = model.mc, mc.bear != nil || mc.base != nil || mc.bull != nil {
            let isExpanded = expandedHistogramKeys.contains(modelKey)
            VStack(alignment: .leading, spacing: 10) {
                // On an iPhone the header and the toggle don't fit on one
                // line, so fall back to stacking them with a wrapping title
                // rather than truncating "…Scenarios (Monte Carlo)" away.
                let histogramToggle = Button {
                    toggleHistogram(modelKey)
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isExpanded ? "chevron.up.circle.fill" : "chart.bar.xaxis")
                        Text(isExpanded ? "Hide Chart" : "Show Distribution")
                            .font(.system(size: 10, weight: .bold))
                    }
                    .foregroundStyle(Color.brandIndigo)
                    .fixedSize(horizontal: true, vertical: false)
                }
                .buttonStyle(.plain)

                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        SectionLabel(title: "Probabilistic Scenarios (Monte Carlo)")
                        Spacer(minLength: 8)
                        histogramToggle
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        SectionLabel(title: "Probabilistic Scenarios (Monte Carlo)", lineLimit: 2)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        histogramToggle
                    }
                }

                HStack(spacing: 8) {
                    scenarioPill(label: "Bear (10th)", value: mc.bear, color: .red)
                    scenarioPill(label: "Median (50th)", value: mc.base, color: Color.brandIndigo)
                    scenarioPill(label: "Bull (90th)", value: mc.bull, color: Color.brandEmerald)
                }

                if isExpanded, let hist = mc.histogram, !hist.isEmpty {
                    VStack(spacing: 8) {
                        HistogramChartView(hist: hist, mc: mc, currentPrice: currentPrice, currency: nativeCur)
                            .frame(height: 140)

                        HStack {
                            Text("Monte Carlo Simulation (10,000 iterations)")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button {
                                selectedDistributionItem = DistributionModelItem(
                                    key: modelKey,
                                    title: title,
                                    color: color,
                                    mc: mc,
                                    currentPrice: currentPrice,
                                    nativeCur: nativeCur
                                )
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                                    Text("Full Modal").font(.system(size: 10, weight: .semibold))
                                }
                                .foregroundStyle(Color.brandIndigo)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.top, 6)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .padding(.top, 12)
        }
    }

    /// The model's identity: icon, name, and any status badges. On compact
    /// widths this gets a full-width row of its own so long names like
    /// "Discounted Cash from Operations" wrap between words instead of being
    /// squeezed to a few characters per line beside the value pill.
    @ViewBuilder
    private func modelCardTitleRow(isCustom: Bool) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: icon).foregroundStyle(color)
            Text(title)
                .font(.headline)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            if let p = primaryBadge {
                Text(p).font(.system(size: 9, weight: .bold)).foregroundStyle(.green)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Color.green.opacity(0.15), in: Capsule())
                    .fixedSize()
            }
            if isCustom {
                Text("CUSTOM")
                    .font(.system(size: 8, weight: .black))
                    .foregroundStyle(.orange)
                    .padding(.horizontal, 5).padding(.vertical, 2)
                    .background(Color.orange.opacity(0.15), in: Capsule())
                    .fixedSize()
            }
        }
    }

    @ViewBuilder
    private func modelCardValuePill(isCustom: Bool, activeVal: Double?) -> some View {
        if model.error == nil, let v = activeVal {
            Text(Fmt.currency(v, code: nativeCur))
                .font(.subheadline.weight(.bold))
                .foregroundStyle(isCustom ? Color.orange : color)
                .lineLimit(1)
                .fixedSize()
                .padding(.horizontal, 10).padding(.vertical, 4)
                .background((isCustom ? Color.orange : color).opacity(0.15), in: Capsule())
        }
    }

    @ViewBuilder
    private func modelCardActions(isCustom: Bool, isEditing: Bool) -> some View {
        Button {
            toggleEditing(modelKey)
        } label: {
            HStack(spacing: 3) {
                Image(systemName: "slider.horizontal.3")
                Text(isEditing ? "Done" : "Edit")
            }
            .font(.system(size: 10, weight: .bold))
            .foregroundStyle(isEditing ? .white : Color.primary)
            .lineLimit(1)
            .fixedSize()
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(isEditing ? Color.indigo : Color.secondary.opacity(0.15), in: RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)

        if isCustom {
            Button {
                customOverrides.removeValue(forKey: modelKey)
            } label: {
                Image(systemName: "arrow.counterclockwise")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.secondary)
                    .padding(5)
                    .background(Color.secondary.opacity(0.12), in: Circle())
            }
            .buttonStyle(.plain)
            .help("Reset to default parameters")
        }
    }

    @ViewBuilder
    private func modelCardDefaultDelta(isCustom: Bool, defaultVal: Double?, activeVal: Double?) -> some View {
        if isCustom, let defVal = defaultVal, let actVal = activeVal, defVal > 0, abs(actVal - defVal) > 0.001 {
            let diff = ((actVal - defVal) / defVal) * 100
            Text("Def: \(Fmt.currency(defVal, code: nativeCur)) (\(Fmt.percent(diff, includeSign: true)))")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(diff >= 0 ? Color.green : Color.red)
                .lineLimit(1)
        }
    }

    @ViewBuilder
    private func modelCardHeader() -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]
        let defaultParams = model.parameters ?? [:]
        let isCustom = overrides.contains { (k, v) in
            if let def = defaultParams[k]?.doubleValue {
                return abs(v - def) > 1e-5
            }
            return true
        }
        let activeVal = isCustom ? customModelValue : model.intrinsicValue

        if hSizeClass == .compact {
            VStack(alignment: .leading, spacing: 10) {
                modelCardTitleRow(isCustom: isCustom)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 6) {
                    modelCardValuePill(isCustom: isCustom, activeVal: activeVal)
                    Spacer(minLength: 8)
                    modelCardActions(isCustom: isCustom, isEditing: isEditing)
                }
                modelCardDefaultDelta(isCustom: isCustom, defaultVal: model.intrinsicValue, activeVal: activeVal)
            }
        } else {
            HStack(alignment: .top, spacing: 12) {
                modelCardTitleRow(isCustom: isCustom)
                Spacer(minLength: 8)
                VStack(alignment: .trailing, spacing: 4) {
                    HStack(spacing: 6) {
                        modelCardValuePill(isCustom: isCustom, activeVal: activeVal)
                        modelCardActions(isCustom: isCustom, isEditing: isEditing)
                    }
                    modelCardDefaultDelta(isCustom: isCustom, defaultVal: model.intrinsicValue, activeVal: activeVal)
                }
                .fixedSize(horizontal: true, vertical: false)
            }
        }
    }

    @ViewBuilder
    private func modelParameterEditor() -> some View {
        if let configs = StockValuationCalculator.configs[modelKey] {
            let overrides = customOverrides[modelKey] ?? [:]
            let defaultParams = model.parameters ?? [:]
            let hasModifications = overrides.contains { (k, v) in
                if let def = defaultParams[k]?.doubleValue {
                    return abs(v - def) > 1e-5
                }
                return true
            }

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("Custom Parameters", systemImage: "slider.horizontal.3")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.indigo)
                    Spacer()
                    if hasModifications {
                        Button("Reset Card Defaults") {
                            customOverrides.removeValue(forKey: modelKey)
                        }
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.orange)
                        .buttonStyle(.plain)
                    }
                }
                .padding(.bottom, 2)

                let columns = hSizeClass == .compact
                    ? [GridItem(.flexible())]
                    : [GridItem(.flexible()), GridItem(.flexible())]

                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(configs) { cfg in
                        let rawDef = model.parameters?[cfg.key]?.doubleValue
                        let defVal = rawDef ?? 0.0
                        let customVal = overrides[cfg.key]
                        let isCustom = customVal != nil && rawDef != nil && abs((customVal ?? 0) - (rawDef ?? 0)) > 1e-5
                        let currentVal = customVal ?? defVal

                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(cfg.label).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                                Spacer()
                                if isCustom {
                                    Button("Revert") {
                                        var map = customOverrides[modelKey] ?? [:]
                                        map.removeValue(forKey: cfg.key)
                                        if map.isEmpty {
                                            customOverrides.removeValue(forKey: modelKey)
                                        } else {
                                            customOverrides[modelKey] = map
                                        }
                                    }
                                    .font(.system(size: 9, weight: .bold))
                                    .foregroundStyle(.orange)
                                    .buttonStyle(.plain)
                                }
                            }

                            HStack(spacing: 8) {
                                let displayVal = cfg.isPercent
                                    ? Fmt.percent(currentVal * 100.0)
                                    : (cfg.unit == .percent
                                        ? "\(Fmt.number(currentVal, fractionDigits: 1))%"
                                        : (cfg.unit == .multiple
                                            ? "\(Fmt.number(currentVal, fractionDigits: 1))x"
                                            : (cfg.unit == .years
                                                ? "\(Int(currentVal)) yrs"
                                                : (cfg.unit == .currency
                                                    ? Fmt.compact(currentVal, code: nativeCur)
                                                    : Fmt.number(currentVal, fractionDigits: 2)))))

                                Text(displayVal)
                                    .font(.subheadline.weight(.bold))
                                    .foregroundStyle(isCustom ? Color.orange : Color.primary)

                                Spacer()

                                Stepper(value: Binding(
                                    get: { currentVal },
                                    set: { newVal in
                                        var map = customOverrides[modelKey] ?? [:]
                                        if let def = rawDef, abs(newVal - def) < 1e-5 {
                                            map.removeValue(forKey: cfg.key)
                                        } else {
                                            map[cfg.key] = newVal
                                        }
                                        if map.isEmpty {
                                            customOverrides.removeValue(forKey: modelKey)
                                        } else {
                                            customOverrides[modelKey] = map
                                        }
                                    }
                                ), in: cfg.min...cfg.max, step: cfg.step) {
                                    EmptyView()
                                }
                                .labelsHidden()
                            }
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(isCustom ? Color.orange.opacity(0.08) : Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(isCustom ? Color.orange.opacity(0.5) : Color.clear, lineWidth: 1))

                            let displayDef = cfg.isPercent
                                ? Fmt.percent(defVal * 100.0)
                                : (cfg.unit == .percent
                                    ? "\(Fmt.number(defVal, fractionDigits: 1))%"
                                    : (cfg.unit == .multiple
                                        ? "\(Fmt.number(defVal, fractionDigits: 1))x"
                                        : (cfg.unit == .currency
                                            ? Fmt.compact(defVal, code: nativeCur)
                                            : Fmt.number(defVal, fractionDigits: 2))))

                            Text("Default: \(displayDef)")
                                .font(.system(size: 9))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(14)
            .background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.secondary.opacity(0.15), lineWidth: 1))
        }
    }
}

/// The cross-method valuation spectrum. Split out of `StockValuationTabView`
/// so its (large) view tree is built in its own stack frame.
private struct ValuationSpectrumSection: View {
    let iv: IntrinsicValueResponse
    let blendedResult: BlendedValuationResult
    let customOverrides: [String: [String: Double]]
    let nativeCur: String

    private struct ValuationSpectrumItem: Identifiable {
        let id: String
        let name: String
        let value: Double
        let defaultValue: Double?
        let isCustom: Bool
        let bear: Double?
        let bull: Double?
        let isRecommended: Bool
        let tint: Color
    }

    /// Title block and the undervalued tally.
    ///
    /// Side by side only where the title still fits on one line: on a phone the
    /// capsule squeezed "Valuation Comparison Spectrum" over three lines with the
    /// tally stranded beside it. `ViewThatFits` measures the row at its one-line
    /// ideal width and falls back to stacking.
    private func spectrumHeader(undervaluedCount: Int, total: Int, currentPrice: Double) -> some View {
        let tint = undervaluedCount > total / 2 ? Color.green : Color.orange
        let title = VStack(alignment: .leading, spacing: 4) {
            Label("Valuation Comparison Spectrum", systemImage: "chart.bar.xaxis")
                .font(.headline)
                .fixedSize(horizontal: false, vertical: true)
            Text("Cross-method valuation distribution compared against current price (\(Fmt.currency(currentPrice, code: nativeCur)))")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        let tally = Text("\(undervaluedCount)/\(total) Undervalued")
            .font(.caption2.weight(.bold))
            .foregroundStyle(tint)
            .fixedSize(horizontal: true, vertical: false)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(tint.opacity(0.12), in: Capsule())

        return ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 12) {
                title
                Spacer(minLength: 8)
                tally
            }
            VStack(alignment: .leading, spacing: 8) {
                title
                tally
            }
        }
    }

    var body: some View {
        let currentPrice = iv.currentPrice ?? 0
        let recKey = iv.recommendedMethod?.methodKey
        let customValues = blendedResult.customModelValues

        let list: [ValuationSpectrumItem] = {
            var items: [ValuationSpectrumItem] = []
            if let models = iv.models {
                let addSpectrum = { (key: String, name: String, origVal: Double?, mc: IntrinsicValueResponse.MC?, tint: Color) in
                    let val = customValues[key] ?? origVal
                    if let v = val, v > 0, v.isFinite {
                        let isCustom = (customOverrides[key]?.isEmpty == false)
                        items.append(.init(
                            id: key,
                            name: name,
                            value: v,
                            defaultValue: origVal,
                            isCustom: isCustom,
                            bear: mc?.bear,
                            bull: mc?.bull,
                            isRecommended: recKey == key,
                            tint: tint
                        ))
                    }
                }

                addSpectrum("dcf", "Discounted Cash Flow (DCF)", models.dcf?.intrinsicValue, models.dcf?.mc, .green)
                addSpectrum("dcfo", "Discounted Cash from Operations (D-CFO)", models.dcfo?.intrinsicValue, models.dcfo?.mc, .teal)
                addSpectrum("dni", "Discounted Net Income (D-NI)", models.dni?.intrinsicValue, models.dni?.mc, .blue)
                addSpectrum("mean_pe", "Mean P/E Valuation", models.meanPe?.intrinsicValue, models.meanPe?.mc, .indigo)
                addSpectrum("peg", "PEG Ratio Fair Value", models.peg?.intrinsicValue, models.peg?.mc, .yellow)
                addSpectrum("mean_pb", "Mean P/B Valuation", models.meanPb?.intrinsicValue, models.meanPb?.mc, .orange)
                addSpectrum("mean_ps", "Mean P/S Valuation", models.meanPs?.intrinsicValue, models.meanPs?.mc, .pink)
                addSpectrum("psg", "Price-to-Sales Growth (PSG)", models.psg?.intrinsicValue, models.psg?.mc, .purple)
                addSpectrum("graham", "Graham Formula", models.graham?.intrinsicValue, models.graham?.mc, .orange)
                addSpectrum("ddm", "Dividend Discount Model (DDM)", models.ddm?.intrinsicValue, models.ddm?.mc, .purple)
                addSpectrum("lynch", "Peter Lynch Fair Value", models.lynch?.intrinsicValue, models.lynch?.mc, .cyan)
                addSpectrum("epv", "Earnings Power Value (EPV)", models.epv?.intrinsicValue, models.epv?.mc, .cyan)
            }
            return items
        }()

        if !list.isEmpty {
            let activeBlended = blendedResult.hasAnyCustom ? blendedResult.customAverage : iv.averageIntrinsicValue
            let isBlendedCustom = blendedResult.hasAnyCustom

            let allVals: [Double] = {
                var vals = list.map(\.value).filter { $0.isFinite && !$0.isNaN && $0 > 0 }
                if currentPrice > 0, currentPrice.isFinite, !currentPrice.isNaN { vals.append(currentPrice) }
                for item in list {
                    if let b = item.bear, b > 0, b.isFinite, !b.isNaN { vals.append(b) }
                    if let b = item.bull, b > 0, b.isFinite, !b.isNaN { vals.append(b) }
                    if let d = item.defaultValue, d > 0, d.isFinite { vals.append(d) }
                }
                if let avg = activeBlended, avg > 0, avg.isFinite, !avg.isNaN { vals.append(avg) }
                if let avgDef = iv.averageIntrinsicValue, avgDef > 0, avgDef.isFinite { vals.append(avgDef) }
                return vals
            }()

            let rawMin = allVals.min() ?? 0
            let rawMax = allVals.max() ?? 100
            let pad = max((rawMax - rawMin) * 0.12, 5)
            let minBound = max(0, rawMin - pad)
            let maxBound = rawMax + pad
            let spread = max(1e-6, maxBound - minBound)
            let undervaluedCount = list.filter { $0.value >= currentPrice }.count

            VStack(alignment: .leading, spacing: 16) {
                spectrumHeader(undervaluedCount: undervaluedCount, total: list.count, currentPrice: currentPrice)

                // Axis Labels
                HStack {
                    Text(Fmt.compact(minBound, code: nativeCur))
                        .font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
                    Spacer()
                    if currentPrice > 0 {
                        Text("Current: \(Fmt.currency(currentPrice, code: nativeCur))")
                            .font(.caption2.weight(.bold))
                            .foregroundStyle(.primary)
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.primary.opacity(0.1), in: Capsule())
                    }
                    Spacer()
                    Text(Fmt.compact(maxBound, code: nativeCur))
                        .font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
                }

                // Blended Intrinsic Value Row (if available)
                if let avg = activeBlended, avg > 0, avg.isFinite, !avg.isNaN {
                    let isUnder = avg >= currentPrice
                    let up = currentPrice > 0 ? ((avg - currentPrice) / currentPrice) * 100 : 0

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            HStack(spacing: 4) {
                                Image(systemName: "sparkles").foregroundStyle(isBlendedCustom ? .orange : .yellow)
                                Text(isBlendedCustom ? "Custom Blended Value" : "Blended Intrinsic Value")
                                    .font(.caption.weight(.bold))
                            }
                            Spacer()
                            Text(Fmt.currency(avg, code: nativeCur))
                                .font(.caption.weight(.bold))
                                .foregroundStyle(isBlendedCustom ? Color.orange : Color.indigo)
                            Text(Fmt.percent(up, includeSign: true))
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(isUnder ? Color.green : Color.red)
                                .padding(.horizontal, 5).padding(.vertical, 1)
                                .background((isUnder ? Color.green : Color.red).opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                        }

                        GeometryReader { geo in
                            let w = max(1, geo.size.width)
                            let spotX = currentPrice > 0 ? max(0, min(w, CGFloat((currentPrice - minBound) / spread) * w)) : nil
                            let valX = max(0, min(w, CGFloat((avg - minBound) / spread) * w))

                            ZStack(alignment: .leading) {
                                Capsule().fill(Color.secondary.opacity(0.12)).frame(height: 10)

                                if let r = iv.range, let bear = r.bear, let bull = r.bull, bear.isFinite, bull.isFinite {
                                    let bearX = max(0, min(w, CGFloat((bear - minBound) / spread) * w))
                                    let bullX = max(0, min(w, CGFloat((bull - minBound) / spread) * w))
                                    if bearX.isFinite && bullX.isFinite {
                                        Capsule().fill(Color.indigo.opacity(0.3))
                                            .frame(width: max(2, abs(bullX - bearX)), height: 10)
                                            .offset(x: min(bearX, bullX))
                                    }
                                }

                                if let sx = spotX, sx.isFinite, valX.isFinite {
                                    Rectangle().fill(isUnder ? Color.green.opacity(0.7) : Color.red.opacity(0.7))
                                        .frame(width: max(2, abs(valX - sx)), height: 3)
                                        .offset(x: min(valX, sx))

                                    Rectangle().fill(Color.primary.opacity(0.5))
                                        .frame(width: 1, height: 16)
                                        .offset(x: sx)
                                }

                                if valX.isFinite {
                                    Circle().fill(isBlendedCustom ? Color.orange : Color.indigo)
                                        .frame(width: 12, height: 12)
                                        .overlay(Circle().stroke(Color.white, lineWidth: 1.5))
                                        .shadow(radius: 1)
                                        .offset(x: max(0, valX - 6))
                                }
                            }
                            .frame(height: 16)
                        }
                        .frame(height: 16)
                    }
                    .padding(10)
                    .background((isBlendedCustom ? Color.orange : Color.indigo).opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke((isBlendedCustom ? Color.orange : Color.indigo).opacity(0.2), lineWidth: 1))
                }

                Divider()

                // Model Rows
                VStack(spacing: 12) {
                    ForEach(list) { item in
                        let isUnder = item.value >= currentPrice
                        let up = currentPrice > 0 ? ((item.value - currentPrice) / currentPrice) * 100 : 0

                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                HStack(spacing: 4) {
                                    Circle().fill(item.tint).frame(width: 6, height: 6)
                                    Text(item.name).font(.caption.weight(.medium)).foregroundStyle(.primary)
                                    if item.isRecommended {
                                        Text("BEST-FIT").font(.system(size: 8, weight: .bold)).foregroundStyle(.white)
                                            .padding(.horizontal, 4).padding(.vertical, 1)
                                            .background(Color.indigo, in: Capsule())
                                    }
                                    if item.isCustom {
                                        Text("CUSTOM").font(.system(size: 7, weight: .bold)).foregroundStyle(.orange)
                                            .padding(.horizontal, 3).padding(.vertical, 1)
                                            .background(Color.orange.opacity(0.15), in: Capsule())
                                    }
                                }
                                Spacer()
                                Text(Fmt.currency(item.value, code: nativeCur))
                                    .font(.caption.weight(.bold))
                                    .foregroundStyle(item.isCustom ? Color.orange : Color.primary)
                                Text(Fmt.percent(up, includeSign: true))
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(isUnder ? Color.green : Color.red)
                                    .padding(.horizontal, 5).padding(.vertical, 1)
                                    .background((isUnder ? Color.green : Color.red).opacity(0.12), in: RoundedRectangle(cornerRadius: 4))
                            }

                            GeometryReader { geo in
                                let w = max(1, geo.size.width)
                                let spotX = currentPrice > 0 ? max(0, min(w, CGFloat((currentPrice - minBound) / spread) * w)) : nil
                                let valX = max(0, min(w, CGFloat((item.value - minBound) / spread) * w))

                                ZStack(alignment: .leading) {
                                    Capsule().fill(Color.secondary.opacity(0.1)).frame(height: 8)

                                    if let bear = item.bear, let bull = item.bull, bear.isFinite, bull.isFinite {
                                        let bearX = max(0, min(w, CGFloat((bear - minBound) / spread) * w))
                                        let bullX = max(0, min(w, CGFloat((bull - minBound) / spread) * w))
                                        if bearX.isFinite && bullX.isFinite {
                                            Capsule().fill((isUnder ? Color.green : Color.red).opacity(0.25))
                                                .frame(width: max(2, abs(bullX - bearX)), height: 8)
                                                .offset(x: min(bearX, bullX))
                                        }
                                    }

                                    if let sx = spotX, sx.isFinite, valX.isFinite {
                                        Rectangle().fill((isUnder ? Color.green : Color.red).opacity(0.6))
                                            .frame(width: max(2, abs(valX - sx)), height: 2)
                                            .offset(x: min(valX, sx))

                                        Rectangle().fill(Color.primary.opacity(0.4))
                                            .frame(width: 1, height: 14)
                                            .offset(x: sx)
                                    }

                                    if valX.isFinite {
                                        Circle().fill(item.isCustom ? Color.orange : item.tint)
                                            .frame(width: 10, height: 10)
                                            .overlay(Circle().stroke(Color.white, lineWidth: 1.2))
                                            .shadow(radius: 1)
                                            .offset(x: max(0, valX - 5))
                                    }
                                }
                                .frame(height: 14)
                            }
                            .frame(height: 14)
                        }
                    }
                }
            }
            .padding(20)
            .card(.standard)
        }
    }
}

struct HistogramChartView: View {
    let hist: [IntrinsicValueResponse.HistogramPoint]
    let mc: IntrinsicValueResponse.MC?
    let currentPrice: Double?
    let currency: String

    var body: some View {
        Chart {
            ForEach(Array(hist.enumerated()), id: \.offset) { _, pt in
                if let price = pt.price, let count = pt.count {
                    AreaMark(
                        x: .value("Price", price),
                        y: .value("Count", count)
                    )
                    .foregroundStyle(
                        LinearGradient(
                            colors: [Color.brandIndigo.opacity(0.4), Color.brandIndigo.opacity(0.05)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .interpolationMethod(.monotone)

                    LineMark(
                        x: .value("Price", price),
                        y: .value("Count", count)
                    )
                    .foregroundStyle(Color.brandIndigo)
                    .interpolationMethod(.monotone)
                }
            }

            if let bear = mc?.bear {
                RuleMark(x: .value("Bear", bear))
                    .foregroundStyle(.red)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .annotation(position: .top) {
                        Text("Bear").font(.system(size: 8, weight: .bold)).foregroundStyle(.red)
                    }
            }

            if let base = mc?.base {
                RuleMark(x: .value("Median", base))
                    .foregroundStyle(Color.brandIndigo)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .annotation(position: .top) {
                        Text("Median").font(.system(size: 8, weight: .bold)).foregroundStyle(Color.brandIndigo)
                    }
            }

            if let bull = mc?.bull {
                RuleMark(x: .value("Bull", bull))
                    .foregroundStyle(Color.brandEmerald)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                    .annotation(position: .top) {
                        Text("Bull").font(.system(size: 8, weight: .bold)).foregroundStyle(Color.brandEmerald)
                    }
            }

            if let cp = currentPrice, cp > 0 {
                RuleMark(x: .value("Current", cp))
                    .foregroundStyle(.purple)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .annotation(position: .bottom) {
                        Text("Price").font(.system(size: 8, weight: .bold)).foregroundStyle(.purple)
                    }
            }
        }
        .chartXAxis {
            // Five currency labels collide below ~390pt of plot width.
            AxisMarks(values: .automatic(desiredCount: isPhoneLayout ? 3 : 5)) { val in
                if let d = val.as(Double.self) {
                    AxisValueLabel {
                        Text(Fmt.currency(d, code: currency))
                            .font(.system(size: 9))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                    }
                }
            }
        }
        .chartYAxis(.hidden)
    }
}

struct DistributionModalView: View {
    let item: DistributionModelItem
    @Environment(\.dismiss) private var dismiss

    /// A Mac sheet can afford 24pt gutters and a 540pt floor; an iPhone sheet
    /// is ~390pt wide, so the same numbers push the chart and the scenario
    /// cards off both edges.
    private var gutter: CGFloat { isPhoneLayout ? 16 : 24 }
    private var chartHeight: CGFloat { isPhoneLayout ? 200 : 260 }

    var body: some View {
        NavigationStack {
            content
        }
        #if os(iOS)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        #else
        .frame(minWidth: 540, minHeight: 440)
        #endif
    }

    private var content: some View {
        VStack(spacing: isPhoneLayout ? 16 : 20) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.title)
                        .font(isPhoneLayout ? .title3.weight(.bold) : .title2.weight(.bold))
                        .lineLimit(2)
                        .minimumScaleFactor(0.7)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Monte Carlo Simulation (10,000 iterations)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                }
                Spacer(minLength: 8)
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, gutter)
            .padding(.top, gutter)

            if let hist = item.mc.histogram, !hist.isEmpty {
                HistogramChartView(
                    hist: hist,
                    mc: item.mc,
                    currentPrice: item.currentPrice,
                    currency: item.nativeCur
                )
                .frame(maxWidth: .infinity)
                .frame(height: chartHeight)
                .padding(.horizontal, gutter)
            }

            HStack(spacing: isPhoneLayout ? 8 : 12) {
                scenarioCard("Bear (10th)", item.mc.bear, .red)
                scenarioCard("Median (50th)", item.mc.base, Color.brandIndigo)
                scenarioCard("Bull (90th)", item.mc.bull, Color.brandEmerald)
            }
            .padding(.horizontal, gutter)
            .padding(.bottom, gutter)

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .top)
    }

    private func scenarioCard(_ label: String, _ val: Double?, _ color: Color) -> some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption2.weight(.bold))
                .foregroundStyle(color)
                .textCase(.uppercase)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(Fmt.currency(val, code: item.nativeCur))
                .font(.headline.weight(.bold))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .padding(.horizontal, isPhoneLayout ? 6 : 12)
        .background(color.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(color.opacity(0.2), lineWidth: 1))
    }
}
