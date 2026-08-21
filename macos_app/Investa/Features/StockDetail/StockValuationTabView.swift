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

                if let rec = iv.recommendedMethod, rec.name != nil, rec.methodKey != "none" {
                    recommendedMethodBanner(rec, currentPrice: iv.currentPrice)
                }

                valuationSpectrumSection(iv)

                if let models = iv.models {
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
    private func valuationModelCards(_ iv: IntrinsicValueResponse, models: IntrinsicValueResponse.Models) -> some View {
        Group {
            if let dcf = models.dcf {
                dcfCard("Discounted Cash Flow", "chart.line.uptrend.xyaxis", .green, dcf, modelKey: "dcf", iv: iv)
            }
            if let dcfo = models.dcfo {
                dcfoCard("Discounted Cash from Operations", "dollarsign.circle", .teal, dcfo, modelKey: "dcfo", iv: iv)
            }
            if let dni = models.dni {
                dniCard("Discounted Net Income", "building.columns", .blue, dni, modelKey: "dni", iv: iv)
            }
            if let pe = models.meanPe {
                meanPeCard("Mean P/E Ratio", "percent", .indigo, pe, modelKey: "mean_pe", iv: iv)
            }
            if let peg = models.peg {
                pegCard("PEG Ratio Fair Value", "bolt", .yellow, peg, modelKey: "peg", iv: iv)
            }
            if let pb = models.meanPb {
                meanPbCard("Mean P/B Ratio", "book", .orange, pb, modelKey: "mean_pb", iv: iv)
            }
        }
        Group {
            if let ps = models.meanPs {
                meanPsCard("Mean P/S Ratio", "chart.line.uptrend.xyaxis", .pink, ps, modelKey: "mean_ps", iv: iv)
            }
            if let psg = models.psg {
                psgCard("Price-to-Sales Growth (PSG)", "sparkles", .purple, psg, modelKey: "psg", iv: iv)
            }
            if let g = models.graham {
                grahamCard("Graham Formula", "scalemass", .orange, g, modelKey: "graham", iv: iv)
            }
            if let ddm = models.ddm {
                ddmCard("Dividend Discount Model", "dollarsign.circle", .purple, ddm, modelKey: "ddm", iv: iv)
            }
            if let lynch = models.lynch {
                lynchCard("Peter Lynch Fair Value", "equal.circle", .cyan, lynch, modelKey: "lynch", iv: iv)
            }
            if let epv = models.epv {
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

    private func scenarioPill(
        label: String,
        value: Double?,
        color: Color,
        modelKey: String,
        title: String,
        mc: IntrinsicValueResponse.MC
    ) -> some View {
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
    private func probabilisticScenariosRow(
        _ mc: IntrinsicValueResponse.MC?,
        modelKey: String,
        title: String,
        color: Color
    ) -> some View {
        if let mc, mc.bear != nil || mc.base != nil || mc.bull != nil {
            let isExpanded = expandedHistogramKeys.contains(modelKey)
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    SectionLabel(title: "Probabilistic Scenarios (Monte Carlo)")
                    Spacer()
                    Button {
                        toggleHistogram(modelKey)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: isExpanded ? "chevron.up.circle.fill" : "chart.bar.xaxis")
                            Text(isExpanded ? "Hide Chart" : "Show Distribution")
                                .font(.system(size: 10, weight: .bold))
                        }
                        .foregroundStyle(Color.brandIndigo)
                    }
                    .buttonStyle(.plain)
                }

                HStack(spacing: 8) {
                    scenarioPill(label: "Bear (10th)", value: mc.bear, color: .red, modelKey: modelKey, title: title, mc: mc)
                    scenarioPill(label: "Median (50th)", value: mc.base, color: Color.brandIndigo, modelKey: modelKey, title: title, mc: mc)
                    scenarioPill(label: "Bull (90th)", value: mc.bull, color: Color.brandEmerald, modelKey: modelKey, title: title, mc: mc)
                }

                if isExpanded, let hist = mc.histogram, !hist.isEmpty {
                    VStack(spacing: 8) {
                        HistogramChartView(hist: hist, mc: mc, currentPrice: viewModel.intrinsic?.currentPrice, currency: nativeCur)
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
                                    currentPrice: viewModel.intrinsic?.currentPrice,
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

    @ViewBuilder
    private func modelCardHeader(
        title: String,
        icon: String,
        color: Color,
        modelKey: String,
        m: IntrinsicValueResponse.Model,
        primaryBadge: String? = nil
    ) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]
        let defaultParams = m.parameters ?? [:]
        let isCustom = overrides.contains { (k, v) in
            if let def = defaultParams[k]?.doubleValue {
                return abs(v - def) > 1e-5
            }
            return true
        }
        let customVal = blendedResult.customModelValues[modelKey]
        let defaultVal = m.intrinsicValue
        let activeVal = isCustom ? customVal : defaultVal

        HStack(alignment: .top, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: icon).foregroundStyle(color)
                Text(title).font(.headline)
                if let p = primaryBadge {
                    Text(p).font(.system(size: 9, weight: .bold)).foregroundStyle(.green)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.green.opacity(0.15), in: Capsule())
                }
                if isCustom {
                    Text("CUSTOM")
                        .font(.system(size: 8, weight: .black))
                        .foregroundStyle(.orange)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(Color.orange.opacity(0.15), in: Capsule())
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                HStack(spacing: 6) {
                    if m.error == nil, let v = activeVal {
                        Text(Fmt.currency(v, code: nativeCur))
                            .font(.subheadline.weight(.bold))
                            .foregroundStyle(isCustom ? Color.orange : color)
                            .padding(.horizontal, 10).padding(.vertical, 4)
                            .background((isCustom ? Color.orange : color).opacity(0.15), in: Capsule())
                    }

                    // Edit Toggle Button
                    Button {
                        toggleEditing(modelKey)
                    } label: {
                        HStack(spacing: 3) {
                            Image(systemName: "slider.horizontal.3")
                            Text(isEditing ? "Done" : "Edit")
                        }
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(isEditing ? .white : Color.primary)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(isEditing ? Color.indigo : Color.secondary.opacity(0.15), in: RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)

                    // Reset Model Button
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

                if isCustom, let defVal = defaultVal, let actVal = activeVal, defVal > 0, abs(actVal - defVal) > 0.001 {
                    let diff = ((actVal - defVal) / defVal) * 100
                    Text("Def: \(Fmt.currency(defVal, code: nativeCur)) (\(Fmt.percent(diff, includeSign: true)))")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(diff >= 0 ? Color.green : Color.red)
                }
            }
        }
    }

    @ViewBuilder
    private func modelParameterEditor(
        modelKey: String,
        m: IntrinsicValueResponse.Model
    ) -> some View {
        if let configs = StockValuationCalculator.configs[modelKey] {
            let overrides = customOverrides[modelKey] ?? [:]
            let defaultParams = m.parameters ?? [:]
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
                        let rawDef = m.parameters?[cfg.key]?.doubleValue
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

    private func dcfCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 20) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m, primaryBadge: "Primary")

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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

                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
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
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 20) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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

                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func ddmCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 20) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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

                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func lynchCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func dcfoCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func dniCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func meanPeCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
                        let eps = overrides["eps"] ?? p["eps"]?.doubleValue ?? 0
                        paramRow("TTM EPS", Fmt.number(eps, fractionDigits: 2), isCustom: overrides["eps"] != nil, defVal: Fmt.number(p["eps"]?.doubleValue, fractionDigits: 2))

                        let pe = overrides["applied_pe"] ?? p["applied_pe"]?.doubleValue ?? 0
                        paramRow("Mean P/E Multiple", "\(Fmt.number(pe, fractionDigits: 1))x", isCustom: overrides["applied_pe"] != nil, defVal: "\(Fmt.number(p["applied_pe"]?.doubleValue, fractionDigits: 1))x")

                        if let v = p["pe_source"]?.stringValue { paramRow("Source", v) }
                    }
                    limitationCallout("Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline.",
                                      "Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges.")
                }
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func pegCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func meanPbCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
                        let bvps = overrides["book_value_per_share"] ?? p["book_value_per_share"]?.doubleValue ?? 0
                        paramRow("Book Value / Share", Fmt.currency(bvps, code: nativeCur), isCustom: overrides["book_value_per_share"] != nil, defVal: Fmt.currency(p["book_value_per_share"]?.doubleValue, code: nativeCur))

                        let pb = overrides["applied_pb"] ?? p["applied_pb"]?.doubleValue ?? 0
                        paramRow("Applied P/B Target", "\(Fmt.number(pb, fractionDigits: 2))x", isCustom: overrides["applied_pb"] != nil, defVal: "\(Fmt.number(p["applied_pb"]?.doubleValue, fractionDigits: 2))x")

                        if let v = p["pb_source"]?.stringValue { paramRow("Benchmark Source", v) }
                    }
                    limitationCallout("Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market.",
                                      "Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property.")
                }
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func meanPsCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
                        let sps = overrides["sales_per_share"] ?? p["sales_per_share"]?.doubleValue ?? 0
                        paramRow("Sales / Share", Fmt.currency(sps, code: nativeCur), isCustom: overrides["sales_per_share"] != nil, defVal: Fmt.currency(p["sales_per_share"]?.doubleValue, code: nativeCur))

                        let ps = overrides["applied_ps"] ?? p["applied_ps"]?.doubleValue ?? 0
                        paramRow("Mean P/S Multiple", "\(Fmt.number(ps, fractionDigits: 2))x", isCustom: overrides["applied_ps"] != nil, defVal: "\(Fmt.number(p["applied_ps"]?.doubleValue, fractionDigits: 2))x")

                        if let v = p["ps_source"]?.stringValue { paramRow("Multiple Source", v) }
                    }
                    limitationCallout("Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction.",
                                      "Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits.")
                }
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func psgCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func epvCard(_ title: String, _ icon: String, _ color: Color, _ m: IntrinsicValueResponse.Model, modelKey: String, iv: IntrinsicValueResponse) -> some View {
        let isEditing = editingModelKeys.contains(modelKey)
        let overrides = customOverrides[modelKey] ?? [:]

        return VStack(alignment: .leading, spacing: 16) {
            modelCardHeader(title: title, icon: icon, color: color, modelKey: modelKey, m: m)

            if let e = m.error {
                Text(e).font(.callout).foregroundStyle(.red)
            } else {
                if isEditing {
                    modelParameterEditor(modelKey: modelKey, m: m)
                } else if let p = m.parameters {
                    let columns = hSizeClass == .compact
                        ? [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)]
                        : [GridItem(.adaptive(minimum: 150), spacing: 24)]
                    LazyVGrid(columns: columns, spacing: 24) {
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
                probabilisticScenariosRow(m.mc, modelKey: modelKey, title: title, color: color)
            }
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    private func limitationCallout(_ bestSuitedFor: String, _ keyCaveats: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 6) {
                Text("Best Suited For:").font(.caption2.weight(.bold)).foregroundStyle(.green)
                Text(bestSuitedFor).font(.caption2).foregroundStyle(.secondary)
            }
            HStack(alignment: .top, spacing: 6) {
                Text("Key Caveats:").font(.caption2.weight(.bold)).foregroundStyle(.orange)
                Text(keyCaveats).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private func recommendedMethodBanner(_ rec: IntrinsicValueResponse.RecommendedMethod, currentPrice: Double?) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                HStack(spacing: 6) {
                    Image(systemName: "sparkles").foregroundStyle(.yellow)
                    Text("Best-Fit Valuation Method").font(.caption.weight(.bold)).foregroundStyle(.indigo)
                }
                .padding(.horizontal, 10).padding(.vertical, 4)
                .background(Color.indigo.opacity(0.15), in: Capsule())

                Spacer()

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
                }
            }

            Text(rec.name ?? "Valuation Method").font(.title3.weight(.bold))
            if let r = rec.rationale {
                Text(r).font(.subheadline).foregroundStyle(.secondary)
            }

            if let suited = rec.bestSuitedFor ?? rec.whenToUse, let caveats = rec.keyCaveats ?? rec.keyLimitation {
                limitationCallout(suited, caveats)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.indigo.opacity(0.05), in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.indigo.opacity(0.2), lineWidth: 1))
    }

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

    @ViewBuilder
    private func valuationSpectrumSection(_ iv: IntrinsicValueResponse) -> some View {
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
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Valuation Comparison Spectrum", systemImage: "chart.bar.xaxis")
                            .font(.headline)
                        Text("Cross-method valuation distribution compared against current price (\(Fmt.currency(currentPrice, code: nativeCur)))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("\(undervaluedCount)/\(list.count) Undervalued")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(undervaluedCount > list.count / 2 ? Color.green : Color.orange)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background((undervaluedCount > list.count / 2 ? Color.green : Color.orange).opacity(0.12), in: Capsule())
                }

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

    private func card<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title).font(.headline)
            content()
        }
        .padding(24).frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
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
            AxisMarks(values: .automatic(desiredCount: 5)) { val in
                if let d = val.as(Double.self) {
                    AxisValueLabel {
                        Text(Fmt.currency(d, code: currency))
                            .font(.system(size: 9))
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

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(item.title)
                            .font(.title2.weight(.bold))
                        Text("Monte Carlo Simulation (10,000 iterations)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 24)
                .padding(.top, 24)

                if let hist = item.mc.histogram, !hist.isEmpty {
                    HistogramChartView(
                        hist: hist,
                        mc: item.mc,
                        currentPrice: item.currentPrice,
                        currency: item.nativeCur
                    )
                    .frame(height: 260)
                    .padding(.horizontal, 24)
                }

                HStack(spacing: 12) {
                    scenarioCard("Bear (10th)", item.mc.bear, .red)
                    scenarioCard("Median (50th)", item.mc.base, Color.brandIndigo)
                    scenarioCard("Bull (90th)", item.mc.bull, Color.brandEmerald)
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 24)

                Spacer()
            }
            .frame(minWidth: 540, minHeight: 440)
        }
    }

    private func scenarioCard(_ label: String, _ val: Double?, _ color: Color) -> some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption2.weight(.bold))
                .foregroundStyle(color)
                .textCase(.uppercase)
            Text(Fmt.currency(val, code: item.nativeCur))
                .font(.headline.weight(.bold))
                .foregroundStyle(.primary)
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(color.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(color.opacity(0.2), lineWidth: 1))
    }
}

