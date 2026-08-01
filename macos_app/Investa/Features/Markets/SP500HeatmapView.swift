import SwiftUI

// MARK: - ViewModel

@MainActor
final class SP500HeatmapViewModel: ObservableObject {
    @Published var items: [SP500HeatmapItem] = []
    @Published var isLoading = true
    @Published var errorMessage: String?

    private let api: APIClient
    private var task: Task<Void, Never>?

    init(api: APIClient = .shared) { self.api = api }

    func load() {
        task?.cancel()
        task = Task { [weak self] in
            guard let self else { return }
            isLoading = true; errorMessage = nil
            do {
                items = try await api.get("/sp500/heatmap")
            } catch is CancellationError {
                // A superseded load must not clear the flag: the replacement
                // task is already in flight and still loading.
                return
            } catch let e as APIError {
                // Auth is handled globally; surface everything else.
                if case .unauthorized = e {} else { errorMessage = e.errorDescription }
            } catch {
                errorMessage = error.localizedDescription
            }
            isLoading = false
        }
    }
}

// MARK: - Metric selector

/// Mirrors `METRICS` in web_app/components/SP500Heatmap.tsx — same keys, same
/// midpoints, same clamps. `mid` values are the index's own median for that
/// metric, measured across all 500 constituents, so roughly half the map falls
/// either side and the scale actually discriminates.
private enum HeatMetric: String, CaseIterable, Identifiable {
    // Performance
    case day = "1-Day Performance"
    case week = "1-Week Performance"
    case month = "1-Month Performance"
    case mtd = "Month to Date"
    case threeMonth = "3-Month Performance"
    case sixMonth = "6-Month Performance"
    case ytd = "Year to Date"
    case oneYear = "1-Year Performance"
    case threeYear = "3-Year Performance"
    case fiveYear = "5-Year Performance"
    case tenYear = "10-Year Performance"
    case drawdown52w = "Drawdown from 52-Week High"
    case gain52wLow = "Gain from 52-Week Low"

    // Valuation
    case pe = "P/E"
    case forwardPE = "Forward P/E"
    case peg = "PEG"
    case ps = "P/S"
    case pb = "P/B"
    case pFcf = "P/FCF"
    case evEbitda = "EV/EBITDA"
    case evSales = "EV/Sales"
    case yield = "Dividend Yield"

    // Earnings & sales
    case epsTtm = "EPS TTM"
    case epsQoQ = "EPS Q/Q"
    case epsGrowth3y = "EPS Growth Past 3 Years"
    case epsGrowth5y = "EPS Growth Past 5 Years"
    case epsSurprise = "EPS Surprise"
    case salesTtm = "Sales TTM"
    case salesQoQ = "Sales Q/Q"
    case salesGrowth3y = "Sales Growth Past 3 Years"
    case salesGrowth5y = "Sales Growth Past 5 Years"

    // Profitability & balance sheet
    case roa = "ROA"
    case roe = "ROE"
    case roic = "ROIC"
    case grossMargin = "Gross Margin"
    case operatingMargin = "Operating Margin"
    case netMargin = "Net Margin"
    case quickRatio = "Quick Ratio"
    case currentRatio = "Current Ratio"
    case ltDebtEquity = "LT Debt/Equity"
    case debtEquity = "Debt/Equity"

    // Market & sentiment
    case relativeVolume = "Relative Volume"
    case floatShort = "Float Short"
    case analystRecom = "Analysts Recom."
    case earningsDays = "Earnings Date"

    var id: String { rawValue }

    static let performance: [HeatMetric] = [
        .day, .week, .month, .mtd, .threeMonth, .sixMonth, .ytd,
        .oneYear, .threeYear, .fiveYear, .tenYear, .drawdown52w, .gain52wLow,
    ]
    static let valuation: [HeatMetric] = [
        .pe, .forwardPE, .peg, .ps, .pb, .pFcf, .evEbitda, .evSales, .yield,
    ]
    static let earnings: [HeatMetric] = [
        .epsTtm, .epsQoQ, .epsGrowth3y, .epsGrowth5y, .epsSurprise,
        .salesTtm, .salesQoQ, .salesGrowth3y, .salesGrowth5y,
    ]
    static let profitability: [HeatMetric] = [
        .roa, .roe, .roic, .grossMargin, .operatingMargin, .netMargin,
        .quickRatio, .currentRatio, .ltDebtEquity, .debtEquity,
    ]
    static let market: [HeatMetric] = [.relativeVolume, .floatShort, .analystRecom, .earningsDays]

    /// Distance from `mid` at which the colour saturates.
    var clamp: Double {
        switch self {
        case .day: return 3
        case .week: return 5
        case .month, .mtd: return 10
        case .threeMonth: return 15
        case .sixMonth: return 25
        case .ytd: return 30
        case .oneYear: return 40
        case .threeYear: return 60
        case .fiveYear: return 100
        case .tenYear: return 200
        case .drawdown52w: return 50
        case .gain52wLow: return 100

        case .pe: return 15
        case .forwardPE: return 10
        case .peg: return 1.5
        case .ps: return 2.5
        case .pb: return 3
        case .pFcf: return 15
        case .evEbitda: return 10
        case .evSales: return 3
        case .yield: return 5

        case .epsTtm: return 15
        case .epsQoQ: return 50
        case .epsGrowth3y, .epsGrowth5y: return 30
        case .epsSurprise: return 10
        case .salesTtm: return 100e9
        case .salesQoQ: return 30
        case .salesGrowth3y, .salesGrowth5y: return 25

        case .roa: return 20
        case .roe: return 50
        case .roic: return 30
        case .grossMargin: return 60
        case .operatingMargin: return 30
        case .netMargin: return 25
        case .quickRatio, .currentRatio: return 1
        case .ltDebtEquity: return 50
        case .debtEquity: return 80

        case .relativeVolume: return 1
        case .floatShort: return 5
        case .analystRecom: return 1.5
        case .earningsDays: return 45
        }
    }

    /// Value that colours neutral. Zero where zero is genuinely neutral (a
    /// return, a margin, a growth rate); a typical large-cap reading for
    /// valuation ratios, which are never meaningfully zero-centred — centring
    /// P/E on zero would paint every profitable company red.
    var mid: Double {
        switch self {
        case .pe: return 25
        case .forwardPE: return 17
        case .peg: return 2
        case .ps: return 3
        case .pb: return 4
        case .pFcf: return 24
        case .evEbitda: return 15
        case .evSales: return 4
        case .quickRatio: return 1
        case .currentRatio: return 1.5
        case .ltDebtEquity: return 50
        case .debtEquity: return 80
        case .relativeVolume: return 1
        case .floatShort: return 5
        case .analystRecom: return 2.5
        case .earningsDays: return 45
        case .drawdown52w: return -50
        default: return 0
        }
    }

    func value(for item: SP500HeatmapItem) -> Double? {
        switch self {
        case .day: return item.changePct
        case .week: return item.weekChangePct
        case .month: return item.monthChangePct
        case .mtd: return item.mtdChangePct
        case .threeMonth: return item.threeMonthChangePct
        case .sixMonth: return item.sixMonthChangePct
        case .ytd: return item.ytdChangePct
        case .oneYear: return item.oneYearChangePct
        case .threeYear: return item.threeYearChangePct
        case .fiveYear: return item.fiveYearChangePct
        case .tenYear: return item.tenYearChangePct
        case .drawdown52w: return item.drawdown52w
        case .gain52wLow: return item.gainFrom52wLow

        case .pe: return item.peRatio
        case .forwardPE: return item.forwardPE
        case .peg: return item.pegRatio
        case .ps: return item.psRatio
        case .pb: return item.pbRatio
        case .pFcf: return item.pFcf
        case .evEbitda: return item.evEbitda
        case .evSales: return item.evSales
        case .yield: return item.dividendYield

        case .epsTtm: return item.epsTtm
        case .epsQoQ: return item.epsQoQ
        case .epsGrowth3y: return item.epsGrowth3y
        case .epsGrowth5y: return item.epsGrowth5y
        case .epsSurprise: return item.epsSurprise
        case .salesTtm: return item.salesTtm
        case .salesQoQ: return item.salesQoQ
        case .salesGrowth3y: return item.salesGrowth3y
        case .salesGrowth5y: return item.salesGrowth5y

        case .roa: return item.roa
        case .roe: return item.roe
        case .roic: return item.roic
        case .grossMargin: return item.grossMargin
        case .operatingMargin: return item.operatingMargin
        case .netMargin: return item.netMargin
        case .quickRatio: return item.quickRatio
        case .currentRatio: return item.currentRatio
        case .ltDebtEquity: return item.ltDebtEquity
        case .debtEquity: return item.debtEquity

        case .relativeVolume: return item.relativeVolume
        case .floatShort: return item.floatShort
        case .analystRecom: return item.analystRecom
        case .earningsDays: return item.earningsDays
        }
    }

    /// True for fields the backend sends as a fraction and we show as a
    /// percentage. `day` is excluded: `change_pct` already arrives in percent
    /// points, as do the ratio-style fields.
    private var isFraction: Bool {
        switch self {
        case .week, .month, .mtd, .threeMonth, .sixMonth, .ytd, .oneYear,
             .threeYear, .fiveYear, .tenYear, .drawdown52w, .gain52wLow,
             .yield, .epsQoQ, .epsGrowth3y, .epsGrowth5y, .epsSurprise,
             .salesQoQ, .salesGrowth3y, .salesGrowth5y,
             .roa, .roe, .roic, .grossMargin, .operatingMargin, .netMargin,
             .floatShort:
            return true
        default:
            return false
        }
    }

    func scaledValue(for item: SP500HeatmapItem) -> Double? {
        guard let v = value(for: item) else { return nil }
        return isFraction ? v * 100 : v
    }

    var inverted: Bool {
        switch self {
        case .pe, .forwardPE, .peg, .ps, .pb, .pFcf, .evEbitda, .evSales,
             .ltDebtEquity, .debtEquity, .floatShort, .analystRecom, .earningsDays:
            return true
        default:
            return false
        }
    }

    /// True for quantities with a hard floor and no "bad" direction, which take
    /// a one-hue magnitude ramp over [mid, mid+clamp] instead of the red-green
    /// diverging scale. Dividend yield is dividends over price, so it cannot go
    /// below zero and half a diverging scale would describe impossible values;
    /// the same holds for revenue and for both 52-week bounds.
    var isSequential: Bool {
        switch self {
        case .yield, .salesTtm, .drawdown52w, .gain52wLow: return true
        default: return false
        }
    }

    /// The [low, high] range the colour scale covers, in display units.
    var domain: (low: Double, high: Double) {
        isSequential ? (mid, mid + clamp) : (mid - clamp, mid + clamp)
    }

    func formatValue(_ v: Double) -> String {
        switch self {
        case .epsTtm:
            return String(format: "$%.2f", v)
        case .salesTtm:
            return formatCompactCap(v)
        case .earningsDays:
            return v >= 0 ? String(format: "%.0fd", v) : String(format: "%.0fd ago", -v)
        case .pe, .forwardPE, .peg, .ps, .pb, .pFcf, .evEbitda, .evSales,
             .quickRatio, .currentRatio, .ltDebtEquity, .debtEquity,
             .relativeVolume, .analystRecom:
            return String(format: "%.2f", v)
        default:
            return String(format: "%@%.2f%%", v >= 0 ? "+" : "", v)
        }
    }
}

private enum SizeMode: String, CaseIterable, Identifiable {
    case cap = "Mkt Cap"
    case equal = "Equal"
    var id: String { rawValue }
}

// MARK: - Color helpers

/// Signed distance from the metric's neutral point, positive meaning "good".
/// Takes an already-scaled (display-unit) value.
private func signedDeviation(_ scaled: Double, _ metric: HeatMetric) -> Double {
    metric.inverted ? metric.mid - scaled : scaled - metric.mid
}

/// A tile fill kept in component form, so the label ink can be resolved against
/// the fill itself — `Color` gives no portable way to read its channels back.
private struct RGB {
    let r, g, b: Double

    init(_ r: Double, _ g: Double, _ b: Double) { self.r = r; self.g = g; self.b = b }

    init(hex: UInt32) {
        self.init(
            Double((hex >> 16) & 0xff) / 255,
            Double((hex >> 8) & 0xff) / 255,
            Double(hex & 0xff) / 255
        )
    }

    var color: Color { Color(red: r, green: g, blue: b) }

    func mixed(to other: RGB, _ t: Double) -> RGB {
        RGB(r + (other.r - r) * t, g + (other.g - g) * t, b + (other.b - b) * t)
    }

    /// WCAG relative luminance.
    var luminance: Double {
        func lin(_ c: Double) -> Double { c <= 0.03928 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
        return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    }

    /// Ink for a label drawn on this fill, picked against the fill itself rather
    /// than against the scheme: the light map runs from near-white at the
    /// neutral point to deep red/green at the ends, so one fixed label colour
    /// would be unreadable over half of it.
    func label(flippingAbove flip: Double) -> Color {
        luminance > flip ? Color(red: 0.082, green: 0.094, blue: 0.114) : .white // #15181d
    }
}

/// Every colour the map paints itself with. Mirrors `HEAT_PALETTES` in
/// web_app/components/SP500Heatmap.tsx. The two schemes are the same scale read
/// against opposite surfaces: on black, "more" means brighter; on white it means
/// more saturated and darker. Only the endpoints and the neutral point change —
/// the geometry of the scale is identical, so a tile means the same thing in
/// either mode.
private struct HeatPalette {
    /// Colour at the metric's neutral point on the diverging scale.
    let mid: RGB
    /// Saturated ends of the diverging scale.
    let pos: RGB
    let neg: RGB
    /// Ends of the one-hue magnitude ramp.
    let seqLow: RGB
    let seqHigh: RGB
    /// A tile with no reading — deliberately achromatic, so it can never be
    /// mistaken for a weak red or a weak green.
    let noData: RGB
    /// Surface behind the tiles; it shows through the 1pt gaps as the grid.
    let surface: RGB
    let sectorHeader: RGB
    let industryHeader: RGB
    let headerText: Color
    let headerRule: RGB
    /// Tile luminance above which a label switches from white to dark ink.
    let labelFlip: Double

    // Black surface: the neutral point is the surface itself, and colour is
    // added as light. The sequential ramp is validated as an ordinal scale
    // against that surface (monotone lightness, sufficient per-step separation,
    // and a low end clearly distinct from the grey used for "no data").
    static let dark = HeatPalette(
        mid: RGB(hex: 0x000000),
        pos: RGB(hex: 0x30cc5a),
        neg: RGB(hex: 0xf63538),
        seqLow: RGB(hex: 0x165433),
        seqHigh: RGB(hex: 0x30cc5a),
        noData: RGB(hex: 0x1a1a1a),
        surface: RGB(hex: 0x000000),
        sectorHeader: RGB(hex: 0x2a2a2a),
        industryHeader: RGB(hex: 0x3a3a3a),
        headerText: .white,
        headerRule: RGB(hex: 0x000000),
        // Above pure white, i.e. never reached: colour is only ever added to
        // black here, so the dark map is white-labelled throughout — the look
        // it has always had.
        labelFlip: 1.01
    )

    // Light surface: the neutral point is a near-white the window can carry, and
    // colour is added as ink. The endpoints are darkened relative to the dark
    // scheme's so white tile labels keep their contrast, and the sequential ramp
    // runs pale -> deep, since on paper "more" reads as darker.
    static let light = HeatPalette(
        mid: RGB(hex: 0xeff1f3),
        pos: RGB(hex: 0x158f47),
        neg: RGB(hex: 0xd02b2f),
        seqLow: RGB(hex: 0xd8efdf),
        seqHigh: RGB(hex: 0x106b38),
        noData: RGB(hex: 0xbcc0c7),
        surface: RGB(hex: 0xb4b9c0),
        sectorHeader: RGB(hex: 0xc3c8ce),
        industryHeader: RGB(hex: 0xdbdfe4),
        headerText: Color(red: 0.122, green: 0.137, blue: 0.157), // #1f2328
        headerRule: RGB(hex: 0xb4b9c0),
        // The crossover that maximises the worst case: white and dark ink reach
        // equal contrast here, so no tile on the light scale falls below 4.2:1.
        labelFlip: 0.2
    )

    static func of(_ scheme: ColorScheme) -> HeatPalette { scheme == .dark ? .dark : .light }
}

private func heatFill(_ value: Double?, metric: HeatMetric, palette: HeatPalette) -> RGB {
    guard let v = value, v.isFinite else { return palette.noData }

    if metric.isSequential {
        let (low, high) = metric.domain
        let t = min(max((v - low) / (high - low), 0), 1)
        return palette.seqLow.mixed(to: palette.seqHigh, t)
    }

    let pct = (signedDeviation(v, metric) / metric.clamp) * 3.0 // scale approximately to -3 / +3

    if pct >= 3.0 { return palette.pos }
    if pct <= -3.0 { return palette.neg }

    if pct > 0 { return palette.mid.mixed(to: palette.pos, pct / 3.0) }
    if pct < 0 { return palette.mid.mixed(to: palette.neg, -pct / 3.0) }
    return palette.mid
}

private func formatCompactCap(_ v: Double) -> String {
    if v >= 1e12 { return String(format: "$%.1fT", v / 1e12) }
    if v >= 1e9  { return String(format: "$%.1fB", v / 1e9) }
    if v >= 1e6  { return String(format: "$%.0fM", v / 1e6) }
    return String(format: "$%.0f", v)
}

// MARK: - Squarified Treemap Layout

private let sectorHeaderHeight: CGFloat = 22
private let industryHeaderHeight: CGFloat = 16
/// Below this width the sub-industry tier costs more space than it conveys.
private let flatLayoutWidth: CGFloat = 640

private struct TreemapNode: Identifiable {
    /// Stable across rebuilds — a fresh UUID per rebuild would change every
    /// row's identity and make SwiftUI tear down and recreate all ~640 views
    /// on each render.
    var id: String
    var name: String
    var type: Int // 0=Root, 1=Sector, 2=Industry, 3=Company
    var value: Double
    var metricVal: Double?
    var color: Color?
    /// Label ink for this tile, resolved against its own fill.
    var textColor: Color?
    var item: SP500HeatmapItem?
    var children: [TreemapNode]
    var frame: CGRect = .zero
    /// False when the box was too short to afford a header, so the label is
    /// suppressed and the space goes to the constituents instead.
    var showHeader: Bool = false

    /// Leaves under this node, i.e. how many constituents the map actually draws.
    var companyCount: Int {
        type == 3 ? 1 : children.reduce(0) { $0 + $1.companyCount }
    }
}

/// Carries the laid-out map width back up so the tree can decide whether the
/// sub-industry tier fits.
private struct MapWidthKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

private func squarifyHierarchy(node: inout TreemapNode, in rect: CGRect) {
    node.frame = rect
    guard !node.children.isEmpty else { return }

    let headerHeight: CGFloat = node.type == 1 ? sectorHeaderHeight : (node.type == 2 ? industryHeaderHeight : 0)
    let sidePad: CGFloat = node.type == 1 ? 2 : (node.type == 2 ? 1 : 0)

    // A header must leave at least as much room for the constituents as it
    // takes for itself; otherwise it is dropped. Reserving it unconditionally
    // used to zero out every box shorter than the header, silently deleting
    // whole sub-industries from the map.
    let showHeader = headerHeight > 0 && rect.height >= headerHeight * 2
    node.showHeader = showHeader
    let topPad = showHeader ? headerHeight : 0

    let innerRect = CGRect(
        x: rect.minX + sidePad,
        y: rect.minY + topPad,
        width: rect.width - sidePad * 2,
        height: rect.height - topPad - (showHeader ? sidePad : 0)
    )

    if innerRect.width <= 0 || innerRect.height <= 0 {
        for i in node.children.indices { node.children[i].frame = .zero }
        return
    }

    squarifyFlat(nodes: &node.children, in: innerRect)

    for i in node.children.indices {
        squarifyHierarchy(node: &node.children[i], in: node.children[i].frame)
    }
}

private func squarifyFlat(nodes: inout [TreemapNode], in rect: CGRect) {
    guard !nodes.isEmpty, rect.width > 0, rect.height > 0 else { return }
    let total = nodes.reduce(0.0) { $0 + $1.value }
    guard total > 0 else { return }

    var remaining = Array(nodes.indices)
    var area = rect

    func layoutStrip(_ indices: [Int], vertical: Bool) {
        let stripTotal = indices.reduce(0.0) { $0 + nodes[$1].value }
        let stripSize = vertical
            ? area.width * (stripTotal / (remaining.reduce(0.0) { $0 + nodes[$1].value }))
            : area.height * (stripTotal / (remaining.reduce(0.0) { $0 + nodes[$1].value }))

        var offset: CGFloat = 0
        for idx in indices {
            let frac = nodes[idx].value / stripTotal
            let length = vertical ? area.height : area.width
            let span = length * frac

            if vertical {
                nodes[idx].frame = CGRect(x: area.minX, y: area.minY + offset, width: stripSize, height: span)
            } else {
                nodes[idx].frame = CGRect(x: area.minX + offset, y: area.minY, width: span, height: stripSize)
            }
            offset += span
        }

        if vertical {
            area = CGRect(x: area.minX + stripSize, y: area.minY, width: area.width - stripSize, height: area.height)
        } else {
            area = CGRect(x: area.minX, y: area.minY + stripSize, width: area.width, height: area.height - stripSize)
        }
    }

    var strip: [Int] = []
    let sortedIndices = remaining.sorted { nodes[$0].value > nodes[$1].value }

    func worstAspect(_ indices: [Int], vertical: Bool) -> Double {
        let stripTotal = indices.reduce(0.0) { $0 + nodes[$1].value }
        let remainingTotal = remaining.reduce(0.0) { $0 + nodes[$1].value }
        guard remainingTotal > 0, stripTotal > 0 else { return .infinity }
        let sideLen = vertical ? area.width * (stripTotal / remainingTotal) : area.height * (stripTotal / remainingTotal)
        guard sideLen > 0 else { return .infinity }
        let length = vertical ? area.height : area.width
        var worst: Double = 0
        for idx in indices {
            let frac = nodes[idx].value / stripTotal
            let span = length * frac
            let aspect = sideLen > 0 && span > 0 ? max(sideLen / span, span / sideLen) : .infinity
            worst = max(worst, aspect)
        }
        return worst
    }

    for idx in sortedIndices {
        let vertical = area.width > area.height
        let currentWorst = strip.isEmpty ? Double.infinity : worstAspect(strip, vertical: vertical)
        let candidateWorst = worstAspect(strip + [idx], vertical: vertical)

        if strip.isEmpty || candidateWorst <= currentWorst {
            strip.append(idx)
        } else {
            layoutStrip(strip, vertical: vertical)
            remaining = remaining.filter { !strip.contains($0) }
            strip = [idx]
        }
    }
    if !strip.isEmpty {
        let vertical = area.width > area.height
        layoutStrip(strip, vertical: vertical)
    }
}

// MARK: - Treemap View

private func flattenNodes(_ node: TreemapNode) -> [TreemapNode] {
    var result = [node]
    for child in node.children {
        result.append(contentsOf: flattenNodes(child))
    }
    return result
}

private struct TreemapView: View {
    let root: TreemapNode
    let metric: HeatMetric
    let palette: HeatPalette
    let onTap: (String) -> Void

    private func layoutNodes(size: CGSize) -> [TreemapNode] {
        var mutableRoot = root
        squarifyHierarchy(node: &mutableRoot, in: CGRect(origin: .zero, size: size))
        return flattenNodes(mutableRoot)
    }

    var body: some View {
        GeometryReader { geo in
            let allNodes = layoutNodes(size: geo.size)

            ZStack(alignment: .topLeading) {
                // The surface acts as thick borders thanks to padding in squarify
                palette.surface.color.frame(width: geo.size.width, height: geo.size.height)

                ForEach(allNodes) { node in
                    if node.frame.width > 0 && node.frame.height > 0 {
                        if node.type == 1 { // Sector
                            if node.showHeader {
                                Text(node.name.uppercased())
                                    .font(.system(size: 11, weight: .bold))
                                    .foregroundStyle(palette.headerText)
                                    // Padding inside the frame, so the header cannot
                                    // overhang the box it belongs to.
                                    .padding(.leading, 4)
                                    .frame(width: node.frame.width, height: sectorHeaderHeight, alignment: .leading)
                                    .background(palette.sectorHeader.color)
                                    .border(palette.headerRule.color, width: 1)
                                    .position(x: node.frame.midX, y: node.frame.minY + sectorHeaderHeight / 2)
                            }
                        } else if node.type == 2 { // Industry
                            if node.showHeader {
                                Text(node.name.uppercased())
                                    .font(.system(size: 9, weight: .semibold))
                                    .foregroundStyle(palette.headerText.opacity(0.9))
                                    .padding(.leading, 3)
                                    .frame(width: node.frame.width, height: industryHeaderHeight, alignment: .leading)
                                    .background(palette.industryHeader.color)
                                    .border(palette.headerRule.color, width: 1)
                                    .position(x: node.frame.midX, y: node.frame.minY + industryHeaderHeight / 2)
                            }
                        } else if node.type == 3 { // Company
                            Button {
                                if let sym = node.item?.symbol { onTap(sym) }
                            } label: {
                                ZStack {
                                    Rectangle().fill(node.color ?? palette.mid.color)
                                        .frame(width: max(0, node.frame.width - 1), height: max(0, node.frame.height - 1)) // 1px gaps between companies

                                    VStack(spacing: 1) {
                                        if node.frame.width > 32 && node.frame.height > 18 {
                                            Text(node.name)
                                                .font(.system(size: node.frame.width > 60 ? 11 : 8, weight: .heavy))
                                                .foregroundStyle(node.textColor ?? .white)
                                                .lineLimit(1)
                                        }
                                        if node.frame.width > 44 && node.frame.height > 30, let mv = node.metricVal {
                                            Text(metric.formatValue(mv))
                                                .font(.system(size: node.frame.width > 60 ? 9 : 7, weight: .semibold))
                                                .foregroundStyle((node.textColor ?? .white).opacity(0.9))
                                                .monospacedDigit()
                                                .lineLimit(1)
                                        }
                                    }
                                }
                                .frame(width: node.frame.width, height: node.frame.height)
                            }
                            .buttonStyle(.plain)
                            .position(x: node.frame.midX, y: node.frame.midY)
                            .help("\(node.item?.name ?? "") (\(node.name)) — \(formatCompactCap(node.item?.marketCap ?? 0))")
                        } else {
                            EmptyView()
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Main View

struct SP500HeatmapView: View {
    @StateObject private var viewModel = SP500HeatmapViewModel()
    @State private var metric: HeatMetric = .day
    @State private var sectorFilter = "All"
    @State private var sizeMode: SizeMode = .cap
    @State private var stockDetail: SymbolID?
    @State private var mapWidth: CGFloat = 0
    @Environment(\.colorScheme) private var colorScheme

    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    #endif


    @ViewBuilder private var colorPicker: some View {
        Picker("Color by", selection: $metric) {
            ForEach(Self.metricGroups, id: \.name) { group in
                Section(header: Text(group.name)) {
                    ForEach(group.metrics) { m in
                        Text(m.rawValue).tag(m)
                    }
                }
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
    }

    private static let metricGroups: [(name: String, metrics: [HeatMetric])] = [
        ("Performance", HeatMetric.performance),
        ("Valuation", HeatMetric.valuation),
        ("Earnings & Sales", HeatMetric.earnings),
        ("Profitability", HeatMetric.profitability),
        ("Market", HeatMetric.market),
    ]

    @ViewBuilder private var sectorPicker: some View {
        Picker("Sector", selection: $sectorFilter) {
            ForEach(sectors, id: \.self) { s in
                Text(s == "All" ? "All Sectors" : s).tag(s)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
    }

    @ViewBuilder private var sizePicker: some View {
        Picker("Size", selection: $sizeMode) {
            ForEach(SizeMode.allCases) { s in Text(s.rawValue).tag(s) }
        }
        .labelsHidden()
        .pickerStyle(.segmented)
    }

    private var sectors: [String] {
        let s = Set(viewModel.items.map(\.sector).filter { !$0.isEmpty })
        return ["All"] + s.sorted()
    }

    private var filteredItems: [SP500HeatmapItem] {
        sectorFilter == "All"
            ? viewModel.items
            : viewModel.items.filter { $0.sector == sectorFilter }
    }

    /// Builds the tree once per (data, filter, metric, sizing, width) change.
    /// Deliberately a function taking every input, so the caller can memoise it
    /// — evaluating this twice per `body` used to regroup all 500 constituents
    /// on every render.
    private static func buildRoot(
        items: [SP500HeatmapItem],
        sectorFilter: String,
        metric: HeatMetric,
        sizeMode: SizeMode,
        flatLayout: Bool,
        palette: HeatPalette
    ) -> TreemapNode {
        var rootChildren: [TreemapNode] = []

        func companyNode(_ item: SP500HeatmapItem) -> TreemapNode? {
            let mv = item.marketCap ?? 0
            if mv <= 0 && sizeMode == .cap { return nil }
            let val = metric.scaledValue(for: item)
            let fill = heatFill(val, metric: metric, palette: palette)
            return TreemapNode(
                id: "c_\(item.symbol)", name: item.symbol, type: 3,
                value: sizeMode == .equal ? 1 : mv, metricVal: val,
                color: fill.color, textColor: fill.label(flippingAbove: palette.labelFlip),
                item: item, children: []
            )
        }

        // Drilled into one sector, the top tier becomes sub-industry.
        let topTier = Dictionary(grouping: items, by: { sectorFilter != "All" ? $0.subIndustry : $0.sector })
        // With a sector selected the leaves already sit one level down, and the
        // narrow layout deliberately skips the sub-industry tier.
        let nestIndustries = sectorFilter == "All" && !flatLayout

        for (sectorName, sectorItems) in topTier {
            var sectorChildren: [TreemapNode] = []

            if !nestIndustries {
                sectorChildren = sectorItems.compactMap(companyNode)
            } else {
                let groupedByIndustry = Dictionary(grouping: sectorItems, by: { $0.subIndustry })
                for (industryName, industryItems) in groupedByIndustry {
                    var industryChildren = industryItems.compactMap(companyNode)
                    if !industryChildren.isEmpty {
                        industryChildren.sort { $0.value > $1.value }
                        let indWeight = industryChildren.reduce(0) { $0 + $1.value }
                        sectorChildren.append(TreemapNode(
                            id: "i_\(sectorName)_\(industryName)", name: industryName, type: 2,
                            value: indWeight, children: industryChildren
                        ))
                    }
                }
            }

            if !sectorChildren.isEmpty {
                sectorChildren.sort { $0.value > $1.value }
                let secWeight = sectorChildren.reduce(0) { $0 + $1.value }
                rootChildren.append(TreemapNode(
                    id: "s_\(sectorName)", name: sectorName, type: 1,
                    value: secWeight, children: sectorChildren
                ))
            }
        }

        // Dictionary iteration order is not stable, so sort ties by name too;
        // otherwise equal-sized boxes could swap places between renders.
        rootChildren.sort { $0.value == $1.value ? $0.name < $1.name : $0.value > $1.value }
        let rootWeight = rootChildren.reduce(0) { $0 + $1.value }
        return TreemapNode(id: "root", name: "Root", type: 0, value: rootWeight, children: rootChildren)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("S&P 500 Heatmap", systemImage: "square.grid.3x3").font(.title3.bold())

// Controls
            #if os(iOS)
            if hSize == .compact {
                VStack(alignment: .leading, spacing: 10) {
                    colorPicker.frame(maxWidth: .infinity, alignment: .leading)
                    sectorPicker.frame(maxWidth: .infinity, alignment: .leading)
                    sizePicker.frame(maxWidth: .infinity)
                }
            } else {
                HStack(spacing: 10) {
                    colorPicker.frame(maxWidth: 180)
                    sectorPicker.frame(maxWidth: 200)
                    Spacer()
                    sizePicker.frame(maxWidth: 160)
                }
            }
            #else
            HStack(spacing: 10) {
                colorPicker.frame(maxWidth: 180)
                sectorPicker.frame(maxWidth: 200)
                Spacer()
                sizePicker.frame(maxWidth: 160)
            }
            #endif

            // Heatmap
            if viewModel.isLoading && viewModel.items.isEmpty {
                HStack { Spacer(); ProgressView("Loading S&P 500 data…").controlSize(.small); Spacer() }
                    .frame(height: 400)
            } else if let error = viewModel.errorMessage, viewModel.items.isEmpty {
                // Previously set but never rendered — a failed load looked
                // identical to an empty index.
                ContentUnavailableView {
                    Label("Couldn't load the heatmap", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(error)
                } actions: {
                    Button("Try Again") { viewModel.load() }
                }
                .frame(height: 400)
            } else if viewModel.items.isEmpty {
                ContentUnavailableView("No data available", systemImage: "square.grid.3x3")
                    .frame(height: 400)
            } else {
                // Built once per body pass and shared with the legend below;
                // evaluating it twice regrouped all 500 constituents each time.
                let palette = HeatPalette.of(colorScheme)
                let node = Self.buildRoot(
                    items: filteredItems, sectorFilter: sectorFilter, metric: metric,
                    sizeMode: sizeMode, flatLayout: mapWidth > 0 && mapWidth < flatLayoutWidth,
                    palette: palette
                )
                if node.children.isEmpty {
                    ContentUnavailableView("No valid data to layout", systemImage: "chart.bar.doc.horizontal")
                        .frame(height: 400)
                } else {
                    TreemapView(root: node, metric: metric, palette: palette, onTap: { stockDetail = SymbolID(id: $0) })
                        .frame(height: 600)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        // `primary` flips with the scheme, so the hairline stays
                        // a faint contrast against the map in either mode.
                        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Color.primary.opacity(0.1), lineWidth: 1))
                        .background(
                            GeometryReader { proxy in
                                Color.clear.preference(key: MapWidthKey.self, value: proxy.size.width)
                            }
                        )
                        .onPreferenceChange(MapWidthKey.self) { mapWidth = $0 }
                }

                // The legend spans exactly the metric's own domain, colouring each
                // stop with the value it represents — so the gradient and the end
                // labels agree for inverted metrics (low P/E green) and the
                // one-sided ones start at their real floor rather than at a
                // negative that cannot occur.
                let (legendLow, legendHigh) = metric.domain
                HStack {
                    Text(metric.formatValue(legendLow))
                        .font(.system(size: 10)).foregroundStyle(.secondary).monospacedDigit()
                    LinearGradient(
                        colors: [0, 0.25, 0.5, 0.75, 1].map { t in
                            heatFill(legendLow + t * (legendHigh - legendLow), metric: metric, palette: palette).color
                        },
                        startPoint: .leading, endPoint: .trailing
                    )
                    .frame(width: 140, height: 8)
                    .clipShape(Capsule())
                    Text(metric.formatValue(legendHigh))
                        .font(.system(size: 10)).foregroundStyle(.secondary).monospacedDigit()
                    Spacer()
                    Text("\(node.companyCount) of \(filteredItems.count) stocks · \(node.children.count) \(sectorFilter == "All" && !(mapWidth > 0 && mapWidth < flatLayoutWidth) ? "sectors" : "groups")")
                        .font(.system(size: 10)).foregroundStyle(.secondary.opacity(0.6))
                }
            }
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
        .task { viewModel.load() }
        .sheet(item: $stockDetail) { StockDetailView(symbol: $0.id, currency: "USD") }
    }
}
