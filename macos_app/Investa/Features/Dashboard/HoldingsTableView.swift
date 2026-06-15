import SwiftUI
import Charts

/// Flattened, sortable presentation row resolved for the active currency.
/// Mirrors the columns the web HoldingsTable exposes.
struct HoldingRow: Identifiable {
    let id: String
    let symbol: String
    let account: String
    let sector: String
    let industry: String
    let quantity: Double
    let price: Double
    let avgCost: Double
    let costBasis: Double
    let marketValue: Double
    let pctOfTotal: Double
    let dayChange: Double
    let dayChangePct: Double
    let unrealGL: Double
    let unrealizedPct: Double
    let realizedGL: Double
    let totalGL: Double
    let totalReturnPct: Double
    let irrPct: Double
    let dividends: Double
    let yieldMktPct: Double
    let estIncome: Double
    let aiScore: Double
    let intrinsicValue: Double
    let sparkline: [Double]

    init(_ h: Holding, currency: String, totalMarketValue: Double) {
        id = h.id
        symbol = h.symbol
        account = h.account ?? ""
        sector = h.string("Sector") ?? ""
        industry = h.string("Industry") ?? ""
        quantity = h.quantity ?? 0
        price = h.currencyValue("Price", currency: currency) ?? 0
        avgCost = h.currencyValue("Avg Cost", currency: currency) ?? 0
        costBasis = h.currencyValue("Cost Basis", currency: currency) ?? 0
        let mv = h.marketValue(currency: currency) ?? 0
        marketValue = mv
        pctOfTotal = totalMarketValue > 0 ? mv / totalMarketValue * 100 : 0
        dayChange = h.currencyValue("Day Change", currency: currency) ?? 0
        dayChangePct = h.dayChangePct ?? 0
        unrealGL = h.currencyValue("Unreal. Gain", currency: currency) ?? 0
        unrealizedPct = h.unrealizedGainPct ?? 0
        realizedGL = h.currencyValue("Realized Gain", currency: currency) ?? 0
        totalGL = h.currencyValue("Total Gain", currency: currency) ?? 0
        totalReturnPct = h.totalReturnPct ?? 0
        irrPct = h.irrPct ?? 0
        dividends = h.currencyValue("Dividends", currency: currency) ?? 0
        yieldMktPct = h.double("Div. Yield (Current) %") ?? 0
        estIncome = h.currencyValue("Est. Ann. Income", currency: currency) ?? 0
        aiScore = h.double("ai_score") ?? 0
        intrinsicValue = h.double("intrinsic_value") ?? 0
        sparkline = h.raw["sparkline_7d"]?.arrayValue?.compactMap { $0.doubleValue } ?? []
    }
}

struct HoldingsTableView: View {
    let holdings: [Holding]
    let currency: String

    @State private var sortOrder = [KeyPathComparator(\HoldingRow.marketValue, order: .reverse)]
    @State private var detail: SymbolID?

    private var totalMarketValue: Double {
        holdings.reduce(0) { $0 + ($1.marketValue(currency: currency) ?? 0) }
    }

    private var rows: [HoldingRow] {
        holdings
            .map { HoldingRow($0, currency: currency, totalMarketValue: totalMarketValue) }
            .sorted(using: sortOrder)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Holdings (\(holdings.count))")
                .font(.headline)

            if holdings.isEmpty {
                ContentUnavailableView("No holdings", systemImage: "tray")
                    .frame(height: 200)
            } else {
                Table(rows, sortOrder: $sortOrder) {
                    coreColumns
                    glColumns
                    returnColumns
                }
                .frame(minHeight: 320)
                .contextMenu(forSelectionType: HoldingRow.ID.self) { ids in
                    if let id = ids.first, let row = rows.first(where: { $0.id == id }) {
                        Button("View Details") { detail = SymbolID(id: row.symbol) }
                    }
                } primaryAction: { ids in
                    if let id = ids.first, let row = rows.first(where: { $0.id == id }) {
                        detail = SymbolID(id: row.symbol)
                    }
                }
            }
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1)
        )
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: currency) }
    }

    // Columns are grouped into sub-builders because @TableColumnBuilder only
    // supports up to 10 columns in a single block.

    @TableColumnBuilder<HoldingRow, KeyPathComparator<HoldingRow>>
    private var coreColumns: some TableColumnContent<HoldingRow, KeyPathComparator<HoldingRow>> {
        TableColumn("Symbol", value: \.symbol) { row in
            HStack(spacing: 6) { StockIcon(symbol: row.symbol, size: 16); Text(row.symbol).fontWeight(.medium) }
        }
        .width(min: 80, ideal: 100)
        TableColumn("7d", value: \.symbol) { sparkline($0) }
            .width(60)
        TableColumn("Qty", value: \.quantity) { Text(Fmt.number($0.quantity)).monospacedDigit() }
            .width(min: 50, ideal: 70)
        TableColumn("% Total", value: \.pctOfTotal) { row in
            Text(String(format: "%.1f%%", row.pctOfTotal)).monospacedDigit().foregroundStyle(.secondary)
        }
        .width(min: 55, ideal: 65)
        TableColumn("Price", value: \.price) {
            Text(Fmt.currency($0.price, code: currency)).monospacedDigit()
        }
        .width(min: 70, ideal: 90)
        TableColumn("Avg Cost", value: \.avgCost) {
            Text(Fmt.currency($0.avgCost, code: currency)).monospacedDigit().foregroundStyle(.secondary)
        }
        .width(min: 70, ideal: 90)
        TableColumn("Cost Basis", value: \.costBasis) {
            Text(Fmt.currency($0.costBasis, code: currency)).monospacedDigit().foregroundStyle(.secondary)
        }
        .width(min: 80, ideal: 100)
        TableColumn("Mkt Val", value: \.marketValue) {
            Text(Fmt.currency($0.marketValue, code: currency)).monospacedDigit()
        }
        .width(min: 90, ideal: 120)
    }

    @TableColumnBuilder<HoldingRow, KeyPathComparator<HoldingRow>>
    private var glColumns: some TableColumnContent<HoldingRow, KeyPathComparator<HoldingRow>> {
        TableColumn("Day %", value: \.dayChangePct) { row in
            Text(Fmt.percent(row.dayChangePct)).monospacedDigit().foregroundStyle(Fmt.tint(for: row.dayChangePct))
        }
        .width(min: 64, ideal: 80)
        TableColumn("Unreal. G/L", value: \.unrealGL) { row in
            Text(Fmt.currency(row.unrealGL, code: currency)).monospacedDigit().foregroundStyle(Fmt.tint(for: row.unrealGL))
        }
        .width(min: 90, ideal: 110)
        TableColumn("Unreal. %", value: \.unrealizedPct) { row in
            Text(Fmt.percent(row.unrealizedPct)).monospacedDigit().foregroundStyle(Fmt.tint(for: row.unrealizedPct))
        }
        .width(min: 72, ideal: 90)
        TableColumn("Real. G/L", value: \.realizedGL) { row in
            Text(Fmt.currency(row.realizedGL, code: currency)).monospacedDigit().foregroundStyle(Fmt.tint(for: row.realizedGL))
        }
        .width(min: 80, ideal: 100)
        TableColumn("Total G/L", value: \.totalGL) { row in
            Text(Fmt.currency(row.totalGL, code: currency)).monospacedDigit().foregroundStyle(Fmt.tint(for: row.totalGL))
        }
        .width(min: 80, ideal: 100)
    }

    @TableColumnBuilder<HoldingRow, KeyPathComparator<HoldingRow>>
    private var returnColumns: some TableColumnContent<HoldingRow, KeyPathComparator<HoldingRow>> {
        TableColumn("Total Ret %", value: \.totalReturnPct) { row in
            Text(Fmt.percent(row.totalReturnPct)).monospacedDigit().foregroundStyle(Fmt.tint(for: row.totalReturnPct))
        }
        .width(min: 84, ideal: 100)
        TableColumn("IRR %", value: \.irrPct) { row in
            Text(Fmt.percent(row.irrPct)).monospacedDigit().foregroundStyle(Fmt.tint(for: row.irrPct))
        }
        .width(min: 64, ideal: 80)
        TableColumn("Divs", value: \.dividends) {
            Text(Fmt.currency($0.dividends, code: currency)).monospacedDigit()
        }
        .width(min: 70, ideal: 90)
        TableColumn("Yield %", value: \.yieldMktPct) {
            Text(Fmt.percent($0.yieldMktPct)).monospacedDigit().foregroundStyle(.secondary)
        }
        .width(min: 64, ideal: 80)
        TableColumn("Sector", value: \.sector) {
            Text($0.sector.isEmpty ? "—" : $0.sector).lineLimit(1)
        }
        .width(min: 90, ideal: 130)
        TableColumn("AI", value: \.aiScore) {
            Text($0.aiScore > 0 ? Fmt.number($0.aiScore, fractionDigits: 0) : "—").monospacedDigit()
        }
        .width(min: 44, ideal: 56)
        TableColumn("Intrinsic", value: \.intrinsicValue) {
            Text($0.intrinsicValue > 0 ? Fmt.currency($0.intrinsicValue, code: currency) : "—")
                .monospacedDigit().foregroundStyle(.secondary)
        }
        .width(min: 80, ideal: 100)
    }

    @ViewBuilder private func sparkline(_ row: HoldingRow) -> some View {
        if row.sparkline.count > 1 {
            let up = (row.sparkline.last ?? 0) >= (row.sparkline.first ?? 0)
            Chart(Array(row.sparkline.enumerated()), id: \.offset) { idx, value in
                LineMark(x: .value("i", idx), y: .value("v", value))
                    .foregroundStyle(up ? Color.up : Color.down)
            }
            .chartXAxis(.hidden).chartYAxis(.hidden)
            .frame(height: 22)
        } else {
            Text("—").foregroundStyle(.secondary)
        }
    }
}
