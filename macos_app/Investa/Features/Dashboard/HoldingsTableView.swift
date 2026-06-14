import SwiftUI

/// Flattened, sortable presentation row resolved for the active currency.
struct HoldingRow: Identifiable {
    let id: String
    let symbol: String
    let account: String
    let quantity: Double
    let marketValue: Double
    let dayChangePct: Double
    let unrealizedPct: Double
    let totalReturnPct: Double

    init(_ h: Holding, currency: String) {
        id = h.id
        symbol = h.symbol
        account = h.account ?? ""
        quantity = h.quantity ?? 0
        marketValue = h.marketValue(currency: currency) ?? 0
        dayChangePct = h.dayChangePct ?? 0
        unrealizedPct = h.unrealizedGainPct ?? 0
        totalReturnPct = h.totalReturnPct ?? 0
    }
}

struct HoldingsTableView: View {
    let holdings: [Holding]
    let currency: String

    @State private var sortOrder = [KeyPathComparator(\HoldingRow.marketValue, order: .reverse)]

    private var rows: [HoldingRow] {
        holdings.map { HoldingRow($0, currency: currency) }.sorted(using: sortOrder)
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
                    TableColumn("Symbol", value: \.symbol) { row in
                        Text(row.symbol).fontWeight(.medium)
                    }
                    .width(min: 70, ideal: 90)

                    TableColumn("Account", value: \.account)
                        .width(min: 80, ideal: 120)

                    TableColumn("Qty", value: \.quantity) { row in
                        Text(Fmt.number(row.quantity, fractionDigits: 2))
                            .monospacedDigit()
                    }
                    .width(min: 60, ideal: 80)

                    TableColumn("Market Value", value: \.marketValue) { row in
                        Text(Fmt.currency(row.marketValue, code: currency))
                            .monospacedDigit()
                    }
                    .width(min: 110, ideal: 140)

                    TableColumn("Day %", value: \.dayChangePct) { row in
                        Text(Fmt.percent(row.dayChangePct))
                            .monospacedDigit()
                            .foregroundStyle(Fmt.tint(for: row.dayChangePct))
                    }
                    .width(min: 70, ideal: 90)

                    TableColumn("Unreal. %", value: \.unrealizedPct) { row in
                        Text(Fmt.percent(row.unrealizedPct))
                            .monospacedDigit()
                            .foregroundStyle(Fmt.tint(for: row.unrealizedPct))
                    }
                    .width(min: 80, ideal: 100)

                    TableColumn("Total Return %", value: \.totalReturnPct) { row in
                        Text(Fmt.percent(row.totalReturnPct))
                            .monospacedDigit()
                            .foregroundStyle(Fmt.tint(for: row.totalReturnPct))
                    }
                    .width(min: 100, ideal: 120)
                }
                .frame(minHeight: 280)
            }
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1)
        )
    }
}
