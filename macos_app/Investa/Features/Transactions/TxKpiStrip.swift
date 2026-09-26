import SwiftUI

/// Activity counts + per-currency cash-flow cards (mirrors transactions/TxKpiStrip.tsx).
/// Sums are never converted across currencies — there's no per-transaction FX.
struct TxKpiStrip: View {
    let transactions: [Transaction]
    let preferredCurrency: String

    /// Width offered to the ledger; 0 until measured, which keeps it stacked.
    @State private var ledgerWidth: CGFloat = 0

    private struct Bucket { var count = 0; var inflow = 0.0; var outflow = 0.0; var fees = 0.0; var tax = 0.0; var traded = 0.0 }
    private struct Row: Identifiable { let currency: String; let b: Bucket; var netFlow: Double { b.inflow - b.outflow }; var id: String { currency } }
    private struct Counts { var total = 0; var buy = 0; var sell = 0; var dividend = 0; var interest = 0; var deposit = 0; var withdrawal = 0; var tax = 0; var fees = 0 }

    private var computed: (counts: Counts, rows: [Row]) {
        var counts = Counts()
        var byCcy: [String: Bucket] = [:]
        for tx in transactions {
            counts.total += 1
            let t = tx.type.lowercased()
            let ccy = tx.localCurrency.uppercased()
            let amount = abs(tx.totalAmount)
            let fee = abs(tx.commission)
            switch t {
            case "buy": counts.buy += 1
            case "sell": counts.sell += 1
            case "dividend": counts.dividend += 1
            case "interest": counts.interest += 1
            case "deposit": counts.deposit += 1
            case "withdrawal": counts.withdrawal += 1
            case "tax": counts.tax += 1
            case "fees": counts.fees += 1
            default: break
            }
            var b = byCcy[ccy] ?? Bucket()
            b.count += 1
            switch t {
            case "deposit", "dividend", "interest": b.inflow += amount
            case "withdrawal": b.outflow += amount
            case "tax": b.tax += amount; b.outflow += amount
            case "fees": b.fees += amount; b.outflow += amount
            case "buy", "sell": b.traded += amount
            default: break
            }
            if fee > 0 && t != "fees" { b.fees += fee }
            byCcy[ccy] = b
        }
        let preferred = preferredCurrency.uppercased()
        let allRows: [Row] = byCcy.map { Row(currency: $0.key, b: $0.value) }
        let active: [Row] = allRows.filter { row in
            let b = row.b
            return abs(row.netFlow) > 0.001 || b.fees > 0.001 || b.tax > 0.001 || b.inflow > 0.001 || b.outflow > 0.001
        }
        let rows: [Row] = active.sorted { a, b in
            if a.currency == preferred && b.currency != preferred { return true }
            if b.currency == preferred && a.currency != preferred { return false }
            return a.b.count > b.b.count
        }
        return (counts, rows)
    }

    private func compact(_ v: Double) -> String {
        let a = abs(v)
        if a >= 1_000_000 { return String(format: "%.2fM", v / 1_000_000) }
        if a >= 10_000 { return String(format: "%.1fK", v / 1_000) }
        if a >= 100 { return String(format: "%.0f", v) }
        return String(format: "%.2f", v)
    }

    var body: some View {
        let c = computed
        return VStack(alignment: .leading, spacing: 12) {
            // Activity counts — one wrapping line on every platform.
            WrappingRow(spacing: 16, lineSpacing: 6) {
                SectionLabel(title: "Activity")
                activity("\(c.counts.total)", "transactions")
                if c.counts.buy + c.counts.sell > 0 {
                    HStack(spacing: 4) { activity("\(c.counts.buy)", "buys"); Text("/").foregroundStyle(.tertiary); activity("\(c.counts.sell)", "sells") }
                }
                if c.counts.dividend + c.counts.interest > 0 {
                    HStack(spacing: 4) {
                        activity("\(c.counts.dividend)", "div", tint: .up)
                        if c.counts.interest > 0 { Text("·").foregroundStyle(.tertiary); activity("\(c.counts.interest)", "int", tint: .up) }
                    }
                }
                if c.counts.deposit + c.counts.withdrawal > 0 { activity("\(c.counts.deposit + c.counts.withdrawal)", "cash flows") }
            }
            if !c.rows.isEmpty {
                Rectangle().fill(Color.line).frame(height: 1)
                ledger(c.rows)
                    .readingContainerWidth { ledgerWidth = $0 }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card(.standard)
    }

    /// One row per currency with shared columns (mirrors the web ledger). Below
    /// `needs` each currency stacks: tag + net on one line, the four figures under it.
    @ViewBuilder
    private func ledger(_ rows: [Row]) -> some View {
        if prefersStackedLayout(measuredWidth: ledgerWidth, needs: 560) {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(rows.enumerated()), id: \.element.id) { i, row in
                    if i > 0 { Rectangle().fill(Color.line).frame(height: 1) }
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            currencyTag(row.currency)
                            Spacer(minLength: 8)
                            netFigure(row)
                        }
                        HStack(alignment: .top, spacing: 12) {
                            figure(row.b.inflow, label: "In")
                            figure(row.b.outflow, label: "Out")
                            figure(row.b.fees, label: "Fees", tone: .warn)
                            figure(row.b.tax, label: "Tax", tone: .warn)
                        }
                    }
                    .padding(.vertical, 10)
                }
            }
        } else {
            Grid(alignment: .trailing, horizontalSpacing: 16, verticalSpacing: 10) {
                GridRow {
                    header("Currency", alignment: .leading).gridColumnAlignment(.leading)
                    header("Net cash flow")
                    header("In")
                    header("Out")
                    header("Fees")
                    header("Tax")
                }
                ForEach(rows) { row in
                    Rectangle().fill(Color.line).frame(height: 1).gridCellUnsizedAxes(.horizontal)
                    GridRow {
                        currencyTag(row.currency)
                        netFigure(row)
                        figure(row.b.inflow)
                        figure(row.b.outflow)
                        figure(row.b.fees, tone: .warn)
                        figure(row.b.tax, tone: .warn)
                    }
                }
            }
        }
    }

    private func activity(_ value: String, _ label: String, tint: Color = .primary) -> some View {
        HStack(spacing: 4) {
            Text(value).appFont(.callout.weight(.semibold)).foregroundStyle(tint).monospacedDigit()
            Text(label).appFont(.caption2).foregroundStyle(.secondary)
        }
        .fixedSize(horizontal: true, vertical: false)
    }

    private func header(_ title: String, alignment: Alignment = .trailing) -> some View {
        Text(title).appFont(.caption2).foregroundStyle(Color.ink3)
            .lineLimit(1).minimumScaleFactor(0.7)
            .frame(maxWidth: alignment == .leading ? nil : .infinity, alignment: alignment)
    }

    private func currencyTag(_ code: String) -> some View {
        // An unbreakable word with no line limit *demands* its width rather
        // than preferring it.
        Text(code).appFont(.caption2.weight(.semibold)).tracking(0.8).lineLimit(1)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Color.inset, in: RoundedRectangle(cornerRadius: 4))
    }

    private func netFigure(_ row: Row) -> some View {
        let positive = row.netFlow >= 0
        return HStack(spacing: 4) {
            Image(systemName: positive ? "arrow.down.right" : "arrow.up.right").imageScale(.small)
            Text("\(positive ? "+" : "−")\(compact(abs(row.netFlow)))").appFont(.title3.weight(.semibold)).monospacedDigit()
        }
        .foregroundStyle(positive ? Color.up : Color.down)
        // On the stack, so a figure added beside it can't opt out. Both parts:
        // `lineLimit(1)` alone converts the wrap into an ellipsis, and half a
        // net figure is a different number.
        .lineLimit(1)
        .minimumScaleFactor(0.6)
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    /// One figure cell. The label is only drawn in the stacked layout — in the
    /// grid the header row names the column.
    private func figure(_ value: Double, label: String? = nil, tone: Color = .primary) -> some View {
        let has = value > 0.001
        return VStack(alignment: label == nil ? .trailing : .leading, spacing: 2) {
            if let label { SectionLabel(title: label) }
            Text(has ? compact(value) : "—")
                .appFont(.callout.weight(.medium)).monospacedDigit()
                .foregroundStyle(has ? tone : Color.secondary.opacity(0.4))
        }
        // Four of these share a line, so at an accessibility size they have to
        // shrink rather than wrap into each other.
        .lineLimit(1)
        .minimumScaleFactor(0.7)
        .frame(maxWidth: .infinity, alignment: label == nil ? .trailing : .leading)
    }
}
