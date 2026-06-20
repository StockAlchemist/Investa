import SwiftUI

/// Activity counts + per-currency cash-flow cards (mirrors transactions/TxKpiStrip.tsx).
/// Sums are never converted across currencies — there's no per-transaction FX.
struct TxKpiStrip: View {
    let transactions: [Transaction]
    let preferredCurrency: String

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
        return VStack(alignment: .leading, spacing: 14) {
            // Activity counts
            #if os(iOS)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    Text("Activity").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                    activity("\(c.counts.total)", "transactions")
                    if c.counts.buy + c.counts.sell > 0 {
                        HStack(spacing: 4) { activity("\(c.counts.buy)", "buys"); Text("/").foregroundStyle(.secondary); activity("\(c.counts.sell)", "sells") }
                    }
                    if c.counts.dividend + c.counts.interest > 0 {
                        activity("\(c.counts.dividend)", "div", tint: .green)
                        if c.counts.interest > 0 { activity("\(c.counts.interest)", "int", tint: .green) }
                    }
                    if c.counts.deposit + c.counts.withdrawal > 0 { activity("\(c.counts.deposit + c.counts.withdrawal)", "cash flows") }
                }
            }
            #else
            HStack(spacing: 16) {
                Text("Activity").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                Spacer()
                activity("\(c.counts.total)", "transactions")
                if c.counts.buy + c.counts.sell > 0 {
                    HStack(spacing: 4) { activity("\(c.counts.buy)", "buys"); Text("/").foregroundStyle(.secondary); activity("\(c.counts.sell)", "sells") }
                }
                if c.counts.dividend + c.counts.interest > 0 {
                    activity("\(c.counts.dividend)", "div", tint: .green)
                    if c.counts.interest > 0 { activity("\(c.counts.interest)", "int", tint: .green) }
                }
                if c.counts.deposit + c.counts.withdrawal > 0 { activity("\(c.counts.deposit + c.counts.withdrawal)", "cash flows") }
            }
            #endif
            if !c.rows.isEmpty {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 240), spacing: 12)], spacing: 12) {
                    ForEach(c.rows) { row in currencyCard(row) }
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func activity(_ value: String, _ label: String, tint: Color = .primary) -> some View {
        HStack(spacing: 4) {
            Text(value).font(.callout.bold()).foregroundStyle(tint).monospacedDigit()
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func currencyCard(_ row: Row) -> some View {
        let positive = row.netFlow >= 0
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                HStack(spacing: 4) {
                    Image(systemName: positive ? "arrow.down.right" : "arrow.up.right")
                    Text("\(positive ? "+" : "−")\(compact(abs(row.netFlow)))").font(.title2.bold()).monospacedDigit()
                }
                .foregroundStyle(positive ? .green : .red)
                Spacer()
                Text(row.currency).font(.caption2.weight(.bold)).padding(.horizontal, 6).padding(.vertical, 2)
                    .background(.quaternary, in: Capsule())
            }
            Text("net cash flow").font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            HStack {
                kv("In", row.b.inflow > 0 ? compact(row.b.inflow) : "—", .primary)
                Spacer()
                kv("Out", row.b.outflow > 0 ? compact(row.b.outflow) : "—", .primary, trailing: true)
            }
            HStack {
                kv("Fees", row.b.fees > 0 ? compact(row.b.fees) : "—", row.b.fees > 0 ? .orange : .secondary)
                Spacer()
                kv("Tax", row.b.tax > 0 ? compact(row.b.tax) : "—", row.b.tax > 0 ? .orange : .secondary, trailing: true)
            }
        }
        .padding(12)
        .background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
    }

    private func kv(_ k: String, _ v: String, _ tone: Color, trailing: Bool = false) -> some View {
        VStack(alignment: trailing ? .trailing : .leading, spacing: 1) {
            Text(k).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(v).font(.caption.bold()).foregroundStyle(tone).monospacedDigit()
        }
    }
}
