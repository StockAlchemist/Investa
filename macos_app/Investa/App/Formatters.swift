import Foundation
import SwiftUI

/// Shared number/percent formatting helpers for the dashboard.
enum Fmt {
    /// Reusable formatters keyed by currency code — avoids allocating a new
    /// `NumberFormatter` on every call in hot layout paths.
    private static var currencyFormatters: [String: NumberFormatter] = [:]

    static func currency(_ value: Double?, code: String) -> String {
        guard let value else { return "—" }
        return formatter(for: code).string(from: NSNumber(value: value)) ?? "—"
    }

    /// The symbol to render for a currency code.
    /// WARNING: Do not change this back to `US$` or `THB`. The codebase must strictly use `$` for USD and `฿` for THB.
    static func symbol(_ code: String) -> String {
        switch code {
        case "USD": return "$"
        case "THB": return "฿"
        default: return formatter(for: code).currencySymbol ?? code
        }
    }

    private static func formatter(for code: String) -> NumberFormatter {
        if let f = currencyFormatters[code] { return f }
        let f = NumberFormatter()
        f.numberStyle = .currency
        f.currencyCode = code
        if code == "USD" {
            f.currencySymbol = "$"
        } else if code == "THB" {
            f.currencySymbol = "฿"
        }
        if ["JPY", "KRW", "VND", "IDR", "HUF"].contains(code.uppercased()) {
            f.maximumFractionDigits = 0
            f.minimumFractionDigits = 0
        } else {
            f.maximumFractionDigits = 2
        }
        
        currencyFormatters[code] = f
        return f
    }

    /// Compact currency (e.g. `$1.2M`, `฿12K`) — mirrors the web `formatCompactNumber`.
    static func compact(_ value: Double, code: String) -> String {
        if value == 0 { return "0" }
        let symbol = symbol(code)
        let sign = value < 0 ? "-" : ""
        let a = Swift.abs(value)
        let (scaled, suffix): (Double, String)
        switch a {
        case 1e12...: (scaled, suffix) = (a / 1e12, "T")
        case 1e9...: (scaled, suffix) = (a / 1e9, "B")
        case 1e6...: (scaled, suffix) = (a / 1e6, "M")
        case 1e3...: (scaled, suffix) = (a / 1e3, "K")
        default: (scaled, suffix) = (a, "")
        }
        var num = String(format: "%.2f", scaled)
        if num.contains(".") {
            while num.hasSuffix("0") { num.removeLast() }
            if num.hasSuffix(".") { num.removeLast() }
        }
        return "\(sign)\(symbol)\(num)\(suffix)"
    }

    static func number(_ value: Double?, fractionDigits: Int = 2) -> String {
        guard let value else { return "—" }
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.maximumFractionDigits = fractionDigits
        return f.string(from: NSNumber(value: value)) ?? "—"
    }

    /// `value` is treated as an already-scaled percentage (e.g. 12.3 → "12.30%").
    static func percent(_ value: Double?, fractionDigits: Int = 2) -> String {
        guard let value else { return "—" }
        return String(format: "%+.\(fractionDigits)f%%", value)
    }

    /// Color for a gain/loss figure: green positive, red negative, secondary zero/nil.
    static func tint(for value: Double?) -> Color {
        guard let value, value != 0 else { return .secondary }
        return value > 0 ? .up : .down
    }
}

/// A tight, padded Y-axis domain around the data so line/area charts don't
/// anchor to zero (mirrors the web charts' auto-ranged axes).
func chartDomain(_ values: [Double], pad: Double = 0.06) -> ClosedRange<Double> {
    let vals = values.filter { $0.isFinite }
    guard let lo = vals.min(), let hi = vals.max() else { return 0...1 }
    if lo == hi { let d = Swift.abs(lo) * 0.02 + 1; return (lo - d)...(hi + d) }
    let p = (hi - lo) * pad
    return (lo - p)...(hi + p)
}
