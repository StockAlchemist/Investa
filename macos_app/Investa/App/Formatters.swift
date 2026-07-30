import Foundation
import SwiftUI

/// Shared number/percent formatting helpers for the dashboard.
enum Fmt {
    /// Reusable formatters keyed by currency code — avoids allocating a new
    /// `NumberFormatter` on every call in hot layout paths.
    ///
    /// Guarded by a lock because both the main thread (UI rendering) and
    /// background `Task`s (e.g. document-import decoding) can call
    /// `formatter(for:)` concurrently.
    private static let _lock = NSLock()
    private static var _currencyFormatters: [String: NumberFormatter] = [:]

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

    /// Re-groups a typed numeric string as it is entered — "1760402" becomes
    /// "1,760,402" (or the locale's own separator).
    ///
    /// Cosmetic only: `ungroup` is the exact inverse, so what is typed and what
    /// is parsed stay the same number. Grouping is applied by hand rather than
    /// through `NumberFormatter` so that a half-finished entry survives — a
    /// trailing "." or an empty string has no numeric value to round-trip.
    static func regroup(_ raw: String) -> String {
        let cleaned = raw.filter { $0.isNumber || $0 == "." }
        let dot = cleaned.firstIndex(of: ".")
        let whole = cleaned[cleaned.startIndex..<(dot ?? cleaned.endIndex)]
        let fraction = dot.map { String(cleaned[cleaned.index(after: $0)...]).filter(\.isNumber) }
        let separator = Locale.current.groupingSeparator ?? ","

        var reversed = ""
        for (offset, digit) in whole.reversed().enumerated() {
            if offset > 0, offset.isMultiple(of: 3) { reversed.append(contentsOf: separator.reversed()) }
            reversed.append(digit)
        }
        let grouped = String(reversed.reversed())
        return fraction.map { "\(grouped).\($0)" } ?? grouped
    }

    /// The number a grouped string stands for — the inverse of `regroup`.
    ///
    /// The separator is removed before anything else, so a locale that groups
    /// with "." (de_DE) does not leave "1.760.402" to be misread as 1.76.
    static func ungroup(_ raw: String) -> Double? {
        var text = raw
        if let separator = Locale.current.groupingSeparator {
            text = text.replacingOccurrences(of: separator, with: "")
        }
        return Double(text.filter { $0.isNumber || $0 == "." })
    }

    private static func formatter(for code: String) -> NumberFormatter {
        _lock.lock()
        defer { _lock.unlock() }
        if let f = _currencyFormatters[code] { return f }
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
        
        _currencyFormatters[code] = f
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
    static func percent(_ value: Double?, fractionDigits: Int = 2, includeSign: Bool = false) -> String {
        guard let value else { return "—" }
        let fmt = includeSign ? "%+.\(fractionDigits)f%%" : "%.\(fractionDigits)f%%"
        return String(format: fmt, value)
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
