import Foundation
import SwiftUI

/// Shared number/percent formatting helpers for the dashboard.
enum Fmt {
    static func currency(_ value: Double?, code: String) -> String {
        guard let value else { return "—" }
        let f = NumberFormatter()
        f.numberStyle = .currency
        f.currencyCode = code
        f.maximumFractionDigits = 2
        return f.string(from: NSNumber(value: value)) ?? "—"
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
        return value > 0 ? .green : .red
    }
}
