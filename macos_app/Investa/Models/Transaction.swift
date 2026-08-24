import Foundation
import SwiftUI

/// A transaction record. Maps to the backend's `TransactionInput`
/// (`src/server/routes/transactions.py`) — the JSON keys are the human-readable
/// aliases (`"Price/Share"`, `"Total Amount"`, …), handled via `CodingKeys`.
struct Transaction: Codable, Sendable, Identifiable {
    var id: Int?
    var date: String
    var account: String
    var symbol: String
    var type: String
    var quantity: Double
    var pricePerShare: Double
    var commission: Double
    var totalAmount: Double
    var localCurrency: String
    var splitRatio: Double?
    var note: String?
    var toAccount: String?
    var autoAddCash: Bool?

    /// `DD MMM YYYY`, the app's one date notation — never the raw ISO string
    /// the API ships.
    var displayDate: String {
        MarketTime.formatted(date)
    }

    enum CodingKeys: String, CodingKey {
        case id
        case date = "Date"
        case account = "Account"
        case symbol = "Symbol"
        case type = "Type"
        case quantity = "Quantity"
        case pricePerShare = "Price/Share"
        case commission = "Commission"
        case totalAmount = "Total Amount"
        case localCurrency = "Local Currency"
        case splitRatio = "Split Ratio"
        case note = "Note"
        case toAccount = "To Account"
        case autoAddCash = "Auto-add Cash"
    }

    // Tolerant decoder: backend rows may carry nulls for numeric columns.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(Int.self, forKey: .id)
        date = try c.decodeIfPresent(String.self, forKey: .date) ?? ""
        account = try c.decodeIfPresent(String.self, forKey: .account) ?? ""
        symbol = try c.decodeIfPresent(String.self, forKey: .symbol) ?? ""
        type = try c.decodeIfPresent(String.self, forKey: .type) ?? "Buy"
        quantity = try c.decodeIfPresent(Double.self, forKey: .quantity) ?? 0
        pricePerShare = try c.decodeIfPresent(Double.self, forKey: .pricePerShare) ?? 0
        commission = try c.decodeIfPresent(Double.self, forKey: .commission) ?? 0
        totalAmount = try c.decodeIfPresent(Double.self, forKey: .totalAmount) ?? 0
        localCurrency = try c.decodeIfPresent(String.self, forKey: .localCurrency) ?? "USD"
        splitRatio = try c.decodeIfPresent(Double.self, forKey: .splitRatio)
        note = try c.decodeIfPresent(String.self, forKey: .note)
        toAccount = try c.decodeIfPresent(String.self, forKey: .toAccount)
        autoAddCash = try c.decodeIfPresent(Bool.self, forKey: .autoAddCash)
    }

    init(
        id: Int? = nil, date: String, account: String, symbol: String, type: String,
        quantity: Double, pricePerShare: Double, commission: Double, totalAmount: Double,
        localCurrency: String, splitRatio: Double? = nil, note: String? = nil,
        toAccount: String? = nil, autoAddCash: Bool? = nil
    ) {
        self.id = id; self.date = date; self.account = account; self.symbol = symbol
        self.type = type; self.quantity = quantity; self.pricePerShare = pricePerShare
        self.commission = commission; self.totalAmount = totalAmount
        self.localCurrency = localCurrency; self.splitRatio = splitRatio; self.note = note
        self.toAccount = toAccount; self.autoAddCash = autoAddCash
    }
}

extension Transaction {
    /// Categorises how a transaction type affects the $CASH balance.
    enum CashImpact {
        /// Reduces cash (Buy, Withdrawal, Fees, Tax, Buy To Cover).
        case outflow
        /// Adds cash (Sell, Deposit, Dividend, Interest, Short Sell).
        case inflow
        /// No cash change (Transfer, Split, Spin-off, …).
        case neutral
    }

    /// Determine whether this transaction adds, reduces, or leaves $CASH unchanged.
    var cashImpact: CashImpact {
        switch type.lowercased().trimmingCharacters(in: .whitespaces) {
        case "buy", "withdrawal", "fees", "fee", "tax", "withholding tax", "buy to cover":
            return .outflow
        case "sell", "deposit", "dividend", "interest", "short sell":
            return .inflow
        default: // transfer, split, stock split, spin-off, spin off, …
            return .neutral
        }
    }

    /// Semantic color for the Total Amount column based on the transaction's
    /// cash impact rather than the raw sign of the stored value.
    var totalAmountColor: Color {
        switch cashImpact {
        case .outflow: return .down   // red
        case .inflow:  return .up     // green
        case .neutral: return .gray
        }
    }

    /// The magnitude shown in the Total column. Transfers are stored with
    /// Total Amount = 0 (they're cash-neutral across the whole portfolio), so
    /// the moved value has to be reconstructed: cash rows carry it in Quantity,
    /// stock rows in Quantity × Price. A non-zero Total Amount is authoritative.
    var displayAmountMagnitude: Double {
        let total = abs(totalAmount)
        if total > 1e-9 { return total }
        if Self.isCashSymbol(symbol) { return abs(quantity) }
        return abs(quantity * pricePerShare)
    }

    /// Total amount formatted for display: negative for outflows, positive
    /// (absolute value) for inflows and neutral types.
    var displayTotalAmount: Double {
        let mag = displayAmountMagnitude
        switch cashImpact {
        case .outflow:          return -mag
        case .inflow, .neutral: return mag
        }
    }

    /// The transaction types the backend accepts (mirrors web TransactionModal).
    static let allTypes = [
        "Buy", "Sell", "Dividend", "Transfer", "Interest", "Fees", "Tax",
        "Deposit", "Withdrawal", "Spin-off", "Split", "Short Sell", "Buy To Cover",
    ]

    /// Resolve a stored/raw Type string to the exact `allTypes` entry so a Picker
    /// shows it selected (an unmatched selection renders blank in SwiftUI).
    /// Ignores case AND hyphen/space differences, since the same action can
    /// arrive as "Spin-off" (option), "Spin-Off" (DB, title-cased) or
    /// "spin off" (engine canonical). Falls back to the raw string.
    static func canonicalType(_ raw: String) -> String {
        let key = raw.lowercased().replacingOccurrences(
            of: "[\\s-]+", with: "", options: .regularExpression)
        return allTypes.first {
            $0.lowercased().replacingOccurrences(
                of: "[\\s-]+", with: "", options: .regularExpression) == key
        } ?? raw
    }

    var shouldHideQtyAndPrice: Bool {
        if quantity == 0 { return true }

        if Self.isCashSymbol(symbol) {
            return true
        }

        let t = type.lowercased().trimmingCharacters(in: .whitespaces)
        
        if quantity == 1 {
            let cashTypes: Set<String> = [
                "dividend", "interest", "fees", "fee", "tax", "withholding tax", "deposit", "withdrawal", "transfer"
            ]
            if cashTypes.contains(t) {
                if abs(pricePerShare - abs(totalAmount)) < 0.01 {
                    return true
                }
            }
        }
        return false
    }

    var quantityDisplay: String {
        shouldHideQtyAndPrice ? "-" : Fmt.number(quantity)
    }

    var priceDisplay: String {
        shouldHideQtyAndPrice ? "-" : Fmt.number(pricePerShare)
    }

    /// Types whose signed Total Amount represents a cash *outflow* (negative).
    private static let outflowTypes: Set<String> = [
        "Buy", "Withdrawal", "Fees", "Tax", "Split", "Buy To Cover",
    ]

    private static func isCashSymbol(_ symbol: String) -> Bool {
        let s = symbol.trimmingCharacters(in: .whitespaces).uppercased()
        return s == "$CASH" || s == "CASH" || s.hasPrefix("CASH (")
    }

    /// Replicates the web modal's Total Amount computation and sign convention.
    /// `enteredTotal` is the user-typed total if any (nil → auto-compute).
    static func computeTotalAmount(
        type: String, symbol: String, quantity: Double, price: Double,
        commission: Double, enteredTotal: Double?
    ) -> Double {
        let txType = type.lowercased()
        var amount = enteredTotal
        if amount == nil {
            if ["buy", "buy to cover"].contains(txType) {
                amount = quantity * price + commission
            } else if ["sell", "short sell"].contains(txType) {
                amount = quantity * price - commission
            } else {
                amount = quantity * price
            }
        }
        if ["transfer", "split"].contains(txType) { amount = 0 }
        if ["deposit", "withdrawal", "buy", "sell"].contains(txType), isCashSymbol(symbol) {
            amount = quantity
        }
        let magnitude = abs(amount ?? 0)
        return outflowTypes.contains(type) ? -magnitude : magnitude
    }
}
