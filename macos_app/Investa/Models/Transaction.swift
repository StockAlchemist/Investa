import Foundation

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
    /// The transaction types the backend accepts (mirrors web TransactionModal).
    static let allTypes = [
        "Buy", "Sell", "Dividend", "Transfer", "Interest", "Fees", "Tax",
        "Deposit", "Withdrawal", "Spin-off", "Split", "Short Sell", "Buy To Cover",
    ]

    /// Types whose signed Total Amount represents a cash *outflow* (negative).
    private static let outflowTypes: Set<String> = [
        "Buy", "Withdrawal", "Fees", "Tax", "Split", "Buy To Cover",
    ]

    private static func isCashSymbol(_ symbol: String) -> Bool {
        let s = symbol.trimmingCharacters(in: .whitespaces).uppercased()
        return s == "$CASH" || s == "CASH"
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
