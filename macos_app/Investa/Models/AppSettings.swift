import Foundation

/// Subset of `GET /api/settings` the MVP needs: currency options, account
/// groupings, and the user's saved defaults. Unused fields are ignored.
struct AppSettings: Codable, Sendable {
    let availableCurrencies: [String]?
    let accountGroups: [String: [String]]?
    let accountGroupOrder: [String]?
    let displayCurrency: String?
    let selectedAccounts: [String]?
    let benchmarks: [String]?
    let showClosed: Bool?
    let manualOverrides: [String: JSONValue]?
    let accountInterestRates: [String: Double]?
    /// bucket (quoteType/sector/country/symbol) → name → target %.
    let targetAllocation: [String: [String: Double]]?

    enum CodingKeys: String, CodingKey {
        case availableCurrencies = "available_currencies"
        case accountGroups = "account_groups"
        case accountGroupOrder = "account_group_order"
        case displayCurrency = "display_currency"
        case selectedAccounts = "selected_accounts"
        case benchmarks
        case showClosed = "show_closed"
        case manualOverrides = "manual_overrides"
        case accountInterestRates = "account_interest_rates"
        case targetAllocation = "target_allocation"
    }

    /// Manual price overrides flattened to symbol → price (handles both the bare
    /// number and the `{ price, ... }` object form the backend may return).
    var manualOverridePrices: [String: Double] {
        var out: [String: Double] = [:]
        for (symbol, value) in manualOverrides ?? [:] {
            if let p = value.doubleValue { out[symbol] = p }
            else if let p = value["price"]?.doubleValue { out[symbol] = p }
        }
        return out
    }

    /// Flattened, de-duplicated list of every account across all groups,
    /// ordered by `accountGroupOrder` when available.
    var allAccounts: [String] {
        guard let groups = accountGroups else { return [] }
        let orderedKeys = accountGroupOrder ?? Array(groups.keys).sorted()
        var seen = Set<String>()
        var result: [String] = []
        for key in orderedKeys {
            for account in groups[key] ?? [] where !seen.contains(account) {
                seen.insert(account)
                result.append(account)
            }
        }
        // Include any groups not named in the order array.
        for (key, accounts) in groups where !orderedKeys.contains(key) {
            for account in accounts where !seen.contains(account) {
                seen.insert(account)
                result.append(account)
            }
        }
        return result
    }
}
