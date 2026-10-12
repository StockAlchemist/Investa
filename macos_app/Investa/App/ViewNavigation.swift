import Foundation

struct ViewNavigation: Equatable {
    let section: AppSection
    let currency: String?
    let accounts: [String]
    let symbol: String?

    static let sections: [String: AppSection] = [
        "performance": .performance, "allocation": .allocation, "asset_change": .assetChange,
        "transactions": .transactions, "dividend": .dividend, "capital_gains": .capitalGains,
        "screener": .market, "buffett_rank": .buffettRank, "strategies": .strategies,
        "watchlist": .watchlist, "markets": .markets, "ai_review": .aiReview, "settings": .settings,
    ]

    init?(url: URL) {
        guard url.scheme == "investa", url.host == "open",
              let components = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
        let items = components.queryItems ?? []
        guard let id = items.first(where: { $0.name == "section" })?.value,
              let section = Self.sections[id] else { return nil }
        self.section = section
        let currency = items.first(where: { $0.name == "currency" })?.value
        self.currency = currency?.range(of: "^[A-Z]{3,6}$", options: .regularExpression) != nil ? currency : nil
        accounts = items.filter { $0.name == "account" }.compactMap(\.value).filter { $0.count <= 100 }
        let symbol = items.first(where: { $0.name == "symbol" })?.value
        self.symbol = symbol?.range(of: "^[A-Za-z0-9.^=\\-$ ]{1,40}$", options: .regularExpression) != nil ? symbol?.uppercased() : nil
    }

    static func url(section: AppSection, currency: String, accounts: [String], symbol: String?) -> URL {
        var components = URLComponents()
        components.scheme = "investa"; components.host = "open"
        let id = sections.first(where: { $0.value == section })!.key
        components.queryItems = [URLQueryItem(name: "section", value: id), URLQueryItem(name: "currency", value: currency)]
            + Array(Set(accounts)).sorted().map { URLQueryItem(name: "account", value: $0) }
        if let symbol { components.queryItems!.append(URLQueryItem(name: "symbol", value: symbol)) }
        return components.url!
    }
}

extension Notification.Name { static let openSharedView = Notification.Name("investa.openSharedView") }
