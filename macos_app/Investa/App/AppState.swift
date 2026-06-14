import Foundation
import Combine

/// History period options matching the backend's accepted `period` values.
enum Period: String, CaseIterable, Identifiable, Sendable {
    case oneMonth = "1m"
    case threeMonths = "3m"
    case sixMonths = "6m"
    case ytd = "ytd"
    case oneYear = "1y"
    case all = "all"

    var id: String { rawValue }
    var label: String {
        switch self {
        case .oneMonth: return "1M"
        case .threeMonths: return "3M"
        case .sixMonths: return "6M"
        case .ytd: return "YTD"
        case .oneYear: return "1Y"
        case .all: return "All"
        }
    }
}

/// Shared selection state driving every dashboard panel: display currency,
/// selected accounts, history period, and whether to include closed accounts.
/// Mirrors the web app's ControlBar/CurrencySelector/AccountSelector state.
@MainActor
final class AppState: ObservableObject {
    @Published var displayCurrency: String = "USD"
    @Published var availableCurrencies: [String] = ["USD"]
    @Published var allAccounts: [String] = []
    /// Empty set means "all accounts" (no `accounts` query param sent).
    @Published var selectedAccounts: Set<String> = []
    @Published var period: Period = .oneYear
    @Published var showClosed: Bool = false
    @Published private(set) var didLoadSettings = false

    private let api: APIClient

    init(api: APIClient = .shared) {
        self.api = api
    }

    /// The `accounts` query value: nil when nothing (or everything) is selected.
    var accountsQuery: [String]? {
        selectedAccounts.isEmpty ? nil : Array(selectedAccounts)
    }

    /// Load currency options, account list, and saved defaults from the backend.
    func loadSettings() async {
        do {
            let settings: AppSettings = try await api.get("/settings")
            availableCurrencies = settings.availableCurrencies?.isEmpty == false
                ? settings.availableCurrencies! : ["USD"]
            allAccounts = settings.allAccounts
            if let cur = settings.displayCurrency, availableCurrencies.contains(cur) {
                displayCurrency = cur
            } else if !availableCurrencies.contains(displayCurrency) {
                displayCurrency = availableCurrencies.first ?? "USD"
            }
            if let saved = settings.selectedAccounts, !saved.isEmpty {
                selectedAccounts = Set(saved.filter { allAccounts.contains($0) })
            }
            showClosed = settings.showClosed ?? false
            didLoadSettings = true
        } catch {
            // Non-fatal: fall back to defaults so the dashboard still loads.
            didLoadSettings = true
        }
    }
}
