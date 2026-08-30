import SwiftUI

/// Dynamic coding key so we can POST a single `settings/update` field by name.
private struct DynamicKey: CodingKey {
    var stringValue: String; var intValue: Int? { nil }
    init(_ s: String) { stringValue = s }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}
private struct KV<T: Encodable>: Encodable {
    let key: String; let value: T
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: DynamicKey.self)
        try c.encode(value, forKey: DynamicKey(key))
    }
}

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var settings: AppSettings?
    @Published var isLoading = false
    @Published var status: String?
    /// Drives the "Sync Now" button's spinner while `/sync/ibkr` is in flight.
    @Published var isSyncingIbkr = false

    private let api: APIClient
    init(api: APIClient = .shared) { self.api = api }

    func load() async {
        isLoading = true; defer { isLoading = false }
        settings = try? await api.get("/settings")
    }

    /// POST a single settings field, then reload.
    @discardableResult
    func update<T: Encodable>(_ key: String, _ value: T, note: String = "Saved.") async -> Bool {
        do {
            let _: StatusResponse = try await api.send(method: "POST", path: "/settings/update", body: KV(key: key, value: value))
            status = note
            await load()
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving settings", style: .error)
            return false
        }
    }

    /// Save account groups and their display order together (they must stay in
    /// sync — the pickers render groups in `account_group_order`).
    @discardableResult
    func updateGroups(_ groups: [String: [String]], order: [String]) async -> Bool {
        struct Body: Encodable {
            let account_groups: [String: [String]]
            let account_group_order: [String]
        }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/settings/update",
                body: Body(account_groups: groups, account_group_order: order))
            status = "Groups saved."
            await load()
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving groups", style: .error)
            return false
        }
    }

    /// Save account preferences in a single POST request.
    @discardableResult
    func updateAccountPreferences(currencyMap: [String: String], cashModeMap: [String: String], closureMap: [String: String]) async -> Bool {
        struct Body: Encodable {
            let account_currency_map: [String: String]
            let account_cash_mode_map: [String: String]
            let account_closure_dates: [String: String]
        }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/settings/update",
                body: Body(
                    account_currency_map: currencyMap,
                    account_cash_mode_map: cashModeMap,
                    account_closure_dates: closureMap
                ))
            status = "Account preferences saved."
            await load()
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving preferences", style: .error)
            return false
        }
    }

    /// Save cash yield settings in a single POST request.
    @discardableResult
    func updateCashYield(rates: [String: Double], thresholds: [String: Double]) async -> Bool {
        struct Body: Encodable {
            let account_interest_rates: [String: Double]
            let interest_free_thresholds: [String: Double]
        }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/settings/update",
                body: Body(account_interest_rates: rates, interest_free_thresholds: thresholds))
            status = "Cash yield settings saved."
            await load()
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving yield settings", style: .error)
            return false
        }
    }

    /// Save IBKR credentials in a single POST request.
    @discardableResult
    func updateIBKR(token: String, queryId: String) async -> Bool {
        struct Body: Encodable {
            let ibkr_token: String
            let ibkr_query_id: String
        }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/settings/update",
                body: Body(ibkr_token: token, ibkr_query_id: queryId))
            status = "Credentials saved."
            await load()
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving credentials", style: .error)
            return false
        }
    }

    /// Save external API keys.
    ///
    /// Takes only the fields the user actually retyped, keyed by their wire
    /// name. The server sends masked previews rather than the real keys and
    /// reads "" as "clear this key", so posting all five would wipe every
    /// stored key whenever the settings fetch had not resolved.
    @discardableResult
    func updateAPIKeys(_ keys: [String: String]) async -> Bool {
        guard !keys.isEmpty else { return false }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/settings/update", body: keys
            )
            status = "API keys saved."
            await load()
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving API keys", style: .error)
            return false
        }
    }

    @discardableResult
    func clearCache() async -> Bool {
        do {
            let _: StatusResponse = try await api.send(method: "POST", path: "/clear_cache")
            status = "Cache cleared."
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error clearing cache", style: .error)
            return false
        }
    }

    @discardableResult
    func triggerRefresh(secret: String) async -> Bool {
        struct Body: Encodable { let secret: String }
        do {
            let _: StatusResponse = try await api.send(method: "POST", path: "/webhook/refresh", body: Body(secret: secret))
            status = "Refresh triggered."
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error triggering refresh", style: .error)
            return false
        }
    }

    @discardableResult
    func syncIbkr() async -> Bool {
        guard !isSyncingIbkr else { return false }
        isSyncingIbkr = true
        defer { isSyncingIbkr = false }
        status = "Syncing IBKR…"
        do {
            let res: StatusResponse = try await api.send(method: "POST", path: "/sync/ibkr")
            status = res.message ?? "IBKR sync complete."
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error syncing IBKR", style: .error)
            return false
        }
    }

    @discardableResult
    func updateProfile(alias: String) async -> Bool {
        struct Body: Encodable { let alias: String }
        do {
            let _: User = try await api.send(method: "PATCH", path: "/auth/me", body: Body(alias: alias))
            status = "Profile updated."
            return true
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error updating profile", style: .error)
            return false
        }
    }

    func deleteAccount() async {
        let _: StatusResponse? = try? await api.send(method: "DELETE", path: "/auth/me")
    }
}

enum SettingsTab: String, CaseIterable, Identifiable {
    case accounts = "Accounts"
    case symbols = "Symbols"
    case overrides = "Overrides"
    case advanced = "Advanced"
    case appearance = "Appearance"
    case account = "Profile & Security"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .accounts: return "person.2"
        case .symbols: return "arrow.left.arrow.right"
        case .overrides: return "slider.horizontal.3"
        case .advanced: return "gearshape"
        case .appearance: return "paintbrush"
        case .account: return "person.crop.circle"
        }
    }

    /// Tooltip on the rail. The tab no longer draws a banner that repeats it.
    var help: String {
        switch self {
        case .accounts: return "Account groups, per-account currency, cash automation, and yield assumptions."
        case .symbols: return "Map portfolio symbols to Yahoo Finance tickers and manage excluded symbols."
        case .overrides: return "Manually override price, sector, asset type, and metadata for specific symbols."
        case .advanced: return "Benchmark comparisons, IBKR Flex Query sync, external API keys, and cache."
        case .appearance: return "Light, dark, or follow the device theme."
        case .account: return "Manage user profile, login credentials, and session security."
        }
    }
}

/// One row of the category rail.
///
/// Deliberately the sidebar's own `NavItem` anatomy — 36pt tall, 8pt radius,
/// `brand/15` fill and a 3pt left rail when active — so the rail reads as part
/// of the app rather than a second, differently-styled navigation.
struct SettingsCategoryRow: View {
    let tab: SettingsTab
    let isActive: Bool
    var count: Int? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                SettingsIcon(icon: tab.icon, size: 18, isActive: isActive)

                Text(tab.rawValue)
                    .appFont(.system(size: 14, weight: isActive ? .semibold : .medium))
                    .foregroundStyle(isActive ? Color.brandInk : Color.secondary)

                Spacer(minLength: 6)

                if let count, count > 0 {
                    Text("\(count)")
                        .appFont(.system(size: 11, weight: .bold))
                        .monospacedDigit()
                        .foregroundStyle(isActive ? Color.brandInk.opacity(0.85) : Color.secondary)
                }
            }
            .lineLimit(1)
            .minimumScaleFactor(0.8)
            .padding(.horizontal, 12)
            .frame(height: Theme.controlDefault)
            .background(
                RoundedRectangle(cornerRadius: Theme.controlRadius, style: .continuous)
                    .fill(isActive ? Color.brand.opacity(0.15) : Color.clear)
            )
            .overlay(alignment: .leading) {
                if isActive {
                    RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                        .fill(Color.brand)
                        .frame(width: 3)
                        .padding(.vertical, 6)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(tab.help)
    }
}

struct SettingsView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var auth: AuthViewModel
    @StateObject private var viewModel = SettingsViewModel()
    @State private var tab: SettingsTab = .accounts

    #if !os(macOS)
    @Environment(\.horizontalSizeClass) private var hSizeClass
    #endif

    private var isCompact: Bool {
        #if os(macOS)
        return false
        #else
        return hSizeClass == .compact
        #endif
    }

    var body: some View {
        Group {
            if isCompact {
                #if os(iOS)
                NavigationStack {
                    SettingsHubIOS(viewModel: viewModel)
                }
                #else
                regularLayout
                #endif
            } else {
                regularLayout
            }
        }
        .task {
            await viewModel.load()
        }
    }

    // MARK: - macOS & iPad

    /// The same shape as every other tab: the control bar names the tab, then
    /// one scrolling column of cards — with the category rail beside it.
    private var regularLayout: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                HStack(alignment: .top, spacing: 24) {
                    rail
                    content
                }
                .padding(20)
            }
        }
        .macMinSize(width: 820, height: 560)
    }

    /// Save status and the loading spinner. The tab's name lives in the control
    /// bar, so with nothing to report this row disappears rather than leaving a
    /// blank band above the first card.
    @ViewBuilder private var header: some View {
        if viewModel.isLoading || viewModel.status != nil {
            HStack(spacing: 10) {
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer(minLength: 8)
                if let status = viewModel.status {
                    Text(status)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.85)
                }
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
        }
    }

    private var rail: some View {
        VStack(alignment: .leading, spacing: 2) {
            SectionLabel(title: "Categories")
                .padding(.horizontal, 12)
                .padding(.bottom, 6)

            ForEach(SettingsTab.allCases) { t in
                SettingsCategoryRow(tab: t, isActive: tab == t, count: count(for: t)) {
                    tab = t
                }
            }
        }
        .frame(width: 216, alignment: .leading)
    }

    @ViewBuilder private var content: some View {
        VStack(spacing: 20) {
            switch tab {
            case .accounts:
                AccountsSettings(
                    vm: viewModel,
                    settings: viewModel.settings,
                    accounts: appState.allAccounts,
                    appState: appState
                )
            case .symbols:
                SymbolsSettings(vm: viewModel, settings: viewModel.settings)
            case .overrides:
                OverridesSettings(vm: viewModel, settings: viewModel.settings)
            case .advanced:
                AdvancedSettings(vm: viewModel, settings: viewModel.settings)
            case .appearance:
                AppearanceSettingsView(embedded: true)
            case .account:
                AccountSecuritySettings(vm: viewModel).environmentObject(auth)
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    /// Counts ride the rail so a category says how much it holds before it opens.
    private func count(for tab: SettingsTab) -> Int? {
        switch tab {
        case .accounts: return viewModel.settings?.accountGroups?.count
        case .symbols: return viewModel.settings?.userSymbolMap?.count
        case .overrides: return viewModel.settings?.manualOverrides?.count
        default: return nil
        }
    }
}
