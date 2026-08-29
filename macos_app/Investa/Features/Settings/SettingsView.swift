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
    func update<T: Encodable>(_ key: String, _ value: T, note: String = "Saved.") async {
        do {
            let _: StatusResponse = try await api.send(method: "POST", path: "/settings/update", body: KV(key: key, value: value))
            status = note
            await load()
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving settings", style: .error)
        }
    }

    /// Save account groups and their display order together (they must stay in
    /// sync — the pickers render groups in `account_group_order`).
    func updateGroups(_ groups: [String: [String]], order: [String]) async {
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
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving groups", style: .error)
        }
    }

    /// Save account preferences in a single POST request.
    func updateAccountPreferences(currencyMap: [String: String], cashModeMap: [String: String], closureMap: [String: String]) async {
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
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving preferences", style: .error)
        }
    }

    /// Save cash yield settings in a single POST request.
    func updateCashYield(rates: [String: Double], thresholds: [String: Double]) async {
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
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving yield settings", style: .error)
        }
    }

    /// Save IBKR credentials in a single POST request.
    func updateIBKR(token: String, queryId: String) async {
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
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving credentials", style: .error)
        }
    }

    /// Save external API keys in a single POST request.
    func updateAPIKeys(gemini: String, fmp: String, secTh: String, bot: String, tiingo: String) async {
        struct Body: Encodable {
            let gemini_api_key: String
            let fmp_api_key: String
            let sec_th_api_key: String
            let bot_api_key: String
            let tiingo_api_key: String
        }
        do {
            let _: StatusResponse = try await api.send(
                method: "POST", path: "/settings/update",
                body: Body(
                    gemini_api_key: gemini,
                    fmp_api_key: fmp,
                    sec_th_api_key: secTh,
                    bot_api_key: bot,
                    tiingo_api_key: tiingo
                )
            )
            status = "API keys saved."
            await load()
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error saving API keys", style: .error)
        }
    }

    func clearCache() async {
        do {
            let _: StatusResponse = try await api.send(method: "POST", path: "/clear_cache")
            status = "Cache cleared."
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error clearing cache", style: .error)
        }
    }

    func triggerRefresh(secret: String) async {
        struct Body: Encodable { let secret: String }
        do {
            let _: StatusResponse = try await api.send(method: "POST", path: "/webhook/refresh", body: Body(secret: secret))
            status = "Refresh triggered."
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error triggering refresh", style: .error)
        }
    }

    func syncIbkr() async {
        guard !isSyncingIbkr else { return }
        isSyncingIbkr = true
        defer { isSyncingIbkr = false }
        status = "Syncing IBKR…"
        do {
            let res: StatusResponse = try await api.send(method: "POST", path: "/sync/ibkr")
            status = res.message ?? "IBKR sync complete."
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error syncing IBKR", style: .error)
        }
    }

    func updateProfile(alias: String) async {
        struct Body: Encodable { let alias: String }
        do {
            let _: User = try await api.send(method: "PATCH", path: "/auth/me", body: Body(alias: alias))
            status = "Profile updated."
        } catch {
            status = (error as? APIError)?.errorDescription ?? error.localizedDescription
            ToastManager.shared.show(message: status ?? "Error updating profile", style: .error)
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
    case advanced = "Advanced Settings"
    case account = "Profile & Security"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .accounts: return "person.2.fill"
        case .symbols: return "arrow.left.arrow.right"
        case .overrides: return "pencil.and.ruler.fill"
        case .advanced: return "gearshape.2.fill"
        case .account: return "person.crop.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .accounts: return .indigo
        case .symbols: return .blue
        case .overrides: return .green
        case .advanced: return .purple
        case .account: return .cyan
        }
    }

    var description: String {
        switch self {
        case .accounts: return "Account groups, per-account currency, cash automation, and yield assumptions."
        case .symbols: return "Map portfolio symbols to Yahoo Finance tickers and manage excluded symbols."
        case .overrides: return "Manually override price, sector, asset type, and metadata for specific symbols."
        case .advanced: return "Benchmark comparisons, IBKR Flex Query sync, external API keys, and cache."
        case .account: return "Manage user profile, login credentials, and session security."
        }
    }
}

struct SettingsSidebarItem: View {
    let tab: SettingsTab
    let isActive: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                SettingsIconBadge(icon: tab.icon, color: tab.color, size: 26, iconSize: 13)

                Text(tab.rawValue)
                    .appFont(.system(size: 14, weight: isActive ? .semibold : .medium))
                    .foregroundStyle(isActive ? .primary : .secondary)

                Spacer()

                if isActive {
                    Circle()
                        .fill(tab.color)
                        .frame(width: 6, height: 6)
                        .shadow(color: tab.color.opacity(0.8), radius: 4, x: 0, y: 0)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(isActive ? Color.primary.opacity(0.06) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(isActive ? Color.primary.opacity(0.12) : Color.clear, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
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
                desktopSplitView
                #endif
            } else {
                desktopSplitView
            }
        }
        .task {
            await viewModel.load()
        }
    }

    // MARK: - Desktop & iPad Split View

    private var desktopSplitView: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Settings")
                        .appFont(.title2.bold())
                    if viewModel.isLoading {
                        ProgressView().controlSize(.small).padding(.leading, 8)
                    }
                    Spacer()
                    if let s = viewModel.status {
                        Text(s)
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Text("Manage application preferences, market data providers, and portfolio configurations.")
                    .appFont(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 24)
            .padding(.vertical, 16)

            HStack(alignment: .top, spacing: 24) {
                // Sidebar Tabs
                VStack(spacing: 6) {
                    ForEach(SettingsTab.allCases) { t in
                        SettingsSidebarItem(tab: t, isActive: tab == t) {
                            tab = t
                        }
                    }
                    Spacer()
                }
                .frame(width: 220)

                // Tab Content Card
                desktopContentArea
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 24)
        }
    }

    private var desktopContentArea: some View {
        VStack(spacing: 0) {
            // Active Tab Header Banner
            HStack(alignment: .center, spacing: 16) {
                SettingsIconBadge(icon: tab.icon, color: tab.color, size: 44, iconSize: 22)

                VStack(alignment: .leading, spacing: 3) {
                    Text(tab.rawValue)
                        .appFont(.title3.bold())
                    Text(tab.description)
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(20)
            .background(Color.primary.opacity(0.02))
            .overlay(
                Rectangle()
                    .frame(height: 1)
                    .foregroundColor(Color.primary.opacity(0.06)),
                alignment: .bottom
            )

            // Scrollable Content
            ScrollView {
                Group {
                    switch tab {
                    case .accounts:
                        AccountsSettings(
                            vm: viewModel,
                            settings: viewModel.settings,
                            accounts: appState.allAccounts,
                            appState: appState
                        )
                    case .symbols:
                        SymbolsSettings(
                            vm: viewModel,
                            settings: viewModel.settings
                        )
                    case .overrides:
                        OverridesSettings(
                            vm: viewModel,
                            settings: viewModel.settings
                        )
                    case .advanced:
                        AdvancedSettings(
                            vm: viewModel,
                            settings: viewModel.settings
                        )
                    case .account:
                        AccountSecuritySettings(
                            vm: viewModel
                        )
                        .environmentObject(auth)
                    }
                }
                .padding(24)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color.primary.opacity(0.02))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }
}
