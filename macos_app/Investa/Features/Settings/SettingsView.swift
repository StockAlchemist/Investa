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
        } catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
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
        } catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
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
        } catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
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
        } catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
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
        } catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }

    func clearCache() async {
        do { let _: StatusResponse = try await api.send(method: "POST", path: "/clear_cache"); status = "Cache cleared." }
        catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }
    func triggerRefresh(secret: String) async {
        struct Body: Encodable { let secret: String }
        do { let _: StatusResponse = try await api.send(method: "POST", path: "/webhook/refresh", body: Body(secret: secret)); status = "Refresh triggered." }
        catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }
    func syncIbkr() async {
        guard !isSyncingIbkr else { return }
        isSyncingIbkr = true
        defer { isSyncingIbkr = false }
        status = "Syncing IBKR…"
        do { let res: StatusResponse = try await api.send(method: "POST", path: "/sync/ibkr"); status = res.message ?? "IBKR sync complete." }
        catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }
    func updateProfile(alias: String) async {
        struct Body: Encodable { let alias: String }
        do { let _: User = try await api.send(method: "PATCH", path: "/auth/me", body: Body(alias: alias)); status = "Profile updated." }
        catch { status = (error as? APIError)?.errorDescription ?? error.localizedDescription }
    }
    func deleteAccount() async {
        let _: StatusResponse? = try? await api.send(method: "DELETE", path: "/auth/me")
    }
}

enum SettingsTab: String, CaseIterable, Identifiable {
    case accounts = "Accounts", symbols = "Symbols", overrides = "Overrides", advanced = "Advanced", account = "Profile & Security"
    var id: String { rawValue }
    var icon: String {
        switch self {
        case .accounts: return "person.2"; case .symbols: return "map"; case .overrides: return "slider.horizontal.3"
        case .advanced: return "gearshape.2"; case .account: return "person.crop.circle"
        }
    }
    var color: Color {
        switch self {
        case .accounts: return .indigo
        case .symbols: return .blue
        case .overrides: return .green
        case .advanced: return .gray
        case .account: return .cyan
        }
    }
    var description: String {
        switch self {
        case .accounts: return "Account groups, per-account currency/cash/closure settings, and cash-yield assumptions."
        case .symbols: return "Map portfolio symbols to their Yahoo Finance ticker and manage excluded symbols."
        case .overrides: return "Manually override price/metadata for specific symbols."
        case .advanced: return "Display, webhook integration, Interactive Brokers sync, and system cache."
        case .account: return "Manage your user profile, password, and login."
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
                Image(systemName: tab.icon)
                    .appFont(.system(size: 18))
                    .foregroundStyle(isActive ? tab.color : .secondary)
                
                Text(tab.rawValue)
                    .appFont(.system(size: 14, weight: .medium))
                    .foregroundStyle(isActive ? .primary : .secondary)
                
                Spacer()
                
                if isActive {
                    Circle()
                        .fill(Color.cyan)
                        .frame(width: 6, height: 6)
                        .shadow(color: .cyan.opacity(0.8), radius: 4, x: 0, y: 0)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(isActive ? Color.primary.opacity(0.05) : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(isActive ? Color.primary.opacity(0.1) : Color.clear, lineWidth: 1)
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
    @State private var tab: SettingsTab = .overrides
    
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
        VStack(alignment: .leading, spacing: 0) {
            // Header
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("Settings").appFont(.title2.bold())
                    if viewModel.isLoading { ProgressView().controlSize(.small).padding(.leading, 8) }
                    Spacer()
                    if let s = viewModel.status { Text(s).appFont(.caption).foregroundStyle(.secondary) }
                }
                Text("Manage application settings, preferences, and account configurations.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            
            if isCompact {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(SettingsTab.allCases) { t in
                            SettingsSidebarItem(tab: t, isActive: tab == t) { tab = t }
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
                }
                mainContentArea
            } else {
                HStack(alignment: .top, spacing: 24) {
                    VStack(spacing: 8) {
                        ForEach(SettingsTab.allCases) { t in
                            SettingsSidebarItem(tab: t, isActive: tab == t) { tab = t }
                        }
                    }
                    .frame(width: 200)
                    
                    mainContentArea
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 24)
            }
        }
        .task { await viewModel.load() }
    }
    
    private var mainContentArea: some View {
        VStack(spacing: 0) {
            // Active Tab Header
            HStack(alignment: .top, spacing: 16) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.primary.opacity(0.05))
                        .shadow(color: .black.opacity(0.05), radius: 4, x: 0, y: 2)
                    
                    Image(systemName: tab.icon)
                        .appFont(.system(size: 24))
                        .foregroundStyle(tab.color)
                }
                .frame(width: 48, height: 48)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(tab.rawValue)
                        .appFont(.title3.bold())
                    Text(tab.description)
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(24)
            .background(Color.primary.opacity(0.02))
            .overlay(Rectangle().frame(height: 1).foregroundColor(Color.primary.opacity(0.05)), alignment: .bottom)
            
            ScrollView {
                Group {
                    switch tab {
                    case .accounts: AccountsSettings(vm: viewModel, settings: viewModel.settings, accounts: appState.allAccounts, appState: appState)
                    case .symbols: SymbolsSettings(vm: viewModel, settings: viewModel.settings)
                    case .overrides: OverridesSettings(vm: viewModel, settings: viewModel.settings)
                    case .advanced: AdvancedSettings(vm: viewModel, settings: viewModel.settings)
                    case .account: AccountSecuritySettings(vm: viewModel).environmentObject(auth)
                    }
                }
                .padding(24)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(Color.primary.opacity(0.02))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.1), lineWidth: 1)
        )
    }
}

// MARK: - Reusable card

struct SettingsCard<Content: View>: View {
    let title: String
    var icon: String? = nil
    var iconColor: Color? = nil
    @ViewBuilder var content: Content
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                if let icon = icon {
                    Image(systemName: icon)
                        .foregroundStyle(iconColor ?? .primary)
                }
                Text(title).appFont(.headline)
            }
            content
        }
        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.1), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.02), radius: 8, x: 0, y: 4)
    }
}

private func deleteButton(_ action: @escaping () -> Void) -> some View {
    Button(role: .destructive, action: action) { Image(systemName: "trash") }.buttonStyle(.borderless).foregroundStyle(.red)
}
