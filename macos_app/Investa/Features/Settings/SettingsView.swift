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
        status = "Syncing IBKR…"
        do { let _: StatusResponse = try await api.send(method: "POST", path: "/sync/ibkr"); status = "IBKR sync complete." }
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
                    .font(.system(size: 18))
                    .foregroundStyle(isActive ? tab.color : .secondary)
                
                Text(tab.rawValue)
                    .font(.system(size: 14, weight: .medium))
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
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Settings")
                        .font(.system(size: 32, weight: .heavy, design: .default))
                        .foregroundStyle(LinearGradient(colors: [.cyan, .blue], startPoint: .leading, endPoint: .trailing))
                    
                    if viewModel.isLoading { ProgressView().controlSize(.small).padding(.leading, 8) }
                    Spacer()
                    if let s = viewModel.status { Text(s).font(.caption).foregroundStyle(.secondary) }
                }
                
                Text("Manage application settings, preferences, and account configurations.")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 24)
            .padding(.top, 24)
            .padding(.bottom, 24)
            
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
                        .font(.system(size: 24))
                        .foregroundStyle(tab.color)
                }
                .frame(width: 48, height: 48)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(tab.rawValue)
                        .font(.title3.bold())
                    Text(tab.description)
                        .font(.subheadline)
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
                Text(title).font(.headline)
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
