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

struct SettingsView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var auth: AuthViewModel
    @StateObject private var viewModel = SettingsViewModel()
    @State private var tab: SettingsTab = .overrides

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Settings").font(.title2.bold())
                    .foregroundStyle(LinearGradient(colors: [.cyan, .blue], startPoint: .leading, endPoint: .trailing))
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
                if let s = viewModel.status { Text(s).font(.caption).foregroundStyle(.secondary) }
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Picker("", selection: $tab) {
                ForEach(SettingsTab.allCases) { Label($0.rawValue, systemImage: $0.icon).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden().padding(.horizontal, 20)
            Text(tab.description).font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 20).padding(.top, 6)
            Divider().padding(.top, 8)
            ScrollView {
                Group {
                    switch tab {
                    case .accounts: AccountsSettings(vm: viewModel, settings: viewModel.settings, accounts: appState.allAccounts)
                    case .symbols: SymbolsSettings(vm: viewModel, settings: viewModel.settings)
                    case .overrides: OverridesSettings(vm: viewModel, settings: viewModel.settings)
                    case .advanced: AdvancedSettings(vm: viewModel, settings: viewModel.settings)
                    case .account: AccountSecuritySettings(vm: viewModel).environmentObject(auth)
                    }
                }
                .padding(20)
            }
        }
        .task { await viewModel.load() }
    }
}

// MARK: - Reusable card

struct SettingsCard<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.headline)
            content
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
}

private func deleteButton(_ action: @escaping () -> Void) -> some View {
    Button(role: .destructive, action: action) { Image(systemName: "trash") }.buttonStyle(.borderless).foregroundStyle(.red)
}
