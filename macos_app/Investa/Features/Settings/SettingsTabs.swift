import SwiftUI

// MARK: - Accounts Tab (Desktop / iPad)

struct AccountsSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    let accounts: [String]
    @ObservedObject var appState: AppState

    var body: some View {
        VStack(spacing: 24) {
            AccountGroupManagerView(
                vm: vm,
                settings: settings,
                availableAccounts: accounts,
                appState: appState,
                embedded: true
            )

            AccountPreferencesView(
                vm: vm,
                settings: settings,
                accounts: accounts,
                appState: appState,
                embedded: true
            )

            CurrencyManagementView(
                vm: vm,
                settings: settings,
                embedded: true
            )

            CashYieldSettingsView(
                vm: vm,
                settings: settings,
                accounts: accounts,
                appState: appState,
                embedded: true
            )
        }
    }
}

// MARK: - Symbols Tab (Desktop / iPad)

struct SymbolsSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var body: some View {
        VStack(spacing: 24) {
            SymbolMappingsView(vm: vm, settings: settings, embedded: true)
            ExcludedSymbolsView(vm: vm, settings: settings, embedded: true)
        }
    }
}

// MARK: - Overrides Tab (Desktop / iPad)

struct OverridesSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var body: some View {
        OverridesListView(vm: vm, settings: settings, embedded: true)
    }
}

// MARK: - Advanced Tab (Desktop / iPad)

struct AdvancedSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    @EnvironmentObject private var appState: AppState

    var body: some View {
        VStack(spacing: 24) {
            BenchmarksSettingsView(vm: vm, settings: settings, embedded: true)
                .environmentObject(appState)

            IntegrationsSettingsView(vm: vm, settings: settings, embedded: true)

            APIKeysSettingsView(vm: vm, settings: settings, embedded: true)

            ServerSettingsView(vm: vm, settings: settings, embedded: true)
        }
    }
}

// MARK: - Profile & Security Tab (Desktop / iPad)

struct AccountSecuritySettings: View {
    @ObservedObject var vm: SettingsViewModel
    @EnvironmentObject private var auth: AuthViewModel

    var body: some View {
        ProfileSecuritySettingsView(vm: vm, embedded: true)
            .environmentObject(auth)
    }
}
