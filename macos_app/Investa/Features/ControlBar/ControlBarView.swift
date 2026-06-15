import SwiftUI

/// Account / currency / period selectors. Mirrors the web ControlBar.
struct ControlBarView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        HStack(spacing: 12) {
            accountPicker
            currencyPicker
            Spacer()
            periodPicker
        }
    }

    // MARK: - Accounts

    private var accountSummary: String {
        if appState.selectedAccounts.isEmpty { return "All Accounts" }
        if appState.selectedAccounts.count == 1 { return appState.selectedAccounts.first! }
        return "\(appState.selectedAccounts.count) Accounts"
    }

    private var accountPicker: some View {
        Menu {
            Button {
                appState.selectedAccounts = []
            } label: {
                Label("All Accounts", systemImage: appState.selectedAccounts.isEmpty ? "checkmark" : "")
            }
            Divider()
            ForEach(appState.allAccounts, id: \.self) { account in
                Button {
                    toggle(account)
                } label: {
                    Label(account, systemImage: appState.selectedAccounts.contains(account) ? "checkmark" : "")
                }
            }
        } label: {
            Label(accountSummary, systemImage: "building.columns")
        }
        .borderlessMenu()
        .fixedSize()
    }

    private func toggle(_ account: String) {
        if appState.selectedAccounts.contains(account) {
            appState.selectedAccounts.remove(account)
        } else {
            appState.selectedAccounts.insert(account)
        }
    }

    // MARK: - Currency

    private var currencyPicker: some View {
        Picker("Currency", selection: $appState.displayCurrency) {
            ForEach(appState.availableCurrencies, id: \.self) { code in
                Text(code).tag(code)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .fixedSize()
    }

    // MARK: - Period

    private var periodPicker: some View {
        Picker("Period", selection: $appState.period) {
            ForEach(Period.allCases) { period in
                Text(period.label).tag(period)
            }
        }
        .pickerStyle(.segmented)
        .fixedSize()
    }
}
