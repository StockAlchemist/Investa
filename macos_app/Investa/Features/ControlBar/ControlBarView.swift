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
        PopoverMenu(minWidth: 200) {
            MenuToggleRow(title: "All Accounts", isOn: appState.selectedAccounts.isEmpty, dismissOnTap: true) {
                appState.selectedAccounts = []
            }
            MenuDivider()
            ForEach(appState.allAccounts, id: \.self) { account in
                MenuToggleRow(title: account, isOn: appState.selectedAccounts.contains(account)) {
                    toggle(account)
                }
            }
        } label: {
            Label(accountSummary, systemImage: "building.columns")
        }
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
        PopoverMenu(minWidth: 100) {
            ForEach(appState.availableCurrencies, id: \.self) { code in
                MenuToggleRow(title: code, isOn: code == appState.displayCurrency, dismissOnTap: true) {
                    appState.displayCurrency = code
                }
            }
        } label: {
            Text(appState.displayCurrency)
        }
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
