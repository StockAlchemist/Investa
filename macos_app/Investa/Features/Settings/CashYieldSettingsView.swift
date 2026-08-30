import SwiftUI

struct CashYieldSettingsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    let accounts: [String]
    @ObservedObject var appState: AppState

    var embedded: Bool = false

    @State private var holdings: [Holding] = []
    @State private var rates: [String: Double] = [:]
    @State private var thresholds: [String: Double] = [:]
    @State private var isSaving = false

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("Cash Yield Assumptions")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
        .onAppear {
            seed()
            fetchHoldings()
        }
        .onChange(of: settings?.accountInterestRates) { _, _ in seed() }
        .onChange(of: settings?.interestFreeThresholds) { _, _ in seed() }
    }

    private var mainContent: some View {
        VStack(spacing: 16) {
            // Header Note
            Text("Configure annual interest rates and interest-free cash thresholds to estimate future cash yield.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 4)

            let cashAccounts = accountsWithCash()
            if cashAccounts.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "banknote")
                        .font(.system(size: 36))
                        .foregroundStyle(.secondary.opacity(0.4))
                        .padding(.vertical, 8)
                    Text("No cash balances found across your accounts.")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 32)
            } else {
                LazyVStack(spacing: 14) {
                    ForEach(cashAccounts, id: \.self) { acc in
                        cashYieldCard(acc)
                    }
                }

                // Save Button
                Button {
                    saveYieldSettings()
                } label: {
                    HStack(spacing: 8) {
                        if isSaving {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "checkmark.circle.fill")
                        }
                        Text("Save Cash Yield Settings")
                    }
                    .frame(maxWidth: .infinity)
                    .fontWeight(.bold)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(Color.brand)
                .disabled(isSaving)
                .padding(.top, 8)
            }
        }
    }

    private func seed() {
        rates = settings?.accountInterestRates ?? [:]
        thresholds = settings?.interestFreeThresholds ?? [:]
    }

    private func fetchHoldings() {
        Task {
            if let result: [Holding] = try? await APIClient.shared.get("/holdings", query: [URLQueryItem(name: "currency", value: appState.displayCurrency)]) {
                await MainActor.run { holdings = result }
            }
        }
    }

    private func accountsWithCash() -> [String] {
        let cashHoldings = holdings.filter { $0.symbol.uppercased().contains("CASH") || $0.symbol.uppercased() == "$CASH" }
        let set = Set(cashHoldings.compactMap { $0.account })
        let list = accounts.filter { set.contains($0) }
        return list.isEmpty ? accounts.filter { $0 != "All Accounts" } : list
    }

    private func cashBalance(for account: String) -> Double {
        let accHoldings = holdings.filter { $0.account == account && ($0.symbol.uppercased().contains("CASH") || $0.symbol.uppercased() == "$CASH") }
        return accHoldings.compactMap { $0.marketValue(currency: appState.displayCurrency) }.reduce(0, +)
    }

    private func cashYieldCard(_ acc: String) -> some View {
        let balance = cashBalance(for: acc)
        let rate = rates[acc] ?? 0.0
        let threshold = thresholds[acc] ?? 0.0
        let interest = max(0, balance - threshold) * (rate / 100.0)

        return VStack(alignment: .leading, spacing: 14) {
            HStack {
                HStack(spacing: 8) {
                    Image(systemName: "percent")
                        .foregroundStyle(Color.brand)
                        .appFont(.body)

                    Text(acc)
                        .appFont(.headline.bold())
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 2) {
                    Text("Cash Balance")
                        .appFont(.caption2)
                        .foregroundStyle(.secondary)
                    Text(balance.formatted(.currency(code: appState.displayCurrency)))
                        .appFont(.subheadline.monospacedDigit().weight(.bold))
                }
            }

            Divider()

            VStack(spacing: 10) {
                HStack {
                    Text("Annual Interest Rate (%)")
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    TextField("0.0", text: Binding(
                        get: { rates[acc].map { String($0) } ?? "" },
                        set: { rates[acc] = Double($0) ?? 0 }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 110)
                    #if os(iOS)
                    .keyboardType(.decimalPad)
                    #endif
                }

                HStack {
                    Text("Exempt Threshold (\(appState.displayCurrency))")
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    TextField("0", text: Binding(
                        get: { thresholds[acc].map { String($0) } ?? "" },
                        set: { thresholds[acc] = Double($0) ?? 0 }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 110)
                    #if os(iOS)
                    .keyboardType(.decimalPad)
                    #endif
                }

                Divider()

                HStack {
                    Text("Est. Annual Interest")
                        .appFont(.subheadline.weight(.medium))
                    Spacer()
                    Text(interest.formatted(.currency(code: appState.displayCurrency)))
                        .appFont(.headline.monospacedDigit().weight(.bold))
                        .foregroundStyle(Color.up)
                }
            }
        }
        .padding(16)
        .card()
    }

    private func saveYieldSettings() {
        isSaving = true
        Task {
            let saved = await vm.updateCashYield(rates: rates, thresholds: thresholds)
            isSaving = false
            guard saved else { return }
            ToastManager.shared.show(message: "Cash yield settings saved", style: .success)
        }
    }
}
