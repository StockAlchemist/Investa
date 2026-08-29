import SwiftUI

struct AccountPreferencesView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    let accounts: [String]
    @ObservedObject var appState: AppState

    @State private var currencyMap: [String: String] = [:]
    @State private var cashModeMap: [String: String] = [:]
    @State private var closureMap: [String: String] = [:]
    @State private var isSaving = false

    private var configurableAccounts: [String] {
        accounts.filter { $0 != "All Accounts" }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Header Note
                Text("Configure base currency, cash automation mode, and closure dates for each account.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 4)

                if configurableAccounts.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "person.2.slash")
                            .font(.system(size: 36))
                            .foregroundStyle(.secondary.opacity(0.4))
                            .padding(.vertical, 8)
                        Text("No active accounts found.")
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 32)
                } else {
                    LazyVStack(spacing: 14) {
                        ForEach(configurableAccounts, id: \.self) { acc in
                            accountCard(acc)
                        }
                    }

                    // Save Button
                    Button {
                        savePreferences()
                    } label: {
                        HStack(spacing: 8) {
                            if isSaving {
                                ProgressView()
                                    .controlSize(.small)
                            } else {
                                Image(systemName: "checkmark.circle.fill")
                            }
                            Text("Save Account Preferences")
                        }
                        .frame(maxWidth: .infinity)
                        .fontWeight(.bold)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.purple)
                    .disabled(isSaving)
                    .padding(.top, 8)
                }
            }
            .padding(16)
        }
        .navigationTitle("Account Preferences")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .onAppear { seed() }
        .onChange(of: settings?.accountCurrencyMap) { _, _ in seed() }
        .onChange(of: settings?.accountCashModeMap) { _, _ in seed() }
        .onChange(of: settings?.accountClosureDates) { _, _ in seed() }
    }

    private func seed() {
        currencyMap = settings?.accountCurrencyMap ?? [:]
        cashModeMap = settings?.accountCashModeMap ?? [:]
        closureMap = settings?.accountClosureDates ?? [:]
    }

    private func accountCard(_ acc: String) -> some View {
        let closureDateStr = closureMap[acc] ?? ""
        let isClosed = !closureDateStr.isEmpty && closureDateStr <= ISO8601DateFormatter().string(from: Date()).prefix(10)

        return VStack(alignment: .leading, spacing: 14) {
            // Account Title & Status
            HStack {
                HStack(spacing: 8) {
                    Image(systemName: isClosed ? "lock.fill" : "creditcard.fill")
                        .foregroundStyle(isClosed ? .secondary : Color.purple)
                        .appFont(.body)

                    Text(acc)
                        .appFont(.headline.bold())
                        .strikethrough(isClosed)
                        .foregroundStyle(isClosed ? .secondary : .primary)
                }

                Spacer()

                if isClosed {
                    Text("CLOSED")
                        .appFont(.system(size: 10, weight: .bold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.secondary.opacity(0.15), in: Capsule())
                } else {
                    Text("ACTIVE")
                        .appFont(.system(size: 10, weight: .bold))
                        .foregroundStyle(.green)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.green.opacity(0.12), in: Capsule())
                }
            }

            Divider()

            // Fields
            VStack(spacing: 12) {
                // Default Currency
                HStack {
                    Text("Default Currency")
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)

                    Spacer()

                    Picker("", selection: Binding(
                        get: { currencyMap[acc] ?? "USD" },
                        set: { currencyMap[acc] = $0 }
                    )) {
                        ForEach(appState.availableCurrencies, id: \.self) { c in
                            Text(c).tag(c)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                }

                // Cash Management Mode
                HStack {
                    Text("Cash Automation")
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)

                    Spacer()

                    Picker("", selection: Binding(
                        get: { cashModeMap[acc] ?? "Manual" },
                        set: { cashModeMap[acc] = $0 }
                    )) {
                        Text("Manual").tag("Manual")
                        Text("Auto").tag("Auto")
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 160)
                }

                // Closure Date
                HStack {
                    Text("Closure Date")
                        .appFont(.subheadline)
                        .foregroundStyle(.secondary)

                    Spacer()

                    if closureDateStr.isEmpty {
                        Button {
                            let formatter = MarketTime.isoFormatter()
                            closureMap[acc] = formatter.string(from: Date())
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "calendar.badge.plus")
                                Text("Set Closed")
                            }
                            .appFont(.caption.weight(.medium))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    } else {
                        let dateBinding = Binding<Date>(
                            get: {
                                MarketTime.isoFormatter().date(from: closureMap[acc] ?? "") ?? Date()
                            },
                            set: { newDate in
                                closureMap[acc] = MarketTime.isoFormatter().string(from: newDate)
                            }
                        )

                        HStack(spacing: 8) {
                            DatePicker("", selection: dateBinding, displayedComponents: .date)
                                .labelsHidden()
                                .datePickerStyle(.compact)
                                .gregorianCalendar()

                            Button(role: .destructive) {
                                closureMap[acc] = ""
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.primary.opacity(isClosed ? 0.015 : 0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
        .opacity(isClosed ? 0.8 : 1.0)
    }

    private func savePreferences() {
        isSaving = true
        Task {
            await vm.updateAccountPreferences(
                currencyMap: currencyMap.filter { !$0.value.isEmpty },
                cashModeMap: cashModeMap,
                closureMap: closureMap.filter { !$0.value.isEmpty }
            )
            isSaving = false
            ToastManager.shared.show(message: "Account preferences saved", style: .success)
        }
    }
}
