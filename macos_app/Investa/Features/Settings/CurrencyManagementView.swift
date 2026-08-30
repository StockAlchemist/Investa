import SwiftUI

struct CurrencyManagementView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var embedded: Bool = false
    @State private var newCurrency = ""

    private var currencies: [String] {
        (settings?.availableCurrencies ?? []).sorted()
    }

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("Currency Management")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            // Add Currency Card
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    SectionLabel(title: "Add Manual Currency")
                    Spacer(minLength: 0)
                }

                Text("Define additional currencies available for manual cash and asset accounts (e.g. SGD, EUR, JPY, GBP).")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)

                HStack(spacing: 10) {
                    TextField("e.g. SGD", text: $newCurrency)
                        .textFieldStyle(.roundedBorder)
                        .uppercaseAutoCapitalization()
                        .autocorrectionDisabled()
                        .frame(maxWidth: 160)

                    Button("Add Currency") {
                        addCurrency()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.brand)
                    .disabled(newCurrency.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(18)
            .card()

            // Active Currencies Section
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    SectionLabel(title: "Available Currencies")
                    SettingsCountBadge(value: currencies.count)
                    Spacer(minLength: 0)
                }

                if currencies.isEmpty {
                    Text("No custom currencies defined (default USD / THB active).")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 8)
                } else {
                    FlowChipsRemovable(items: currencies, color: .brand) { curr in
                        removeCurrency(curr)
                    }
                    .padding(.top, 4)
                }
            }
            .padding(18)
            .card()
        }
    }

    private func addCurrency() {
        let c = newCurrency.trimmingCharacters(in: .whitespaces).uppercased()
        guard !c.isEmpty else { return }
        newCurrency = ""

        let current = settings?.availableCurrencies ?? []
        guard !current.contains(c) else { return }

        let updated = Array(Set(current + [c])).sorted()
        Task {
            guard await vm.update("available_currencies", updated, note: "Added currency \(c)") else { return }
            ToastManager.shared.show(message: "Added currency \(c)", style: .success)
        }
    }

    private func removeCurrency(_ curr: String) {
        let current = settings?.availableCurrencies ?? []
        let updated = current.filter { $0 != curr }
        Task {
            await vm.update("available_currencies", updated, note: "Removed currency \(curr)")
            ToastManager.shared.show(message: "Removed currency \(curr)", style: .info)
        }
    }
}
