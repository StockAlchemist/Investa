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
                    Image(systemName: "dollarsign.circle.fill")
                        .foregroundStyle(Color.orange)
                        .appFont(.title3)
                    Text("Add Manual Currency")
                        .appFont(.headline.bold())
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
                    .tint(.orange)
                    .disabled(newCurrency.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(18)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.primary.opacity(0.03))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
            )

            // Active Currencies Section
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Available Currencies")
                        .appFont(.headline.bold())
                    Spacer()
                    Text("\(currencies.count) defined")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }

                if currencies.isEmpty {
                    Text("No custom currencies defined (default USD / THB active).")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 8)
                } else {
                    FlowChipsRemovable(items: currencies, color: .orange) { curr in
                        removeCurrency(curr)
                    }
                    .padding(.top, 4)
                }
            }
            .padding(18)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.primary.opacity(0.03))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
            )
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
            await vm.update("available_currencies", updated, note: "Added currency \(c)")
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
