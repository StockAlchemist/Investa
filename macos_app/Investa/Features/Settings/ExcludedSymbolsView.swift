import SwiftUI

struct ExcludedSymbolsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var embedded: Bool = false
    @State private var newSymbol = ""

    private var excludedList: [String] {
        (settings?.userExcludedSymbols ?? []).sorted()
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
                .navigationTitle("Excluded Symbols")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            // Add Exclude Card
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(Color.red)
                        .appFont(.title3)
                    Text("Exclude a Symbol")
                        .appFont(.headline.bold())
                }

                Text("Excluded tickers are completely skipped during portfolio calculations, performance returns, and market data queries.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)

                HStack(spacing: 10) {
                    TextField("e.g. TEST-TICKER, CASH-USD", text: $newSymbol)
                        .textFieldStyle(.roundedBorder)
                        .uppercaseAutoCapitalization()
                        .autocorrectionDisabled()

                    Button("Exclude") {
                        addSymbol()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .disabled(newSymbol.trimmingCharacters(in: .whitespaces).isEmpty)
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

            // Current Excluded Section
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Currently Excluded Symbols")
                        .appFont(.headline.bold())
                    Spacer()
                    Text("\(excludedList.count) symbols")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }

                if excludedList.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "checkmark.circle")
                            .font(.system(size: 32))
                            .foregroundStyle(.green.opacity(0.6))
                            .padding(.vertical, 8)
                        Text("No symbols are currently excluded.")
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 24)
                } else {
                    FlowChipsRemovable(items: excludedList, color: .red) { sym in
                        removeSymbol(sym)
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

    private func addSymbol() {
        let sym = newSymbol.trimmingCharacters(in: .whitespaces).uppercased()
        guard !sym.isEmpty else { return }
        newSymbol = ""

        let current = settings?.userExcludedSymbols ?? []
        guard !current.contains(sym) else { return }

        let updated = Array(Set(current + [sym])).sorted()
        Task {
            await vm.update("user_excluded_symbols", updated, note: "Excluded \(sym)")
            ToastManager.shared.show(message: "Excluded symbol \(sym)", style: .info)
        }
    }

    private func removeSymbol(_ sym: String) {
        let current = settings?.userExcludedSymbols ?? []
        let updated = current.filter { $0 != sym }
        Task {
            guard await vm.update("user_excluded_symbols", updated, note: "Removed \(sym) from exclusion list") else { return }
            ToastManager.shared.show(message: "Re-included \(sym)", style: .success)
        }
    }
}
