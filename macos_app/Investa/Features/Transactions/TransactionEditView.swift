import SwiftUI

/// Add/edit form presented as a sheet. Computes the signed Total Amount the same
/// way the web TransactionModal does, unless the user overrides it explicitly.
struct TransactionEditView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var appState: AppState

    /// nil → creating a new transaction.
    let existing: Transaction?
    let onSave: (Transaction) async -> Bool

    @State private var date = Date()
    @State private var type = "Buy"
    @State private var symbol = ""
    @State private var account = ""
    @State private var toAccount = ""
    @State private var quantity = ""
    @State private var price = ""
    @State private var commission = ""
    @State private var note = ""
    @State private var currency = "USD"
    @State private var overrideTotal = ""
    @State private var autoAddCash = false

    @State private var isSaving = false
    @State private var error: String?

    private var isTransfer: Bool { type.lowercased() == "transfer" }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(existing == nil ? "Add Transaction" : "Edit Transaction")
                .font(.title2.bold())
                .padding(20)
            Divider()

            Form {
                Section {
                    Picker("Type", selection: $type) {
                        ForEach(Transaction.allTypes, id: \.self) { Text($0).tag($0) }
                    }
                    DatePicker("Date", selection: $date, displayedComponents: .date)
                    TextField("Symbol", text: $symbol)
                        .textCase(.uppercase)
                }

                Section {
                    if isTransfer {
                        TextField("From Account", text: $account)
                        TextField("To Account", text: $toAccount)
                    } else {
                        accountField
                    }
                    Picker("Currency", selection: $currency) {
                        ForEach(appState.availableCurrencies, id: \.self) { Text($0).tag($0) }
                    }
                }

                Section {
                    TextField("Quantity", text: $quantity)
                    TextField("Price / Share", text: $price)
                    TextField("Commission", text: $commission)
                    TextField("Total Amount (optional override)", text: $overrideTotal)
                    Text("Computed total: \(Fmt.currency(computedTotal, code: currency))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section {
                    TextField("Note", text: $note)
                    if ["buy", "sell"].contains(type.lowercased()) {
                        Toggle("Auto-add matching cash transaction", isOn: $autoAddCash)
                    }
                }

                if let error {
                    Text(error).foregroundStyle(.red).font(.callout)
                }
            }
            .formStyle(.grouped)

            Divider()
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button(existing == nil ? "Add" : "Save") { submit() }
                    .buttonStyle(.borderedProminent)
                    .disabled(isSaving || symbol.isEmpty)
            }
            .padding(16)
        }
        .frame(width: 460, height: 620)
        .onAppear(perform: populate)
    }

    @ViewBuilder private var accountField: some View {
        if appState.allAccounts.isEmpty {
            TextField("Account", text: $account)
        } else {
            Picker("Account", selection: $account) {
                Text("Select…").tag("")
                ForEach(appState.allAccounts, id: \.self) { Text($0).tag($0) }
            }
        }
    }

    private var computedTotal: Double {
        Transaction.computeTotalAmount(
            type: type, symbol: symbol,
            quantity: Double(quantity) ?? 0, price: Double(price) ?? 0,
            commission: Double(commission) ?? 0,
            enteredTotal: overrideTotal.isEmpty ? nil : Double(overrideTotal)
        )
    }

    private func populate() {
        currency = appState.availableCurrencies.contains(appState.displayCurrency)
            ? appState.displayCurrency : (appState.availableCurrencies.first ?? "USD")
        if !appState.selectedAccounts.isEmpty {
            account = appState.selectedAccounts.first ?? ""
        }
        guard let tx = existing else { return }
        type = tx.type
        symbol = tx.symbol
        account = tx.account
        toAccount = tx.toAccount ?? ""
        quantity = String(tx.quantity)
        price = String(tx.pricePerShare)
        commission = String(tx.commission)
        note = tx.note ?? ""
        currency = tx.localCurrency
        autoAddCash = tx.autoAddCash ?? false
        if let parsed = Self.dateFormatter.date(from: String(tx.date.prefix(10))) {
            date = parsed
        }
    }

    private func submit() {
        error = nil
        isSaving = true
        let payload = Transaction(
            id: existing?.id,
            date: Self.dateFormatter.string(from: date),
            account: account,
            symbol: symbol.trimmingCharacters(in: .whitespaces).uppercased(),
            type: type,
            quantity: Double(quantity) ?? 0,
            pricePerShare: Double(price) ?? 0,
            commission: Double(commission) ?? 0,
            totalAmount: computedTotal,
            localCurrency: currency,
            note: note.isEmpty ? nil : note,
            toAccount: isTransfer ? toAccount : nil,
            autoAddCash: autoAddCash
        )
        Task {
            let ok = await onSave(payload)
            isSaving = false
            if ok { dismiss() } else { error = "Failed to save. Check the fields and try again." }
        }
    }

    static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()
}
