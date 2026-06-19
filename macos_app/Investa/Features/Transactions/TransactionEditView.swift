import SwiftUI

/// Add/edit form presented as a sheet. Mirrors the web TransactionModal:
/// field disabling, type-change clearing, cash-symbol locking, account→currency
/// auto-mapping, Auto cash mode awareness, symbol/account autocomplete, live
/// total recompute, and per-type validation.
///
/// Reactive side effects (clearing fields on type change, currency mapping on
/// account change, total recompute) run **only on user edits** via custom
/// `Binding`s — not `.onChange` — so loading an existing transaction never wipes
/// its quantity/price or overrides its stored currency.
struct TransactionEditView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var appState: AppState

    /// nil → creating a new transaction.
    let existing: Transaction?
    let onSave: (Transaction) async -> Bool
    /// Symbols already used in the table, for autocomplete suggestions.
    var existingSymbols: [String] = []

    @State private var date = Date()
    @State private var type = "Buy"
    @State private var symbol = ""
    @State private var account = ""
    @State private var toAccount = ""
    @State private var quantity = ""
    @State private var price = ""
    @State private var commission = ""
    @State private var splitRatio = ""
    @State private var note = ""
    @State private var currency = "USD"
    @State private var overrideTotal = ""
    @State private var autoAddCash = false
    /// Set when the user types directly into the Total field; suppresses the
    /// auto-recompute from quantity/price/commission (mirrors web).
    @State private var totalLockedByUser = false

    @State private var isSaving = false
    @State private var error: String?

    @FocusState private var focusedField: Field?
    private enum Field: Hashable { case symbol, account, toAccount }

    // MARK: - Derived flags (mirrors web modal)

    private var isTransfer: Bool { type.lowercased() == "transfer" }
    private var isSplit: Bool { ["split", "stock split"].contains(type.lowercased()) }
    private var isCash: Bool {
        let s = symbol.trimmingCharacters(in: .whitespaces).uppercased()
        return s == "$CASH" || s == "CASH" || s.hasPrefix("CASH (")
    }
    private var canAutoAddCash: Bool {
        ["buy", "sell", "short sell", "buy to cover"].contains(type.lowercased()) && !isCash
    }
    private var isAccountAutoCash: Bool {
        (appState.accountCashModeMap[account] ?? "Manual") == "Auto"
    }

    // Field disable flags
    private var isQtyDisabled: Bool { isSplit }
    private var isPriceDisabled: Bool {
        isTransfer || isSplit ||
        (isCash && ["deposit", "withdrawal", "buy", "sell"].contains(type.lowercased()))
    }
    private var isTotalDisabled: Bool { isTransfer || isSplit }
    private var isCommDisabled: Bool { isTransfer || isSplit }

    // MARK: - Body

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(existing == nil ? "Add Transaction" : "Edit Transaction")
                .font(.title2.bold())
                .padding(20)
            Divider()

            Form {
                Section {
                    Picker("Type", selection: typeBinding) {
                        ForEach(Transaction.allTypes, id: \.self) { Text($0).tag($0) }
                    }
                    DatePicker("Date", selection: $date, displayedComponents: .date)
                    TextField("Symbol", text: symbolBinding)
                        .textCase(.uppercase)
                        .focused($focusedField, equals: .symbol)
                    suggestionRows(for: .symbol, current: symbolBinding, source: existingSymbols)
                }

                Section {
                    if isTransfer {
                        TextField("From Account", text: accountBinding)
                            .focused($focusedField, equals: .account)
                        suggestionRows(for: .account, current: accountBinding, source: appState.allAccounts)
                        TextField("To Account", text: $toAccount)
                            .focused($focusedField, equals: .toAccount)
                        suggestionRows(for: .toAccount, current: $toAccount, source: appState.allAccounts)
                    } else if isSplit {
                        LabeledContent("Account") {
                            Text("All Accounts").foregroundStyle(.secondary)
                        }
                    } else {
                        accountField
                    }
                    Picker("Currency", selection: $currency) {
                        ForEach(appState.availableCurrencies, id: \.self) { Text($0).tag($0) }
                    }
                }

                Section {
                    TextField("Quantity", text: quantityBinding)
                        .decimalKeyboard()
                        .disabled(isQtyDisabled)
                    TextField("Price / Share", text: priceBinding)
                        .decimalKeyboard()
                        .disabled(isPriceDisabled)
                    TextField("Commission", text: commissionBinding)
                        .decimalKeyboard()
                        .disabled(isCommDisabled)
                    if isSplit {
                        TextField("Split Ratio (e.g. 2 for 2:1)", text: $splitRatio)
                            .decimalKeyboard()
                    } else {
                        TextField("Total Amount", text: totalBinding)
                            .decimalKeyboard()
                            .disabled(isTotalDisabled)
                    }
                }

                Section {
                    TextField("Note", text: $note)
                    if canAutoAddCash {
                        Toggle("Auto-add matching cash transaction", isOn: $autoAddCash)
                            .disabled(isAccountAutoCash)
                        if isAccountAutoCash {
                            Text("Not available: this account uses Auto cash mode")
                                .font(.caption).foregroundStyle(.secondary)
                        }
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
        .frame(width: 460, height: 660)
        .onAppear(perform: populate)
    }

    // MARK: - Account field

    @ViewBuilder private var accountField: some View {
        if appState.allAccounts.isEmpty {
            TextField("Account", text: accountBinding)
                .focused($focusedField, equals: .account)
            suggestionRows(for: .account, current: accountBinding, source: appState.allAccounts)
        } else {
            Picker("Account", selection: accountBinding) {
                Text("Select…").tag("")
                ForEach(appState.allAccounts, id: \.self) { Text($0).tag($0) }
            }
        }
    }

    // MARK: - Autocomplete

    private func filteredSuggestions(_ input: String, _ source: [String]) -> [String] {
        let q = input.trimmingCharacters(in: .whitespaces).uppercased()
        let base = source.filter { !$0.isEmpty }
        let matches = q.isEmpty
            ? base
            : base.filter { $0.uppercased().contains(q) && $0.uppercased() != q }
        return Array(matches.prefix(8))
    }

    @ViewBuilder
    private func suggestionRows(for field: Field, current: Binding<String>, source: [String]) -> some View {
        if focusedField == field {
            let matches = filteredSuggestions(current.wrappedValue, source)
            ForEach(matches, id: \.self) { suggestion in
                Button {
                    current.wrappedValue = suggestion
                    focusedField = nil
                } label: {
                    HStack {
                        Text(suggestion)
                        Spacer()
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .font(.callout)
                .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Custom bindings (run reactive logic on user edits only)

    private var typeBinding: Binding<String> {
        Binding(get: { type }, set: { newType in
            type = newType
            handleTypeChange(newType)
        })
    }

    private var symbolBinding: Binding<String> {
        Binding(get: { symbol }, set: { newValue in
            symbol = newValue.uppercased()
            handleSymbolChange()
        })
    }

    private var accountBinding: Binding<String> {
        Binding(get: { account }, set: { newAccount in
            account = newAccount
            handleAccountChange(newAccount)
        })
    }

    private var quantityBinding: Binding<String> {
        Binding(get: { quantity }, set: { quantity = $0; totalLockedByUser = false; recomputeTotal() })
    }

    private var priceBinding: Binding<String> {
        Binding(get: { price }, set: { price = $0; totalLockedByUser = false; recomputeTotal() })
    }

    private var commissionBinding: Binding<String> {
        Binding(get: { commission }, set: { commission = $0; totalLockedByUser = false; recomputeTotal() })
    }

    private var totalBinding: Binding<String> {
        Binding(get: { overrideTotal }, set: { overrideTotal = $0; totalLockedByUser = !$0.isEmpty })
    }

    // MARK: - Total computation

    private var computedTotal: Double {
        Transaction.computeTotalAmount(
            type: type, symbol: symbol,
            quantity: Double(quantity) ?? 0, price: Double(price) ?? 0,
            commission: Double(commission) ?? 0,
            enteredTotal: overrideTotal.isEmpty ? nil : Double(overrideTotal)
        )
    }

    /// Live-fills the Total field for cash and trade types, mirroring the web
    /// modal's update effect. Cash always tracks quantity; trade types track
    /// quantity × price ± commission unless the user has locked the total.
    private func recomputeTotal() {
        let t = type.lowercased()
        if isCash {
            if ["deposit", "withdrawal", "buy", "sell"].contains(t), !quantity.isEmpty {
                overrideTotal = quantity
            }
            return
        }
        guard !totalLockedByUser else { return }
        if ["buy", "sell", "short sell", "buy to cover"].contains(t) {
            if let q = Double(quantity), let p = Double(price) {
                let comm = Double(commission) ?? 0
                let total = ["buy", "buy to cover"].contains(t) ? q * p + comm : q * p - comm
                overrideTotal = String(format: "%.2f", total)
            } else {
                overrideTotal = ""
            }
        }
    }

    // MARK: - Change handlers

    private func handleTypeChange(_ newType: String) {
        let t = newType.lowercased()
        if t == "dividend" {
            quantity = ""
            price = ""
        } else if t == "split" || t == "stock split" {
            quantity = ""
            price = ""
            overrideTotal = ""
            commission = ""
        } else if t == "transfer" {
            price = ""
            overrideTotal = ""
            commission = ""
        }
        if !canAutoAddCash { autoAddCash = false }
        recomputeTotal()
    }

    private func handleSymbolChange() {
        guard isCash else { return }
        let t = type.lowercased()
        if ["deposit", "withdrawal", "buy", "sell"].contains(t) {
            price = "1"
        }
        recomputeTotal()
    }

    private func handleAccountChange(_ newAccount: String) {
        if let mapped = appState.accountCurrencyMap[newAccount], !mapped.isEmpty {
            currency = mapped
        }
    }

    // MARK: - Populate from existing

    private func populate() {
        currency = appState.availableCurrencies.contains(appState.displayCurrency)
            ? appState.displayCurrency : (appState.availableCurrencies.first ?? "USD")
        if !appState.selectedAccounts.isEmpty {
            account = appState.selectedAccounts.first ?? ""
        }
        guard let tx = existing else { return }
        let rawType = tx.type
        let matched = Transaction.allTypes.first { $0.lowercased() == rawType.lowercased() }
        type = matched ?? rawType
        symbol = tx.symbol
        account = tx.account
        toAccount = tx.toAccount ?? ""
        quantity = tx.quantity == 0 ? "" : String(tx.quantity)
        price = tx.pricePerShare == 0 ? "" : String(tx.pricePerShare)
        commission = tx.commission == 0 ? "" : String(tx.commission)
        splitRatio = tx.splitRatio.map { $0 > 0 ? String($0) : "" } ?? ""
        note = tx.note ?? ""
        currency = tx.localCurrency
        autoAddCash = tx.autoAddCash ?? false
        // Total Amount is stored signed; show absolute value for the field
        // so the user sees "15.00" not "-15.00" when editing a Buy.
        let absTotal = abs(tx.totalAmount)
        overrideTotal = absTotal > 0 ? String(absTotal) : ""
        if let parsed = Self.dateFormatter.date(from: String(tx.date.prefix(10))) {
            date = parsed
        }
    }

    // MARK: - Submit

    private func submit() {
        error = nil
        let txType = type.lowercased()
        let sym = symbol.trimmingCharacters(in: .whitespaces).uppercased()
        let q = Double(quantity)
        let p = Double(price)

        // Required fields
        if sym.isEmpty { error = "Symbol cannot be empty."; return }
        if txType == "transfer" {
            if account.isEmpty || toAccount.isEmpty {
                error = "From and To accounts are required for a Transfer."; return
            }
        } else if !isSplit && account.isEmpty {
            error = "Account cannot be empty."; return
        }

        // Per-type value validation (mirrors web modal)
        if isCash {
            if ["deposit", "withdrawal", "buy", "sell"].contains(txType), (q ?? 0) <= 0 {
                error = "Amount (Quantity) must be positive for cash operations."; return
            }
        } else if txType == "transfer" {
            if (q ?? 0) <= 0 { error = "Quantity must be positive for a Transfer."; return }
        } else if ["buy", "sell", "short sell", "buy to cover"].contains(txType) {
            if (q ?? 0) <= 0 { error = "Quantity must be positive."; return }
            // Allow zero price (free-stock acquisitions); reject only negatives.
            if p == nil || p! < 0 { error = "Price/Unit cannot be negative."; return }
        } else if txType == "dividend" {
            if let total = Double(overrideTotal) {
                if total < 0 { error = "Dividend Total Amount cannot be negative."; return }
            } else {
                if (q ?? 0) <= 0 { error = "Dividend Quantity must be positive if Total is missing."; return }
                if (p ?? 0) <= 0 { error = "Dividend Price must be positive if Total is missing."; return }
            }
        }

        isSaving = true
        let payload = Transaction(
            id: existing?.id,
            date: Self.dateFormatter.string(from: date),
            account: isSplit ? "All Accounts" : account,
            symbol: sym,
            type: type,
            quantity: q ?? 0,
            pricePerShare: p ?? 0,
            commission: Double(commission) ?? 0,
            totalAmount: computedTotal,
            localCurrency: currency,
            splitRatio: isSplit ? Double(splitRatio) : nil,
            note: note.isEmpty ? nil : note,
            toAccount: isTransfer ? toAccount : nil,
            autoAddCash: canAutoAddCash ? autoAddCash : nil
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
