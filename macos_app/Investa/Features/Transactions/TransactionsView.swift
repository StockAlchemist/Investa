import SwiftUI
import UniformTypeIdentifiers

enum TxDatePreset: String, CaseIterable, Identifiable {
    case all, mtd, ytd, d30, d90, y1, custom
    var id: String { rawValue }
    var label: String {
        switch self {
        case .all: return "All"; case .mtd: return "This month"; case .ytd: return "YTD"
        case .d30: return "30D"; case .d90: return "90D"; case .y1: return "1Y"; case .custom: return "Custom"
        }
    }
}

struct TransactionsView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = TransactionsViewModel()

    @State private var symbolFilter = ""
    @State private var accountFilter = ""
    @State private var filterTypes: Set<String> = []
    @State private var datePreset: TxDatePreset = .all
    @State private var customFrom = Date()
    @State private var customTo = Date()
    @State private var showFilters = false
    @State private var showDuplicatesOnly = false
    @State private var showInternalCash = false

    @State private var sortOrder = [KeyPathComparator(\Transaction.date, order: .reverse)]
    @State private var selection: Set<Int> = []
    @State private var editing: Transaction?
    @State private var showingAdd = false
    @State private var pendingDelete: Transaction?
    @State private var pendingBulkDelete = false
    // Import / IBKR
    @State private var reviewTransactions: [Transaction] = []
    @State private var showingReview = false
    @State private var importAccount = ""
    @State private var autoAddCash = true
    @State private var showImporter = false

    private var cur: String { appState.displayCurrency }
    private var isImportAccountAutoCash: Bool {
        !importAccount.isEmpty && (appState.accountCashModeMap[importAccount] ?? "Manual") == "Auto"
    }

    // MARK: - Derived data

    private func isInternalCash(_ tx: Transaction) -> Bool {
        let s = tx.symbol.uppercased()
        let cash = s == "$CASH" || s == "CASH" || s.hasPrefix("CASH (")
        return cash && (tx.note ?? "").lowercased().hasPrefix("auto-cash")
    }

    private func dupKey(_ tx: Transaction) -> String {
        "\(tx.symbol)|\(tx.date.prefix(10))|\(tx.type)|\(abs(tx.quantity))|\(tx.totalAmount)|\(tx.account)|\(tx.note ?? "")"
    }

    // Mirrors web app importMatchKey: symbol|date|type (dividend/tax omit qty to handle parser qty=0 cases)
    private func importMatchKey(_ tx: Transaction) -> String {
        let date = String(tx.date.prefix(10))
        let sym = tx.symbol.uppercased()
        let type = tx.type.lowercased()
        if type == "dividend" || type == "tax" {
            return "\(sym)|\(date)|\(type)"
        }
        return "\(sym)|\(date)|\(type)|\(abs(tx.quantity))"
    }

    private var existingTxKeys: Set<String> {
        Set(viewModel.transactions.map { importMatchKey($0) })
    }

    private var reviewDuplicateCount: Int {
        let keys = existingTxKeys
        return reviewTransactions.filter { keys.contains(importMatchKey($0)) }.count
    }

    private var duplicateKeys: Set<String> {
        var counts: [String: Int] = [:]
        for tx in viewModel.transactions where !isInternalCash(tx) { counts[dupKey(tx), default: 0] += 1 }
        return Set(counts.filter { $0.value > 1 }.keys)
    }
    private var duplicateCount: Int {
        let keys = duplicateKeys
        return viewModel.transactions.filter { !isInternalCash($0) && keys.contains(dupKey($0)) }.count
    }

    private var dateRange: (from: String?, to: String?) {
        let cal = Calendar.current; let now = Date()
        func iso(_ d: Date) -> String {
            let f = DateFormatter(); f.locale = Locale(identifier: "en_US_POSIX"); f.dateFormat = "yyyy-MM-dd"; return f.string(from: d)
        }
        switch datePreset {
        case .all: return (nil, nil)
        case .mtd: return (iso(cal.date(from: cal.dateComponents([.year, .month], from: now))!), nil)
        case .ytd: return (iso(cal.date(from: cal.dateComponents([.year], from: now))!), nil)
        case .d30: return (iso(cal.date(byAdding: .day, value: -30, to: now)!), nil)
        case .d90: return (iso(cal.date(byAdding: .day, value: -90, to: now)!), nil)
        case .y1: return (iso(cal.date(byAdding: .year, value: -1, to: now)!), nil)
        case .custom: return (iso(customFrom), iso(customTo))
        }
    }

    private var filtered: [Transaction] {
        let range = dateRange
        let types = Set(filterTypes.map { $0.lowercased() })
        return viewModel.transactions.filter { tx in
            let symOK = symbolFilter.isEmpty || tx.symbol.lowercased().contains(symbolFilter.lowercased())
            let accOK = accountFilter.isEmpty || tx.account.lowercased().contains(accountFilter.lowercased())
            let typeOK = types.isEmpty || types.contains(tx.type.lowercased())
            let d = String(tx.date.prefix(10))
            let dateOK = (range.from == nil || d >= range.from!) && (range.to == nil || d <= range.to!)
            let dupOK = !showDuplicatesOnly || duplicateKeys.contains(dupKey(tx))
            let cashOK = (showInternalCash || showDuplicatesOnly) || !isInternalCash(tx)
            return symOK && accOK && typeOK && dateOK && dupOK && cashOK
        }
    }
    private var sorted: [Transaction] { filtered.sorted(using: sortOrder) }
    private var existingTypes: [String] { Array(Set(viewModel.transactions.map { $0.type })).sorted() }
    private var existingSymbols: [String] {
        Array(Set(viewModel.transactions.map { $0.symbol }.filter { !$0.isEmpty })).sorted()
    }

    private static func typeColor(_ type: String) -> Color {
        switch type.uppercased() {
        case "BUY", "DEPOSIT", "BUY TO COVER": return .green
        case "SELL", "WITHDRAWAL", "SHORT SELL": return .red
        case "DIVIDEND", "INTEREST": return .indigo
        default: return .gray
        }
    }

    private func iconForType(_ type: String) -> String {
        switch type.uppercased() {
        case "BUY": return "bag.fill"
        case "SELL": return "tag.fill"
        case "DEPOSIT": return "arrow.down.circle.fill"
        case "WITHDRAWAL": return "arrow.up.circle.fill"
        case "DIVIDEND": return "dollarsign.circle.fill"
        case "INTEREST": return "percent"
        case "SHORT SELL": return "arrow.down.right.circle.fill"
        case "BUY TO COVER": return "arrow.up.left.circle.fill"
        default: return "doc.text.fill"
        }
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Transactions").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
                Text("Showing \(sorted.count) of \(viewModel.transactions.count)")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            if let error = viewModel.errorMessage { errorBanner(error) }
            contentBody
        }
        .macMinSize(width: 820, height: 560)
        .task(id: accountSignature) { reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
        .sheet(isPresented: $showingAdd) { TransactionEditView(existing: nil, onSave: handleSave, existingSymbols: existingSymbols).environmentObject(appState) }
        .sheet(item: $editing) { tx in TransactionEditView(existing: tx, onSave: handleSave, existingSymbols: existingSymbols).environmentObject(appState) }
        .sheet(isPresented: $showingReview) { reviewSheet }
        .alert("Delete transaction?", isPresented: deleteBinding, presenting: pendingDelete) { tx in
            Button("Delete", role: .destructive) { Task { await viewModel.delete(tx); reload() } }
            Button("Cancel", role: .cancel) {}
        } message: { tx in Text("\(tx.type) \(tx.symbol) on \(tx.displayDate) will be removed.") }
        .alert("Delete \(selection.count) transactions?", isPresented: $pendingBulkDelete) {
            Button("Delete", role: .destructive) { bulkDelete() }
            Button("Cancel", role: .cancel) {}
        }
    }

    // MARK: - Content

    /// macOS keeps the controls fixed at the top and lets the `Table` fill the
    /// remaining window height (the table scrolls internally). iOS scrolls the
    /// whole page since its rows are a plain `LazyVStack`.
    @ViewBuilder private var contentBody: some View {
        #if os(macOS)
        VStack(spacing: 16) {
            TxKpiStrip(transactions: sorted, preferredCurrency: cur)
            if !viewModel.pendingIbkr.isEmpty { ibkrPendingCard }
            if duplicateCount > 0 { duplicateBanner }
            toolbar
            if showFilters { filterPanel }
            table
        }
        .padding(20)
        #else
        ScrollView {
            VStack(spacing: 16) {
                TxKpiStrip(transactions: sorted, preferredCurrency: cur)
                if !viewModel.pendingIbkr.isEmpty { ibkrPendingCard }
                if duplicateCount > 0 { duplicateBanner }
                toolbar
                if showFilters { filterPanel }
                table
            }
            .padding(20)
        }
        #endif
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        #if os(iOS)
        HStack(spacing: 12) {
            Button { showingAdd = true } label: { Image(systemName: "plus") }
                .buttonStyle(.borderedProminent)
            importControl
            Button { Task { await viewModel.syncIbkr() } } label: { Image(systemName: "arrow.triangle.2.circlepath") }
                .buttonStyle(.bordered)
            Spacer()
            Button { showFilters.toggle(); if !showFilters { resetFilters() } } label: {
                Image(systemName: "line.3.horizontal.decrease.circle")
            }
            .buttonStyle(.bordered).tint(showFilters ? .accentColor : nil)
            Button { exportCSV() } label: { Image(systemName: "square.and.arrow.up") }
                .buttonStyle(.bordered)
            if !selection.isEmpty {
                Button(role: .destructive) { pendingBulkDelete = true } label: {
                    Image(systemName: "trash")
                }.buttonStyle(.bordered)
            }
        }
        #else
        HStack {
            Button { showingAdd = true } label: { Label("Add", systemImage: "plus") }
                .buttonStyle(.borderedProminent)
            importControl
            Button { Task { await viewModel.syncIbkr() } } label: { Label("IBKR Sync", systemImage: "arrow.triangle.2.circlepath") }
                .buttonStyle(.bordered)
            if !selection.isEmpty {
                Button(role: .destructive) { pendingBulkDelete = true } label: {
                    Label("Delete (\(selection.count))", systemImage: "trash")
                }.buttonStyle(.bordered)
            }
            Spacer()
            Button { showFilters.toggle(); if !showFilters { resetFilters() } } label: {
                Label("Filters", systemImage: "line.3.horizontal.decrease.circle")
            }
            .buttonStyle(.bordered).tint(showFilters ? .accentColor : nil)
            Button { exportCSV() } label: { Label("Export", systemImage: "square.and.arrow.up") }
                .buttonStyle(.bordered)
        }
        #endif
    }

    // MARK: - Import (PDF/IBKR document parsing)

    private var importControl: some View {
        Menu {
            Picker("Import to account", selection: $importAccount) {
                Text("Default").tag("")
                ForEach(appState.allAccounts, id: \.self) { Text($0).tag($0) }
            }
            .onChange(of: importAccount) { _, _ in
                if isImportAccountAutoCash { autoAddCash = false }
            }
            Toggle("Auto-add cash", isOn: $autoAddCash)
                .disabled(isImportAccountAutoCash)
            Divider()
            Button("Choose PDF / Image…") { showImporter = true }
        } label: {
            if viewModel.isImporting { ProgressView().controlSize(.small) }
            else { Label("Import", systemImage: "doc.badge.plus") }
        }
        .borderlessMenu().fixedSize()
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.pdf, .image]) { result in
            if case .success(let url) = result { openImportFile(url) }
        }
    }

    private static let outflowTypes: Set<String> = [
        "buy", "withdrawal", "fees", "fee", "tax", "withholding tax",
        "split", "stock split", "buy to cover"
    ]

    private func openImportFile(_ url: URL) {
        Task { @MainActor in
            let access = url.startAccessingSecurityScopedResource()
            var parsed = await viewModel.parseDocument(url)
            if access { url.stopAccessingSecurityScopedResource() }
            parsed = parsed.map { tx in
                var t = tx
                // Mirror web: importAccount overrides parsed account; fall back to "Default" if both empty.
                t.account = importAccount.isEmpty ? (t.account.isEmpty ? "Default" : t.account) : importAccount
                // Apply sign convention: parsers emit positive values; outflows store negative.
                let absTotal = abs(t.totalAmount)
                if absTotal > 1e-9 {
                    let isOutflow = Self.outflowTypes.contains(t.type.lowercased().trimmingCharacters(in: .whitespaces))
                    t.totalAmount = isOutflow ? -absTotal : absTotal
                }
                return t
            }
            if !parsed.isEmpty { reviewTransactions = parsed; showingReview = true }
        }
    }

    private var reviewSheet: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Review Import (\(reviewTransactions.count))").font(.title2.bold())
                    if reviewDuplicateCount > 0 {
                        Label("\(reviewDuplicateCount) already in your table (highlighted)", systemImage: "exclamationmark.triangle.fill")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                Spacer()
                Toggle("Auto-add cash", isOn: $autoAddCash)
            }
            .padding(16)
            Divider()
            if reviewTransactions.isEmpty {
                ContentUnavailableView("Nothing to import", systemImage: "doc")
            } else {
                #if os(macOS)
                // Column headers
                HStack(spacing: 0) {
                    Text("Date").frame(width: 90, alignment: .leading)
                    Text("Type").frame(width: 88, alignment: .leading)
                    Text("Symbol").frame(width: 82, alignment: .leading)
                    Text("Qty").frame(width: 58, alignment: .trailing)
                    Text("Price").frame(width: 65, alignment: .trailing)
                    Text("Total").frame(width: 92, alignment: .trailing)
                    Text("Account").frame(width: 114, alignment: .leading).padding(.leading, 6)
                    Spacer()
                }
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 16).padding(.vertical, 6)
                Divider()
                #endif
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(Array(reviewTransactions.enumerated()), id: \.offset) { i, tx in
                            let isDuplicate = existingTxKeys.contains(importMatchKey(tx))
                            // Normalize stored type casing to match allTypes tags for Picker selection.
                            let normalizedType = Transaction.allTypes.first { $0.lowercased() == tx.type.lowercased() } ?? tx.type
                            #if os(iOS)
                            VStack(spacing: 6) {
                                // Row 1: Symbol (editable) + duplicate badge | Total (editable)
                                HStack {
                                    TextField("Symbol", text: Binding(
                                        get: { reviewTransactions[i].symbol },
                                        set: { reviewTransactions[i].symbol = $0.uppercased() }))
                                        .fontWeight(.bold).textCase(.uppercase)
                                    if isDuplicate {
                                        Label("Duplicate", systemImage: "exclamationmark.triangle.fill")
                                            .font(.system(size: 10, weight: .bold))
                                            .foregroundStyle(.orange)
                                            .padding(.horizontal, 5).padding(.vertical, 2)
                                            .background(.orange.opacity(0.15), in: Capsule())
                                    }
                                    Spacer()
                                    TextField("Total", text: Binding(
                                        get: { reviewTransactions[i].totalAmount == 0 ? "" : String(reviewTransactions[i].totalAmount) },
                                        set: { if let v = Double($0) { reviewTransactions[i].totalAmount = v } }))
                                        .decimalKeyboard()
                                        .multilineTextAlignment(.trailing).monospacedDigit().frame(width: 90)
                                }
                                // Row 2: Type (Picker) | Qty (editable) | Price (editable)
                                HStack {
                                    Picker("", selection: Binding(
                                        get: { normalizedType },
                                        set: { reviewTransactions[i].type = $0 })) {
                                        ForEach(Transaction.allTypes, id: \.self) { Text($0).tag($0) }
                                    }
                                    .labelsHidden().pickerStyle(.menu)
                                    .font(.caption.weight(.bold))
                                    Spacer()
                                    TextField("Qty", text: Binding(
                                        get: { reviewTransactions[i].quantity == 0 ? "" : Fmt.number(reviewTransactions[i].quantity) },
                                        set: { if let v = Double($0) { reviewTransactions[i].quantity = v } }))
                                        .decimalKeyboard()
                                        .multilineTextAlignment(.trailing).monospacedDigit().font(.caption).frame(width: 70)
                                    Text("@").font(.caption2).foregroundStyle(.tertiary)
                                    TextField("Price", text: Binding(
                                        get: { reviewTransactions[i].pricePerShare == 0 ? "" : Fmt.number(reviewTransactions[i].pricePerShare) },
                                        set: { if let v = Double($0) { reviewTransactions[i].pricePerShare = v } }))
                                        .decimalKeyboard()
                                        .multilineTextAlignment(.trailing).monospacedDigit().font(.caption).frame(width: 70)
                                }
                                // Row 3: Date (editable) | Account (editable) | Delete
                                HStack {
                                    TextField("YYYY-MM-DD", text: Binding(
                                        get: { String(reviewTransactions[i].date.prefix(10)) },
                                        set: { reviewTransactions[i].date = $0 }))
                                        .font(.caption2).foregroundStyle(.secondary).frame(width: 90)
                                    Spacer()
                                    TextField("Account", text: Binding(
                                        get: { reviewTransactions[i].account },
                                        set: { reviewTransactions[i].account = $0 }))
                                        .textFieldStyle(.roundedBorder).frame(width: 130)
                                    Button(role: .destructive) { reviewTransactions.remove(at: i) } label: { Image(systemName: "trash") }
                                        .buttonStyle(.borderless).foregroundStyle(.red)
                                }
                            }
                            .padding(.vertical, 12).padding(.horizontal, 16)
                            .background(isDuplicate ? Color.orange.opacity(0.08) : Color.clear)
                            Divider()
                            #else
                            HStack(spacing: 0) {
                                // Date
                                TextField("YYYY-MM-DD", text: Binding(
                                    get: { String(reviewTransactions[i].date.prefix(10)) },
                                    set: { reviewTransactions[i].date = $0 }))
                                    .foregroundStyle(.secondary).frame(width: 90, alignment: .leading)
                                // Type
                                Picker("", selection: Binding(
                                    get: { normalizedType },
                                    set: { reviewTransactions[i].type = $0 })) {
                                    ForEach(Transaction.allTypes, id: \.self) { Text($0).tag($0) }
                                }
                                .labelsHidden().pickerStyle(.menu).frame(width: 88, alignment: .leading)
                                // Symbol + duplicate badge
                                HStack(spacing: 4) {
                                    TextField("Symbol", text: Binding(
                                        get: { reviewTransactions[i].symbol },
                                        set: { reviewTransactions[i].symbol = $0.uppercased() }))
                                        .fontWeight(.medium).textCase(.uppercase)
                                    if isDuplicate {
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .font(.system(size: 10))
                                            .foregroundStyle(.orange)
                                            .help("This transaction already exists in your table")
                                    }
                                }
                                .frame(width: 82, alignment: .leading)
                                // Qty
                                TextField("Qty", text: Binding(
                                    get: { reviewTransactions[i].quantity == 0 ? "" : Fmt.number(reviewTransactions[i].quantity) },
                                    set: { if let v = Double($0) { reviewTransactions[i].quantity = v } }))
                                    .multilineTextAlignment(.trailing).monospacedDigit().frame(width: 58, alignment: .trailing)
                                // Price
                                TextField("Price", text: Binding(
                                    get: { reviewTransactions[i].pricePerShare == 0 ? "" : Fmt.number(reviewTransactions[i].pricePerShare) },
                                    set: { if let v = Double($0) { reviewTransactions[i].pricePerShare = v } }))
                                    .multilineTextAlignment(.trailing).monospacedDigit().frame(width: 65, alignment: .trailing)
                                // Total
                                TextField("Total", text: Binding(
                                    get: { reviewTransactions[i].totalAmount == 0 ? "" : String(reviewTransactions[i].totalAmount) },
                                    set: { if let v = Double($0) { reviewTransactions[i].totalAmount = v } }))
                                    .multilineTextAlignment(.trailing).monospacedDigit().frame(width: 92, alignment: .trailing)
                                // Account
                                TextField("Account", text: Binding(
                                    get: { reviewTransactions[i].account },
                                    set: { reviewTransactions[i].account = $0 }))
                                    .textFieldStyle(.roundedBorder).frame(width: 108)
                                    .padding(.leading, 6)
                                Spacer()
                                Button(role: .destructive) { reviewTransactions.remove(at: i) } label: { Image(systemName: "trash") }
                                    .buttonStyle(.borderless).foregroundStyle(.red)
                            }
                            .font(.callout).padding(.vertical, 6).padding(.horizontal, 16)
                            .background(isDuplicate ? Color.orange.opacity(0.08) : Color.clear)
                            Divider()
                            #endif
                        }
                    }
                }
            }
            Divider()
            HStack {
                Spacer()
                Button("Cancel") { showingReview = false; reviewTransactions = [] }
                Button("Import \(reviewTransactions.count)") {
                    Task {
                        let ok = await viewModel.addBatch(reviewTransactions, autoAddCash: autoAddCash)
                        if ok { showingReview = false; reviewTransactions = []; reload() }
                    }
                }
                .buttonStyle(.borderedProminent).disabled(reviewTransactions.isEmpty || viewModel.isImporting)
            }
            .padding(16)
        }
        #if os(macOS)
        .frame(width: 760, height: 560)
        #endif
    }

    // MARK: - IBKR pending

    private var ibkrPendingCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            #if os(iOS)
            VStack(alignment: .leading, spacing: 8) {
                Label("\(viewModel.pendingIbkr.count) pending IBKR", systemImage: "tray.full").font(.headline)
                HStack {
                    Button("Approve All") { Task { await viewModel.approvePending(viewModel.pendingIbkr.compactMap { $0.id }) } }
                        .buttonStyle(.borderedProminent).tint(.green)
                    Button("Reject All", role: .destructive) { Task { await viewModel.rejectPending(viewModel.pendingIbkr.compactMap { $0.id }) } }
                        .buttonStyle(.bordered)
                }
            }
            ForEach(viewModel.pendingIbkr) { tx in
                VStack(spacing: 8) {
                    HStack {
                        Text(tx.symbol).font(.headline).fontWeight(.bold)
                        Spacer()
                        Text(Fmt.currency(tx.totalAmount, code: tx.localCurrency)).fontWeight(.bold).monospacedDigit()
                    }
                    HStack {
                        Text(tx.type).font(.caption2.weight(.bold)).padding(.horizontal, 6).padding(.vertical, 2).background(.quaternary, in: Capsule())
                        Spacer()
                        if tx.quantity != 0 {
                            Text("\(Fmt.number(tx.quantity))").font(.caption).monospacedDigit().foregroundStyle(.secondary)
                        }
                    }
                    HStack {
                        Text(tx.displayDate).font(.caption2).foregroundStyle(.secondary)
                        Spacer()
                        if let id = tx.id {
                            Button { Task { await viewModel.approvePending([id]) } } label: { Image(systemName: "checkmark.circle.fill") }.buttonStyle(.borderless).foregroundStyle(.green).font(.title3)
                            Button { Task { await viewModel.rejectPending([id]) } } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.borderless).foregroundStyle(.red).font(.title3)
                        }
                    }
                }
                .padding(12).background(.background, in: RoundedRectangle(cornerRadius: 10))
            }
            #else
            HStack {
                Label("\(viewModel.pendingIbkr.count) pending IBKR transactions", systemImage: "tray.full").font(.headline)
                Spacer()
                Button("Approve All") { Task { await viewModel.approvePending(viewModel.pendingIbkr.compactMap { $0.id }) } }
                    .buttonStyle(.borderedProminent).tint(.green)
                Button("Reject All", role: .destructive) { Task { await viewModel.rejectPending(viewModel.pendingIbkr.compactMap { $0.id }) } }
                    .buttonStyle(.bordered)
            }
            ForEach(viewModel.pendingIbkr) { tx in
                HStack {
                    Text(tx.displayDate).foregroundStyle(.secondary).frame(width: 90, alignment: .leading)
                    Text(tx.type).frame(width: 80, alignment: .leading)
                    Text(tx.symbol).fontWeight(.medium).frame(width: 70, alignment: .leading)
                    Text(Fmt.number(tx.quantity)).monospacedDigit().frame(width: 60, alignment: .trailing)
                    Text(Fmt.currency(tx.totalAmount, code: tx.localCurrency)).monospacedDigit().frame(width: 100, alignment: .trailing)
                    Text(tx.account).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    if let id = tx.id {
                        Button { Task { await viewModel.approvePending([id]) } } label: { Image(systemName: "checkmark.circle") }
                            .buttonStyle(.borderless).foregroundStyle(.green)
                        Button { Task { await viewModel.rejectPending([id]) } } label: { Image(systemName: "xmark.circle") }
                            .buttonStyle(.borderless).foregroundStyle(.red)
                    }
                }
                .font(.callout).padding(.vertical, 3)
                Divider()
            }
            #endif
        }
        .padding(16)
        .background(.cyan.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.cyan.opacity(0.3), lineWidth: 1))
    }

    private var duplicateBanner: some View {
        Button { showDuplicatesOnly.toggle() } label: {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                Text("\(duplicateCount) potential duplicate \(duplicateCount == 1 ? "transaction" : "transactions") detected")
                    .font(.callout)
                Spacer()
                Text(showDuplicatesOnly ? "Showing — tap to clear" : "Review").font(.caption.bold())
            }
            .padding(12)
            .background(.orange.opacity(showDuplicatesOnly ? 0.2 : 0.1), in: RoundedRectangle(cornerRadius: 8))
            .foregroundStyle(.orange)
        }
        .buttonStyle(.plain)
    }

    private var filterPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            #if os(iOS)
            VStack(alignment: .leading, spacing: 12) {
                TextField("Filter symbol…", text: $symbolFilter).textFieldStyle(.roundedBorder)
                TextField("Filter account…", text: $accountFilter).textFieldStyle(.roundedBorder)
                Toggle("Internal cash", isOn: $showInternalCash)
            }
            #else
            HStack {
                TextField("Filter symbol…", text: $symbolFilter).textFieldStyle(.roundedBorder).frame(width: 160)
                TextField("Filter account…", text: $accountFilter).textFieldStyle(.roundedBorder).frame(width: 180)
                Toggle("Internal cash", isOn: $showInternalCash)
                Spacer()
            }
            #endif
            // Type chips
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(existingTypes, id: \.self) { type in
                        let on = filterTypes.contains(type)
                        Button {
                            if on { filterTypes.remove(type) } else { filterTypes.insert(type) }
                        } label: {
                            Text(type).font(.caption.weight(.medium))
                                .padding(.horizontal, 10).padding(.vertical, 4)
                                .background(on ? Self.typeColor(type).opacity(0.2) : Color.gray.opacity(0.15), in: Capsule())
                                .foregroundStyle(on ? Self.typeColor(type) : Color.gray)
                        }.buttonStyle(.plain)
                    }
                }
            }
            // Date presets
            #if os(iOS)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(TxDatePreset.allCases) { p in
                        Button { datePreset = p } label: {
                            Text(p.label).font(.caption.weight(.medium))
                                .padding(.horizontal, 10).padding(.vertical, 4)
                                .background(datePreset == p ? Color.accentColor.opacity(0.2) : Color.gray.opacity(0.15), in: Capsule())
                                .foregroundStyle(datePreset == p ? Color.accentColor : Color.gray)
                        }.buttonStyle(.plain)
                    }
                }
            }
            if datePreset == .custom {
                VStack(alignment: .leading, spacing: 8) {
                    DatePicker("From", selection: $customFrom, displayedComponents: .date)
                        .gregorianCalendar()
                    DatePicker("To", selection: $customTo, displayedComponents: .date)
                        .gregorianCalendar()
                }
            }
            #else
            HStack(spacing: 6) {
                ForEach(TxDatePreset.allCases) { p in
                    Button { datePreset = p } label: {
                        Text(p.label).font(.caption.weight(.medium))
                            .padding(.horizontal, 10).padding(.vertical, 4)
                            .background(datePreset == p ? Color.accentColor.opacity(0.2) : Color.gray.opacity(0.15), in: Capsule())
                            .foregroundStyle(datePreset == p ? Color.accentColor : Color.gray)
                    }.buttonStyle(.plain)
                }
                if datePreset == .custom {
                    DatePicker("", selection: $customFrom, displayedComponents: .date).labelsHidden()
                        .gregorianCalendar()
                    Text("→").foregroundStyle(.secondary)
                    DatePicker("", selection: $customTo, displayedComponents: .date).labelsHidden()
                        .gregorianCalendar()
                }
            }
            #endif
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Table

    private var table: some View {
        #if os(iOS)
        LazyVStack(spacing: 12) {
            ForEach(sorted) { tx in
                iosTransactionRow(tx)
            }
        }
        #else
        Table(sorted, selection: tableSelection, sortOrder: $sortOrder) {
            TableColumn("Date", value: \.date) { Text($0.displayDate).foregroundStyle(.secondary) }
                .width(min: 90, ideal: 110)
            TableColumn("Type", value: \.type) { tx in
                Text(tx.type.uppercased()).font(.caption2.weight(.bold))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Self.typeColor(tx.type).opacity(0.15), in: Capsule())
                    .foregroundStyle(Self.typeColor(tx.type))
            }
            .width(min: 80, ideal: 100)
            TableColumn("Symbol", value: \.symbol) { Text($0.symbol).fontWeight(.medium) }
                .width(min: 70, ideal: 90)
            TableColumn("Qty", value: \.quantity) { Text($0.quantity == 0 ? "-" : Fmt.number($0.quantity)).monospacedDigit() }
                .width(min: 60, ideal: 80)
            TableColumn("Price", value: \.pricePerShare) { Text($0.pricePerShare == 0 ? "-" : Fmt.number($0.pricePerShare)).monospacedDigit() }
                .width(min: 64, ideal: 84)
            TableColumn("Total", value: \.totalAmount) { tx in
                Text(Fmt.currency(tx.totalAmount, code: tx.localCurrency)).fontWeight(.bold).monospacedDigit()
                    .foregroundStyle(Fmt.tint(for: tx.totalAmount))
            }
            .width(min: 90, ideal: 120)
            TableColumn("Account", value: \.account) { Text($0.account).font(.caption).foregroundStyle(.secondary) }
                .width(min: 90, ideal: 130)
            TableColumn("") { tx in
                HStack(spacing: 4) {
                    Button { editing = tx } label: { Image(systemName: "pencil") }.buttonStyle(.borderless)
                    Button { pendingDelete = tx } label: { Image(systemName: "trash") }.buttonStyle(.borderless).foregroundStyle(.red)
                }
            }
            .width(56)
        }
        .frame(maxHeight: .infinity)
        #endif
    }

    private func iosTransactionRow(_ tx: Transaction) -> some View {
        HStack(alignment: .center, spacing: 16) {
            // Icon
            StockIcon(symbol: tx.symbol.isEmpty ? "CASH" : tx.symbol, size: 42)
                .frame(width: 48, height: 48)

            // Middle: Symbol, Type, Date
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(tx.symbol.isEmpty ? tx.type : tx.symbol)
                        .font(.headline.weight(.bold))
                        .lineLimit(1)
                    if !tx.symbol.isEmpty {
                        Text(tx.type.uppercased())
                            .font(.system(size: 10, weight: .bold))
                            .lineLimit(1)
                            .padding(.horizontal, 6).padding(.vertical, 3)
                            .background(Self.typeColor(tx.type).opacity(0.15), in: Capsule())
                            .foregroundStyle(Self.typeColor(tx.type))
                            .fixedSize(horizontal: true, vertical: false)
                    }
                }
                
                HStack(spacing: 4) {
                    Text(tx.displayDate)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if tx.quantity != 0 {
                        Text("•")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        Text("\(Fmt.number(tx.quantity)) @ \(Fmt.number(tx.pricePerShare))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }

            Spacer(minLength: 8)

            // Right: Amount, Account, Menu
            VStack(alignment: .trailing, spacing: 6) {
                Text(Fmt.currency(tx.totalAmount, code: tx.localCurrency))
                    .font(.subheadline.weight(.bold))
                    .monospacedDigit()
                    .foregroundStyle(Fmt.tint(for: tx.totalAmount))
                
                HStack(spacing: 8) {
                    Text(tx.account)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                    
                    Menu {
                        Button { editing = tx } label: { Label("Edit", systemImage: "pencil") }
                        Button(role: .destructive) { pendingDelete = tx } label: { Label("Delete", systemImage: "trash") }
                    } label: {
                        Image(systemName: "ellipsis.circle.fill")
                            .font(.system(size: 18))
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
    }

    /// Table selection works on the row id (`Int?`); bridge to a `Set<Int>`.
    private var tableSelection: Binding<Set<Int?>> {
        Binding(
            get: { Set(selection.map { Optional($0) }) },
            set: { selection = Set($0.compactMap { $0 }) }
        )
    }

    private func errorBanner(_ message: String) -> some View {
        HStack { Image(systemName: "exclamationmark.triangle.fill"); Text(message); Spacer(); Button("Retry") { reload() } }
            .font(.callout).padding(12).foregroundStyle(.red)
            .background(.red.opacity(0.12), in: RoundedRectangle(cornerRadius: 8)).padding(.horizontal, 20).padding(.top, 12)
    }

    // MARK: - Actions

    private var accountSignature: String { appState.selectedAccounts.sorted().joined(separator: ",") }
    private var deleteBinding: Binding<Bool> { Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }) }
    private func reload() { viewModel.reload(accounts: appState.accountsQuery) }
    private func resetFilters() {
        symbolFilter = ""; accountFilter = ""; filterTypes = []; datePreset = .all; showInternalCash = false; showDuplicatesOnly = false
    }
    private func handleSave(_ tx: Transaction) async -> Bool {
        let ok = await viewModel.save(tx); if ok { reload() }; return ok
    }
    private func bulkDelete() {
        let toDelete = viewModel.transactions.filter { $0.id.map { selection.contains($0) } ?? false }
        selection = []
        Task { for tx in toDelete { await viewModel.delete(tx) }; reload() }
    }

    private func exportCSV() {
        let header = "Date,Type,Symbol,Quantity,Price/Share,Commission,Total Amount,Local Currency,Account,Note"
        func esc(_ s: String) -> String { "\"\(s.replacingOccurrences(of: "\"", with: "\"\""))\"" }
        let lines = sorted.map { tx in
            [tx.date, tx.type, tx.symbol, String(tx.quantity), String(tx.pricePerShare), String(tx.commission),
             String(tx.totalAmount), tx.localCurrency, tx.account, tx.note ?? ""].map(esc).joined(separator: ",")
        }
        exportText(([header] + lines).joined(separator: "\n"), filename: "transactions.csv")
    }
}
