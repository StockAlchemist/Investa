import SwiftUI

struct TransactionsView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = TransactionsViewModel()

    @State private var editing: Transaction?
    @State private var showingAdd = false
    @State private var sortOrder = [KeyPathComparator(\Transaction.date, order: .reverse)]
    @State private var pendingDelete: Transaction?

    private var rows: [Transaction] { viewModel.transactions.sorted(using: sortOrder) }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if let error = viewModel.errorMessage {
                errorBanner(error)
            }
            content
        }
        .frame(minWidth: 700, minHeight: 500)
        .task(id: accountSignature) { reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
        .sheet(isPresented: $showingAdd) {
            TransactionEditView(existing: nil, onSave: handleSave)
                .environmentObject(appState)
        }
        .sheet(item: $editing) { tx in
            TransactionEditView(existing: tx, onSave: handleSave)
                .environmentObject(appState)
        }
        .alert("Delete transaction?", isPresented: deleteAlertBinding, presenting: pendingDelete) { tx in
            Button("Delete", role: .destructive) { Task { await viewModel.delete(tx) } }
            Button("Cancel", role: .cancel) {}
        } message: { tx in
            Text("\(tx.type) \(tx.symbol) on \(tx.date) will be permanently removed.")
        }
    }

    private var header: some View {
        HStack {
            Text("Transactions").font(.title2.bold())
            if viewModel.isLoading { ProgressView().controlSize(.small) }
            Spacer()
            Button { showingAdd = true } label: { Label("Add", systemImage: "plus") }
                .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
    }

    @ViewBuilder private var content: some View {
        if viewModel.transactions.isEmpty && !viewModel.isLoading {
            ContentUnavailableView("No transactions", systemImage: "list.bullet.rectangle",
                                   description: Text("Add one with the + button."))
        } else {
            Table(rows, sortOrder: $sortOrder) {
                TableColumn("Date", value: \.date) { Text($0.date) }
                    .width(min: 90, ideal: 100)
                TableColumn("Type", value: \.type) { Text($0.type) }
                    .width(min: 80, ideal: 100)
                TableColumn("Symbol", value: \.symbol) { Text($0.symbol).fontWeight(.medium) }
                    .width(min: 70, ideal: 90)
                TableColumn("Account", value: \.account) { Text($0.account) }
                    .width(min: 80, ideal: 120)
                TableColumn("Qty", value: \.quantity) {
                    Text(Fmt.number($0.quantity)).monospacedDigit()
                }
                .width(min: 60, ideal: 80)
                TableColumn("Price", value: \.pricePerShare) {
                    Text(Fmt.number($0.pricePerShare)).monospacedDigit()
                }
                .width(min: 70, ideal: 90)
                TableColumn("Total", value: \.totalAmount) { tx in
                    Text(Fmt.currency(tx.totalAmount, code: tx.localCurrency))
                        .monospacedDigit()
                        .foregroundStyle(Fmt.tint(for: tx.totalAmount))
                }
                .width(min: 90, ideal: 120)
                TableColumn("") { tx in
                    HStack(spacing: 4) {
                        Button { editing = tx } label: { Image(systemName: "pencil") }
                            .buttonStyle(.borderless)
                        Button { pendingDelete = tx } label: { Image(systemName: "trash") }
                            .buttonStyle(.borderless)
                            .foregroundStyle(.red)
                    }
                }
                .width(60)
            }
            .contextMenu(forSelectionType: Transaction.ID.self) { _ in } primaryAction: { ids in
                if let id = ids.first, let tx = viewModel.transactions.first(where: { $0.id == id }) {
                    editing = tx
                }
            }
        }
    }

    private func errorBanner(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(message)
            Spacer()
            Button("Retry") { reload() }
        }
        .font(.callout).padding(12)
        .background(.red.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(.red)
        .padding(.horizontal, 20).padding(.top, 12)
    }

    private var accountSignature: String {
        appState.selectedAccounts.sorted().joined(separator: ",")
    }

    private var deleteAlertBinding: Binding<Bool> {
        Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })
    }

    private func reload() { viewModel.reload(accounts: appState.accountsQuery) }

    private func handleSave(_ tx: Transaction) async -> Bool {
        let ok = await viewModel.save(tx)
        if ok { reload() }
        return ok
    }
}
