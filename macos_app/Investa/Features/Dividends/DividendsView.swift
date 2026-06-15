import SwiftUI

@MainActor
final class DividendsViewModel: ObservableObject {
    @Published var dividends: [Dividend] = []
    @Published var projected: [ProjectedIncome] = []
    @Published var calendar: [DividendEvent] = []
    @Published var metrics: Metrics?
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: APIClient
    private var task: Task<Void, Never>?

    init(api: APIClient = .shared) { self.api = api }

    func reload(currency: String, accounts: [String]?) {
        task?.cancel()
        task = Task { [weak self] in await self?.load(currency: currency, accounts: accounts) }
    }

    private func load(currency: String, accounts: [String]?) async {
        isLoading = true; errorMessage = nil
        defer { isLoading = false }
        let q = [URLQueryItem(name: "currency", value: currency)] + APIClient.arrayQuery("accounts", accounts)
        async let divR: [Dividend] = api.get("/dividends", query: q)
        async let projR: [ProjectedIncome] = api.get("/projected_income", query: q)
        async let calR: [DividendEvent] = api.get("/dividend_calendar", query: q)
        async let summaryR: SummaryResponse = api.get("/summary", query: q)
        do {
            dividends = try await divR
        } catch is CancellationError { return }
        catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
        projected = (try? await projR) ?? []
        calendar = (try? await calR) ?? []
        metrics = (try? await summaryR)?.metrics
    }
}

/// Income tab — mirrors the web "Dividend" tab (Dividend.tsx, layout group "Income Sections").
struct DividendsView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = DividendsViewModel()
    @State private var detail: SymbolID?

    private var cur: String { appState.displayCurrency }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Income").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            ScrollView {
                VStack(spacing: 20) {
                    if vis("incomeKpis") {
                        IncomeKpiStrip(dividends: viewModel.dividends, currency: cur,
                                       expectedDividends: viewModel.metrics?.estAnnualIncomeDisplay,
                                       dividendYield: viewModel.metrics?.dividendYieldPct)
                    }
                    if vis("incomeProjector") { IncomeProjectorCard(income: viewModel.projected, currency: cur) }
                    if vis("dividendCalendar") {
                        DividendCalendarSection(events: viewModel.calendar, currency: cur, onSelect: { detail = SymbolID(id: $0) })
                    }
                    payersRow
                    if vis("annualDividends") { AnnualDividendsCard(dividends: viewModel.dividends, currency: cur) }
                    if vis("dividendTransactions") { DividendTransactionsCard(dividends: viewModel.dividends, currency: cur) }
                }
                .padding(20)
            }
        }
        .frame(minWidth: 820, minHeight: 560)
        .task(id: signature) { reload() }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in reload() }
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: cur) }
    }

    private func vis(_ id: String) -> Bool { appState.isVisible(.dividend, id) }

    @ViewBuilder private var payersRow: some View {
        let payers = vis("topPayers"); let byAcct = vis("byAccount")
        if payers && byAcct {
            twoColumn(TopPayersCard(dividends: viewModel.dividends, currency: cur, onSelect: { detail = SymbolID(id: $0) }),
                      ByAccountCard(dividends: viewModel.dividends, currency: cur))
        } else if payers {
            TopPayersCard(dividends: viewModel.dividends, currency: cur, onSelect: { detail = SymbolID(id: $0) })
        } else if byAcct {
            ByAccountCard(dividends: viewModel.dividends, currency: cur)
        }
    }

    @ViewBuilder private func twoColumn<L: View, R: View>(_ left: L, _ right: R) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 20) { left; right }
            VStack(spacing: 20) { left; right }
        }
    }

    private var signature: String {
        "\(cur)|\(appState.selectedAccounts.sorted().joined(separator: ","))"
    }
    private func reload() {
        viewModel.reload(currency: cur, accounts: appState.accountsQuery)
    }
}
