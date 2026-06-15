import SwiftUI

/// Sidebar sections — order, labels, icons, and grouping match the web app's
/// left sidebar exactly (see screenshot): two groups separated by a divider.
enum AppSection: String, CaseIterable, Identifiable {
    // Group 1
    case performance = "Dashboard"
    case allocation = "Portfolio"
    case assetChange = "Performance"
    case transactions = "Transactions"
    case dividend = "Income"
    case capitalGains = "Capital Gains"
    // Group 2
    case market = "Screener"
    case watchlist = "Watchlist"
    case markets = "Markets"
    case aiReview = "AI Insights"

    var id: String { rawValue }

    static let group1: [AppSection] = [.performance, .allocation, .assetChange, .transactions, .dividend, .capitalGains]
    static let group2: [AppSection] = [.market, .watchlist, .markets, .aiReview]

    var icon: String {
        switch self {
        case .performance: return "square.grid.2x2"
        case .allocation: return "chart.pie"
        case .assetChange: return "chart.line.uptrend.xyaxis"
        case .transactions: return "arrow.left.arrow.right"
        case .dividend: return "dollarsign"
        case .capitalGains: return "chart.bar"
        case .market: return "magnifyingglass"
        case .watchlist: return "star"
        case .markets: return "globe"
        case .aiReview: return "sparkles"
        }
    }
}

/// Sidebar-based shell hosting the feature tabs.
struct MainView: View {
    @EnvironmentObject private var auth: AuthViewModel
    @EnvironmentObject private var appState: AppState
    @State private var selection: AppSection = .performance
    @State private var showingSettings = false
    @State private var showingPalette = false
    @State private var paletteStock: SymbolID?
    /// nil = follow system; true/false = forced.
    @AppStorage("investa.forceDark") private var forceDark = false
    @AppStorage("investa.appearanceSet") private var appearanceSet = false

    var body: some View {
        NavigationSplitView {
            // iOS requires an optional single-selection binding; bridge to the
            // non-optional state (ignore deselection).
            List(selection: Binding(get: { selection }, set: { selection = $0 ?? selection })) {
                Section {
                    ForEach(AppSection.group1) { row($0) }
                }
                Section {
                    ForEach(AppSection.group2) { row($0) }
                }
            }
            .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 260)
            .safeAreaInset(edge: .bottom) { footer }
        } detail: {
            VStack(spacing: 0) {
                GlobalControlBar(section: selection)
                Divider()
                detail
            }
            .task { if !appState.didLoadSettings { await appState.loadSettings() } }
        }
        .preferredColorScheme(appearanceSet ? (forceDark ? .dark : .light) : nil)
        .sheet(isPresented: $showingSettings) {
            SettingsSheet().environmentObject(appState).environmentObject(auth)
        }
        .sheet(isPresented: $showingPalette) {
            CommandPaletteView(
                onNavigate: { selection = $0 },
                onOpenSettings: { showingSettings = true },
                onOpenStock: { paletteStock = SymbolID(id: $0) })
        }
        .sheet(item: $paletteStock) { StockDetailView(symbol: $0.id, currency: appState.displayCurrency) }
        .onReceive(NotificationCenter.default.publisher(for: .commandPalette)) { _ in showingPalette = true }
    }

    private func row(_ section: AppSection) -> some View {
        Label(section.rawValue, systemImage: section.icon).tag(section)
    }

    @ViewBuilder private var detail: some View {
        switch selection {
        case .performance: DashboardView()
        case .allocation: AllocationView()
        case .assetChange: AssetChangeView()
        case .transactions: TransactionsView()
        case .dividend: DividendsView()
        case .capitalGains: CapitalGainsView()
        case .market: ScreenerView()
        case .watchlist: WatchlistView()
        case .markets: MarketsView()
        case .aiReview: AIView()
        }
    }

    private var footer: some View {
        VStack(spacing: 0) {
            Divider()
            VStack(alignment: .leading, spacing: 2) {
                footerButton("Settings", "gearshape") { showingSettings = true }
                footerButton("Dark mode", forceDark ? "sun.max" : "moon") {
                    appearanceSet = true
                    forceDark.toggle()
                }
                Menu {
                    if let user = auth.currentUser { Text("Signed in as \(user.displayName)"); Divider() }
                    Button("Refresh") { NotificationCenter.default.post(name: .refreshRequested, object: nil) }
                    Button("Log Out") { auth.logout() }
                } label: {
                    Label(auth.currentUser?.displayName ?? "Account", systemImage: "person.crop.circle")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .borderlessMenu()
                .padding(.horizontal, 8).padding(.vertical, 6)
            }
            .padding(.vertical, 6)
        }
    }

    private func footerButton(_ title: String, _ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: icon).frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 8).padding(.vertical, 6)
    }
}

/// Wraps SettingsView in a dismissible sheet container.
private struct SettingsSheet: View {
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(spacing: 0) {
            HStack { Spacer(); Button("Done") { dismiss() }.keyboardShortcut(.defaultAction) }
                .padding(12)
            SettingsView()
        }
        .frame(width: 720, height: 640)
    }
}
