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
    @State private var visitedSections: Set<AppSection> = [.performance]
    @State private var showingSettings = false
    @State private var showingPalette = false
    @State private var paletteStock: SymbolID?
    /// nil = follow system; true/false = forced.
    @AppStorage("investa.forceDark") private var forceDark = false
    @AppStorage("investa.appearanceSet") private var appearanceSet = false
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    @Environment(\.verticalSizeClass) private var vSize
    #endif

    var body: some View {
        shell
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

    @ViewBuilder private var shell: some View {
        #if os(iOS)
        if hSize == .compact { phoneShell } else { splitShell }
        #else
        splitShell
        #endif
    }

    /// Sidebar shell — macOS and iPad (regular width).
    private var splitShell: some View {
        NavigationSplitView {
            // iOS requires an optional single-selection binding; bridge to the
            // non-optional state (ignore deselection).
            List(selection: Binding(get: { selection }, set: { selection = $0 ?? selection })) {
                Section { ForEach(AppSection.group1) { row($0) } }
                Section { ForEach(AppSection.group2) { row($0) } }
            }
            .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 260)
            .safeAreaInset(edge: .bottom) { footer }
        } detail: {
            VStack(spacing: 0) {
                GlobalControlBar(section: selection)
                Divider()
                ZStack {
                    ForEach(AppSection.allCases) { section in
                        if visitedSections.contains(section) {
                            sectionView(section)
                                .opacity(selection == section ? 1 : 0)
                                .allowsHitTesting(selection == section)
                        }
                    }
                }
            }
            .task { if !appState.didLoadSettings { await appState.loadSettings() } }
            .onChange(of: selection) { _, newSelection in
                visitedSections.insert(newSelection)
            }
            #if os(iOS)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if !appState.indices.isEmpty {
                        IndexStrip(indices: appState.indices)
                    }
                }
            }
            #endif
        }
        #if os(macOS)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                if !appState.indices.isEmpty {
                    IndexStrip(indices: appState.indices)
                }
            }
        }
        #endif
    }

    private func row(_ section: AppSection) -> some View {
        Label(section.rawValue, systemImage: section.icon).tag(section)
    }

    @ViewBuilder private func sectionView(_ section: AppSection) -> some View {
        switch section {
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

    #if os(iOS)
    // MARK: - iPhone shell (TabView; iOS auto-adds a "More" tab beyond 5 items)

    private var phoneShell: some View {
        TabView(selection: $selection) {
            ForEach(AppSection.allCases) { section in
                phoneTab(section)
                    .tabItem { Label(section.rawValue, systemImage: section.icon) }
                    .tag(section)
            }
        }
        .task { if !appState.didLoadSettings { await appState.loadSettings() } }
    }

    @ViewBuilder
    private func phoneTab(_ section: AppSection) -> some View {
        let isMainTab = Array(AppSection.allCases.prefix(4)).contains(section)
        if isMainTab {
            NavigationStack {
                phoneTabContent(section)
            }
        } else {
            phoneTabContent(section)
        }
    }

    private func phoneTabContent(_ section: AppSection) -> some View {
        GeometryReader { geo in
            VStack(spacing: 0) {
                GlobalControlBar(section: section) {
                    HStack(spacing: 16) {
                        Button { NotificationCenter.default.post(name: .refreshRequested, object: nil) } label: { Image(systemName: "arrow.clockwise") }
                        accountToolbarMenu
                    }
                    .font(.system(size: 20))
                }
                Divider()
                sectionView(section)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 6) {
                        Image("AppLogo")
                            .resizable()
                            .scaledToFit()
                            .frame(height: 24)
                        
                        if geo.size.width > 450 {
                            Text("Investa")
                                .font(.title3).bold()
                                .foregroundColor(.primary)
                        }
                    }
                    .fixedSize(horizontal: true, vertical: false)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    if !appState.indices.isEmpty {
                        IndexStrip(indices: appState.indices)
                    }
                }
            }
        }
    }

    private var accountToolbarMenu: some View {
        Menu {
            if let user = auth.currentUser { Text("Signed in as \(user.displayName)"); Divider() }
            Button { showingSettings = true } label: { Label("Settings", systemImage: "gearshape") }
            Button { appearanceSet = true; forceDark.toggle() } label: {
                Label(forceDark ? "Light mode" : "Dark mode", systemImage: forceDark ? "sun.max" : "moon")
            }
            Divider()
            Button(role: .destructive) { auth.logout() } label: { Label("Log Out", systemImage: "rectangle.portrait.and.arrow.right") }
        } label: { Image(systemName: "person.crop.circle") }
    }
    #endif
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
        #if os(macOS)
        .frame(width: 720, height: 640)
        #endif
    }
}

/// A horizontally scrolling strip showing market indices, typically placed in the app title bar.
struct IndexStrip: View {
    let indices: [IndexQuote]

    private func shortName(_ name: String?) -> String {
        guard let name = name else { return "IDX" }
        let upper = name.uppercased()
        if upper.contains("DOW") { return "DOW" }
        if upper.contains("S&P") { return "S&P" }
        if upper.contains("NASDAQ") || upper.contains("NAS") { return "NAS" }
        if upper.contains("RUSSELL") { return "RUT" }
        return String(upper.prefix(3))
    }

    var body: some View {
        ViewThatFits(in: .horizontal) {
            fullStrip
            shortStrip
        }
    }

    private var fullStrip: some View {
        HStack(spacing: 12) {
            ForEach(indices) { index in
                let isUp = (index.change ?? 0) >= 0
                HStack(spacing: 2) {
                    Text(shortName(index.name))
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.primary)
                    
                    Text(Fmt.number(index.price))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .padding(.leading, 2)
                        
                    if let change = index.change {
                        Text("\(isUp ? "+" : "")\(Fmt.number(change))")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(isUp ? Color.green : Color.red)
                            .padding(.leading, 2)
                    }
                    
                    HStack(spacing: 0) {
                        Text("(")
                            .font(.caption.monospacedDigit())
                        Image(systemName: isUp ? "arrowtriangle.up.fill" : "arrowtriangle.down.fill")
                            .font(.system(size: 8))
                        Text(String(format: "%.2f%%)", abs(index.changesPercentage ?? 0)))
                            .font(.caption.monospacedDigit())
                    }
                    .foregroundStyle(isUp ? Color.green : Color.red)
                    .padding(.leading, 2)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .lineLimit(1)
        .minimumScaleFactor(0.8)
    }

    private var shortStrip: some View {
        HStack(spacing: 12) {
            ForEach(indices) { index in
                let isUp = (index.change ?? 0) >= 0
                HStack(spacing: 2) {
                    Text(shortName(index.name))
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.primary)
                    
                    HStack(spacing: 0) {
                        Image(systemName: isUp ? "arrowtriangle.up.fill" : "arrowtriangle.down.fill")
                            .font(.system(size: 8))
                        Text(String(format: "%.2f%%", abs(index.changesPercentage ?? 0)))
                            .font(.caption.monospacedDigit())
                    }
                    .foregroundStyle(isUp ? Color.green : Color.red)
                    .padding(.leading, 2)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .lineLimit(1)
        .minimumScaleFactor(0.8)
    }
}
