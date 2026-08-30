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
    case buffettRank = "Rankings"
    case strategies = "Strategies"
    case watchlist = "Watchlist"
    case markets = "Markets"
    case aiReview = "AI Insights"
    // Settings
    case settings = "Settings"

    var id: String { rawValue }

    static let group1: [AppSection] = [.performance, .allocation, .assetChange, .transactions, .dividend, .capitalGains]
    static let group2: [AppSection] = [.market, .buffettRank, .strategies, .watchlist, .markets, .aiReview]

    var icon: String {
        switch self {
        case .performance: return "square.grid.2x2"
        case .allocation: return "chart.pie"
        case .assetChange: return "chart.line.uptrend.xyaxis"
        case .transactions: return "arrow.left.arrow.right"
        case .dividend: return "dollarsign"
        case .capitalGains: return "chart.bar"
        case .market: return "magnifyingglass"
        case .buffettRank: return "trophy"
        case .strategies: return "square.stack.3d.up"
        case .watchlist: return "star"
        case .markets: return "globe"
        case .aiReview: return "sparkles"
        case .settings: return "gearshape"
        }
    }
}

/// Sidebar-based shell hosting the feature tabs.
struct MainView: View {
    @EnvironmentObject private var auth: AuthViewModel
    @EnvironmentObject private var appState: AppState
    @State private var selection: AppSection = .performance
    @State private var visitedSections: Set<AppSection> = [.performance]
    @State private var showingPalette = false
    /// The appearance preference set in Settings ▸ Appearance
    /// (`AppearanceSettingsView`): until the user picks a side,
    /// `appearanceSet` is false and the app follows the system.
    @AppStorage("investa.forceDark") private var forceDark = false
    @AppStorage("investa.appearanceSet") private var appearanceSet = false
    @Environment(\.scenePhase) private var scenePhase
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    @Environment(\.verticalSizeClass) private var vSize
    #endif

    var body: some View {
        shell
            .overlay {
                // Draggable floating bubble; it manages its own position and
                // edge clearance, so no alignment/padding is needed here.
                AIChatLauncher()
            }
            .preferredColorScheme(appearanceSet ? (forceDark ? .dark : .light) : nil)
            .sheet(isPresented: $showingPalette) {
                CommandPaletteView(
                    onNavigate: { selection = $0; visitedSections.insert($0); appState.clearStock() },
                    onOpenSettings: { selection = .settings; visitedSections.insert(.settings); appState.clearStock() },
                    onOpenStock: { appState.openStock($0) })
            }
            .onReceive(NotificationCenter.default.publisher(for: .commandPalette)) { _ in showingPalette = true }
            .onReceive(NotificationCenter.default.publisher(for: .navigateToSection)) { note in
                if let section = note.object as? AppSection {
                    appState.clearStock()
                    selection = section
                    visitedSections.insert(section)
                }
            }
            .onReceive(NotificationCenter.default.publisher(for: .openSettings)) { _ in
                appState.clearStock()
                selection = .settings
                visitedSections.insert(.settings)
            }
            .onReceive(NotificationCenter.default.publisher(for: .toggleDarkMode)) { _ in
                appearanceSet = true; forceDark.toggle()
            }
            // Poll for fresh prices only while the app is foregrounded (the poll
            // itself no-ops unless the market is open).
            .onChange(of: scenePhase, initial: true) { _, phase in
                appState.setAutoRefresh(phase == .active)
            }
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
                if let selectedStock = appState.selectedStock {
                    StockDetailView(symbol: selectedStock, currency: appState.displayCurrency)
                        .id(selectedStock)
                } else {
                    ZStack(alignment: .top) {
                        ForEach(AppSection.allCases) { section in
                            if visitedSections.contains(section) {
                                sectionView(section)
                                    .opacity(selection == section ? 1 : 0)
                                    .allowsHitTesting(selection == section)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                }
            }
            .navigationTitle(appState.selectedStock ?? "Investa")
            .task { if !appState.didLoadSettings { await appState.loadSettings() } }
            .onChange(of: selection) { _, newSelection in
                appState.clearStock()
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
            ToolbarItem(placement: .navigation) {
                Image("AppLogoNoText")
                    .resizable()
                    .scaledToFill()
                    .frame(width: 28, height: 28)
                    .clipShape(Circle())
            }
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
        case .buffettRank: BuffettRankView()
        case .strategies: StrategiesView()
        case .watchlist: WatchlistView()
        case .markets: MarketsView()
        case .aiReview: AIView()
        case .settings: SettingsView()
        }
    }

    private var footer: some View {
        VStack(spacing: 0) {
            Divider()
            VStack(alignment: .leading, spacing: 2) {
                footerNavButton(.settings)
                userRow
            }
            .padding(.vertical, 6)
        }
    }

    /// Who is signed in, with sign-out beside it — the web sidebar's own footer
    /// anatomy. It used to be a popover menu that also repeated Refresh; the
    /// control bar above every tab already carries refresh (and ⌘R), so the
    /// footer states the account rather than hiding it behind a click.
    private var userRow: some View {
        HStack(spacing: 0) {
            Label(auth.currentUser?.displayName ?? "Account", systemImage: "person.crop.circle")
                .lineLimit(1)
                .minimumScaleFactor(0.85)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button {
                auth.logout()
            } label: {
                Image(systemName: "rectangle.portrait.and.arrow.right")
                    .foregroundStyle(Color.down)
                    .padding(4)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Sign out")
            .accessibilityLabel("Sign out")
        }
        .padding(.horizontal, 8).padding(.vertical, 6)
    }

    private func footerNavButton(_ section: AppSection) -> some View {
        let isSelected = selection == section && appState.selectedStock == nil
        return Button {
            appState.clearStock()
            selection = section
            visitedSections.insert(section)
        } label: {
            Label(section.rawValue, systemImage: section.icon)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(isSelected ? Color.accentColor.opacity(0.15) : Color.clear, in: RoundedRectangle(cornerRadius: 6))
                .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
        }
        .buttonStyle(.plain)
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
        .onChange(of: selection) { _, _ in
            appState.clearStock()
        }
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
                // Settings sits at the right edge of the bar as itself, where a
                // "•••" menu used to hold it alongside sign-out. Settings is
                // the one of the two that gets used; signing out stays a tap
                // further in, on the Settings hub's Profile & Security row.
                GlobalControlBar(section: section) {
                    Button {
                        appState.clearStock()
                        selection = .settings
                        visitedSections.insert(.settings)
                    } label: {
                        Image(systemName: "gearshape")
                            .appFont(.body)
                            .frame(width: 32, height: 32)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.primary)
                    .accessibilityLabel("Settings")
                }
                Divider()
                // Pinned to the shell's width, so a section that momentarily
                // overflows can't widen the VStack around it — a wider VStack
                // centers the control bar over the section's width and slides
                // it off the right edge of the screen.
                if let selectedStock = appState.selectedStock {
                    StockDetailView(symbol: selectedStock, currency: appState.displayCurrency)
                        .id(selectedStock)
                        .frame(width: geo.size.width, alignment: .topLeading)
                } else {
                    sectionView(section)
                        .frame(width: geo.size.width, alignment: .topLeading)
                }
            }

            // Pinned to the container rather than `maxWidth: .infinity`, which
            // grows to fit an oversized child instead of clamping it: one card
            // that overflows would otherwise widen the shell and push the
            // control bar off the right edge of the screen.
            .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 6) {
                        appIcon
                            .resizable()
                            .scaledToFit()
                            .frame(height: 34)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        
                        if geo.size.width > 450 {
                            Text("Investa")
                                .appFont(.title3).bold()
                                .foregroundColor(.primary)
                        }
                    }
                    .fixedSize(horizontal: true, vertical: false)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    if !appState.indices.isEmpty {
                        IndexStrip(indices: appState.indices)
                    }
                }
            }
        }
    }

    private var appIcon: Image {
        if let icons = Bundle.main.infoDictionary?["CFBundleIcons"] as? [String: Any],
           let primary = icons["CFBundlePrimaryIcon"] as? [String: Any],
           let files = primary["CFBundleIconFiles"] as? [String],
           let name = files.last,
           let uiImage = UIImage(named: name) {
            return Image(uiImage: uiImage)
        }
        return Image("AppLogoNoText")
    }

    #endif
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
                        .appFont(.caption.weight(.bold))
                        .foregroundStyle(.primary)
                    
                    Text(Fmt.number(index.price))
                        .appFont(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .padding(.leading, 2)
                        
                    if let change = index.change {
                        Text("\(isUp ? "+" : "")\(Fmt.number(change))")
                            .appFont(.caption.monospacedDigit())
                            .foregroundStyle(isUp ? Color.green : Color.red)
                            .padding(.leading, 2)
                    }
                    
                    HStack(spacing: 0) {
                        Text("(")
                            .appFont(.caption.monospacedDigit())
                        Image(systemName: isUp ? "arrowtriangle.up.fill" : "arrowtriangle.down.fill")
                            .appFont(.system(size: 9))
                        Text(String(format: "%.2f%%)", abs(index.changesPercentage ?? 0)))
                            .appFont(.caption.monospacedDigit())
                    }
                    .foregroundStyle(isUp ? Color.green : Color.red)
                    .padding(.leading, 2)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .lineLimit(1)
        // Render at intrinsic width when chosen so the toolbar doesn't truncate
        // the last index with an ellipsis. ViewThatFits still falls back to
        // shortStrip when this doesn't fit the available width.
        .fixedSize(horizontal: true, vertical: false)
    }

    private var shortStrip: some View {
        HStack(spacing: 12) {
            ForEach(indices) { index in
                let isUp = (index.change ?? 0) >= 0
                HStack(spacing: 2) {
                    Text(shortName(index.name))
                        .appFont(.caption.weight(.bold))
                        .foregroundStyle(.primary)
                    
                    HStack(spacing: 0) {
                        Image(systemName: isUp ? "arrowtriangle.up.fill" : "arrowtriangle.down.fill")
                            .appFont(.system(size: 9))
                        Text(String(format: "%.2f%%", abs(index.changesPercentage ?? 0)))
                            .appFont(.caption.monospacedDigit())
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
