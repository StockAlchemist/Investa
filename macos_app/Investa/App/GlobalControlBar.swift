import SwiftUI

/// Persistent header controls shown on every tab (mirrors the web PageHeader):
/// account selector (+ closed indicator), currency, show-closed toggle, and a
/// per-tab Layout configurator menu.
struct GlobalControlBar<Trailing: View>: View {
    @EnvironmentObject private var appState: AppState
    let section: AppSection
    let trailing: Trailing

    init(section: AppSection, @ViewBuilder trailing: () -> Trailing = { EmptyView() }) {
        self.section = section
        self.trailing = trailing()
    }
    #if os(iOS)
    @Environment(\.horizontalSizeClass) private var hSize
    #endif

    var body: some View {
        #if os(iOS)
        compactBar
            .zIndex(1) // keep the search dropdown above the tab content below
        #else
        regularBar
            .zIndex(1)
        #endif
    }

    private var regularBar: some View {
        HStack(spacing: 12) {
            accountMenu
            if TabLayout.hasLayout(section) { layoutMenu }
            benchmarkButton
            showClosedToggle
            Spacer()
            marketStatusBadge
            lastUpdatedLabel
            StockSearchBar(currency: appState.displayCurrency)
            currencyMenu
            trailing
        }
        .padding(.horizontal, 20).padding(.vertical, 8)
        .liquidGlass()
    }

    /// "Live" / "Closed" market-status pill (mirrors the web header badge).
    /// Shown only in the regular (macOS/iPad) bar, matching the web's hide-on-small.
    @ViewBuilder private var marketStatusBadge: some View {
        if let open = appState.marketIsOpen {
            HStack(spacing: 5) {
                Image(systemName: "circle.fill")
                    .font(.system(size: 6))
                    .symbolEffect(.pulse, options: .repeating, isActive: open)
                Text(open ? "LIVE" : "CLOSED")
                    .font(.system(size: 10, weight: .bold)).tracking(0.5)
            }
            .foregroundStyle(open ? Color.green : .secondary)
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background((open ? Color.green : Color.secondary).opacity(0.12), in: Capsule())
        }
    }

    /// Last time the market data was refreshed (mirrors the web header time).
    @ViewBuilder private var lastUpdatedLabel: some View {
        if let ts = appState.lastUpdated {
            Text(ts.formatted(date: .omitted, time: .shortened))
                .font(.system(size: 10, weight: .medium)).monospacedDigit()
                .foregroundStyle(.secondary)
        }
    }

    /// Space-saving market status for the compact (iPhone/iPad) bar: a colored
    /// dot (green pulse when open, muted when closed) + the last-updated time.
    /// The dot color conveys open/closed without the "LIVE/CLOSED" word.
    @ViewBuilder private var marketStatusCompact: some View {
        if let open = appState.marketIsOpen {
            HStack(spacing: 4) {
                Image(systemName: "circle.fill")
                    .font(.system(size: 6))
                    .symbolEffect(.pulse, options: .repeating, isActive: open)
                    .foregroundStyle(open ? Color.green : .secondary)
                if let ts = appState.lastUpdated {
                    Text(ts.formatted(date: .omitted, time: .shortened))
                        .font(.system(size: 10, weight: .medium)).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
            .accessibilityLabel(open ? "Market open" : "Market closed")
            .fixedSize()
        }
    }

    private var compactBar: some View {
        HStack(spacing: 6) {
            // Primary scope controls — kept fixed/leftmost so they're always visible.
            accountMenu
                .labelStyle(.iconOnly)
                .font(.body)
                .padding(.leading, 12)
            currencyMenu
            // Secondary config (show-closed, benchmarks, layout) scrolls if tight.
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    showClosedToggle
                    benchmarkButton
                    if TabLayout.hasLayout(section) { layoutMenu }
                }
                .labelStyle(.iconOnly)
                .font(.body) // compact glyphs so the dense control row isn't cramped
                .padding(.vertical, 8)
            }
            // Take only the leftover width and scroll internally — otherwise the
            // ScrollView claims its full content width and overlaps the search box.
            .frame(maxWidth: .infinity)
            StockSearchBar(currency: appState.displayCurrency)
                .layoutPriority(1)
            marketStatusCompact
            trailing
                .padding(.trailing, 12)
        }
        .liquidGlass()
    }

    // MARK: - Benchmarks

    @State private var showBenchmarks = false
    @State private var customBenchmark = ""
    private let presetBenchmarks = [
        "S&P 500", "Dow Jones", "NASDAQ", "Russell 2000",
        "SPY (S&P 500 ETF)", "QQQ (Nasdaq 100 ETF)", "DIA (Dow Jones ETF)", "S&P 500 Total Return",
    ]

    private var benchmarkButton: some View {
        Button { showBenchmarks.toggle() } label: {
            Label("Benchmarks (\(appState.benchmarks.count))", systemImage: "chart.xyaxis.line")
        }
        .buttonStyle(.borderless)
        .interactiveGlass()
        .popover(isPresented: $showBenchmarks, arrowEdge: .bottom) { benchmarkPopover }
    }

    private var benchmarkPopover: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Benchmarks").font(.headline)
            ForEach(presetBenchmarks, id: \.self) { b in
                let on = appState.benchmarks.contains(b)
                Button { toggleBenchmark(b) } label: {
                    HStack {
                        Image(systemName: on ? "checkmark.square.fill" : "square").foregroundStyle(on ? Color.accentColor : .secondary)
                        Text(b); Spacer()
                    }
                }.buttonStyle(.plain)
            }
            // Custom tickers currently selected.
            let custom = appState.benchmarks.filter { !presetBenchmarks.contains($0) }
            if !custom.isEmpty {
                Divider()
                ForEach(custom, id: \.self) { c in
                    HStack { Text(c).fontWeight(.medium); Spacer()
                        Button { toggleBenchmark(c) } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).foregroundStyle(.secondary)
                    }
                }
            }
            Divider()
            HStack {
                TextField("Custom ticker", text: $customBenchmark).textFieldStyle(.roundedBorder).frame(width: 120)
                Button("Add") {
                    let t = customBenchmark.trimmingCharacters(in: .whitespaces).uppercased()
                    guard !t.isEmpty, !appState.benchmarks.contains(t) else { return }
                    customBenchmark = ""; appState.setBenchmarks(appState.benchmarks + [t])
                }.buttonStyle(.borderedProminent)
            }
        }
        .padding(14)
        #if os(iOS)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        #else
        .frame(width: 240)
        #endif
    }

    private func toggleBenchmark(_ b: String) {
        if appState.benchmarks.contains(b) { appState.setBenchmarks(appState.benchmarks.filter { $0 != b }) }
        else { appState.setBenchmarks(appState.benchmarks + [b]) }
    }

    // MARK: - Accounts

    private var accountSummary: String {
        if appState.selectedAccounts.isEmpty { return "All Accounts" }
        if appState.selectedAccounts.count == 1 { return appState.selectedAccounts.first! }
        return "\(appState.selectedAccounts.count) Accounts"
    }

    private var orderedGroups: [(name: String, accounts: [String])] {
        let g = appState.accountGroups
        let order = appState.accountGroupOrder.isEmpty ? Array(g.keys).sorted() : appState.accountGroupOrder
        return order.compactMap { name in g[name].map { (name, $0) } }
    }

    @State private var showAccounts = false

    private var accountMenu: some View {
        Button { showAccounts.toggle() } label: {
            Label(accountSummary, systemImage: "building.columns")
        }
        .buttonStyle(.borderless)
        .interactiveGlass()
        .popover(isPresented: $showAccounts, arrowEdge: .bottom) { accountPopover }
    }

    private var accountPopover: some View {
        let individuals = appState.allAccounts.filter { $0 != "All Accounts" }
        return ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                AccountMenuRow(title: "All Accounts", checked: appState.selectedAccounts.isEmpty) {
                    appState.selectedAccounts = []
                }
                if !orderedGroups.isEmpty {
                    accountSectionHeader("Groups")
                    ForEach(orderedGroups, id: \.name) { group in
                        let selected = !appState.selectedAccounts.isEmpty && appState.selectedAccounts == Set(group.accounts)
                        AccountMenuRow(title: group.name, checked: selected) {
                            appState.selectedAccounts = Set(group.accounts); showAccounts = false
                        }
                    }
                    accountSectionHeader("Individual")
                }
                ForEach(individuals, id: \.self) { account in
                    AccountMenuRow(title: account,
                                   checked: appState.selectedAccounts.contains(account),
                                   closed: appState.closedAccounts.contains(account)) {
                        toggle(account)
                    }
                }
            }
            .padding(.vertical, 4)
        }
        #if os(iOS)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        #else
        .frame(minWidth: 220, maxHeight: 440)
        #endif
    }

    private func accountSectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            .padding(.horizontal, 12).padding(.top, 8).padding(.bottom, 2)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func toggle(_ account: String) {
        if appState.selectedAccounts.contains(account) { appState.selectedAccounts.remove(account) }
        else { appState.selectedAccounts.insert(account) }
    }

    // MARK: - Currency / show-closed

    private var currencyMenu: some View {
        HStack(spacing: 8) {
            if appState.displayCurrency != "USD", let rate = appState.currentFXRateToUSD {
                Text("1 USD = \(String(format: "%.2f", rate)) \(appState.displayCurrency)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Menu {
                ForEach(appState.availableCurrencies, id: \.self) { cur in
                    Button(cur) { appState.displayCurrency = cur }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(appState.displayCurrency)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 14))
                }
            }
            .borderlessMenu().fixedSize()
            .interactiveGlass()
        }
        .onChange(of: appState.displayCurrency) {
            Task { await appState.fetchFXRate() }
        }
    }

    private var showClosedToggle: some View {
        Toggle(isOn: $appState.showClosed) {
            Label("Show Closed", systemImage: appState.showClosed ? "eye" : "eye.slash")
        }
        .toggleStyle(.button).controlSize(.small)
        .interactiveGlass()
    }

    // MARK: - Layout configurator

    private var layoutMenu: some View {
        Menu {
            Text(TabLayout.sectionTitle(for: section))
            Divider()
            let items = TabLayout.items(for: section)
            // Render grouped (group header as a disabled label) preserving order.
            ForEach(Array(groupedItems(items).enumerated()), id: \.offset) { _, group in
                if let label = group.label { Section(label) { itemButtons(group.items) } }
                else { itemButtons(group.items) }
            }
        } label: {
            Label("Layout", systemImage: "slider.horizontal.3")
        }
        .borderlessMenu().fixedSize()
        .interactiveGlass()
    }

    private func itemButtons(_ items: [LayoutItem]) -> some View {
        ForEach(items) { item in
            Button { appState.toggle(section, item.id) } label: {
                Label(item.title, systemImage: appState.isVisible(section, item.id) ? "checkmark" : "")
            }
        }
    }

    private func groupedItems(_ items: [LayoutItem]) -> [(label: String?, items: [LayoutItem])] {
        var groups: [(label: String?, items: [LayoutItem])] = []
        var indexByLabel: [String: Int] = [:]
        for item in items {
            let label = item.group
            let key = label ?? "__none"
            if let idx = indexByLabel[key] { groups[idx].items.append(item) }
            else { indexByLabel[key] = groups.count; groups.append((label, [item])) }
        }
        return groups
    }
}

/// A plain menu-style account row that can show a right-adjusted "Closed" tag
/// (a native `Menu` can't render trailing accessories, so the dropdown is drawn
/// as a lightweight popover instead).
private struct AccountMenuRow: View {
    let title: String
    let checked: Bool
    var closed: Bool = false
    let action: () -> Void
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "checkmark").font(.caption.weight(.bold)).opacity(checked ? 1 : 0)
                Text(title).lineLimit(1)
                Spacer(minLength: 16)
                if closed {
                    Text("Closed").font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                }
            }
            .font(.system(size: 13))
            .padding(.horizontal, 12).padding(.vertical, 5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .background(hovered ? Color.accentColor.opacity(0.15) : Color.clear)
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
    }
}
