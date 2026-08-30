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
            // Left cluster, in the web header's order (PageHeader.tsx):
            // tab name, portfolio KPI, market status + last-updated time.
            sectionTitle
            barDivider
            if appState.headlineMarketValue != nil {
                headlineKPI
                barDivider
            }
            HStack(spacing: 8) {
                marketStatusBadge
                lastUpdatedLabel
            }

            Spacer()

            // Right cluster, likewise: search, then the controls. The web puts
            // its indices ticker between the two; the native bar leaves it out
            // (the Markets tab is where indices live here). Refresh has no web
            // twin — the web polls — so it goes last, with the app chrome.
            StockSearchBar(currency: appState.displayCurrency)
            barDivider
            if TabLayout.hasLayout(section) { layoutMenu }
            currencyMenu
            showClosedToggle
            accountMenu
            refreshControl
            trailing
        }
        .padding(.horizontal, 20).padding(.vertical, 8)
        .liquidGlass()
    }

    /// The current tab's name at the leading edge of the bar — the twin of the
    /// web header's `navLabel(activeTab)` title (PageHeader.tsx). The tab is
    /// named here and nowhere else, so no section repeats it at the top of its
    /// own content. Hidden on iPhone, where the web hides it too and the tab
    /// bar already names the tab.
    /// The web hides its header title below `md`; the iPhone bar does the same
    /// (`hSize` only exists on iOS, so the check is compiled per-platform).
    private var showsSectionTitle: Bool {
        #if os(iOS)
        return hSize != .compact
        #else
        return true
        #endif
    }

    private var sectionTitle: some View {
        Text(section.rawValue)
            .appFont(.headline)
            .lineLimit(1)
            .minimumScaleFactor(0.85)
            // Priority, not `fixedSize`: the title keeps its room ahead of the
            // flexible controls without *demanding* width from the bar.
            .layoutPriority(1)
            .accessibilityAddTraits(.isHeader)
    }

    /// Hairline between the title and the controls, matching the web header's
    /// `w-px h-5` separators.
    private var barDivider: some View {
        Rectangle()
            .fill(Color.secondary.opacity(0.25))
            .frame(width: 1, height: 16)
    }

    /// Manual refresh (⌘R's twin), or the in-flight spinner while a reload runs.
    @ViewBuilder private var refreshControl: some View {
        if appState.isRefreshing {
            ProgressView().controlSize(.small).frame(width: 16, height: 16)
        } else {
            Button { NotificationCenter.default.post(name: .refreshRequested, object: nil) } label: {
                Image(systemName: "arrow.clockwise").appFont(.body)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.primary)
        }
    }

    /// Total portfolio value + day-change pill, beside the section title — the
    /// twin of the web header's mini KPI (PageHeader.tsx). It follows the same
    /// currency and account selection as the rest of the app, and the web's
    /// `hidden lg:flex`: shown wherever the section title is, so the crowded
    /// iPhone bar stays as it was.
    @ViewBuilder private var headlineKPI: some View {
        if let value = appState.headlineMarketValue {
            HStack(spacing: 6) {
                Text(Fmt.compact(value, code: appState.displayCurrency, forceDecimals: true))
                    .appFont(.system(size: 15, weight: .bold)).monospacedDigit()
                    .foregroundStyle(.primary)
                if let pct = appState.headlineDayChangePct {
                    let up = pct >= 0
                    HStack(spacing: 2) {
                        Image(systemName: up ? "arrow.up.right" : "arrow.down.right")
                            .appFont(.system(size: 9, weight: .bold))
                        Text(Fmt.percent(pct, includeSign: true))
                            .appFont(.system(size: 11, weight: .bold)).monospacedDigit()
                    }
                    .foregroundStyle(up ? Color.up : Color.down)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background((up ? Color.up : Color.down).opacity(0.12), in: Capsule())
                    .overlay(Capsule().strokeBorder((up ? Color.up : Color.down).opacity(0.25), lineWidth: 1))
                }
            }
            // A truncated figure is a different number, so the pair shrinks
            // together rather than ellipsising, and keeps its room ahead of the
            // flexible controls without demanding width (no `fixedSize`).
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            .layoutPriority(1)
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Portfolio value \(Fmt.currency(value, code: appState.displayCurrency))")
            .accessibilityValue(appState.headlineDayChangePct.map {
                "Day change \(Fmt.percent($0, includeSign: true))"
            } ?? "")
        }
    }

    /// "Live" / "Closed" market-status pill (mirrors the web header badge).
    /// Shown only in the regular (macOS/iPad) bar, matching the web's hide-on-small.
    @ViewBuilder private var marketStatusBadge: some View {
        if let open = appState.marketIsOpen {
            HStack(spacing: 5) {
                Image(systemName: "circle.fill")
                    .appFont(.system(size: 7))
                    .symbolEffect(.pulse, options: .repeating, isActive: open)
                Text(open ? "LIVE" : "CLOSED")
                    .appFont(.system(size: 11, weight: .bold)).tracking(0.5)
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
                .appFont(.system(size: 11, weight: .medium)).monospacedDigit()
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
                    .appFont(.system(size: 7))
                    .symbolEffect(.pulse, options: .repeating, isActive: open)
                    .foregroundStyle(open ? Color.green : .secondary)
                if let ts = appState.lastUpdated {
                    Text(ts.formatted(date: .omitted, time: .shortened))
                        .appFont(.system(size: 11, weight: .medium)).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
            .accessibilityLabel(open ? "Market open" : "Market closed")
            .fixedSize()
        }
    }

    /// Compact bar for iPhone (and iPad portrait). The previous design crammed
    /// nine controls + a horizontal scroll into one row; this keeps only the
    /// frequently-used controls inline (account, search, currency, market status)
    /// and folds the rest — Layout, Benchmarks, Show Closed, plus the host's
    /// refresh/settings/account actions — into a single overflow menu.
    /// While the search field is focused it takes over the whole bar (the other
    /// controls hide) — the standard iOS search pattern. This prevents the
    /// expanded field from shoving the currency menu off-screen and the glass
    /// container from ballooning when everything no longer fits in one row.
    @State private var searchActive = false

    private var compactBar: some View {
        HStack(spacing: 10) {
            if !searchActive {
                if showsSectionTitle {
                    sectionTitle.padding(.leading, 12)
                    barDivider
                    if appState.headlineMarketValue != nil {
                        headlineKPI
                        barDivider
                    }
                }
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        accountMenu
                            .labelStyle(.iconOnly)
                            .appFont(.body)
                            .padding(.leading, 12)
                        if TabLayout.hasLayout(section) {
                            PopoverMenu { layoutMenuContent } label: {
                                Image(systemName: "slider.horizontal.3").appFont(.body)
                            }
                        }
                        Button { appState.showClosed.toggle() } label: {
                            Image(systemName: appState.showClosed ? "eye" : "eye.slash").appFont(.body)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(appState.showClosed ? .primary : .secondary)
                    }
                }
                
                Spacer(minLength: 8)
                
                marketStatusCompact

                refreshControl
            }
            // Single instance kept across the active/inactive switch so focus and
            // typed text survive when the sibling controls show/hide.
            StockSearchBar(currency: appState.displayCurrency,
                           fillExpanded: true,
                           onActiveChange: { active in
                               withAnimation(.easeInOut(duration: 0.2)) { searchActive = active }
                           })
                .layoutPriority(1)
                .padding(.leading, searchActive ? 12 : 0)
            if !searchActive {
                currencyMenu
                if Trailing.self != EmptyView.self {
                    overflowMenu
                }
            }
        }
        .padding(.trailing, 12)
        .padding(.vertical, 4)
        .liquidGlass()
    }

    /// The secondary/rarely-used controls, collapsed into one "•••" menu so the
    /// compact bar stays uncluttered. (Customize Layout sits inline next to the
    /// account menu.) The host-supplied `trailing` (refresh / settings / account)
    /// is appended below a divider.
    private var overflowMenu: some View {
        PopoverMenu {
            trailing
        } label: {
            Image(systemName: "ellipsis.circle")
                .appFont(.title3)
                .foregroundStyle(.secondary)
                .frame(width: 32, height: 32)
                .contentShape(Rectangle())
        }
    }



    // MARK: - Accounts

    private var accountSummary: String {
        if appState.selectedAccounts.isEmpty { return "All Accounts" }
        if appState.selectedAccounts.count == 1 { return appState.selectedAccounts.first! }
        return "\(appState.selectedAccounts.count) Accounts"
    }

    private var orderedGroups: [(name: String, accounts: [String])] {
        let g = appState.accountGroups
        var order = appState.accountGroupOrder.isEmpty ? Array(g.keys).sorted() : appState.accountGroupOrder
        // Groups missing from the saved order (e.g. newly created) go last,
        // matching the web selector, instead of being dropped.
        order += g.keys.filter { !order.contains($0) }.sorted()
        return order.compactMap { name in g[name].map { (name, $0) } }
    }

    private var accountMenu: some View {
        PopoverMenu(minWidth: 220, maxHeight: 440) {
            // Open accounts first, then closed, each alphabetical (matches the web selector).
            let individuals = appState.allAccounts
                .filter { $0 != "All Accounts" }
                .sorted { a, b in
                    let aClosed = appState.closedAccounts.contains(a)
                    let bClosed = appState.closedAccounts.contains(b)
                    if aClosed != bClosed { return !aClosed }
                    return a.localizedStandardCompare(b) == .orderedAscending
                }
            MenuToggleRow(title: "All Accounts", isOn: appState.selectedAccounts.isEmpty, dismissOnTap: true) {
                appState.selectedAccounts = []
            }
            if !orderedGroups.isEmpty {
                MenuSectionHeader("Groups")
                ForEach(orderedGroups, id: \.name) { group in
                    let selected = !appState.selectedAccounts.isEmpty && appState.selectedAccounts == Set(group.accounts)
                    MenuToggleRow(title: group.name, isOn: selected, dismissOnTap: true) {
                        appState.selectedAccounts = Set(group.accounts)
                    }
                }
                MenuSectionHeader("Individual")
            }
            ForEach(individuals, id: \.self) { account in
                MenuToggleRow(title: account,
                              isOn: appState.selectedAccounts.contains(account),
                              trailing: appState.closedAccounts.contains(account) ? "Closed" : nil) {
                    toggle(account)
                }
            }
        } label: {
            Label(accountSummary, systemImage: "building.columns")
        }
        .interactiveGlass()
    }

    private func toggle(_ account: String) {
        if appState.selectedAccounts.contains(account) { appState.selectedAccounts.remove(account) }
        else { appState.selectedAccounts.insert(account) }
    }

    // MARK: - Currency / show-closed

    /// The FX rate caption is only shown where there's room. On iPhone (compact)
    /// it's hidden — otherwise, when the bar is tight, the untruncated string
    /// wraps to several lines and balloons the glass container.
    private var showFXRate: Bool {
        #if os(iOS)
        return hSize != .compact
        #else
        return true
        #endif
    }

    private var currencyMenu: some View {
        HStack(spacing: 8) {
            if showFXRate, appState.displayCurrency != "USD", let rate = appState.currentFXRateToUSD {
                Text("1 USD = \(String(format: "%.2f", rate)) \(appState.displayCurrency)")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
            }
            PopoverMenu(minWidth: 130) {
                ForEach(appState.availableCurrencies, id: \.self) { cur in
                    MenuToggleRow(title: cur, isOn: cur == appState.displayCurrency, dismissOnTap: true) {
                        appState.displayCurrency = cur
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(appState.displayCurrency)
                    Image(systemName: "chevron.up.chevron.down")
                        .appFont(.system(size: 16))
                }
            }
            .fixedSize()
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
        PopoverMenu { layoutMenuContent } label: {
            Label("Layout", systemImage: "slider.horizontal.3")
        }
        .fixedSize()
        .interactiveGlass()
    }

    /// The per-tab visible-section toggles, reusable both as the macOS bar's
    /// Layout menu and inside the compact overflow menu.
    @ViewBuilder private var layoutMenuContent: some View {
        MenuSectionHeader(TabLayout.sectionTitle(for: section))
        let items = TabLayout.items(for: section)
        ForEach(Array(groupedItems(items).enumerated()), id: \.offset) { _, group in
            if let label = group.label { MenuSectionHeader(label) }
            ForEach(group.items) { item in
                MenuToggleRow(title: item.title, isOn: appState.isVisible(section, item.id)) {
                    appState.toggle(section, item.id)
                }
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
