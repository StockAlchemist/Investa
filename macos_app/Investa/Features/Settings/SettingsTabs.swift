import SwiftUI

// MARK: - Accounts

struct AccountsSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    let accounts: [String]

    @State private var newGroup = ""
    @State private var currencyMap: [String: String] = [:]
    @State private var cashModeMap: [String: String] = [:]
    @State private var closureMap: [String: String] = [:]
    @State private var rates: [String: Double] = [:]
    @State private var thresholds: [String: Double] = [:]

    private var groups: [(name: String, accounts: [String])] {
        let g = settings?.accountGroups ?? [:]
        let order = settings?.accountGroupOrder ?? Array(g.keys).sorted()
        return order.compactMap { name in g[name].map { (name, $0) } } + g.keys.filter { !order.contains($0) }.sorted().map { ($0, g[$0] ?? []) }
    }

    var body: some View {
        VStack(spacing: 16) {
            groupsCard
            perAccountCard
        }
        .onAppear(perform: seed)
        .onChange(of: settings?.displayCurrency) { _, _ in seed() }
    }

    private func seed() {
        currencyMap = settings?.accountCurrencyMap ?? [:]
        cashModeMap = settings?.accountCashModeMap ?? [:]
        closureMap = settings?.accountClosureDates ?? [:]
        rates = settings?.accountInterestRates ?? [:]
        thresholds = settings?.interestFreeThresholds ?? [:]
    }

    private var groupsCard: some View {
        SettingsCard(title: "Custom Account Groups") {
            ForEach(groups, id: \.name) { group in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(group.name).fontWeight(.bold)
                        Spacer()
                        Button(role: .destructive) { deleteGroup(group.name) } label: { Image(systemName: "trash") }
                            .buttonStyle(.borderless).foregroundStyle(.red)
                    }
                    // Account membership toggles.
                    FlowToggles(all: accounts, selected: Set(group.accounts)) { acc, on in
                        toggleAccount(group.name, acc, on)
                    }
                    Divider()
                }
                .padding(.vertical, 2)
            }
            HStack {
                TextField("e.g. Retirement, Short Term", text: $newGroup).textFieldStyle(.roundedBorder)
                Button("Add Group") {
                    let name = newGroup.trimmingCharacters(in: .whitespaces); guard !name.isEmpty else { return }
                    var g = settings?.accountGroups ?? [:]; g[name] = []
                    var order = settings?.accountGroupOrder ?? []; order.append(name)
                    newGroup = ""
                    Task { await vm.update("account_groups", g); await vm.update("account_group_order", order) }
                }.buttonStyle(.borderedProminent)
            }
        }
    }

    private func toggleAccount(_ group: String, _ account: String, _ on: Bool) {
        var g = settings?.accountGroups ?? [:]
        var members = Set(g[group] ?? [])
        if on { members.insert(account) } else { members.remove(account) }
        g[group] = Array(members).sorted()
        Task { await vm.update("account_groups", g) }
    }
    private func deleteGroup(_ name: String) {
        var g = settings?.accountGroups ?? [:]; g.removeValue(forKey: name)
        let order = (settings?.accountGroupOrder ?? []).filter { $0 != name }
        Task { await vm.update("account_groups", g); await vm.update("account_group_order", order) }
    }

    private var perAccountCard: some View {
        SettingsCard(title: "Per-Account Settings") {
            if accounts.isEmpty { Text("No accounts.").foregroundStyle(.secondary) }
            ForEach(accounts, id: \.self) { acc in
                VStack(alignment: .leading, spacing: 6) {
                    Text(acc).fontWeight(.bold)
                    HStack(spacing: 12) {
                        labeled("Currency") {
                            TextField("USD", text: bind($currencyMap, acc)).frame(width: 70).textFieldStyle(.roundedBorder)
                        }
                        labeled("Cash Mode") {
                            Picker("", selection: bind($cashModeMap, acc, default: "Manual")) {
                                Text("Manual").tag("Manual"); Text("Auto").tag("Auto")
                            }.labelsHidden().fixedSize()
                        }
                        labeled("Closure Date") {
                            TextField("YYYY-MM-DD", text: bind($closureMap, acc)).frame(width: 110).textFieldStyle(.roundedBorder)
                        }
                        labeled("Interest %") {
                            TextField("0", text: bindNum($rates, acc)).frame(width: 56).textFieldStyle(.roundedBorder)
                        }
                        labeled("Free Threshold") {
                            TextField("0", text: bindNum($thresholds, acc)).frame(width: 70).textFieldStyle(.roundedBorder)
                        }
                    }
                    Divider()
                }
                .padding(.vertical, 2)
            }
            Button("Save Account Settings") {
                Task {
                    await vm.update("account_currency_map", currencyMap.filter { !$0.value.isEmpty })
                    await vm.update("account_cash_mode_map", cashModeMap)
                    await vm.update("account_closure_dates", closureMap.filter { !$0.value.isEmpty })
                    await vm.update("account_interest_rates", rates)
                    await vm.update("interest_free_thresholds", thresholds)
                }
            }.buttonStyle(.borderedProminent)
        }
    }

    private func labeled<C: View>(_ label: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 2) { Text(label).font(.caption2).foregroundStyle(.secondary); content() }
    }
    private func bind(_ map: Binding<[String: String]>, _ key: String, default def: String = "") -> Binding<String> {
        Binding(get: { map.wrappedValue[key] ?? def }, set: { map.wrappedValue[key] = $0 })
    }
    private func bindNum(_ map: Binding<[String: Double]>, _ key: String) -> Binding<String> {
        Binding(get: { map.wrappedValue[key].map { String($0) } ?? "" }, set: { map.wrappedValue[key] = Double($0) ?? 0 })
    }
}

/// Wrapping toggle chips for account membership.
private struct FlowToggles: View {
    let all: [String]; let selected: Set<String>; let onToggle: (String, Bool) -> Void
    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 6)], alignment: .leading, spacing: 6) {
            ForEach(all, id: \.self) { acc in
                let on = selected.contains(acc)
                Button { onToggle(acc, !on) } label: {
                    Label(acc, systemImage: on ? "checkmark.square.fill" : "square")
                        .font(.caption).foregroundStyle(on ? Color.accentColor : .secondary)
                }.buttonStyle(.plain)
            }
        }
    }
}

// MARK: - Symbols

struct SymbolsSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    @State private var mapFrom = ""; @State private var mapTo = ""; @State private var excludeSym = ""

    var body: some View {
        VStack(spacing: 16) {
            SettingsCard(title: "Symbol Mapping") {
                ForEach((settings?.userSymbolMap ?? [:]).sorted(by: { $0.key < $1.key }), id: \.key) { from, to in
                    HStack { Text(from).fontWeight(.medium); Image(systemName: "arrow.right").foregroundStyle(.secondary); Text(to)
                        Spacer()
                        Button(role: .destructive) { removeMap(from) } label: { Image(systemName: "trash") }.buttonStyle(.borderless).foregroundStyle(.red)
                    }
                    Divider()
                }
                HStack {
                    TextField("From (e.g. FB)", text: $mapFrom).textFieldStyle(.roundedBorder).frame(width: 140)
                    Image(systemName: "arrow.right").foregroundStyle(.secondary)
                    TextField("To (e.g. META)", text: $mapTo).textFieldStyle(.roundedBorder).frame(width: 140)
                    Button("Add") { addMap() }.buttonStyle(.borderedProminent)
                        .disabled(mapFrom.isEmpty || mapTo.isEmpty)
                }
            }
            SettingsCard(title: "Excluded Symbols") {
                let excluded = (settings?.userExcludedSymbols ?? []).sorted()
                if excluded.isEmpty { Text("None excluded.").foregroundStyle(.secondary) }
                ForEach(excluded, id: \.self) { sym in
                    HStack { Text(sym).fontWeight(.medium); Spacer()
                        Button(role: .destructive) { setExcluded(excluded.filter { $0 != sym }) } label: { Image(systemName: "trash") }.buttonStyle(.borderless).foregroundStyle(.red)
                    }
                    Divider()
                }
                HStack {
                    TextField("Symbol to exclude", text: $excludeSym).textFieldStyle(.roundedBorder).frame(width: 180)
                    Button("Exclude") {
                        let s = excludeSym.trimmingCharacters(in: .whitespaces).uppercased(); guard !s.isEmpty else { return }
                        excludeSym = ""; setExcluded(Array(Set(excluded + [s])))
                    }.buttonStyle(.borderedProminent)
                }
            }
        }
    }
    private func addMap() {
        var m = settings?.userSymbolMap ?? [:]; m[mapFrom.uppercased()] = mapTo.uppercased(); mapFrom = ""; mapTo = ""
        Task { await vm.update("user_symbol_map", m) }
    }
    private func removeMap(_ from: String) {
        var m = settings?.userSymbolMap ?? [:]; m.removeValue(forKey: from)
        Task { await vm.update("user_symbol_map", m) }
    }
    private func setExcluded(_ list: [String]) { Task { await vm.update("user_excluded_symbols", list) } }
}

// MARK: - Overrides

struct OverridesSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    @State private var sym = ""; @State private var price = ""; @State private var assetType = ""
    @State private var sector = ""; @State private var geo = ""; @State private var industry = ""; @State private var exchange = ""

    private var overrides: [(symbol: String, price: Double?, meta: [String: String])] {
        (settings?.manualOverrides ?? [:]).sorted(by: { $0.key < $1.key }).map { sym, val in
            if let p = val.doubleValue { return (sym, p, [:]) }
            var meta: [String: String] = [:]
            for k in ["asset_type", "sector", "geography", "industry", "exchange"] { if let s = val[k]?.stringValue { meta[k] = s } }
            return (sym, val["price"]?.doubleValue, meta)
        }
    }

    var body: some View {
        SettingsCard(title: "Manual Overrides") {
            ForEach(overrides, id: \.symbol) { o in
                HStack {
                    Text(o.symbol).fontWeight(.bold).frame(width: 80, alignment: .leading)
                    Text(o.price.map { Fmt.number($0) } ?? "—").monospacedDigit().frame(width: 80, alignment: .trailing)
                    Text([o.meta["asset_type"], o.meta["sector"], o.meta["geography"], o.meta["industry"], o.meta["exchange"]].compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    Spacer()
                    Button(role: .destructive) { remove(o.symbol) } label: { Image(systemName: "trash") }.buttonStyle(.borderless).foregroundStyle(.red)
                }
                Divider()
            }
            VStack(spacing: 8) {
                HStack {
                    TextField("Symbol", text: $sym).frame(width: 100)
                    TextField("Price", text: $price).frame(width: 90)
                    TextField("Asset Type", text: $assetType)
                    TextField("Sector", text: $sector)
                }
                HStack {
                    TextField("Geography", text: $geo)
                    TextField("Industry", text: $industry)
                    TextField("Exchange", text: $exchange)
                    Button("Add Override") { add() }.buttonStyle(.borderedProminent).disabled(sym.isEmpty)
                }
            }
            .textFieldStyle(.roundedBorder)
        }
    }
    private func add() {
        var map = settings?.manualOverrides ?? [:]
        var obj: [String: JSONValue] = [:]
        if let p = Double(price) { obj["price"] = .double(p) }
        for (k, v) in [("asset_type", assetType), ("sector", sector), ("geography", geo), ("industry", industry), ("exchange", exchange)] where !v.isEmpty { obj[k] = .string(v) }
        map[sym.uppercased()] = .object(obj)
        sym = ""; price = ""; assetType = ""; sector = ""; geo = ""; industry = ""; exchange = ""
        Task { await vm.update("manual_price_overrides", map) }
    }
    private func remove(_ symbol: String) {
        var map = settings?.manualOverrides ?? [:]; map.removeValue(forKey: symbol)
        Task { await vm.update("manual_price_overrides", map) }
    }
}

// MARK: - Advanced

struct AdvancedSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    @EnvironmentObject private var appState: AppState
    @State private var newCurrency = ""
    @State private var ibkrToken = ""; @State private var ibkrQuery = ""
    @State private var refreshSecret = ""
    @State private var serverURL = APIConfig.baseURL
    @State private var benchmarksText = ""

    var body: some View {
        VStack(spacing: 16) {
            SettingsCard(title: "Display") {
                Picker("Display Currency", selection: $appState.displayCurrency) {
                    ForEach(appState.availableCurrencies, id: \.self) { Text($0).tag($0) }
                }
                Toggle("Include closed accounts", isOn: $appState.showClosed)
                HStack {
                    TextField("Benchmarks (e.g. SPY, QQQ)", text: $benchmarksText).textFieldStyle(.roundedBorder)
                    Button("Save") {
                        let list = benchmarksText.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces).uppercased() }.filter { !$0.isEmpty }
                        Task { await vm.update("benchmarks", list) }
                    }
                }
            }
            SettingsCard(title: "Available Currencies") {
                let currencies = (settings?.availableCurrencies ?? []).sorted()
                FlowChipsRemovable(items: currencies) { remove($0, currencies) }
                HStack {
                    TextField("Add currency (e.g. SGD)", text: $newCurrency).textFieldStyle(.roundedBorder).frame(width: 160)
                    Button("Add") { let c = newCurrency.uppercased(); guard !c.isEmpty else { return }; newCurrency = ""; setCurrencies(Array(Set(currencies + [c]))) }
                        .buttonStyle(.borderedProminent)
                }
            }
            SettingsCard(title: "Interactive Brokers") {
                TextField("Flex Token", text: $ibkrToken).textFieldStyle(.roundedBorder)
                TextField("Query ID", text: $ibkrQuery).textFieldStyle(.roundedBorder)
                HStack {
                    Button("Save Credentials") { Task { await vm.update("ibkr_token", ibkrToken); await vm.update("ibkr_query_id", ibkrQuery) } }
                        .buttonStyle(.bordered)
                    Button("Sync Now") { Task { await vm.syncIbkr() } }.buttonStyle(.borderedProminent)
                }
            }
            SettingsCard(title: "System") {
                HStack {
                    TextField("Webhook secret", text: $refreshSecret).textFieldStyle(.roundedBorder)
                    Button("Trigger Refresh") { Task { await vm.triggerRefresh(secret: refreshSecret) } }.buttonStyle(.bordered)
                }
                Button("Clear Server Cache") { Task { await vm.clearCache() } }.buttonStyle(.bordered)
            }
            SettingsCard(title: "Backend Server") {
                TextField("Base URL", text: $serverURL).textFieldStyle(.roundedBorder)
                Button("Save Server URL") { APIConfig.baseURL = serverURL }.buttonStyle(.bordered)
            }
        }
        .onAppear {
            ibkrToken = settings?.ibkrToken ?? ""; ibkrQuery = settings?.ibkrQueryId ?? ""
            benchmarksText = (settings?.benchmarks ?? []).joined(separator: ", ")
        }
        .onChange(of: settings?.ibkrToken) { _, new in ibkrToken = new ?? ibkrToken }
    }
    private func setCurrencies(_ list: [String]) { Task { await vm.update("available_currencies", list) } }
    private func remove(_ c: String, _ list: [String]) { setCurrencies(list.filter { $0 != c }) }
}

private struct FlowChipsRemovable: View {
    let items: [String]; let onRemove: (String) -> Void
    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 90), spacing: 6)], alignment: .leading, spacing: 6) {
            ForEach(items, id: \.self) { item in
                HStack(spacing: 4) {
                    Text(item).font(.caption)
                    Button { onRemove(item) } label: { Image(systemName: "xmark.circle.fill").font(.caption2) }.buttonStyle(.plain).foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(.background.tertiary, in: Capsule())
            }
        }
    }
}

// MARK: - Profile & Security

struct AccountSecuritySettings: View {
    @ObservedObject var vm: SettingsViewModel
    @EnvironmentObject private var auth: AuthViewModel
    @State private var alias = ""
    @State private var showPassword = false
    @State private var confirmDelete = false

    var body: some View {
        VStack(spacing: 16) {
            SettingsCard(title: "Profile") {
                if let user = auth.currentUser {
                    LabeledContent("Username", value: user.username)
                }
                HStack {
                    TextField("Display name", text: $alias).textFieldStyle(.roundedBorder)
                    Button("Save") { Task { await vm.updateProfile(alias: alias) } }.buttonStyle(.borderedProminent)
                }
            }
            SettingsCard(title: "Security") {
                Button("Change Password…") { showPassword = true }.buttonStyle(.bordered)
                Button("Log Out", role: .destructive) { auth.logout() }.buttonStyle(.bordered)
            }
            SettingsCard(title: "Danger Zone") {
                Text("Permanently delete your account and all associated data.").font(.caption).foregroundStyle(.secondary)
                Button("Delete Account", role: .destructive) { confirmDelete = true }.buttonStyle(.borderedProminent).tint(.red)
            }
        }
        .onAppear { alias = auth.currentUser?.alias ?? "" }
        .sheet(isPresented: $showPassword) { ChangePasswordView().environmentObject(auth) }
        .alert("Delete account?", isPresented: $confirmDelete) {
            Button("Delete", role: .destructive) { Task { await vm.deleteAccount(); auth.logout() } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("This permanently deletes your account and all data. This cannot be undone.") }
    }
}
