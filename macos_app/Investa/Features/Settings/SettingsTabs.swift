import SwiftUI

// MARK: - Accounts

struct AccountsSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    let accounts: [String]
    @ObservedObject var appState: AppState

    @State private var holdings: [Holding] = []

    @State private var newGroup = ""
    @State private var currencyMap: [String: String] = [:]
    @State private var cashModeMap: [String: String] = [:]
    @State private var closureMap: [String: String] = [:]
    @State private var rates: [String: Double] = [:]
    @State private var thresholds: [String: Double] = [:]

    @State private var isCreatingGroup = false
    @State private var editingGroupAccounts: Set<String> = []
    
    @State private var newCurrency = ""

    private var groups: [(name: String, accounts: [String])] {
        let g = settings?.accountGroups ?? [:]
        let order = settings?.accountGroupOrder ?? Array(g.keys).sorted()
        return order.compactMap { name in g[name].map { (name, $0) } } + g.keys.filter { !order.contains($0) }.sorted().map { ($0, g[$0] ?? []) }
    }

    var body: some View {
        VStack(spacing: 16) {
            groupsCard
            perAccountCard
            cashYieldCard
            availableCurrenciesCard
        }
        .onAppear {
            seed()
            fetchHoldings()
        }
        .onChange(of: settings?.displayCurrency) { _, _ in seed() }
    }

    private func seed() {
        currencyMap = settings?.accountCurrencyMap ?? [:]
        cashModeMap = settings?.accountCashModeMap ?? [:]
        closureMap = settings?.accountClosureDates ?? [:]
        rates = settings?.accountInterestRates ?? [:]
        thresholds = settings?.interestFreeThresholds ?? [:]
    }

    private func fetchHoldings() {
        Task {
            if let result: [Holding] = try? await APIClient.shared.get("/holdings", query: [URLQueryItem(name: "currency", value: appState.displayCurrency)]) {
                await MainActor.run { holdings = result }
            }
        }
    }

    private var availableCurrenciesCard: some View {
        SettingsCard(title: "Available Currencies") {
            let currencies = (settings?.availableCurrencies ?? []).sorted()
            FlowChipsRemovable(items: currencies) { remove($0, currencies) }
            HStack {
                TextField("Add currency (e.g. SGD)", text: $newCurrency).textFieldStyle(.roundedBorder).frame(width: 160)
                Button("Add") { let c = newCurrency.uppercased(); guard !c.isEmpty else { return }; newCurrency = ""; setCurrencies(Array(Set(currencies + [c]))) }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private func setCurrencies(_ currs: [String]) { Task { await vm.update("available_currencies", currs) } }
    private func remove(_ c: String, _ all: [String]) { let rest = all.filter { $0 != c }; setCurrencies(rest) }

    private var groupsCard: some View {
        SettingsCard(title: "Custom Account Groups") {
            Text("Create custom groups of accounts for quick filtering.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            if groups.isEmpty && !isCreatingGroup {
                Text("No groups defined yet.")
                    .font(.caption).foregroundStyle(.secondary)
                    .padding()
            } else {
                ForEach(groups, id: \.name) { group in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(group.name).fontWeight(.bold)
                            Spacer()
                            Button(role: .destructive) { deleteGroup(group.name) } label: { Image(systemName: "trash") }
                                .buttonStyle(.borderless).foregroundStyle(.red)
                        }
                        Text(group.accounts.joined(separator: ", "))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                        Divider()
                    }
                    .padding(.vertical, 2)
                }
            }
            
            if isCreatingGroup {
                VStack(alignment: .leading, spacing: 8) {
                    Text("New Account Group").fontWeight(.bold)
                    TextField("Group Name (e.g. Retirement)", text: $newGroup)
                        .textFieldStyle(.roundedBorder)
                    
                    Text("Select Accounts:").font(.caption).foregroundStyle(.secondary)
                    
                    FlowToggles(all: accounts, selected: editingGroupAccounts) { acc, on in
                        if on { editingGroupAccounts.insert(acc) }
                        else { editingGroupAccounts.remove(acc) }
                    }
                    
                    HStack {
                        Button("Cancel") {
                            isCreatingGroup = false
                            newGroup = ""
                            editingGroupAccounts.removeAll()
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Save Group") {
                            let name = newGroup.trimmingCharacters(in: .whitespaces)
                            guard !name.isEmpty, !editingGroupAccounts.isEmpty else { return }
                            var g = settings?.accountGroups ?? [:]
                            g[name] = Array(editingGroupAccounts).sorted()
                            var order = settings?.accountGroupOrder ?? []
                            if !order.contains(name) { order.append(name) }
                            
                            isCreatingGroup = false
                            newGroup = ""
                            editingGroupAccounts.removeAll()
                            
                            Task { 
                                await vm.update("account_groups", g)
                                await vm.update("account_group_order", order) 
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(newGroup.trimmingCharacters(in: .whitespaces).isEmpty || editingGroupAccounts.isEmpty)
                    }
                    .padding(.top, 4)
                }
                .padding()
                .background(Color.secondary.opacity(0.1))
                .cornerRadius(8)
            } else {
                Button("Create Group") {
                    isCreatingGroup = true
                    newGroup = ""
                    editingGroupAccounts.removeAll()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private func deleteGroup(_ name: String) {
        var g = settings?.accountGroups ?? [:]; g.removeValue(forKey: name)
        let order = (settings?.accountGroupOrder ?? []).filter { $0 != name }
        Task { await vm.update("account_groups", g); await vm.update("account_group_order", order) }
    }

    private var perAccountCard: some View {
        SettingsCard(title: "Account Preferences") {
            Text("Configure currency, cash management mode, and closure date for each account.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            let configurableAccounts = accounts.filter { $0 != "All Accounts" }
            if configurableAccounts.isEmpty { 
                Text("No accounts.").foregroundStyle(.secondary) 
            }
            
            // On macOS, we can use a LazyVGrid or just let them wrap if we had a wrapping layout.
            // For simplicity and matching the exact visual of the web app cards:
            VStack(spacing: 12) {
                ForEach(configurableAccounts, id: \.self) { acc in
                    let closureDateStr = closureMap[acc] ?? ""
                    let isClosed = !closureDateStr.isEmpty && closureDateStr <= ISO8601DateFormatter().string(from: Date()).prefix(10)

                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            if isClosed {
                                Text(acc).font(.headline).strikethrough().foregroundStyle(.secondary)
                                Text("CLOSED")
                                    .font(.system(size: 11, weight: .bold))
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(Color.secondary.opacity(0.2))
                                    .cornerRadius(4)
                            } else {
                                Text(acc).font(.headline)
                            }
                            Spacer()
                        }
                        
                        Divider()
                        
                        // Use a grid so it wraps on iOS but stays in a row on macOS
                        #if os(macOS)
                        let cols = [GridItem(.adaptive(minimum: 120), spacing: 16)]
                        #else
                        let cols = [GridItem(.adaptive(minimum: 140), spacing: 16)]
                        #endif
                        
                        LazyVGrid(columns: cols, alignment: .leading, spacing: 12) {
                            labeled("Currency") {
                                Picker("", selection: bind($currencyMap, acc, default: "USD")) {
                                    ForEach(appState.availableCurrencies, id: \.self) { c in
                                        Text(c).tag(c)
                                    }
                                }
                                .labelsHidden()
                                .pickerStyle(.menu)
                                .fixedSize()
                                #if os(iOS)
                                .padding(.horizontal, 12).padding(.vertical, 8)
                                .background(Color(uiColor: .systemGray6))
                                .cornerRadius(8)
                                #endif
                            }
                            
                            labeled("Cash Mode") {
                                Picker("", selection: bind($cashModeMap, acc, default: "Manual")) {
                                    Text("Manual").tag("Manual"); Text("Auto").tag("Auto")
                                }
                                .labelsHidden()
                                .pickerStyle(.menu)
                                .fixedSize()
                                #if os(iOS)
                                .padding(.horizontal, 12).padding(.vertical, 8)
                                .background(Color(uiColor: .systemGray6))
                                .cornerRadius(8)
                                #endif
                            }
                            
                            labeled("Closure Date") {
                                HStack(spacing: 8) {
                                    TextField("YYYY-MM-DD", text: bind($closureMap, acc))
                                        .textFieldStyle(.roundedBorder)
                                        .frame(maxWidth: 130)
                                    if !closureDateStr.isEmpty {
                                        Button(role: .destructive) { closureMap[acc] = "" } label: { Image(systemName: "trash") }
                                            .buttonStyle(.borderless).foregroundStyle(.red)
                                    }
                                }
                            }
                        }
                    }
                    .padding()
                    #if os(macOS)
                    .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
                    #else
                    .background(Color(uiColor: .secondarySystemGroupedBackground))
                    #endif
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.secondary.opacity(0.1), lineWidth: 1)
                    )
                    .opacity(isClosed ? 0.6 : 1.0)
                }
            }
            .padding(.top, 4)
            
            Button("Save Account Preferences") {
                Task {
                    await vm.update("account_currency_map", currencyMap.filter { !$0.value.isEmpty })
                    await vm.update("account_cash_mode_map", cashModeMap)
                    await vm.update("account_closure_dates", closureMap.filter { !$0.value.isEmpty })
                }
            }.buttonStyle(.borderedProminent)
            .padding(.top, 8)
        }
    }

    private var cashYieldCard: some View {
        SettingsCard(title: "Cash Yield Management") {
            Text("Configure annual interest rates and interest-free thresholds for your cash balances to estimate future yield.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            let cashAccounts = accountsWithCash()
            if cashAccounts.isEmpty { 
                Text("No accounts with cash balances found.")
                    .font(.caption).foregroundStyle(.secondary)
                    .padding(.vertical, 8)
            } else {
                ScrollView(.horizontal, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 16) {
                            Text("Account").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 120, alignment: .leading)
                            Text("Cash Balance").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 120, alignment: .trailing)
                            Text("Est. Annual Interest").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 130, alignment: .trailing)
                            Text("Annual Rate (%)").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 120, alignment: .leading)
                            Text("Exempt Threshold").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 120, alignment: .leading)
                        }
                        Divider()
                        ForEach(cashAccounts, id: \.self) { acc in
                            let balance = cashBalance(for: acc)
                            let rate = rates[acc] ?? 0.0
                            let threshold = thresholds[acc] ?? 0.0
                            let interest = max(0, balance - threshold) * (rate / 100.0)
                            
                            HStack(spacing: 16) {
                                Text(acc).fontWeight(.bold).frame(width: 120, alignment: .leading).lineLimit(1)
                                Text(balance.formatted(.currency(code: appState.displayCurrency))).frame(width: 120, alignment: .trailing).monospacedDigit()
                                Text(interest.formatted(.currency(code: appState.displayCurrency))).frame(width: 130, alignment: .trailing).monospacedDigit().foregroundStyle(.green)
                                TextField("0.0", text: bindNum($rates, acc)).frame(width: 120).textFieldStyle(.roundedBorder)
                                TextField("0", text: bindNum($thresholds, acc)).frame(width: 120).textFieldStyle(.roundedBorder)
                            }
                            Divider()
                        }
                    }
                    .padding(.bottom, 8)
                }
            }
            Button("Save Cash Yield Settings") {
                Task {
                    await vm.update("account_interest_rates", rates)
                    await vm.update("interest_free_thresholds", thresholds)
                }
            }.buttonStyle(.borderedProminent)
        }
    }

    private func accountsWithCash() -> [String] {
        let cashHoldings = holdings.filter { $0.symbol.uppercased().contains("CASH") || $0.symbol.uppercased() == "$CASH" }
        let set = Set(cashHoldings.compactMap { $0.account })
        return accounts.filter { set.contains($0) }
    }

    private func cashBalance(for account: String) -> Double {
        let accHoldings = holdings.filter { $0.account == account && ($0.symbol.uppercased().contains("CASH") || $0.symbol.uppercased() == "$CASH") }
        return accHoldings.compactMap { $0.marketValue(currency: appState.displayCurrency) }.reduce(0, +)
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
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), spacing: 8)], alignment: .leading, spacing: 8) {
            ForEach(all, id: \.self) { acc in
                let on = selected.contains(acc)
                Button { onToggle(acc, !on) } label: {
                    Label(acc, systemImage: on ? "checkmark.square.fill" : "square")
                        .font(.caption).foregroundStyle(on ? Color.accentColor : .secondary)
                        .lineLimit(1)
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
                VStack(spacing: 8) {
                    HStack {
                        TextField("From (e.g. FB)", text: $mapFrom).textFieldStyle(.roundedBorder)
                        Image(systemName: "arrow.right").foregroundStyle(.secondary)
                        TextField("To (e.g. META)", text: $mapTo).textFieldStyle(.roundedBorder)
                    }
                    Button("Add Mapping") { addMap() }.buttonStyle(.borderedProminent)
                        .disabled(mapFrom.isEmpty || mapTo.isEmpty)
                        .frame(maxWidth: .infinity, alignment: .trailing)
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
        VStack(spacing: 16) {
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
                        TextField("Symbol", text: $sym)
                        TextField("Price", text: $price)
                    }
                    HStack {
                        TextField("Asset Type", text: $assetType)
                        TextField("Sector", text: $sector)
                    }
                    HStack {
                        TextField("Geography", text: $geo)
                        TextField("Industry", text: $industry)
                    }
                    HStack {
                        TextField("Exchange", text: $exchange)
                        Button("Add Override") { add() }.buttonStyle(.borderedProminent).disabled(sym.isEmpty)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                    }
                }
                .textFieldStyle(.roundedBorder)
            }
            
            ValuationOverridesSettings(vm: vm, settings: settings)
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

struct ValuationOverridesSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    @State private var sym = ""
    @State private var dcfDiscountRate = ""
    @State private var dcfGrowthRate = ""
    @State private var dcfTerminalGrowth = ""
    @State private var dcfProjectionYears = ""
    @State private var dcfFcf = ""
    @State private var targetFcfMargin = ""
    @State private var grahamEps = ""
    @State private var grahamGrowthRate = ""
    @State private var grahamBondYield = ""

    private var overrides: [(symbol: String, values: [String: Double])] {
        (settings?.valuationOverrides ?? [:]).sorted(by: { $0.key < $1.key }).map { (symbol: $0.key, values: $0.value) }
    }

    var body: some View {
        SettingsCard(title: "Valuation Overrides") {
            ForEach(overrides, id: \.symbol) { o in
                HStack {
                    Text(o.symbol).fontWeight(.bold).frame(width: 80, alignment: .leading)
                    
                    Text(o.values.map { "\($0.key): \($0.value)" }.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    Spacer()
                    Button(role: .destructive) { remove(o.symbol) } label: { Image(systemName: "trash") }.buttonStyle(.borderless).foregroundStyle(.red)
                }
                Divider()
            }
            VStack(spacing: 8) {
                HStack {
                    TextField("Symbol", text: $sym)
                    TextField("DCF Discount %", text: $dcfDiscountRate)
                }
                HStack {
                    TextField("DCF Growth %", text: $dcfGrowthRate)
                    TextField("DCF Term. Growth %", text: $dcfTerminalGrowth)
                }
                HStack {
                    TextField("DCF Proj Years", text: $dcfProjectionYears)
                    TextField("DCF Base FCF", text: $dcfFcf)
                }
                HStack {
                    TextField("Target FCF Margin %", text: $targetFcfMargin)
                    TextField("Graham EPS", text: $grahamEps)
                }
                HStack {
                    TextField("Graham Growth Rate", text: $grahamGrowthRate)
                    TextField("Graham Bond Yield %", text: $grahamBondYield)
                }
                Button("Add Override") { add() }.buttonStyle(.borderedProminent).disabled(sym.isEmpty)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
            .textFieldStyle(.roundedBorder)
        }
    }

    private func add() {
        var map = settings?.valuationOverrides ?? [:]
        var obj: [String: Double] = [:]
        
        func addIfValid(_ str: String, key: String, isPercent: Bool = false) {
            if let v = Double(str) { obj[key] = isPercent ? v / 100.0 : v }
        }
        
        addIfValid(dcfDiscountRate, key: "dcf_discount_rate", isPercent: true)
        addIfValid(dcfGrowthRate, key: "dcf_growth_rate", isPercent: true)
        addIfValid(dcfTerminalGrowth, key: "dcf_terminal_growth", isPercent: true)
        addIfValid(dcfProjectionYears, key: "dcf_projection_years")
        addIfValid(dcfFcf, key: "dcf_fcf")
        addIfValid(targetFcfMargin, key: "target_fcf_margin", isPercent: true)
        addIfValid(grahamEps, key: "graham_eps")
        addIfValid(grahamGrowthRate, key: "graham_growth_rate")
        addIfValid(grahamBondYield, key: "graham_bond_yield")
        
        if !obj.isEmpty {
            map[sym.uppercased()] = obj
            Task { await vm.update("valuation_overrides", map) }
        }
        
        sym = ""; dcfDiscountRate = ""; dcfGrowthRate = ""; dcfTerminalGrowth = ""; dcfProjectionYears = ""; dcfFcf = ""; targetFcfMargin = ""; grahamEps = ""; grahamGrowthRate = ""; grahamBondYield = ""
    }
    private func remove(_ symbol: String) {
        var map = settings?.valuationOverrides ?? [:]; map.removeValue(forKey: symbol)
        Task { await vm.update("valuation_overrides", map) }
    }
}

// MARK: - Advanced

struct AdvancedSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    @EnvironmentObject private var appState: AppState
    @State private var ibkrToken = ""; @State private var ibkrQuery = ""
    @State private var refreshSecret = ""
    @State private var serverURL = APIConfig.baseURL
    @State private var benchmarksText = ""

    var body: some View {
        VStack(spacing: 16) {

            benchmarksCard
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
        }
        .onChange(of: settings?.ibkrToken) { _, new in ibkrToken = new ?? ibkrToken }
    }

    private let presetBenchmarks = [
        "S&P 500", "Dow Jones", "NASDAQ", "Russell 2000",
        "SPY (S&P 500 ETF)", "QQQ (Nasdaq 100 ETF)", "DIA (Dow Jones ETF)", "S&P 500 Total Return",
    ]

    private var benchmarksCard: some View {
        SettingsCard(title: "Benchmarks (\(appState.benchmarks.count))") {
            ForEach(presetBenchmarks, id: \.self) { b in
                let on = appState.benchmarks.contains(b)
                Button { toggleBenchmark(b) } label: {
                    HStack {
                        Image(systemName: on ? "checkmark.square.fill" : "square").foregroundStyle(on ? Color.accentColor : .secondary)
                        Text(b); Spacer()
                    }
                }.buttonStyle(.plain)
            }
            
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
                TextField("Custom ticker", text: $benchmarksText).textFieldStyle(.roundedBorder).frame(maxWidth: 160)
                Button("Add") {
                    let t = benchmarksText.trimmingCharacters(in: .whitespaces).uppercased()
                    guard !t.isEmpty, !appState.benchmarks.contains(t) else { return }
                    benchmarksText = ""; appState.setBenchmarks(appState.benchmarks + [t])
                }.buttonStyle(.borderedProminent)
            }
        }
    }

    private func toggleBenchmark(_ b: String) {
        if appState.benchmarks.contains(b) { appState.setBenchmarks(appState.benchmarks.filter { $0 != b }) }
        else { appState.setBenchmarks(appState.benchmarks + [b]) }
    }
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
                    .fixedSize(horizontal: false, vertical: true)
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
