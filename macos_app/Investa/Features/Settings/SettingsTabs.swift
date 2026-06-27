import SwiftUI

// MARK: - Accounts

struct AccountsSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    let accounts: [String]
    @ObservedObject var appState: AppState

    @State private var holdings: [Holding] = []

    @State private var currencyMap: [String: String] = [:]
    @State private var cashModeMap: [String: String] = [:]
    @State private var closureMap: [String: String] = [:]
    @State private var rates: [String: Double] = [:]
    @State private var thresholds: [String: Double] = [:]
    
    @State private var newCurrency = ""

    var body: some View {
        VStack(spacing: 16) {
            availableCurrenciesCard
            perAccountCard
            cashYieldCard
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
        SettingsCard(title: "Currency Management", icon: "dollarsign.circle", iconColor: .orange) {
            Text("Add or remove currencies available for manual accounts.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            let currencies = (settings?.availableCurrencies ?? []).sorted()
            if currencies.isEmpty {
                Text("No additional currencies defined.")
                    .font(.caption).foregroundStyle(.secondary)
                    .padding(.vertical, 8)
            } else {
                FlowChipsRemovable(items: currencies) { remove($0, currencies) }
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Add a Currency").font(.caption2).foregroundStyle(.secondary)
                HStack {
                    TextField("e.g. SGD", text: $newCurrency)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 120)
                    Button("Add") { 
                        let c = newCurrency.uppercased()
                        guard !c.isEmpty else { return }
                        newCurrency = ""
                        setCurrencies(Array(Set(currencies + [c]))) 
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .disabled(newCurrency.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(.top, 8)
        }
    }

    private func setCurrencies(_ currs: [String]) { Task { await vm.update("available_currencies", currs) } }
    private func remove(_ c: String, _ all: [String]) { let rest = all.filter { $0 != c }; setCurrencies(rest) }

    private var perAccountCard: some View {
        SettingsCard(title: "Account Preferences", icon: "slider.horizontal.3", iconColor: .purple) {
            Text("Configure currency, cash management mode, and closure date for each account.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            let configurableAccounts = accounts.filter { $0 != "All Accounts" }
            if configurableAccounts.isEmpty { 
                Text("No accounts found.")
                    .font(.caption).foregroundStyle(.secondary) 
                    .padding(.vertical, 8)
            } else {
                VStack(spacing: 12) {
                    ForEach(configurableAccounts, id: \.self) { acc in
                        let closureDateStr = closureMap[acc] ?? ""
                        let isClosed = !closureDateStr.isEmpty && closureDateStr <= ISO8601DateFormatter().string(from: Date()).prefix(10)

                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                if isClosed {
                                    Text(acc).font(.headline).strikethrough().foregroundStyle(.secondary)
                                    Text("CLOSED")
                                        .font(.system(size: 10, weight: .bold))
                                        .padding(.horizontal, 6).padding(.vertical, 2)
                                        .background(Color.secondary.opacity(0.2))
                                        .cornerRadius(4)
                                } else {
                                    Text(acc).font(.headline).fontWeight(.bold)
                                }
                                Spacer()
                            }
                            
                            Divider()
                            
                            #if os(macOS)
                            let cols = [GridItem(.adaptive(minimum: 140), spacing: 16)]
                            #else
                            let cols = [GridItem(.adaptive(minimum: 120), spacing: 16)]
                            #endif
                            
                            LazyVGrid(columns: cols, alignment: .leading, spacing: 12) {
                                labeled("Default Currency") {
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
                                
                                labeled("Cash Management") {
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
                                        let binding = Binding<Date>(
                                            get: {
                                                let formatter = DateFormatter(); formatter.dateFormat = "yyyy-MM-dd"
                                                return formatter.date(from: closureMap[acc] ?? "") ?? Date()
                                            },
                                            set: { newDate in
                                                let formatter = DateFormatter(); formatter.dateFormat = "yyyy-MM-dd"
                                                closureMap[acc] = formatter.string(from: newDate)
                                            }
                                        )
                                        if closureDateStr.isEmpty {
                                            Button("Set Date") {
                                                let formatter = DateFormatter(); formatter.dateFormat = "yyyy-MM-dd"
                                                closureMap[acc] = formatter.string(from: Date())
                                            }
                                            .buttonStyle(.bordered)
                                        } else {
                                            #if os(iOS)
                                            DatePicker("", selection: binding, displayedComponents: .date)
                                                .labelsHidden()
                                                .datePickerStyle(.graphical)
                                            #else
                                            DatePicker("", selection: binding, displayedComponents: .date)
                                                .labelsHidden()
                                                .datePickerStyle(.compact)
                                            #endif
                                        }
                                        if !closureDateStr.isEmpty {
                                            Button(role: .destructive) { closureMap[acc] = "" } label: { Image(systemName: "trash") }
                                                .buttonStyle(.borderless).foregroundStyle(.red)
                                        }
                                    }
                                }
                            }
                        }
                        .padding()
                        .background(Color.primary.opacity(0.03))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.primary.opacity(0.05), lineWidth: 1)
                        )
                        .opacity(isClosed ? 0.7 : 1.0)
                    }
                }
                .padding(.top, 4)
            }
            
            Button("Save Account Preferences") {
                Task {
                    await vm.update("account_currency_map", currencyMap.filter { !$0.value.isEmpty })
                    await vm.update("account_cash_mode_map", cashModeMap)
                    await vm.update("account_closure_dates", closureMap.filter { !$0.value.isEmpty })
                }
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 8)
        }
    }

    private var cashYieldCard: some View {
        SettingsCard(title: "Cash Yield Management", icon: "percent", iconColor: .teal) {
            Text("Configure annual interest rates and interest-free thresholds for your cash balances to estimate future yield.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            let cashAccounts = accountsWithCash()
            if cashAccounts.isEmpty { 
                Text("No accounts with cash balances found.")
                    .font(.caption).foregroundStyle(.secondary)
                    .padding(.vertical, 8)
            } else {
                VStack(spacing: 12) {
                    ForEach(cashAccounts, id: \.self) { acc in
                        let balance = cashBalance(for: acc)
                        let rate = rates[acc] ?? 0.0
                        let threshold = thresholds[acc] ?? 0.0
                        let interest = max(0, balance - threshold) * (rate / 100.0)
                        
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Text(acc).font(.headline).fontWeight(.bold)
                                Spacer()
                                VStack(alignment: .trailing, spacing: 2) {
                                    Text("Balance").font(.caption2).foregroundStyle(.secondary)
                                    Text(balance.formatted(.currency(code: appState.displayCurrency))).font(.subheadline).monospacedDigit().fontWeight(.bold)
                                }
                            }
                            
                            Divider()
                            
                            #if os(macOS)
                            let cols = [GridItem(.adaptive(minimum: 140), spacing: 16)]
                            #else
                            let cols = [GridItem(.adaptive(minimum: 120), spacing: 16)]
                            #endif
                            
                            LazyVGrid(columns: cols, alignment: .leading, spacing: 12) {
                                labeled("Annual Rate (%)") {
                                    TextField("0.0", text: bindNum($rates, acc))
                                        .textFieldStyle(.roundedBorder)
                                }
                                labeled("Exempt Threshold") {
                                    TextField("0", text: bindNum($thresholds, acc))
                                        .textFieldStyle(.roundedBorder)
                                }
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Est. Annual Interest").font(.caption2).foregroundStyle(.secondary)
                                    Text(interest.formatted(.currency(code: appState.displayCurrency)))
                                        .font(.body).monospacedDigit().fontWeight(.bold).foregroundStyle(.green)
                                        .padding(.vertical, 4)
                                }
                            }
                        }
                        .padding()
                        .background(Color.primary.opacity(0.03))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.primary.opacity(0.05), lineWidth: 1)
                        )
                    }
                }
                .padding(.top, 4)
            }
            Button("Save Cash Yield Settings") {
                Task {
                    await vm.update("account_interest_rates", rates)
                    await vm.update("interest_free_thresholds", thresholds)
                }
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 8)
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

// MARK: - Symbols

struct SymbolsSettings: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?
    @State private var mapFrom = ""; @State private var mapTo = ""; @State private var excludeSym = ""

    var body: some View {
        VStack(spacing: 16) {
            SettingsCard(title: "Add Symbol Mapping", icon: "map", iconColor: .blue) {
                Text("Resolve custom or broker-specific tickers to a real Yahoo Finance symbol.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                
                HStack(alignment: .bottom, spacing: 16) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Portfolio Symbol").font(.caption2).foregroundStyle(.secondary)
                        TextField("e.g. MY-FUND", text: $mapFrom)
                            .textFieldStyle(.roundedBorder)
                    }
                    
                    Image(systemName: "arrow.right")
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 6)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Yahoo Finance Ticker").font(.caption2).foregroundStyle(.secondary)
                        TextField("e.g. VTSAX", text: $mapTo)
                            .textFieldStyle(.roundedBorder)
                    }
                    
                    Button("Map") { addMap() }
                        .buttonStyle(.borderedProminent)
                        .disabled(mapFrom.trimmingCharacters(in: .whitespaces).isEmpty || mapTo.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                .padding(.top, 4)
            }
            
            SettingsCard(title: "Active Mappings", icon: "map", iconColor: .blue) {
                let sortedMap = (settings?.userSymbolMap ?? [:]).sorted(by: { $0.key < $1.key })
                if sortedMap.isEmpty {
                    Text("No symbol mappings defined.")
                        .font(.caption).foregroundStyle(.secondary)
                        .padding(.vertical, 8)
                } else {
                    ScrollView(.horizontal, showsIndicators: true) {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 16) {
                                Text("Portfolio Symbol").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 140, alignment: .leading)
                                Text("").frame(width: 20)
                                Text("Mapped Ticker").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 140, alignment: .leading)
                                Spacer().frame(width: 40) // Actions
                            }
                            Divider()
                            ForEach(sortedMap, id: \.key) { from, to in
                                HStack(spacing: 16) {
                                    Text(from).fontWeight(.bold).frame(width: 140, alignment: .leading).lineLimit(1)
                                    Image(systemName: "arrow.right").foregroundStyle(.secondary).frame(width: 20)
                                    Text(to).font(.body).monospacedDigit().foregroundStyle(.blue).frame(width: 140, alignment: .leading).lineLimit(1)
                                    
                                    Button(role: .destructive) { removeMap(from) } label: { Image(systemName: "trash") }
                                        .buttonStyle(.borderless).foregroundStyle(.red)
                                        .frame(width: 40, alignment: .trailing)
                                }
                                Divider()
                            }
                        }
                        .padding(.bottom, 8)
                    }
                }
            }
            
            SettingsCard(title: "Exclude a Symbol", icon: "xmark.circle", iconColor: .red) {
                Text("Excluded symbols are skipped during portfolio calculations and data fetches.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                
                HStack(alignment: .bottom, spacing: 16) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Symbol to Exclude").font(.caption2).foregroundStyle(.secondary)
                        TextField("e.g. TEST-SYM", text: $excludeSym)
                            .textFieldStyle(.roundedBorder)
                    }
                    Button("Exclude") {
                        let s = excludeSym.trimmingCharacters(in: .whitespaces).uppercased()
                        guard !s.isEmpty else { return }
                        excludeSym = ""
                        let excluded = settings?.userExcludedSymbols ?? []
                        setExcluded(Array(Set(excluded + [s])))
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .disabled(excludeSym.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                .padding(.top, 4)
            }
            
            SettingsCard(title: "Excluded Symbols", icon: "xmark.circle", iconColor: .red) {
                let excluded = (settings?.userExcludedSymbols ?? []).sorted()
                if excluded.isEmpty {
                    Text("No excluded symbols.")
                        .font(.caption).foregroundStyle(.secondary)
                        .padding(.vertical, 8)
                } else {
                    FlowChipsRemovable(items: excluded, backgroundColor: Color.red.opacity(0.1), borderColor: Color.red.opacity(0.3), textColor: .red) { sym in
                        setExcluded(excluded.filter { $0 != sym })
                    }
                }
            }
        }
    }
    private func addMap() {
        var m = settings?.userSymbolMap ?? [:]
        m[mapFrom.trimmingCharacters(in: .whitespaces).uppercased()] = mapTo.trimmingCharacters(in: .whitespaces).uppercased()
        mapFrom = ""; mapTo = ""
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

    @State private var isEditing = false

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
            if isEditing {
                SettingsCard(title: sym.isEmpty ? "Add Override" : "Edit Override", icon: "pencil.and.outline", iconColor: .green) {
                    Text("Set a manual price, asset type, or any metadata field for a symbol.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    
                    #if os(macOS)
                    let cols = [GridItem(.adaptive(minimum: 160), spacing: 16)]
                    #else
                    let cols = [GridItem(.adaptive(minimum: 140), spacing: 16)]
                    #endif
                    
                    LazyVGrid(columns: cols, alignment: .leading, spacing: 12) {
                        labeled("Symbol") { TextField("AAPL", text: $sym).textFieldStyle(.roundedBorder).disabled(!sym.isEmpty && overrides.contains(where: { $0.symbol == sym })) }
                        labeled("Price") { TextField("0.00", text: $price).textFieldStyle(.roundedBorder) }
                        labeled("Asset Type") { TextField("Select...", text: $assetType).textFieldStyle(.roundedBorder) }
                        labeled("Sector") { TextField("Select...", text: $sector).textFieldStyle(.roundedBorder) }
                        labeled("Country") { TextField("Select...", text: $geo).textFieldStyle(.roundedBorder) }
                        labeled("Industry") { TextField("Select...", text: $industry).textFieldStyle(.roundedBorder) }
                        labeled("Market") { TextField("NASDAQ", text: $exchange).textFieldStyle(.roundedBorder) }
                    }
                    .padding(.top, 4)
                    
                    HStack {
                        Button("Cancel") { isEditing = false; clear() }.buttonStyle(.bordered)
                        Spacer()
                        Button("Save Override") { add(); isEditing = false }
                            .buttonStyle(.borderedProminent)
                            .disabled(sym.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                    .padding(.top, 8)
                }
            } else {
                HStack {
                    Spacer()
                    Button(action: { clear(); isEditing = true }) {
                        HStack { Image(systemName: "plus"); Text("Add New Override") }
                    }.buttonStyle(.borderedProminent).tint(.green)
                }
            }
            
            SettingsCard(title: "Active Overrides", icon: "slider.horizontal.3", iconColor: .green) {
                if overrides.isEmpty {
                    Text("No manual overrides defined.")
                        .font(.caption).foregroundStyle(.secondary)
                        .padding(.vertical, 8)
                } else {
                    ScrollView(.horizontal, showsIndicators: true) {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 16) {
                                Text("Symbol").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 80, alignment: .leading)
                                Text("Price").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 80, alignment: .trailing)
                                Text("Asset Type").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 100, alignment: .leading)
                                Text("Sector").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 120, alignment: .leading)
                                Text("Country").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 80, alignment: .leading)
                                Text("Industry").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 120, alignment: .leading)
                                Text("Market").font(.caption).fontWeight(.bold).foregroundStyle(.secondary).frame(width: 80, alignment: .leading)
                                Spacer().frame(width: 60) // Actions
                            }
                            Divider()
                            ForEach(overrides, id: \.symbol) { o in
                                HStack(spacing: 16) {
                                    Text(o.symbol).fontWeight(.bold).frame(width: 80, alignment: .leading).lineLimit(1)
                                    Text(o.price.map { Fmt.number($0) } ?? "—").monospacedDigit().frame(width: 80, alignment: .trailing)
                                    
                                    Text(o.meta["asset_type"] ?? "—").font(.caption).frame(width: 100, alignment: .leading).lineLimit(1)
                                    Text(o.meta["sector"] ?? "—").font(.caption).frame(width: 120, alignment: .leading).lineLimit(1)
                                    Text(o.meta["geography"] ?? "—").font(.caption).frame(width: 80, alignment: .leading).lineLimit(1)
                                    Text(o.meta["industry"] ?? "—").font(.caption).frame(width: 120, alignment: .leading).lineLimit(1)
                                    Text(o.meta["exchange"] ?? "—").font(.caption).frame(width: 80, alignment: .leading).lineLimit(1)
                                    
                                    HStack(spacing: 8) {
                                        Button { edit(o) } label: { Image(systemName: "pencil") }
                                            .buttonStyle(.borderless).foregroundStyle(.blue)
                                        Button(role: .destructive) { remove(o.symbol) } label: { Image(systemName: "trash") }
                                            .buttonStyle(.borderless).foregroundStyle(.red)
                                    }.frame(width: 60, alignment: .trailing)
                                }
                                Divider()
                            }
                        }
                        .padding(.bottom, 8)
                    }
                }
            }
            
            ValuationOverridesSettings(vm: vm, settings: settings)
        }
    }
    
    private func labeled<C: View>(_ label: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 2) { Text(label).font(.caption2).foregroundStyle(.secondary); content() }
    }

    private func clear() { sym = ""; price = ""; assetType = ""; sector = ""; geo = ""; industry = ""; exchange = "" }
    
    private func add() {
        var map = settings?.manualOverrides ?? [:]
        var obj: [String: JSONValue] = [:]
        if let p = Double(price) { obj["price"] = .double(p) }
        for (k, v) in [("asset_type", assetType), ("sector", sector), ("geography", geo), ("industry", industry), ("exchange", exchange)] where !v.isEmpty { obj[k] = .string(v) }
        map[sym.uppercased()] = .object(obj)
        clear()
        Task { await vm.update("manual_price_overrides", map) }
    }
    
    private func edit(_ o: (symbol: String, price: Double?, meta: [String: String])) {
        sym = o.symbol
        price = o.price.map { String($0) } ?? ""
        assetType = o.meta["asset_type"] ?? ""
        sector = o.meta["sector"] ?? ""
        geo = o.meta["geography"] ?? ""
        industry = o.meta["industry"] ?? ""
        exchange = o.meta["exchange"] ?? ""
        isEditing = true
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

    @State private var isEditing = false

    private var overrides: [(symbol: String, values: [String: Double])] {
        (settings?.valuationOverrides ?? [:]).sorted(by: { $0.key < $1.key }).map { (symbol: $0.key, values: $0.value) }
    }

    var body: some View {
        VStack(spacing: 16) {
            if isEditing {
                SettingsCard(title: sym.isEmpty ? "Customize Valuation" : "Edit Valuation", icon: "pencil.and.outline", iconColor: .purple) {
                    #if os(macOS)
                    let cols = [GridItem(.adaptive(minimum: 160), spacing: 16)]
                    #else
                    let cols = [GridItem(.adaptive(minimum: 140), spacing: 16)]
                    #endif
                    
                    labeled("Symbol") { TextField("e.g. AAPL", text: $sym).textFieldStyle(.roundedBorder).disabled(!sym.isEmpty && overrides.contains(where: { $0.symbol == sym })) }
                        .padding(.bottom, 8)
                    
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Discounted Cash Flow (DCF)").font(.caption).fontWeight(.bold).foregroundStyle(.cyan)
                        LazyVGrid(columns: cols, alignment: .leading, spacing: 12) {
                            labeled("Discount Rate %") { TextField("", text: $dcfDiscountRate).textFieldStyle(.roundedBorder) }
                            labeled("Growth Rate %") { TextField("", text: $dcfGrowthRate).textFieldStyle(.roundedBorder) }
                            labeled("Term. Growth %") { TextField("", text: $dcfTerminalGrowth).textFieldStyle(.roundedBorder) }
                            labeled("Proj. Years") { TextField("", text: $dcfProjectionYears).textFieldStyle(.roundedBorder) }
                            labeled("Base FCF") { TextField("", text: $dcfFcf).textFieldStyle(.roundedBorder) }
                            labeled("Target FCF Margin %") { TextField("", text: $targetFcfMargin).textFieldStyle(.roundedBorder) }
                        }
                    }
                    .padding().background(Color.cyan.opacity(0.1)).cornerRadius(8)
                    
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Graham's Formula").font(.caption).fontWeight(.bold).foregroundStyle(.orange)
                        LazyVGrid(columns: cols, alignment: .leading, spacing: 12) {
                            labeled("Graham EPS") { TextField("", text: $grahamEps).textFieldStyle(.roundedBorder) }
                            labeled("Graham Growth Rate") { TextField("", text: $grahamGrowthRate).textFieldStyle(.roundedBorder) }
                            labeled("Graham Bond Yield %") { TextField("", text: $grahamBondYield).textFieldStyle(.roundedBorder) }
                        }
                    }
                    .padding().background(Color.orange.opacity(0.1)).cornerRadius(8)
                    
                    HStack {
                        Button("Cancel") { isEditing = false; clear() }.buttonStyle(.bordered)
                        Spacer()
                        Button("Save Parameters") { add(); isEditing = false }
                            .buttonStyle(.borderedProminent)
                            .disabled(sym.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                    .padding(.top, 8)
                }
            } else {
                HStack {
                    Spacer()
                    Button(action: { clear(); isEditing = true }) {
                        HStack { Image(systemName: "plus"); Text("Add Valuation Override") }
                    }.buttonStyle(.borderedProminent).tint(.purple)
                }
            }

            SettingsCard(title: "Active Valuation Overrides", icon: "chart.line.uptrend.xyaxis", iconColor: .mint) {
                if overrides.isEmpty {
                    Text("No valuation overrides defined.")
                        .font(.caption).foregroundStyle(.secondary)
                        .padding(.vertical, 8)
                } else {
                    VStack(spacing: 16) {
                        ForEach(overrides, id: \.symbol) { o in
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Text(o.symbol).font(.headline).fontWeight(.black).foregroundStyle(.purple)
                                    Spacer()
                                    Button { edit(o) } label: { Image(systemName: "pencil") }
                                        .buttonStyle(.borderless).foregroundStyle(.blue)
                                    Button(role: .destructive) { remove(o.symbol) } label: { Image(systemName: "trash") }
                                        .buttonStyle(.borderless).foregroundStyle(.red)
                                }
                                
                                let dcfKeys = ["dcf_discount_rate", "dcf_growth_rate", "dcf_terminal_growth", "dcf_projection_years", "dcf_fcf", "target_fcf_margin"]
                                let dcfValues = o.values.filter { dcfKeys.contains($0.key) }
                                
                                let grahamKeys = ["graham_eps", "graham_growth_rate", "graham_bond_yield"]
                                let grahamValues = o.values.filter { grahamKeys.contains($0.key) }
                                
                                #if os(macOS)
                                let detailCols = [GridItem(.adaptive(minimum: 140), spacing: 8)]
                                #else
                                let detailCols = [GridItem(.adaptive(minimum: 120), spacing: 8)]
                                #endif

                                if !dcfValues.isEmpty {
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text("DCF Model").font(.caption2).fontWeight(.bold).foregroundStyle(.cyan)
                                        LazyVGrid(columns: detailCols, alignment: .leading, spacing: 8) {
                                            ForEach(dcfValues.sorted(by: { $0.key < $1.key }), id: \.key) { k, v in
                                                VStack(alignment: .leading, spacing: 2) {
                                                    Text(formatKey(k)).font(.caption2).foregroundStyle(.secondary)
                                                    Text(formatValue(k, v)).font(.caption).monospacedDigit().fontWeight(.bold)
                                                }
                                            }
                                        }
                                    }
                                    .padding(8).background(Color.cyan.opacity(0.1)).cornerRadius(8)
                                }
                                
                                if !grahamValues.isEmpty {
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text("Graham's Formula").font(.caption2).fontWeight(.bold).foregroundStyle(.orange)
                                        LazyVGrid(columns: detailCols, alignment: .leading, spacing: 8) {
                                            ForEach(grahamValues.sorted(by: { $0.key < $1.key }), id: \.key) { k, v in
                                                VStack(alignment: .leading, spacing: 2) {
                                                    Text(formatKey(k)).font(.caption2).foregroundStyle(.secondary)
                                                    Text(formatValue(k, v)).font(.caption).monospacedDigit().fontWeight(.bold)
                                                }
                                            }
                                        }
                                    }
                                    .padding(8).background(Color.orange.opacity(0.1)).cornerRadius(8)
                                }
                            }
                            .padding()
                            .background(Color.primary.opacity(0.03))
                            .cornerRadius(12)
                        }
                    }
                }
            }
        }
    }
    
    private func labeled<C: View>(_ label: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 2) { Text(label).font(.caption2).foregroundStyle(.secondary); content() }
    }
    
    private func formatKey(_ key: String) -> String {
        switch key {
        case "dcf_discount_rate": return "Discount Rate"
        case "dcf_growth_rate": return "Growth Rate"
        case "dcf_terminal_growth": return "Term. Growth"
        case "dcf_projection_years": return "Proj. Years"
        case "dcf_fcf": return "Base FCF"
        case "target_fcf_margin": return "Target FCF Margin"
        case "graham_eps": return "Graham EPS"
        case "graham_growth_rate": return "Growth Rate"
        case "graham_bond_yield": return "Bond Yield"
        default: return key
        }
    }
    
    private func formatValue(_ key: String, _ val: Double) -> String {
        let isPercent = ["dcf_discount_rate", "dcf_growth_rate", "dcf_terminal_growth", "target_fcf_margin"].contains(key)
        if isPercent { return String(format: "%.2f%%", val * 100) }
        if key == "graham_bond_yield" { return "\(val)%" }
        if key == "dcf_fcf" { return Fmt.number(val) }
        return String(format: "%.2f", val)
    }

    private func edit(_ o: (symbol: String, values: [String: Double])) {
        sym = o.symbol
        let v = o.values
        
        func formatPct(_ key: String) -> String { v[key].map { String(format: "%.2f", $0 * 100) } ?? "" }
        func formatRaw(_ key: String) -> String { v[key].map { String(format: "%.2f", $0) } ?? "" }
        func formatExact(_ key: String) -> String { v[key].map { String($0) } ?? "" }
        
        dcfDiscountRate = formatPct("dcf_discount_rate")
        dcfGrowthRate = formatPct("dcf_growth_rate")
        dcfTerminalGrowth = formatPct("dcf_terminal_growth")
        targetFcfMargin = formatPct("target_fcf_margin")
        dcfProjectionYears = formatRaw("dcf_projection_years")
        dcfFcf = formatExact("dcf_fcf")
        
        grahamEps = formatRaw("graham_eps")
        grahamGrowthRate = formatRaw("graham_growth_rate")
        grahamBondYield = formatExact("graham_bond_yield")
        isEditing = true
    }

    private func clear() {
        sym = ""; dcfDiscountRate = ""; dcfGrowthRate = ""; dcfTerminalGrowth = ""; dcfProjectionYears = ""; dcfFcf = ""; targetFcfMargin = ""
        grahamEps = ""; grahamGrowthRate = ""; grahamBondYield = ""
    }
    
    private func add() {
        var map = settings?.valuationOverrides ?? [:]
        var vals: [String: Double] = [:]
        func put(_ key: String, _ text: String, isPct: Bool = false) {
            guard let v = Double(text) else { return }
            vals[key] = isPct ? v / 100 : v
        }
        put("dcf_discount_rate", dcfDiscountRate, isPct: true)
        put("dcf_growth_rate", dcfGrowthRate, isPct: true)
        put("dcf_terminal_growth", dcfTerminalGrowth, isPct: true)
        put("dcf_projection_years", dcfProjectionYears)
        put("dcf_fcf", dcfFcf)
        put("target_fcf_margin", targetFcfMargin, isPct: true)
        put("graham_eps", grahamEps)
        put("graham_growth_rate", grahamGrowthRate)
        put("graham_bond_yield", grahamBondYield)
        map[sym.uppercased()] = vals
        clear()
        Task { await vm.update("valuation_overrides", map) }
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
            webhookCard
            ibkrCard
            serverCard
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
        SettingsCard(title: "Benchmarks", icon: "chart.line.uptrend.xyaxis", iconColor: .purple) {
            Text("Select indices and specific symbols to compare your portfolio performance against.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            #if os(macOS)
            let cols = [GridItem(.adaptive(minimum: 160), spacing: 8)]
            #else
            let cols = [GridItem(.adaptive(minimum: 140), spacing: 8)]
            #endif
            
            LazyVGrid(columns: cols, alignment: .leading, spacing: 8) {
                ForEach(presetBenchmarks, id: \.self) { b in
                    let on = appState.benchmarks.contains(b)
                    Button { toggleBenchmark(b) } label: {
                        HStack {
                            Image(systemName: on ? "checkmark.square.fill" : "square")
                                .foregroundStyle(on ? Color.purple : .secondary)
                            Text(b).font(.caption).fontWeight(.medium)
                            Spacer()
                        }
                        .padding(8)
                        .background(on ? Color.purple.opacity(0.1) : Color.primary.opacity(0.05))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(on ? Color.purple.opacity(0.3) : Color.clear, lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.top, 4)
            
            Divider().padding(.vertical, 4)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Custom Ticker").font(.caption2).foregroundStyle(.secondary)
                HStack {
                    TextField("e.g. AAPL", text: $benchmarksText)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 200)
                    Button {
                        let t = benchmarksText.trimmingCharacters(in: .whitespaces).uppercased()
                        guard !t.isEmpty, !appState.benchmarks.contains(t) else { return }
                        benchmarksText = ""; appState.setBenchmarks(appState.benchmarks + [t])
                    } label: {
                        Image(systemName: "plus")
                    }
                    .buttonStyle(.bordered)
                    .disabled(benchmarksText.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                
                let custom = appState.benchmarks.filter { !presetBenchmarks.contains($0) }
                if !custom.isEmpty {
                    FlowChipsRemovable(items: custom, backgroundColor: Color.purple.opacity(0.1), borderColor: Color.purple.opacity(0.3), textColor: .purple) { sym in
                        toggleBenchmark(sym)
                    }
                    .padding(.top, 4)
                }
            }
        }
    }

    private var webhookCard: some View {
        SettingsCard(title: "Webhook Integration", icon: "waveform.path.ecg", iconColor: .cyan) {
            Text("Trigger a background data refresh externally by providing the secret key.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            HStack(spacing: 12) {
                TextField("Enter Webhook Secret", text: $refreshSecret)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 300)
                Button("Test") { Task { await vm.triggerRefresh(secret: refreshSecret) } }
                    .buttonStyle(.bordered)
            }
            .padding(.top, 4)
        }
    }

    private var ibkrCard: some View {
        SettingsCard(title: "Interactive Brokers Sync", icon: "slider.horizontal.3", iconColor: .blue) {
            Text("Sync transactions using IBKR Flex Web Service. Requires an active Activity Flex Query.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                
            #if os(macOS)
            let cols = [GridItem(.adaptive(minimum: 200), spacing: 16)]
            #else
            let cols = [GridItem(.adaptive(minimum: 140), spacing: 16)]
            #endif
            
            LazyVGrid(columns: cols, alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Flex Token").font(.caption2).foregroundStyle(.secondary)
                    TextField("Token", text: $ibkrToken).textFieldStyle(.roundedBorder)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Query ID").font(.caption2).foregroundStyle(.secondary)
                    TextField("ID", text: $ibkrQuery).textFieldStyle(.roundedBorder)
                }
            }
            .padding(.top, 4)
            
            HStack(spacing: 12) {
                Button("Save Credentials") { Task { await vm.update("ibkr_token", ibkrToken); await vm.update("ibkr_query_id", ibkrQuery) } }
                    .buttonStyle(.borderedProminent)
                Button("Sync Now") { Task { await vm.syncIbkr() } }
                    .buttonStyle(.bordered)
            }
            .padding(.top, 4)
        }
    }

    private var serverCard: some View {
        SettingsCard(title: "Advanced System Settings", icon: "gearshape", iconColor: .gray) {
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Backend Base URL").font(.caption2).foregroundStyle(.secondary)
                    HStack(spacing: 12) {
                        TextField("Base URL", text: $serverURL).textFieldStyle(.roundedBorder).frame(maxWidth: 300)
                        Button("Save") { APIConfig.baseURL = serverURL }.buttonStyle(.bordered)
                    }
                }
                Divider()
                HStack {
                    Text("Clear Server Cache").font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                    Button("Clear Cache") { Task { await vm.clearCache() } }.buttonStyle(.bordered)
                }
            }
        }
    }

    private func toggleBenchmark(_ b: String) {
        if appState.benchmarks.contains(b) { appState.setBenchmarks(appState.benchmarks.filter { $0 != b }) }
        else { appState.setBenchmarks(appState.benchmarks + [b]) }
    }
}

private struct FlowChipsRemovable: View {
    let items: [String]
    var backgroundColor: Color = Color.secondary.opacity(0.1)
    var borderColor: Color = Color.secondary.opacity(0.3)
    var textColor: Color = .primary
    let onRemove: (String) -> Void
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
            SettingsCard(title: "Profile Information", icon: "person.crop.circle", iconColor: .cyan) {
                Text("Identifiers and display name shown across the app.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                
                if let user = auth.currentUser {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack(spacing: 16) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Username").font(.caption2).foregroundStyle(.secondary)
                                Text(user.username)
                                    .font(.system(.body, design: .monospaced))
                                    .padding(.horizontal, 12).padding(.vertical, 8)
                                    .background(Color.primary.opacity(0.05))
                                    .cornerRadius(8)
                            }
                            VStack(alignment: .leading, spacing: 4) {
                                Text("User ID").font(.caption2).foregroundStyle(.secondary)
                                Text("\(user.id)")
                                    .font(.system(.body, design: .monospaced))
                                    .padding(.horizontal, 12).padding(.vertical, 8)
                                    .background(Color.primary.opacity(0.05))
                                    .cornerRadius(8)
                            }
                        }
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Alias (Display Name)").font(.caption2).foregroundStyle(.secondary)
                            HStack {
                                TextField("e.g. My Portfolio", text: $alias)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(maxWidth: 300)
                                Button("Save") { Task { await vm.updateProfile(alias: alias) } }
                                    .buttonStyle(.bordered)
                                    .disabled(alias == user.alias)
                            }
                            HStack(spacing: 4) {
                                Image(systemName: "info.circle").font(.caption2).foregroundStyle(.secondary)
                                Text("This name will be displayed in the user menu. Leave empty to use username.")
                                    .font(.system(size: 11)).foregroundStyle(.secondary)
                            }.padding(.top, 2)
                        }
                    }
                    .padding(.top, 4)
                }
            }
            
            SettingsCard(title: "Security", icon: "lock.shield", iconColor: .orange) {
                Text("Change your login password.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                
                Button("Change Password...") { showPassword = true }
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 4)
            }
            
            HStack(spacing: 16) {
                // Sign Out
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "iphone").font(.headline)
                        Text("Sign Out Device").font(.headline)
                    }
                    Text("End your current session on this device.").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button(action: { auth.logout() }) {
                        HStack {
                            Image(systemName: "arrow.right.square")
                            Text("Sign Out")
                        }.frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.primary.opacity(0.03))
                .cornerRadius(12)
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.1), lineWidth: 1))
                
                // Danger Zone
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill").font(.headline).foregroundStyle(.red)
                        Text("Danger Zone").font(.headline).foregroundStyle(.red)
                    }
                    Text("Permanently delete your account and all associated data.").font(.caption).foregroundStyle(.red.opacity(0.8))
                    Spacer()
                    Button(action: { confirmDelete = true }) {
                        HStack {
                            Image(systemName: "trash")
                            Text("Delete Account")
                        }.frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.red.opacity(0.05))
                .cornerRadius(12)
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.red.opacity(0.2), lineWidth: 1))
            }
            .fixedSize(horizontal: false, vertical: true)
        }
        .onAppear { alias = auth.currentUser?.alias ?? "" }
        .sheet(isPresented: $showPassword) { ChangePasswordView().environmentObject(auth) }
        .alert("Delete account?", isPresented: $confirmDelete) {
            Button("Delete", role: .destructive) { Task { await vm.deleteAccount(); auth.logout() } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("This permanently deletes your account and all data. This cannot be undone.") }
    }
}
