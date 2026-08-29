import SwiftUI

/// Modal sheet for adding or editing a symbol's manual price or metadata override.
struct OverrideEditSheet: View {
    @Environment(\.dismiss) private var dismiss
    let initialSymbol: String?
    let initialPrice: Double?
    let initialMeta: [String: String]
    let onSave: (String, Double?, [String: String]) -> Void

    @State private var symbol: String = ""
    @State private var priceText: String = ""
    @State private var assetType: String = ""
    @State private var sector: String = ""
    @State private var geography: String = ""
    @State private var industry: String = ""
    @State private var exchange: String = ""

    private let assetTypes = ["Stock", "ETF", "Mutual Fund", "Crypto", "Cash", "Bond", "Commodity", "REIT"]
    private let commonExchanges = ["NASDAQ", "NYSE", "BKK", "AMEX", "LSE", "TSE", "HKEX"]

    init(
        symbol: String? = nil,
        price: Double? = nil,
        meta: [String: String] = [:],
        onSave: @escaping (String, Double?, [String: String]) -> Void
    ) {
        self.initialSymbol = symbol
        self.initialPrice = price
        self.initialMeta = meta
        self.onSave = onSave
    }

    private var isEditing: Bool { initialSymbol != nil && !initialSymbol!.isEmpty }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    HStack {
                        Text("Symbol")
                            .appFont(.body)
                        Spacer()
                        if isEditing {
                            Text(symbol)
                                .appFont(.headline.bold())
                                .foregroundStyle(.secondary)
                        } else {
                            TextField("e.g. AAPL, BTC-USD", text: $symbol)
                                .multilineTextAlignment(.trailing)
                                .uppercaseAutoCapitalization()
                                .autocorrectionDisabled()
                        }
                    }

                    HStack {
                        Text("Manual Price")
                            .appFont(.body)
                        Spacer()
                        TextField("0.00 (optional)", text: $priceText)
                            .multilineTextAlignment(.trailing)
                            #if os(iOS)
                            .keyboardType(.decimalPad)
                            #endif
                    }
                } header: {
                    Text("Basic Information")
                } footer: {
                    Text("Leave price empty to use automated live market price feeds.")
                }

                Section {
                    Picker("Asset Type", selection: $assetType) {
                        Text("Select...").tag("")
                        ForEach(assetTypes, id: \.self) { type in
                            Text(type).tag(type)
                        }
                    }

                    HStack {
                        Text("Sector")
                            .appFont(.body)
                        Spacer()
                        TextField("e.g. Technology", text: $sector)
                            .multilineTextAlignment(.trailing)
                    }

                    HStack {
                        Text("Industry")
                            .appFont(.body)
                        Spacer()
                        TextField("e.g. Consumer Electronics", text: $industry)
                            .multilineTextAlignment(.trailing)
                    }
                } header: {
                    Text("Classification")
                }

                Section {
                    HStack {
                        Text("Country / Region")
                            .appFont(.body)
                        Spacer()
                        TextField("e.g. United States, Thailand", text: $geography)
                            .multilineTextAlignment(.trailing)
                    }

                    Picker("Market / Exchange", selection: $exchange) {
                        Text("Select / Other").tag("")
                        ForEach(commonExchanges, id: \.self) { ex in
                            Text(ex).tag(ex)
                        }
                    }
                } header: {
                    Text("Market & Geography")
                }
            }
            .navigationTitle(isEditing ? "Edit Override" : "Add Override")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        save()
                        dismiss()
                    }
                    .disabled(symbol.trimmingCharacters(in: .whitespaces).isEmpty)
                    .fontWeight(.bold)
                }
            }
            .onAppear {
                symbol = initialSymbol ?? ""
                if let p = initialPrice, p > 0 {
                    priceText = String(p)
                }
                assetType = initialMeta["asset_type"] ?? ""
                sector = initialMeta["sector"] ?? ""
                geography = initialMeta["geography"] ?? ""
                industry = initialMeta["industry"] ?? ""
                exchange = initialMeta["exchange"] ?? ""
            }
        }
        #if os(macOS)
        .frame(minWidth: 440, minHeight: 460)
        #endif
    }

    private func save() {
        let sym = symbol.trimmingCharacters(in: .whitespaces).uppercased()
        guard !sym.isEmpty else { return }

        let priceVal = Double(priceText.trimmingCharacters(in: .whitespaces))
        var meta: [String: String] = [:]
        if !assetType.isEmpty { meta["asset_type"] = assetType }
        if !sector.isEmpty { meta["sector"] = sector.trimmingCharacters(in: .whitespaces) }
        if !geography.isEmpty { meta["geography"] = geography.trimmingCharacters(in: .whitespaces) }
        if !industry.isEmpty { meta["industry"] = industry.trimmingCharacters(in: .whitespaces) }
        if !exchange.isEmpty { meta["exchange"] = exchange.trimmingCharacters(in: .whitespaces) }

        onSave(sym, priceVal, meta)
    }
}
