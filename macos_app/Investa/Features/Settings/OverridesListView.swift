import SwiftUI

struct OverrideItem: Identifiable {
    var id: String { symbol }
    let symbol: String
    let price: Double?
    let meta: [String: String]
    let currency: String
}

struct OverridesListView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var embedded: Bool = false
    @State private var searchText = ""
    @State private var showingAddSheet = false
    @State private var editingItem: OverrideItem? = nil
    @State private var deletingSymbol: String? = nil
    @State private var showDeleteAlert = false

    private var overridesList: [OverrideItem] {
        (settings?.manualOverrides ?? [:]).sorted(by: { $0.key < $1.key }).map { sym, val in
            let currency = val["currency"]?.stringValue ?? (sym.hasSuffix(".BK") || sym.contains(":BKK") ? "THB" : "USD")
            if let p = val.doubleValue {
                return OverrideItem(symbol: sym, price: p > 0 ? p : nil, meta: [:], currency: currency)
            }
            var meta: [String: String] = [:]
            for k in ["asset_type", "sector", "geography", "industry", "exchange"] {
                if let s = val[k]?.stringValue { meta[k] = s }
            }
            let p = val["price"]?.doubleValue
            return OverrideItem(symbol: sym, price: (p != nil && p! > 0) ? p : nil, meta: meta, currency: currency)
        }
    }

    private var filteredOverrides: [OverrideItem] {
        let q = searchText.trimmingCharacters(in: .whitespaces).lowercased()
        if q.isEmpty { return overridesList }
        return overridesList.filter { item in
            item.symbol.lowercased().contains(q) ||
            (item.meta["sector"]?.lowercased().contains(q) ?? false) ||
            (item.meta["asset_type"]?.lowercased().contains(q) ?? false) ||
            (item.meta["geography"]?.lowercased().contains(q) ?? false) ||
            (item.meta["industry"]?.lowercased().contains(q) ?? false) ||
            (item.meta["exchange"]?.lowercased().contains(q) ?? false)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Search & Add Bar
            HStack(spacing: 12) {
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search symbols, sectors, asset types...", text: $searchText)
                        .textFieldStyle(.plain)
                    if !searchText.isEmpty {
                        Button { searchText = "" } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

                Button {
                    showingAddSheet = true
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "plus")
                        Text("Add")
                    }
                    .fontWeight(.semibold)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
            }
            .padding(.horizontal, embedded ? 0 : 16)
            .padding(.top, embedded ? 0 : 12)
            .padding(.bottom, 12)

            if filteredOverrides.isEmpty {
                emptyView
            } else {
                if embedded {
                    LazyVStack(spacing: 12) {
                        ForEach(filteredOverrides) { item in
                            overrideCard(item)
                        }
                    }
                } else {
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(filteredOverrides) { item in
                                overrideCard(item)
                            }
                        }
                        .padding(16)
                    }
                }
            }
        }
        #if os(iOS)
        // Standalone only: embedded, these would retitle the host column and
        // hang a stray "+" off the app's own toolbar. The in-view "Add" button
        // above already covers the embedded case.
        .modifier(OverridesNavigationChrome(embedded: embedded, showingAddSheet: $showingAddSheet))
        #endif
        .sheet(isPresented: $showingAddSheet) {
            OverrideEditSheet { sym, price, meta in
                saveOverride(symbol: sym, price: price, meta: meta)
            }
        }
        .sheet(item: $editingItem) { item in
            OverrideEditSheet(
                symbol: item.symbol,
                price: item.price,
                meta: item.meta
            ) { sym, price, meta in
                saveOverride(symbol: sym, price: price, meta: meta)
            }
        }
        .alert("Delete Override", isPresented: $showDeleteAlert, presenting: deletingSymbol) { sym in
            Button("Delete", role: .destructive) {
                removeOverride(sym)
            }
            Button("Cancel", role: .cancel) {}
        } message: { sym in
            Text("Are you sure you want to remove the manual override for \(sym)?")
        }
    }

    private func overrideCard(_ item: OverrideItem) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center) {
                HStack(spacing: 10) {
                    StockIcon(symbol: item.symbol, size: 36)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.symbol)
                            .appFont(.headline.bold())
                            .foregroundStyle(.primary)

                        if let ex = item.meta["exchange"], !ex.isEmpty {
                            Text(ex)
                                .appFont(.caption2.weight(.medium))
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Spacer()

                if let p = item.price, p > 0 {
                    let currSymbol = item.currency == "THB" ? "฿" : "$"
                    HStack(spacing: 2) {
                        Text("\(currSymbol)\(Fmt.number(p, fractionDigits: 4))")
                            .appFont(.subheadline.monospacedDigit().weight(.bold))
                            .foregroundStyle(.green)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.green.opacity(0.12), in: Capsule())
                    .overlay(Capsule().strokeBorder(Color.green.opacity(0.3), lineWidth: 1))
                } else {
                    Text("Auto Price")
                        .appFont(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.primary.opacity(0.05), in: Capsule())
                }
            }

            // Metadata Badges Flow
            let tags = metadataTags(item.meta)
            if !tags.isEmpty {
                FlowChipsView(tags: tags)
            }

            Divider().padding(.top, 2)

            // Actions
            HStack {
                Spacer()

                Button {
                    editingItem = item
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "pencil")
                        Text("Edit")
                    }
                    .appFont(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button(role: .destructive) {
                    deletingSymbol = item.symbol
                    showDeleteAlert = true
                } label: {
                    Image(systemName: "trash")
                        .appFont(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(.red)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    private func metadataTags(_ meta: [String: String]) -> [(title: String, icon: String, color: Color)] {
        var tags: [(title: String, icon: String, color: Color)] = []
        if let val = meta["asset_type"], !val.isEmpty { tags.append((val, "cube.fill", .blue)) }
        if let val = meta["sector"], !val.isEmpty { tags.append((val, "chart.pie.fill", .indigo)) }
        if let val = meta["geography"], !val.isEmpty { tags.append((val, "globe.americas.fill", .orange)) }
        if let val = meta["industry"], !val.isEmpty { tags.append((val, "building.2.fill", .purple)) }
        return tags
    }

    private var emptyView: some View {
        VStack(spacing: 12) {
            Image(systemName: "slider.horizontal.3")
                .font(.system(size: 44))
                .foregroundStyle(.secondary.opacity(0.5))
                .padding(.bottom, 4)

            Text(searchText.isEmpty ? "No Manual Overrides" : "No Matching Overrides")
                .appFont(.headline)

            Text(searchText.isEmpty
                 ? "You haven't set any manual prices or metadata overrides yet."
                 : "Try changing your search keywords.")
                .appFont(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 320)

            if searchText.isEmpty {
                Button {
                    showingAddSheet = true
                } label: {
                    Label("Add First Override", systemImage: "plus")
                        .fontWeight(.semibold)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .padding(.top, 8)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }

    private func saveOverride(symbol: String, price: Double?, meta: [String: String]) {
        var map = settings?.manualOverrides ?? [:]
        var obj: [String: JSONValue] = [:]
        if let p = price, p > 0 { obj["price"] = .double(p) }
        for (k, v) in meta where !v.isEmpty { obj[k] = .string(v) }
        map[symbol.uppercased()] = .object(obj)

        Task {
            guard await vm.update("manual_price_overrides", map, note: "Override saved for \(symbol)") else { return }
            ToastManager.shared.show(message: "Override saved for \(symbol)", style: .success)
        }
    }

    private func removeOverride(_ symbol: String) {
        var map = settings?.manualOverrides ?? [:]
        map.removeValue(forKey: symbol)
        Task {
            await vm.update("manual_price_overrides", map, note: "Override removed for \(symbol)")
            ToastManager.shared.show(message: "Override removed for \(symbol)", style: .info)
        }
    }
}

private struct FlowChipsView: View {
    let tags: [(title: String, icon: String, color: Color)]

    var body: some View {
        HStack(spacing: 6) {
            ForEach(Array(tags.enumerated()), id: \.offset) { _, tag in
                HStack(spacing: 4) {
                    Image(systemName: tag.icon)
                        .font(.system(size: 9))
                    Text(tag.title)
                        .appFont(.caption2.weight(.medium))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .foregroundStyle(tag.color)
                .background(tag.color.opacity(0.1), in: Capsule())
                .overlay(Capsule().strokeBorder(tag.color.opacity(0.2), lineWidth: 1))
            }
            Spacer()
        }
    }
}


#if os(iOS)
/// Navigation chrome that must apply only when OverridesListView owns its own
/// navigation stack - never when it is embedded in another column.
private struct OverridesNavigationChrome: ViewModifier {
    let embedded: Bool
    @Binding var showingAddSheet: Bool

    func body(content: Content) -> some View {
        if embedded {
            content
        } else {
            content
                .navigationTitle("Manual Overrides")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button {
                            showingAddSheet = true
                        } label: {
                            Image(systemName: "plus")
                        }
                    }
                }
        }
    }
}
#endif
