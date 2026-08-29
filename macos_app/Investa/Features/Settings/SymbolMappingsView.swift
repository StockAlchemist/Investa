import SwiftUI

struct SymbolMappingsView: View {
    @ObservedObject var vm: SettingsViewModel
    let settings: AppSettings?

    var embedded: Bool = false
    @State private var mapFrom = ""
    @State private var mapTo = ""
    @State private var searchText = ""
    @State private var isAdding = false

    private var sortedMap: [(from: String, to: String)] {
        (settings?.userSymbolMap ?? [:]).sorted(by: { $0.key < $1.key }).map { (from: $0.key, to: $0.value) }
    }

    private var filteredMap: [(from: String, to: String)] {
        let q = searchText.trimmingCharacters(in: .whitespaces).lowercased()
        if q.isEmpty { return sortedMap }
        return sortedMap.filter {
            $0.from.lowercased().contains(q) || $0.to.lowercased().contains(q)
        }
    }

    var body: some View {
        Group {
            if embedded {
                mainContent
            } else {
                ScrollView {
                    mainContent
                        .padding(16)
                }
                .navigationTitle("Symbol Mappings")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
            }
        }
    }

    private var mainContent: some View {
        VStack(spacing: 20) {
            // Add Mapping Card
            addMappingCard

            // Active Mappings Section
            activeMappingsSection
        }
    }

    private var addMappingCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "plus.circle.fill")
                    .foregroundStyle(Color.blue)
                    .appFont(.title3)
                Text("Map a Symbol")
                    .appFont(.headline.bold())
                Spacer()
            }

            Text("Resolve non-standard or broker-specific tickers to a real Yahoo Finance symbol for live prices.")
                .appFont(.caption)
                .foregroundStyle(.secondary)

            VStack(spacing: 12) {
                #if os(iOS)
                VStack(spacing: 10) {
                    TextField("Portfolio Symbol (e.g. MY-FUND)", text: $mapFrom)
                        .textFieldStyle(.roundedBorder)
                        .uppercaseAutoCapitalization()
                        .autocorrectionDisabled()

                    HStack {
                        Image(systemName: "arrow.down")
                            .foregroundStyle(.secondary)
                        Text("maps to")
                            .appFont(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    TextField("Yahoo Ticker (e.g. VTSAX)", text: $mapTo)
                        .textFieldStyle(.roundedBorder)
                        .uppercaseAutoCapitalization()
                        .autocorrectionDisabled()
                }
                #else
                HStack(spacing: 12) {
                    TextField("Portfolio Symbol (e.g. MY-FUND)", text: $mapFrom)
                        .textFieldStyle(.roundedBorder)

                    Image(systemName: "arrow.right")
                        .foregroundStyle(.secondary)

                    TextField("Yahoo Ticker (e.g. VTSAX)", text: $mapTo)
                        .textFieldStyle(.roundedBorder)
                }
                #endif

                Button {
                    addMapping()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "link.badge.plus")
                        Text("Create Mapping")
                    }
                    .frame(maxWidth: .infinity)
                    .fontWeight(.semibold)
                    .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .disabled(mapFrom.trimmingCharacters(in: .whitespaces).isEmpty || mapTo.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(18)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        )
    }

    private var activeMappingsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Active Mappings")
                    .appFont(.headline.bold())

                Spacer()

                if !sortedMap.isEmpty {
                    Text("\(sortedMap.count) total")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if !sortedMap.isEmpty {
                // Search field
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Filter mappings...", text: $searchText)
                        .textFieldStyle(.plain)
                    if !searchText.isEmpty {
                        Button { searchText = "" } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }

            if sortedMap.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "map")
                        .font(.system(size: 32))
                        .foregroundStyle(.secondary.opacity(0.4))
                        .padding(.vertical, 8)
                    Text("No symbol mappings configured.")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else if filteredMap.isEmpty {
                Text("No mappings match \"\(searchText)\".")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 16)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(filteredMap, id: \.from) { item in
                        mappingRow(from: item.from, to: item.to)
                    }
                }
            }
        }
    }

    private func mappingRow(from: String, to: String) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("PORTFOLIO")
                    .appFont(.system(size: 9, weight: .bold))
                    .foregroundStyle(.secondary)
                Text(from)
                    .appFont(.headline.bold())
                    .foregroundStyle(.primary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Image(systemName: "arrow.right")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(Color.blue)

            VStack(alignment: .trailing, spacing: 2) {
                Text("YAHOO FINANCE")
                    .appFont(.system(size: 9, weight: .bold))
                    .foregroundStyle(.secondary)
                Text(to)
                    .appFont(.headline.bold().monospacedDigit())
                    .foregroundStyle(Color.blue)
            }
            .frame(maxWidth: .infinity, alignment: .trailing)

            Button(role: .destructive) {
                removeMapping(from)
            } label: {
                Image(systemName: "trash")
                    .appFont(.subheadline)
                    .foregroundStyle(.red)
                    .padding(8)
                    .background(Color.red.opacity(0.08), in: Circle())
            }
            .buttonStyle(.plain)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.primary.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.06), lineWidth: 1)
        )
    }

    private func addMapping() {
        let f = mapFrom.trimmingCharacters(in: .whitespaces).uppercased()
        let t = mapTo.trimmingCharacters(in: .whitespaces).uppercased()
        guard !f.isEmpty && !t.isEmpty else { return }

        var m = settings?.userSymbolMap ?? [:]
        m[f] = t
        mapFrom = ""
        mapTo = ""

        Task {
            guard await vm.update("user_symbol_map", m, note: "Mapped \(f) ➔ \(t)") else { return }
            ToastManager.shared.show(message: "Mapped \(f) ➔ \(t)", style: .success)
        }
    }

    private func removeMapping(_ from: String) {
        var m = settings?.userSymbolMap ?? [:]
        m.removeValue(forKey: from)
        Task {
            await vm.update("user_symbol_map", m, note: "Removed mapping for \(from)")
            ToastManager.shared.show(message: "Removed mapping for \(from)", style: .info)
        }
    }
}
