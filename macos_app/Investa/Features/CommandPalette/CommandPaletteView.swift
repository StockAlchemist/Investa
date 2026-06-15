import SwiftUI

/// ⌘K command palette: navigate tabs and search symbols (mirrors CommandPalette.tsx).
struct CommandPaletteView: View {
    @Environment(\.dismiss) private var dismiss
    let onNavigate: (AppSection) -> Void
    let onOpenSettings: () -> Void
    let onOpenStock: (String) -> Void

    @State private var query = ""
    @State private var results: [SymbolSearchResult] = []
    @State private var loading = false
    @State private var searchTask: Task<Void, Never>?
    @FocusState private var focused: Bool

    private struct NavCommand: Identifiable { let id = UUID(); let section: AppSection?; let label: String; let icon: String; let group: String; let settings: Bool }
    private let navCommands: [NavCommand] = [
        .init(section: .performance, label: "Dashboard", icon: "rectangle.3.group", group: "Portfolio", settings: false),
        .init(section: .allocation, label: "Portfolio", icon: "chart.pie", group: "Portfolio", settings: false),
        .init(section: .assetChange, label: "Performance", icon: "chart.line.uptrend.xyaxis", group: "Portfolio", settings: false),
        .init(section: .transactions, label: "Transactions", icon: "arrow.left.arrow.right", group: "Portfolio", settings: false),
        .init(section: .dividend, label: "Income", icon: "dollarsign", group: "Portfolio", settings: false),
        .init(section: .capitalGains, label: "Capital Gains", icon: "chart.bar", group: "Portfolio", settings: false),
        .init(section: .market, label: "Screener", icon: "magnifyingglass", group: "Tools", settings: false),
        .init(section: .watchlist, label: "Watchlist", icon: "star", group: "Tools", settings: false),
        .init(section: .markets, label: "Markets", icon: "globe", group: "Tools", settings: false),
        .init(section: .aiReview, label: "AI Insights", icon: "sparkles", group: "Tools", settings: false),
        .init(section: nil, label: "Settings", icon: "gearshape", group: "Settings", settings: true),
    ]

    private var filteredNav: [NavCommand] {
        let q = query.lowercased()
        return q.isEmpty ? navCommands : navCommands.filter { $0.label.lowercased().contains(q) || $0.group.lowercased().contains(q) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search pages or symbols…", text: $query).textFieldStyle(.plain).font(.title3).focused($focused)
                if loading { ProgressView().controlSize(.small) }
                Text("esc").font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 5).padding(.vertical, 2)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 4))
            }
            .padding(16)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(["Portfolio", "Tools", "Settings"], id: \.self) { group in
                        let items = filteredNav.filter { $0.group == group }
                        if !items.isEmpty {
                            Text(group.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                                .padding(.horizontal, 12).padding(.top, 8)
                            ForEach(items) { cmd in navRow(cmd) }
                        }
                    }
                    if !results.isEmpty {
                        Text("STOCKS").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                            .padding(.horizontal, 12).padding(.top, 8)
                        ForEach(results) { stockRow($0) }
                    }
                }
                .padding(.bottom, 8)
            }
        }
        .frame(width: 560, height: 460)
        .onAppear { focused = true }
        .onChange(of: query) { _, q in search(q) }
        #if os(macOS)
        .onExitCommand { dismiss() }
        #endif
    }

    private func navRow(_ cmd: NavCommand) -> some View {
        Button {
            if cmd.settings { onOpenSettings() } else if let s = cmd.section { onNavigate(s) }
            dismiss()
        } label: {
            HStack { Image(systemName: cmd.icon).frame(width: 20).foregroundStyle(.secondary); Text(cmd.label); Spacer()
                Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.secondary) }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func stockRow(_ r: SymbolSearchResult) -> some View {
        Button { onOpenStock(r.symbol); dismiss() } label: {
            HStack {
                StockIcon(symbol: r.symbol, size: 20)
                Text(r.symbol).fontWeight(.bold).frame(width: 70, alignment: .leading)
                Text(r.name).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                if !r.type.isEmpty {
                    Text(r.type.uppercased()).font(.caption2.weight(.bold))
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(.tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 8).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func search(_ q: String) {
        searchTask?.cancel()
        let query = q.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { results = []; loading = false; return }
        loading = true
        searchTask = Task {
            try? await Task.sleep(nanoseconds: 280_000_000)
            if Task.isCancelled { return }
            let data: [SymbolSearchResult] = (try? await APIClient.shared.get("/search", query: [URLQueryItem(name: "q", value: query)])) ?? []
            if Task.isCancelled { return }
            results = data; loading = false
        }
    }
}
