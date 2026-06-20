import SwiftUI

/// Persistent symbol-search box (mirrors the web `StockSearchBar`): a bordered
/// input with a debounced `/search` lookup and a floating results dropdown.
/// Selecting a result opens `StockDetailView` as a sheet. Used in the global
/// control bar across macOS, iPad, and iPhone.
struct StockSearchBar: View {
    let currency: String
    var placeholder: String = "Search symbol…"

    @State private var query = ""
    @State private var results: [SymbolSearchResult] = []
    @State private var loading = false
    @State private var open = false
    @State private var selected: SymbolID?
    @State private var searchTask: Task<Void, Never>?
    @State private var closeWork: DispatchWorkItem?
    @FocusState private var focused: Bool

    private var trimmed: String { query.trimmingCharacters(in: .whitespaces) }
    /// Show the dropdown when focused and there's something to display.
    private var showDropdown: Bool {
        open && (!results.isEmpty || (!trimmed.isEmpty && !loading))
    }

    /// Collapsed to an icon-sized pill when idle; expands once focused or typed
    /// into (mirrors the web search bar). Keeps the bar compact in portrait.
    private var expanded: Bool { focused || !query.isEmpty }
    private var collapsedWidth: CGFloat { 38 }
    private var expandedWidth: CGFloat { 230 }

    var body: some View {
        field
            .overlay(alignment: .topLeading) {
                if showDropdown { dropdown.offset(y: 40) }
            }
            .sheet(item: $selected) { StockDetailView(symbol: $0.id, currency: currency) }
    }

    // MARK: - Input field

    private var field: some View {
        HStack(spacing: 6) {
            Group {
                if loading {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "magnifyingglass").font(.caption).foregroundStyle(.secondary)
                }
            }
            .frame(width: 16, height: 16)

            // The text field stays in the hierarchy (so it can be focused on tap)
            // but is hidden and clipped away while the bar is collapsed.
            TextField(placeholder, text: $query)
                .textFieldStyle(.plain)
                .font(.callout)
                .focused($focused)
                .autocorrectionDisabled()
                #if os(iOS)
                .textInputAutocapitalization(.characters)
                #endif
                .onSubmit(submit)
                .onChange(of: query) { _, q in open = true; runSearch(q) }
                .onChange(of: focused) { _, isFocused in handleFocus(isFocused) }
                .opacity(expanded ? 1 : 0)

            if expanded && !query.isEmpty {
                Button {
                    query = ""; results = []; focused = true
                } label: {
                    Image(systemName: "xmark.circle.fill").font(.caption).foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .frame(width: expanded ? expandedWidth : collapsedWidth, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.separator.opacity(0.6), lineWidth: 1))
        .contentShape(Rectangle())
        .onTapGesture { if !focused { focused = true } }
        .animation(.easeInOut(duration: 0.2), value: expanded)
    }

    // MARK: - Results dropdown

    private var dropdown: some View {
        VStack(alignment: .leading, spacing: 0) {
            if results.isEmpty {
                // No API matches — offer a direct lookup of the typed text.
                Button { openDetail(trimmed.uppercased()) } label: {
                    HStack(spacing: 10) {
                        Image(systemName: "chart.bar").foregroundStyle(.tint)
                        Text(trimmed.uppercased()).fontWeight(.bold)
                        Spacer()
                    }
                    .padding(.horizontal, 12).padding(.vertical, 10)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            } else {
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(results) { resultRow($0) }
                    }
                }
                .frame(maxHeight: 320)
            }
        }
        .frame(width: 300)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.separator.opacity(0.5), lineWidth: 1))
        .shadow(color: .black.opacity(0.2), radius: 16, y: 6)
        .zIndex(100)
    }

    private func resultRow(_ r: SymbolSearchResult) -> some View {
        Button { openDetail(r.symbol) } label: {
            HStack(spacing: 10) {
                StockIcon(symbol: r.symbol, size: 26)
                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 6) {
                        Text(r.symbol).font(.callout.bold())
                        typeBadge(r.type)
                    }
                    if !r.name.isEmpty {
                        Text(r.name).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer()
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder private func typeBadge(_ type: String) -> some View {
        if !type.isEmpty {
            Text(typeLabel(type))
                .font(.system(size: 9, weight: .bold))
                .textCase(.uppercase)
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(typeColor(type).opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                .foregroundStyle(typeColor(type))
        }
    }

    private func typeLabel(_ type: String) -> String {
        switch type.lowercased() {
        case "equity", "stock": return "Equity"
        case "etf": return "ETF"
        case "mutualfund", "mutual fund": return "Fund"
        case "index": return "Index"
        case "crypto", "cryptocurrency": return "Crypto"
        default: return type
        }
    }

    private func typeColor(_ type: String) -> Color {
        switch type.lowercased() {
        case "equity", "stock": return .indigo
        case "etf": return .cyan
        case "mutualfund", "mutual fund": return .purple
        case "index": return .orange
        case "crypto", "cryptocurrency": return .orange
        default: return .secondary
        }
    }

    // MARK: - Behavior

    private func handleFocus(_ isFocused: Bool) {
        closeWork?.cancel()
        if isFocused {
            open = true
        } else {
            // Delay so a click on a result row registers before the dropdown closes.
            let work = DispatchWorkItem { open = false }
            closeWork = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.18, execute: work)
        }
    }

    private func submit() {
        if let first = results.first {
            openDetail(first.symbol)
        } else if !trimmed.isEmpty {
            openDetail(trimmed.uppercased())
        }
    }

    private func openDetail(_ symbol: String) {
        guard !symbol.isEmpty else { return }
        closeWork?.cancel()
        selected = SymbolID(id: symbol)
        open = false
        query = ""
        results = []
        focused = false
    }

    private func runSearch(_ q: String) {
        searchTask?.cancel()
        let query = q.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty else { results = []; loading = false; return }
        loading = true
        searchTask = Task {
            try? await Task.sleep(nanoseconds: 280_000_000)
            if Task.isCancelled { return }
            let data: [SymbolSearchResult] = (try? await APIClient.shared.get(
                "/search", query: [URLQueryItem(name: "q", value: query)])) ?? []
            if Task.isCancelled { return }
            results = data
            loading = false
        }
    }
}
