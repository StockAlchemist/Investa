import SwiftUI
import Charts

@MainActor
final class WatchlistViewModel: ObservableObject {
    @Published var watchlists: [WatchlistMeta] = []
    @Published var activeId: Int = 1
    @Published var items: [WatchlistItem] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: APIClient
    init(api: APIClient = .shared) { self.api = api }

    func loadLists() async {
        watchlists = (try? await api.get("/watchlists")) ?? []
        if !watchlists.contains(where: { $0.id == activeId }), let first = watchlists.first { activeId = first.id }
    }

    func loadItems(currency: String) async {
        isLoading = true; errorMessage = nil
        defer { isLoading = false }
        do {
            items = try await api.get("/watchlist",
                query: [URLQueryItem(name: "currency", value: currency), URLQueryItem(name: "id", value: String(activeId))])
        } catch let error as APIError {
            if case .unauthorized = error { return }
            errorMessage = error.errorDescription
        } catch { errorMessage = error.localizedDescription }
    }

    func add(symbol: String, note: String, currency: String) async {
        struct Body: Encodable { let symbol: String; let note: String; let watchlist_id: Int }
        let _: StatusResponse? = try? await api.send(method: "POST", path: "/watchlist",
            body: Body(symbol: symbol.uppercased(), note: note, watchlist_id: activeId))
        await loadItems(currency: currency)
    }
    func remove(symbol: String, currency: String) async {
        let _: StatusResponse? = try? await api.send(method: "DELETE", path: "/watchlist/\(symbol)",
            query: [URLQueryItem(name: "id", value: String(activeId))])
        items.removeAll { $0.symbol == symbol }
    }
    func createList(name: String, currency: String) async {
        struct Body: Encodable { let name: String }
        if let meta: WatchlistMeta = try? await api.send(method: "POST", path: "/watchlists", body: Body(name: name)) {
            await loadLists(); activeId = meta.id; await loadItems(currency: currency)
        }
    }
    func renameList(name: String) async {
        struct Body: Encodable { let name: String }
        let _: StatusResponse? = try? await api.send(method: "PUT", path: "/watchlists/\(activeId)", body: Body(name: name))
        await loadLists()
    }
    func deleteList(currency: String) async {
        let id = activeId
        let _: StatusResponse? = try? await api.send(method: "DELETE", path: "/watchlists/\(id)")
        await loadLists()
        await loadItems(currency: currency)
    }
}

private enum WLSort: String { case symbol, name, price, day, mktcap, pe, div, ai, intrinsic, note }

struct WatchlistView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = WatchlistViewModel()

    @State private var newSymbol = ""
    @State private var newNote = ""
    @State private var search = ""
    @State private var sortKey: WLSort = .symbol
    @State private var sortAsc = true
    @State private var creatingList = false
    @State private var newListName = ""
    @State private var renaming = false
    @State private var renameName = ""
    @State private var editingNoteSymbol: String?
    @State private var editNoteText = ""
    @State private var detail: SymbolID?

    private var cur: String { appState.displayCurrency }
    private var currentListName: String { viewModel.watchlists.first { $0.id == viewModel.activeId }?.name ?? "Watchlist" }

    private var sorted: [WatchlistItem] {
        func key(_ i: WatchlistItem) -> Double? {
            switch sortKey {
            case .price: return i.price; case .day: return i.dayChangePct; case .mktcap: return i.marketCap
            case .pe: return i.peRatio; case .div: return i.dividendYield; case .ai: return i.aiScore
            case .intrinsic: return i.intrinsicValue; default: return nil
            }
        }
        return viewModel.items.sorted { a, b in
            switch sortKey {
            case .symbol: return sortAsc ? a.symbol < b.symbol : a.symbol > b.symbol
            case .name: return sortAsc ? (a.name ?? "") < (b.name ?? "") : (a.name ?? "") > (b.name ?? "")
            case .note: return sortAsc ? a.note < b.note : a.note > b.note
            default:
                let av = key(a) ?? 0, bv = key(b) ?? 0
                return sortAsc ? av < bv : av > bv
            }
        }
    }
    private var filtered: [WatchlistItem] {
        let q = search.lowercased()
        return q.isEmpty ? sorted : sorted.filter { $0.symbol.lowercased().contains(q) || ($0.name?.lowercased().contains(q) ?? false) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Watchlist").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    listSelector
                    if !viewModel.items.isEmpty { WatchlistKpiStrip(items: viewModel.items) }
                    card
                }
                .padding(20)
            }
        }
        .macMinSize(width: 860, height: 560)
        .task { await viewModel.loadLists(); await viewModel.loadItems(currency: cur) }
        .onChange(of: viewModel.activeId) { _, _ in Task { await viewModel.loadItems(currency: cur) } }
        .onChange(of: cur) { _, _ in Task { await viewModel.loadItems(currency: cur) } }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in
            Task { await viewModel.loadItems(currency: cur) }
        }
        .sheet(item: $detail) { StockDetailView(symbol: $0.id, currency: cur) }
    }

    // MARK: - List selector

    private var listSelector: some View {
        #if os(iOS)
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(viewModel.watchlists) { wl in
                    Button { viewModel.activeId = wl.id } label: {
                        Text(wl.name).font(.callout.weight(.medium))
                            .padding(.horizontal, 12).padding(.vertical, 6)
                            .background(viewModel.activeId == wl.id ? Color.accentColor : Color.gray.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
                            .foregroundStyle(viewModel.activeId == wl.id ? .white : .primary)
                    }.buttonStyle(.plain)
                }
                if creatingList {
                    TextField("List name", text: $newListName).textFieldStyle(.roundedBorder).frame(width: 130)
                    Button { Task { await viewModel.createList(name: newListName, currency: cur); newListName = ""; creatingList = false } } label: { Image(systemName: "checkmark") }
                    Button { creatingList = false } label: { Image(systemName: "xmark") }
                } else {
                    Button { creatingList = true } label: { Label("New List", systemImage: "plus") }.buttonStyle(.bordered)
                }
                Spacer()
            }
        }
        #else
        HStack(spacing: 8) {
            ForEach(viewModel.watchlists) { wl in
                Button { viewModel.activeId = wl.id } label: {
                    Text(wl.name).font(.callout.weight(.medium))
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(viewModel.activeId == wl.id ? Color.accentColor : Color.gray.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
                        .foregroundStyle(viewModel.activeId == wl.id ? .white : .primary)
                }.buttonStyle(.plain)
            }
            if creatingList {
                TextField("List name", text: $newListName).textFieldStyle(.roundedBorder).frame(width: 130)
                Button { Task { await viewModel.createList(name: newListName, currency: cur); newListName = ""; creatingList = false } } label: { Image(systemName: "checkmark") }
                Button { creatingList = false } label: { Image(systemName: "xmark") }
            } else {
                Button { creatingList = true } label: { Label("New List", systemImage: "plus") }.buttonStyle(.bordered)
            }
            Spacer()
        }
        #endif
    }

    // MARK: - Card (header + add form + table)

    private var card: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                if renaming {
                    TextField("Name", text: $renameName).textFieldStyle(.roundedBorder).frame(width: 200)
                    Button { Task { await viewModel.renameList(name: renameName); renaming = false } } label: { Image(systemName: "checkmark") }
                    Button { renaming = false } label: { Image(systemName: "xmark") }
                } else {
                    Text(currentListName).font(.headline)
                    Button { renameName = currentListName; renaming = true } label: { Image(systemName: "pencil") }.buttonStyle(.borderless)
                    if viewModel.watchlists.count > 1 {
                        Button { Task { await viewModel.deleteList(currency: cur) } } label: { Image(systemName: "trash") }
                            .buttonStyle(.borderless).foregroundStyle(.red)
                    }
                }
                Spacer()
                TextField("Search this list…", text: $search).textFieldStyle(.roundedBorder).frame(width: 180)
            }
            addForm
            table
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }

    private var addForm: some View {
        #if os(iOS)
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Symbol").font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                TextField("e.g. AAPL, BTC-USD", text: $newSymbol).textFieldStyle(.roundedBorder)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("Note (optional)").font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                TextField("Add a description…", text: $newNote).textFieldStyle(.roundedBorder)
            }
            Button {
                let sym = newSymbol.trimmingCharacters(in: .whitespaces)
                guard !sym.isEmpty else { return }
                let note = newNote; newSymbol = ""; newNote = ""
                Task { await viewModel.add(symbol: sym, note: note, currency: cur) }
            } label: { Label("Add to List", systemImage: "plus") }
                .buttonStyle(.borderedProminent)
                .disabled(newSymbol.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(12).background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
        #else
        HStack(alignment: .bottom, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Symbol").font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                TextField("e.g. AAPL, BTC-USD", text: $newSymbol).textFieldStyle(.roundedBorder).frame(width: 180)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("Note (optional)").font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase)
                TextField("Add a description…", text: $newNote).textFieldStyle(.roundedBorder)
            }
            Button {
                let sym = newSymbol.trimmingCharacters(in: .whitespaces)
                guard !sym.isEmpty else { return }
                let note = newNote; newSymbol = ""; newNote = ""
                Task { await viewModel.add(symbol: sym, note: note, currency: cur) }
            } label: { Label("Add to List", systemImage: "plus") }
                .buttonStyle(.borderedProminent)
                .disabled(newSymbol.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(12).background(.background.tertiary, in: RoundedRectangle(cornerRadius: 10))
        #endif
    }

    // MARK: - Table

    private var table: some View {
        Group {
            if viewModel.items.isEmpty && !viewModel.isLoading {
                ContentUnavailableView("No symbols in this list yet", systemImage: "star",
                                       description: Text("Add a symbol above to start tracking it."))
            } else {
                #if os(iOS)
                LazyVStack(spacing: 12) {
                    ForEach(filtered) { item in
                        iosWatchlistRow(item)
                    }
                }
                #else
                ScrollView(.horizontal, showsIndicators: true) {
                    Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 0) {
                        headerRow
                        Divider().gridCellColumns(14)
                        ForEach(filtered) { item in
                            rowView(item)
                            Divider().gridCellColumns(14)
                        }
                    }
                }
                #endif
            }
        }
    }

    #if os(iOS)
    private func iosWatchlistRow(_ item: WatchlistItem) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button { detail = SymbolID(id: item.symbol) } label: {
                    HStack(spacing: 6) { StockIcon(symbol: item.symbol, size: 27); Text(item.symbol).font(.headline).fontWeight(.bold) }
                }.buttonStyle(.plain)
                Spacer()
                Text(item.price.map { Fmt.currency($0, code: item.currency ?? cur) } ?? "-").font(.headline).monospacedDigit()
            }
            HStack {
                Text(item.name ?? "-").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                Text(Fmt.percent(item.dayChangePct)).font(.subheadline.bold()).monospacedDigit().foregroundStyle(Fmt.tint(for: item.dayChangePct))
            }
            Divider()
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Mkt Cap").font(.caption2).foregroundStyle(.secondary)
                    Text(item.marketCap.map { Fmt.number($0, fractionDigits: 0) } ?? "-").monospacedDigit().font(.caption)
                }
                Spacer()
                VStack(alignment: .leading, spacing: 2) {
                    Text("Intrinsic").font(.caption2).foregroundStyle(.secondary)
                    Text(item.intrinsicValue.map { Fmt.currency($0, code: item.currency ?? cur) } ?? "-").monospacedDigit().font(.caption)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("AI").font(.caption2).foregroundStyle(.secondary)
                    Text(item.aiScore.map { String(format: "%.1f", $0) } ?? "-").monospacedDigit().font(.caption)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("7D").font(.caption2).foregroundStyle(.secondary)
                    sparkline(item.sparkline)
                }
            }
            Divider()
            HStack {
                noteCell(item)
                Spacer()
                Button { Task { await viewModel.remove(symbol: item.symbol, currency: cur) } } label: { Image(systemName: "trash") }
                    .buttonStyle(.borderless).foregroundStyle(.red)
            }
        }
        .padding(12)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
    #endif

    private var headerRow: some View {
        GridRow {
            sortHeader("Symbol", .symbol, .leading)
            sortHeader("Name", .name, .leading)
            sortHeader("Price", .price, .trailing)
            sortHeader("Day %", .day, .trailing)
            sortHeader("Mkt Cap", .mktcap, .trailing)
            sortHeader("PE", .pe, .trailing)
            sortHeader("Div", .div, .trailing)
            sortHeader("AI", .ai, .trailing)
            sortHeader("Intrinsic", .intrinsic, .trailing)
            Text("Sent.").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            Text("Cat.").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            Text("7D").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).gridColumnAlignment(.leading)
            sortHeader("Note", .note, .leading)
            Text("").gridColumnAlignment(.trailing)
        }
    }

    private func sortHeader(_ label: String, _ key: WLSort, _ align: Alignment) -> some View {
        Button { if sortKey == key { sortAsc.toggle() } else { sortKey = key; sortAsc = true } } label: {
            HStack(spacing: 2) {
                Text(label).font(.caption2.weight(.semibold))
                if sortKey == key { Image(systemName: sortAsc ? "arrow.up" : "arrow.down").font(.system(size: 9)) }
            }.foregroundStyle(sortKey == key ? Color.accentColor : .secondary)
        }
        .buttonStyle(.plain)
        .gridColumnAlignment(align == .leading ? .leading : .trailing)
    }

    private func rowView(_ item: WatchlistItem) -> some View {
        GridRow {
            Button { detail = SymbolID(id: item.symbol) } label: {
                HStack(spacing: 6) { StockIcon(symbol: item.symbol, size: 18); Text(item.symbol).fontWeight(.bold) }
            }
            .buttonStyle(.plain).gridColumnAlignment(.leading)
            Text(item.name ?? "-").foregroundStyle(.secondary).lineLimit(1).frame(maxWidth: 160, alignment: .leading).gridColumnAlignment(.leading)
            Text(item.price.map { Fmt.currency($0, code: item.currency ?? cur) } ?? "-").monospacedDigit()
            Text(Fmt.percent(item.dayChangePct)).monospacedDigit().foregroundStyle(Fmt.tint(for: item.dayChangePct))
            Text(item.marketCap.map { Fmt.number($0, fractionDigits: 0) } ?? "-").monospacedDigit().foregroundStyle(.secondary)
            Text(item.peRatio.map { String(format: "%.1f", $0) } ?? "-").monospacedDigit().foregroundStyle(.secondary)
            Text(Fmt.percent(item.dividendYield)).monospacedDigit().foregroundStyle(.secondary)
            Text(item.aiScore.map { String(format: "%.1f", $0) } ?? "-").monospacedDigit()
            Text(item.intrinsicValue.map { Fmt.currency($0, code: item.currency ?? cur) } ?? "-").monospacedDigit().foregroundStyle(.secondary)
            sentimentCell(item.sentiment)
            Text(item.catalystCount > 0 ? "\(item.catalystCount)" : "–").font(.caption2)
                .foregroundStyle(item.catalystCount > 0 ? .orange : .secondary)
            sparkline(item.sparkline).gridColumnAlignment(.leading)
            noteCell(item).gridColumnAlignment(.leading)
            Button { Task { await viewModel.remove(symbol: item.symbol, currency: cur) } } label: { Image(systemName: "trash") }
                .buttonStyle(.borderless).foregroundStyle(.red).gridColumnAlignment(.trailing)
        }
        .font(.caption)
        .padding(.vertical, 6)
    }

    @ViewBuilder private func sentimentCell(_ s: Double?) -> some View {
        if let s {
            Circle().fill(s > 0.1 ? .green : (s < -0.1 ? .red : .gray)).frame(width: 8, height: 8)
        } else { Text("–").foregroundStyle(.secondary) }
    }

    @ViewBuilder private func sparkline(_ data: [Double]) -> some View {
        if data.count > 1 {
            let up = (data.last ?? 0) >= (data.first ?? 0)
            Chart(Array(data.enumerated()), id: \.offset) { i, v in
                LineMark(x: .value("i", i), y: .value("v", v)).foregroundStyle(up ? .green : .red)
            }
            .chartXAxis(.hidden).chartYAxis(.hidden).frame(width: 70, height: 22)
        } else { Text("–").foregroundStyle(.secondary) }
    }

    @ViewBuilder private func noteCell(_ item: WatchlistItem) -> some View {
        if editingNoteSymbol == item.symbol {
            HStack(spacing: 4) {
                TextField("Note", text: $editNoteText).textFieldStyle(.roundedBorder).frame(width: 140)
                Button { Task { await viewModel.add(symbol: item.symbol, note: editNoteText, currency: cur); editingNoteSymbol = nil } } label: { Image(systemName: "checkmark") }.buttonStyle(.borderless)
            }
        } else {
            HStack(spacing: 4) {
                Text(item.note.isEmpty ? "—" : item.note).foregroundStyle(.secondary).lineLimit(1).frame(maxWidth: 160, alignment: .leading)
                Button { editingNoteSymbol = item.symbol; editNoteText = item.note } label: { Image(systemName: "pencil") }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - KPI strip (mirrors watchlist/WatchlistKpiStrip.tsx)

struct WatchlistKpiStrip: View {
    let items: [WatchlistItem]

    var body: some View {
        let changes = items.compactMap { $0.dayChangePct }
        let avg = changes.isEmpty ? nil : changes.reduce(0, +) / Double(changes.count)
        let best = items.compactMap { i in i.dayChangePct.map { (i.symbol, $0) } }.max { $0.1 < $1.1 }
        let worst = items.compactMap { i in i.dayChangePct.map { (i.symbol, $0) } }.min { $0.1 < $1.1 }
        let opportunities = items.filter { ($0.marginOfSafety ?? 0) > 0 }.count
        return KpiRow(count: 5, minTileWidth: 140) {
            tile("Symbols", "\(items.count)", "tracked", .primary)
            tile("Avg Day Change", avg.map { Fmt.percent($0) } ?? "–", nil, Fmt.tint(for: avg))
            tile("Best Today", best.map { Fmt.percent($0.1) } ?? "–", best?.0, .green)
            tile("Worst Today", worst.map { Fmt.percent($0.1) } ?? "–", worst?.0, .red)
            tile("Opportunities", "\(opportunities)", "below fair value", opportunities > 0 ? .green : .primary)
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
    private func tile(_ label: String, _ value: String, _ sub: String?, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(value).font(.title3.bold()).foregroundStyle(tone)
            if let sub { Text(sub).font(.caption2).foregroundStyle(.secondary) }
        }
        .padding(.horizontal, 16).frame(maxWidth: .infinity, alignment: .leading)
    }
}
