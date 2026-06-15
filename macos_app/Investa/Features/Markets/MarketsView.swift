import SwiftUI
import Charts

@MainActor
final class MarketsViewModel: ObservableObject {
    @Published var indices: [IndexQuote] = []
    @Published var marketNews: [MarketNewsItem] = []
    @Published var stockNews: [MarketNewsItem] = []
    @Published var holdings: [Holding] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let api: APIClient
    private var task: Task<Void, Never>?

    init(api: APIClient = .shared) { self.api = api }

    func reload(currency: String) {
        task?.cancel()
        task = Task { [weak self] in await self?.load(currency: currency) }
    }

    private func load(currency: String) async {
        isLoading = true; errorMessage = nil
        defer { isLoading = false }
        async let indicesR: [String: IndexQuote] = api.get("/indices")
        async let newsR: [MarketNewsItem] = api.get("/markets/news", query: [URLQueryItem(name: "limit", value: "20")])
        async let holdingsR: [Holding] = api.get("/holdings", query: [URLQueryItem(name: "currency", value: currency)])
        async let wlR: [WatchlistItem] = api.get("/watchlist",
            query: [URLQueryItem(name: "currency", value: currency), URLQueryItem(name: "id", value: "1")])

        if let map = try? await indicesR { indices = Array(map.values) }
        do { marketNews = try await newsR } catch let e as APIError {
            if case .unauthorized = e { return }; errorMessage = e.errorDescription
        } catch is CancellationError { return } catch { errorMessage = error.localizedDescription }
        holdings = (try? await holdingsR) ?? []

        // Stock news for portfolio + watchlist symbols.
        func isCash(_ s: String) -> Bool { let u = s.uppercased(); return u == "$CASH" || u == "CASH" || u.hasPrefix("CASH (") }
        var symbols = Set(holdings.map { $0.symbol }.filter { !isCash($0) })
        if let wl = try? await wlR { symbols.formUnion(wl.map { $0.symbol }) }
        if !symbols.isEmpty {
            stockNews = (try? await api.get("/markets/news",
                query: [URLQueryItem(name: "symbols", value: symbols.sorted().joined(separator: ",")),
                        URLQueryItem(name: "limit", value: "30")])) ?? []
        }
    }
}

struct MarketsView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var viewModel = MarketsViewModel()
    @Environment(\.openURL) private var openURL
    @State private var newsQuery = ""
    @State private var indexDetail: IndexQuote?
    @State private var stockDetail: SymbolID?

    private var cur: String { appState.displayCurrency }

    private func filterNews(_ news: [MarketNewsItem]) -> [MarketNewsItem] {
        let q = newsQuery.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return news }
        return news.filter { $0.title.lowercased().contains(q) || $0.provider.lowercased().contains(q) || ($0.symbol?.lowercased().contains(q) ?? false) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Markets").font(.title2.bold())
                if viewModel.isLoading { ProgressView().controlSize(.small) }
                Spacer()
            }
            .padding(.horizontal, 20).padding(.vertical, 12)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    if !viewModel.indices.isEmpty { summaryBar }
                    indicesSection
                    YourMoversSection(holdings: viewModel.holdings, currency: cur, onPick: { stockDetail = SymbolID(id: $0) })
                    searchField
                    if !viewModel.stockNews.isEmpty {
                        newsSection("Your Stock News", filterNews(viewModel.stockNews))
                    }
                    newsSection("Market News", filterNews(viewModel.marketNews))
                }
                .padding(20)
            }
        }
        .frame(minWidth: 820, minHeight: 560)
        .task { viewModel.reload(currency: cur) }
        .onReceive(NotificationCenter.default.publisher(for: .refreshRequested)) { _ in viewModel.reload(currency: cur) }
        .sheet(item: $indexDetail) { idx in IndexGraphSheet(index: idx) }
        .sheet(item: $stockDetail) { StockDetailView(symbol: $0.id) }
    }

    // MARK: - Summary bar

    private var summaryBar: some View {
        let list = viewModel.indices
        let up = list.filter { ($0.changesPercentage ?? 0) >= 0 }.count
        let down = list.count - up
        let best = list.max { ($0.changesPercentage ?? 0) < ($1.changesPercentage ?? 0) }
        let worst = list.min { ($0.changesPercentage ?? 0) < ($1.changesPercentage ?? 0) }
        return HStack(spacing: 0) {
            tile("Breadth", "\(up) ▲ / \(down) ▼", "\(list.count) indices", up >= down ? .green : .red)
            Divider().frame(height: 36)
            tile("Best", best.map { Fmt.percent($0.changesPercentage) } ?? "–", best?.name, .green)
            Divider().frame(height: 36)
            tile("Worst", worst.map { Fmt.percent($0.changesPercentage) } ?? "–", worst?.name, .red)
            Spacer()
        }
        .padding(16)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
    }
    private func tile(_ label: String, _ value: String, _ sub: String?, _ tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption2).foregroundStyle(.secondary).textCase(.uppercase)
            Text(value).font(.title3.bold()).foregroundStyle(tone)
            if let sub { Text(sub).font(.caption2).foregroundStyle(.secondary).lineLimit(1) }
        }
        .padding(.horizontal, 16).frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Indices

    private var indicesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Market Indices").font(.title3.bold())
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), spacing: 16)], spacing: 16) {
                ForEach(viewModel.indices) { idx in
                    Button { indexDetail = idx } label: { IndexCard(index: idx) }.buttonStyle(.plain)
                }
            }
        }
    }

    // MARK: - News

    private var searchField: some View {
        HStack {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("Search news by headline, ticker, or source…", text: $newsQuery)
                .textFieldStyle(.plain)
            if !newsQuery.isEmpty { Button { newsQuery = "" } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).foregroundStyle(.secondary) }
        }
        .padding(10)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }

    private func newsSection(_ title: String, _ news: [MarketNewsItem]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: "newspaper").font(.title3.bold())
            if news.isEmpty {
                Text("No news available.").font(.callout).foregroundStyle(.secondary)
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 320), spacing: 12)], spacing: 12) {
                    ForEach(news) { item in NewsCard(item: item) { if let u = URL(string: item.url) { openURL(u) } } }
                }
            }
        }
    }
}

// MARK: - Index card (mirrors MarketsTab IndexCard)

private struct IndexCard: View {
    let index: IndexQuote
    private var isUp: Bool { (index.change ?? 0) >= 0 }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(index.name ?? "Index").font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase).lineLimit(1)
                        Text(Fmt.number(index.price)).font(.system(size: 28, weight: .bold)).monospacedDigit()
                    }
                    Spacer()
                    HStack(spacing: 2) {
                        Image(systemName: isUp ? "arrow.up.right" : "arrow.down.right").font(.caption2)
                        Text(Fmt.percent(index.changesPercentage)).fontWeight(.bold)
                    }
                    .font(.caption).foregroundStyle(isUp ? .green : .red)
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background((isUp ? Color.green : .red).opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
                }
                Text("\(isUp ? "+" : "")\(Fmt.number(index.change)) pts").font(.caption.weight(.semibold))
                    .foregroundStyle(isUp ? .green : .red)
            }
            .padding(16)
            if index.sparkline.count > 1 {
                Chart(Array(index.sparkline.enumerated()), id: \.offset) { i, v in
                    AreaMark(x: .value("i", i), y: .value("v", v)).foregroundStyle((isUp ? Color.green : .red).opacity(0.2))
                    LineMark(x: .value("i", i), y: .value("v", v)).foregroundStyle(isUp ? .green : .red)
                }
                .chartXAxis(.hidden).chartYAxis(.hidden).frame(height: 70)
            }
            Text("7D Trend").font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 16).padding(.bottom, 10).padding(.top, 6)
        }
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.quaternary, lineWidth: 1))
    }
}

// MARK: - Index graph sheet (mirrors IndexGraphModal)

private struct IndexGraphSheet: View {
    @Environment(\.dismiss) private var dismiss
    let index: IndexQuote
    private var isUp: Bool { (index.change ?? 0) >= 0 }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading) {
                    Text(index.name ?? "Index").font(.title2.bold())
                    HStack(spacing: 10) {
                        Text(Fmt.number(index.price)).font(.title3).monospacedDigit()
                        Text("\(isUp ? "+" : "")\(Fmt.number(index.change)) (\(Fmt.percent(index.changesPercentage)))")
                            .foregroundStyle(isUp ? .green : .red).fontWeight(.semibold)
                    }
                }
                Spacer()
                Button { dismiss() } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).font(.title2).foregroundStyle(.secondary)
            }
            if index.sparkline.count > 1 {
                Chart(Array(index.sparkline.enumerated()), id: \.offset) { i, v in
                    AreaMark(x: .value("i", i), y: .value("v", v))
                        .foregroundStyle(.linearGradient(colors: [(isUp ? Color.green : .red).opacity(0.3), .clear], startPoint: .top, endPoint: .bottom))
                    LineMark(x: .value("i", i), y: .value("v", v)).foregroundStyle(isUp ? .green : .red).interpolationMethod(.monotone)
                }
                .frame(height: 280)
            } else {
                ContentUnavailableView("No chart data", systemImage: "chart.xyaxis.line").frame(height: 280)
            }
        }
        .padding(24).frame(width: 560, height: 420)
    }
}

// MARK: - Your Movers Today (from holdings)

private struct YourMoversSection: View {
    let holdings: [Holding]
    let currency: String
    let onPick: (String) -> Void

    private struct Mover { let symbol: String; let pct: Double; let price: Double? }

    private var movers: (gainers: [Mover], losers: [Mover]) {
        func isCash(_ s: String) -> Bool { let u = s.uppercased(); return u == "$CASH" || u == "CASH" || u.hasPrefix("CASH (") }
        var bySymbol: [String: Mover] = [:]
        for h in holdings where !isCash(h.symbol) {
            guard let pct = h.dayChangePct, bySymbol[h.symbol] == nil else { continue }
            bySymbol[h.symbol] = Mover(symbol: h.symbol, pct: pct, price: h.currencyValue("Price", currency: currency))
        }
        let sorted = bySymbol.values.sorted { $0.pct > $1.pct }
        return (Array(sorted.filter { $0.pct > 0 }.prefix(5)),
                Array(sorted.filter { $0.pct < 0 }.suffix(5).reversed()))
    }

    var body: some View {
        let m = movers
        if !m.gainers.isEmpty || !m.losers.isEmpty {
            VStack(alignment: .leading, spacing: 12) {
                Label("Your Movers Today", systemImage: "chart.bar").font(.title3.bold())
                HStack(alignment: .top, spacing: 32) {
                    column("Top Gainers", m.gainers, true)
                    column("Top Losers", m.losers, false)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 1))
            }
        }
    }

    private func column(_ title: String, _ rows: [Mover], _ positive: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: positive ? "chart.line.uptrend.xyaxis" : "chart.line.downtrend.xyaxis")
                .font(.caption2).textCase(.uppercase).foregroundStyle(positive ? .green : .red)
            if rows.isEmpty { Text("No movers.").font(.caption).foregroundStyle(.secondary) }
            ForEach(rows, id: \.symbol) { r in
                Button { onPick(r.symbol) } label: {
                    HStack {
                        Text(r.symbol).fontWeight(.bold)
                        Spacer()
                        if let p = r.price { Text(Fmt.currency(p, code: currency)).font(.caption).foregroundStyle(.secondary) }
                        Text("\(r.pct >= 0 ? "+" : "")\(String(format: "%.2f%%", r.pct))").fontWeight(.bold).foregroundStyle(positive ? .green : .red)
                    }.font(.caption)
                }.buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - News card (mirrors MarketsTab NewsCard)

private struct NewsCard: View {
    let item: MarketNewsItem
    let onTap: () -> Void

    private func timeAgo(_ iso: String) -> String {
        let f = ISO8601DateFormatter()
        guard let d = f.date(from: iso) ?? ISO8601DateFormatter().date(from: iso + "Z") else { return "" }
        let m = Int(Date().timeIntervalSince(d) / 60)
        if m < 1 { return "just now" }; if m < 60 { return "\(m)m ago" }
        let h = m / 60; if h < 24 { return "\(h)h ago" }
        return "\(h / 24)d ago"
    }

    var body: some View {
        Button(action: onTap) {
            HStack(alignment: .top, spacing: 10) {
                if let thumb = item.thumbnail, let url = URL(string: thumb) {
                    AsyncImage(url: url) { img in img.resizable().aspectRatio(contentMode: .fill) } placeholder: { Color.gray.opacity(0.15) }
                        .frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 8))
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.title).font(.callout.weight(.semibold)).multilineTextAlignment(.leading).lineLimit(2)
                    HStack(spacing: 6) {
                        if let sym = item.symbol, !sym.isEmpty {
                            Text(sym).font(.caption2.weight(.bold)).foregroundStyle(.indigo)
                                .padding(.horizontal, 5).padding(.vertical, 1).background(.indigo.opacity(0.12), in: RoundedRectangle(cornerRadius: 4))
                        }
                        Text(item.provider).font(.caption2.weight(.bold)).foregroundStyle(.secondary).textCase(.uppercase).lineLimit(1)
                        if !item.pubDate.isEmpty {
                            Text("· \(timeAgo(item.pubDate))").font(.caption2).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: "arrow.up.right.square").font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary, lineWidth: 1))
        }
        .buttonStyle(.plain)
    }
}
