import SwiftUI
import WebKit
#if canImport(AppKit)
import AppKit
#endif

// MARK: - Symbol mapping

/// Translate Investa's (Yahoo Finance) symbols into the `EXCHANGE:TICKER` form
/// TradingView's embedded chart expects.
///
/// Mirrors `web_app/lib/tradingview.ts` — keep the two tables in step. A miss
/// would silently show a *different company's* chart, so anything we can't map
/// confidently resolves to `nil` and the caller falls back rather than guessing.
enum TradingViewSymbol {
    /// Yahoo listing suffix → TradingView exchange prefix.
    private static let suffixExchange: [String: String] = [
        // Asia-Pacific
        "BK": "SET", "HK": "HKEX", "T": "TSE", "SS": "SSE", "SZ": "SZSE",
        "KS": "KRX", "KQ": "KRX", "TW": "TWSE", "TWO": "TPEX", "SI": "SGX",
        "KL": "MYX", "JK": "IDX", "NS": "NSE", "BO": "BSE", "AX": "ASX", "NZ": "NZX",
        // Europe
        "L": "LSE", "DE": "XETR", "F": "FWB", "PA": "EURONEXT", "AS": "EURONEXT",
        "BR": "EURONEXT", "LS": "EURONEXT", "MI": "MIL", "MC": "BME", "SW": "SIX",
        "VI": "VIE", "ST": "OMXSTO", "HE": "OMXHEX", "CO": "OMXCOP", "OL": "OSL",
        "IC": "OMXICE", "IS": "BIST", "AT": "ATHEX", "WA": "GPW",
        // Americas / Africa / Middle East
        "TO": "TSX", "V": "TSXV", "NE": "NEO", "MX": "BMV", "SA": "BMFBOVESPA",
        "JO": "JSE", "TA": "TASE", "SR": "TADAWUL",
    ]

    /// yfinance `info.exchange` code → TradingView exchange prefix (US venues).
    private static let exchangeCode: [String: String] = [
        "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ",
        "NYQ": "NYSE", "NYS": "NYSE",
        "PCX": "AMEX",  // NYSE Arca — TradingView files Arca listings under AMEX
        "ASE": "AMEX",  // NYSE American
        "BTS": "BATS", "PNK": "OTC", "OTC": "OTC",
    ]

    /// Yahoo index/FX/commodity tickers → their TradingView counterparts.
    ///
    /// The free embedded widget will not draw a licensed index feed: `SP:SPX`,
    /// `DJ:DJI`, `NASDAQ:IXIC` and every `TVC:` index answer with "This symbol
    /// is only available on TradingView" and an empty chart. What it does draw
    /// are the freely redistributable index CFDs TradingView's own free widgets
    /// use, so each index maps to the venue that actually renders.
    ///
    /// The Nasdaq Composite is the one index with no free feed at any venue, so
    /// it falls back to the ETF that tracks it — a different scale, but the same
    /// underlying, and the widget's legend names it. Where nothing free tracks
    /// the instrument at all (`^TNX`, `^SP500TR`, `^SET.BK`) there is no entry:
    /// the caller falls back rather than showing a chart that never draws.
    private static let specialSymbol: [String: String] = [
        "^GSPC": "FOREXCOM:SPXUSD",
        "^IXIC": "NASDAQ:ONEQ",  // Fidelity Nasdaq Composite Index ETF
        "^NDX": "FOREXCOM:NSXUSD", "^DJI": "FOREXCOM:DJI", "^RUT": "OANDA:US2000USD",
        "^VIX": "CAPITALCOM:VIX", "^FTSE": "FOREXCOM:UKXGBP", "^GDAXI": "XETR:DAX",
        "^FCHI": "OANDA:FR40EUR", "^STOXX50E": "CAPITALCOM:EU50", "^N225": "INDEX:NKY",
        "^HSI": "CAPITALCOM:HK50", "^AXJO": "ASX:XJO",
        "GC=F": "TVC:GOLD", "SI=F": "TVC:SILVER", "CL=F": "TVC:USOIL",
        "NG=F": "CAPITALCOM:NATURALGAS",
    ]

    private static let quoteCurrencies: Set<String> = ["USD", "USDT", "EUR", "BTC", "ETH", "THB"]

    /// Investa's benchmark display names and `/indices` map keys → Yahoo tickers.
    ///
    /// Indices reach the UI under two aliases that are neither Yahoo symbols:
    /// `/indices` keys them by the backend's own codes (`config.INDICES_FOR_HEADER`)
    /// and the graph sheets name them the way `config.BENCHMARK_MAPPING` does.
    private static let benchmarkYahoo: [String: String] = [
        // `/indices` map keys
        ".DJI": "^DJI", "IXIC": "^IXIC", ".INX": "^GSPC",
        // Display names — INDEX_DISPLAY_NAMES and BENCHMARK_MAPPING
        "DOW": "^DJI", "DOW JONES": "^DJI", "NASDAQ": "^IXIC", "S&P 500": "^GSPC",
        "RUSSELL 2000": "^RUT", "S&P 500 TOTAL RETURN": "^SP500TR",
        "SPY (S&P 500 ETF)": "SPY", "QQQ (NASDAQ 100 ETF)": "QQQ", "DIA (DOW JONES ETF)": "DIA",
    ]

    /// The Yahoo symbol behind a benchmark name or `/indices` key, or `nil` when
    /// the label isn't one we carry — the caller then has nothing to chart
    /// rather than a guess at what "Nikkei" might have meant.
    static func benchmark(_ nameOrKey: String) -> String? {
        benchmarkYahoo[nameOrKey.trimmingCharacters(in: .whitespaces).uppercased()]
    }

    /// The TradingView symbol for `symbol`, or `nil` when no confident mapping
    /// exists (cash rows, unknown listing venues, futures we don't carry).
    ///
    /// - Parameter exchange: yfinance's `info.exchange` code, used to pin US
    ///   listings to the right venue. Optional — a bare US ticker resolves fine
    ///   on TradingView's side without it.
    static func map(_ symbol: String, exchange: String? = nil) -> String? {
        let raw = symbol.trimmingCharacters(in: .whitespaces).uppercased()
        guard !raw.isEmpty, raw != "$CASH", raw != "CASH" else { return nil }

        if let special = specialSymbol[raw] { return special }

        // Crypto pairs: BTC-USD → CRYPTO:BTCUSD. Checked before the share-class
        // rewrite below, which also keys on the hyphen.
        if let dash = raw.lastIndex(of: "-") {
            let base = String(raw[raw.startIndex..<dash])
            let quote = String(raw[raw.index(after: dash)...])
            if quoteCurrencies.contains(quote), !base.isEmpty, base.allSatisfy(\.isLetter) {
                return "CRYPTO:\(base)\(quote)"
            }
        }

        // FX: EURUSD=X (pair) and THB=X (implicit USD base).
        if raw.hasSuffix("=X") {
            let pair = String(raw.dropLast(2))
            guard pair.count >= 3, pair.allSatisfy(\.isLetter) else { return nil }
            return "FX:\(pair.count == 3 ? "USD\(pair)" : pair)"
        }

        // Anything else carrying Yahoo's index caret or a futures suffix is
        // outside the tables above — don't guess at it.
        if raw.hasPrefix("^") || raw.hasSuffix("=F") { return nil }

        if let dot = raw.lastIndex(of: "."), dot != raw.startIndex {
            let root = String(raw[raw.startIndex..<dot])
            let suffix = String(raw[raw.index(after: dot)...])
            // An unmapped suffix would otherwise fall through to `root` alone and
            // resolve against some unrelated US listing of the same name.
            guard let prefix = suffixExchange[suffix] else { return nil }
            return "\(prefix):\(root.replacingOccurrences(of: "-", with: "."))"
        }

        // US listing. Yahoo writes share classes as BRK-B, TradingView as BRK.B.
        let ticker = raw.replacingOccurrences(of: "-", with: ".")
        if let code = exchange?.uppercased(), let prefix = exchangeCode[code] {
            return "\(prefix):\(ticker)"
        }
        return ticker
    }
}

// MARK: - View

/// TradingView's Advanced Chart, embedded as their free widget in a `WKWebView`.
///
/// Everything the chart draws comes from TradingView, not from Investa's
/// backend — prices, splits and dividends are theirs, and can differ slightly
/// from the Yahoo-sourced series the Price/Return views draw.
///
/// Unlike the web client, which uses TradingView's loader script to inject the
/// widget as an iframe into the page, the native clients navigate straight to
/// the widget URL. A WKWebView will not paint that iframe at anything near a
/// full-height container — it renders only when the iframe collapses to its
/// ~150pt default — so the web view is pointed at the widget itself, which
/// sidesteps the iframe entirely.
struct TradingViewChartView: View {
    let symbol: String
    var exchange: String?
    /// Fixed chart height, or `nil` to fill whatever the parent offers — which
    /// is what the full-screen presentation wants.
    var height: CGFloat? = 520

    @Environment(\.colorScheme) private var colorScheme
    @State private var state: TradingViewWeb.LoadState = .loading

    var body: some View {
        Group {
            if let tvSymbol = TradingViewSymbol.map(symbol, exchange: exchange),
               let url = Self.widgetURL(symbol: tvSymbol, dark: colorScheme == .dark) {
                ZStack {
                    TradingViewWeb(url: url) { state = $0 }
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    switch state {
                    case .loading:
                        ProgressView()
                    case .failed:
                        Text("TradingView couldn't be reached. Check your connection, or use the Price or Return % view.")
                            .appFont(.callout).foregroundStyle(.secondary)
                            .multilineTextAlignment(.center).padding(.horizontal, 24)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .background(.background)
                    case .loaded:
                        EmptyView()
                    }
                }
            } else {
                ContentUnavailableView(
                    "No TradingView listing",
                    systemImage: "chart.xyaxis.line",
                    description: Text("\(symbol) has no TradingView listing we can resolve — use the Price or Return % view.")
                )
            }
        }
        .modifier(FixedOrFillHeight(height: height))
    }

    /// The widget page TradingView's loader would otherwise put in an iframe:
    /// its config travels as a URL-encoded JSON fragment.
    ///
    /// Keys are sorted so the same inputs always produce the same URL — the web
    /// view reloads on URL change, and SwiftUI rebuilds this view often.
    private static func widgetURL(symbol: String, dark: Bool) -> URL? {
        let config: [String: Any] = [
            "symbol": symbol,
            "interval": "D",
            // Market-local, per Investa's time convention — never the device's
            // zone. TradingView resolves "exchange" to the listing venue.
            "timezone": "exchange",
            "theme": dark ? "dark" : "light",
            "style": "1",  // candlesticks
            "locale": "en",
            "withdateranges": true,
            "hide_side_toolbar": false,
            "allow_symbol_change": false,
            "details": false,
            "calendar": false,
            "support_host": "https://www.tradingview.com",
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8),
              let fragment = json.addingPercentEncoding(withAllowedCharacters: .alphanumerics)
        else { return nil }
        return URL(string: "https://www.tradingview-widget.com/embed-widget/advanced-chart/?locale=en#\(fragment)")
    }
}

/// Applies a fixed height, or lets the view fill its parent when `height` is
/// `nil`. `.frame(height: nil)` would instead leave the web view at its own
/// ideal size, which for a `WKWebView` is zero.
private struct FixedOrFillHeight: ViewModifier {
    let height: CGFloat?

    func body(content: Content) -> some View {
        if let height {
            content.frame(height: height)
        } else {
            content.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

// MARK: - Full screen

/// Enters or leaves the full-screen chart. Always given a row of its own —
/// TradingView's widget puts its own button in the top-right of its toolbar,
/// which anything laid over the chart there would cover.
struct TradingViewFullScreenChip: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: systemImage)
                Text(title.uppercased())
            }
            .appFont(.system(size: 11, weight: .bold))
            .lineLimit(1).fixedSize(horizontal: true, vertical: false)
            .padding(.horizontal, 9).padding(.vertical, 4)
            .background(Color.gray.opacity(0.15), in: Capsule())
            .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
    }
}

/// The chart on its own, filling a cover (iOS) or a screen-sized sheet (macOS).
///
/// This builds a second web view rather than moving the inline one, so the
/// chart reloads on the way in and out: TradingView's own state — interval,
/// drawings, zoom — starts fresh each time.
struct TradingViewFullScreenView: View {
    let symbol: String
    var exchange: String?

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Spacer()
                TradingViewFullScreenChip(
                    title: "Exit Full Screen",
                    systemImage: "arrow.down.right.and.arrow.up.left"
                ) { dismiss() }
                .keyboardShortcut(.escape, modifiers: [])
            }
            TradingViewChartView(symbol: symbol, exchange: exchange, height: nil)
        }
        .padding(8)
        #if os(macOS)
        // A sheet takes its size from its content, so ask for very nearly the
        // whole display rather than the web view's own ideal size.
        .frame(width: Self.macSheetSize.width, height: Self.macSheetSize.height)
        #else
        .ignoresSafeArea(.container, edges: .bottom)
        #endif
    }

    #if os(macOS)
    private static var macSheetSize: CGSize {
        let visible = NSScreen.main?.visibleFrame.size ?? CGSize(width: 1280, height: 800)
        return CGSize(width: visible.width * 0.94, height: visible.height * 0.94)
    }
    #endif
}

// MARK: - WKWebView bridge

private struct TradingViewWeb {
    enum LoadState { case loading, loaded, failed }

    let url: URL
    let onState: (LoadState) -> Void

    func makeWebView(_ coordinator: Coordinator) -> WKWebView {
        let webView = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        webView.navigationDelegate = coordinator
        #if canImport(UIKit)
        webView.scrollView.bounces = false
        #endif
        load(webView, coordinator)
        return webView
    }

    /// Reloads only on a real URL change — SwiftUI re-runs `update…` for any
    /// parent redraw, and reloading there would restart the chart constantly.
    func load(_ webView: WKWebView, _ coordinator: Coordinator) {
        coordinator.onState = onState
        guard coordinator.loadedURL != url else { return }
        coordinator.loadedURL = url
        // Hops off this view-update pass before touching @State.
        DispatchQueue.main.async { onState(.loading) }
        webView.load(URLRequest(url: url))
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var loadedURL: URL?
        var onState: ((LoadState) -> Void)?

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            onState?(.loaded)
        }
        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            onState?(.failed)
        }
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            onState?(.failed)
        }
    }
}

#if canImport(UIKit)
extension TradingViewWeb: UIViewRepresentable {
    func makeUIView(context: Context) -> WKWebView { makeWebView(context.coordinator) }
    func updateUIView(_ webView: WKWebView, context: Context) { load(webView, context.coordinator) }
}
#elseif canImport(AppKit)
extension TradingViewWeb: NSViewRepresentable {
    func makeNSView(context: Context) -> WKWebView { makeWebView(context.coordinator) }
    func updateNSView(_ webView: WKWebView, context: Context) { load(webView, context.coordinator) }
}
#endif
