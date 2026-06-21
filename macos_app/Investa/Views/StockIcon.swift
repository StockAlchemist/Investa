import SwiftUI

/// Loads a company logo by trying several sources in order and caching the
/// result. Falls back to nil (→ monogram) only when every source fails.
@MainActor
@Observable
final class LogoLoader {
    var image: PlatformImage?
    var resolved = false

    @ObservationIgnored private static var cache: [String: PlatformImage] = [:]
    @ObservationIgnored private static var misses: Set<String> = []

    func load(symbol: String, sources: [URL]) async {
        if let cached = Self.cache[symbol] { image = cached; resolved = true; return }
        if Self.misses.contains(symbol) { resolved = true; return }
        for url in sources {
            guard let (data, response) = try? await URLSession.shared.data(from: url) else { continue }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { continue }
            // Reject tiny / empty placeholder responses.
            if let img = PlatformImage.decode(data), img.size.width >= 8, img.size.height >= 8 {
                Self.cache[symbol] = img
                image = img; resolved = true
                return
            }
        }
        Self.misses.insert(symbol)
        resolved = true
    }
}

/// Stock/company logo, mirroring the web `StockIcon`. Prefers Clearbit for the
/// (many) tickers FMP gets wrong, falling back to FMP, a favicon, then a
/// colored monogram of the first letter.
struct StockIcon: View {
    let symbol: String
    var size: CGFloat = 18
    @State private var loader = LogoLoader()

    // Ticker → brand domain. Used both for Clearbit (accurate logos) and the
    // favicon fallback. Covers the large/mid caps FMP commonly mis-serves.
    private static let brandDomains: [String: String] = [
        "AAPL": "apple.com", "MSFT": "microsoft.com", "GOOG": "google.com", "GOOGL": "google.com",
        "AMZN": "amazon.com", "META": "meta.com", "NVDA": "nvidia.com", "TSLA": "tesla.com",
        "NFLX": "netflix.com", "AMD": "amd.com", "INTC": "intel.com", "IBM": "ibm.com",
        "ORCL": "oracle.com", "CRM": "salesforce.com", "ADBE": "adobe.com", "CSCO": "cisco.com",
        "QCOM": "qualcomm.com", "TXN": "ti.com", "AVGO": "broadcom.com", "AMAT": "appliedmaterials.com",
        "MU": "micron.com", "NOW": "servicenow.com", "UBER": "uber.com", "ABNB": "airbnb.com",
        "PYPL": "paypal.com", "SQ": "block.xyz", "SHOP": "shopify.com", "SNOW": "snowflake.com",
        "PLTR": "palantir.com", "DDOG": "datadoghq.com", "NET": "cloudflare.com", "CRWD": "crowdstrike.com",
        "ZM": "zoom.us", "DOCU": "docusign.com", "TWLO": "twilio.com", "ROKU": "roku.com",
        "SPOT": "spotify.com", "PINS": "pinterest.com", "SNAP": "snap.com", "DIS": "disney.com",
        "CMCSA": "comcast.com", "T": "att.com", "VZ": "verizon.com", "TMUS": "t-mobile.com",
        "KO": "coca-cola.com", "PEP": "pepsi.com", "MCD": "mcdonalds.com", "SBUX": "starbucks.com",
        "NKE": "nike.com", "WMT": "walmart.com", "COST": "costco.com", "TGT": "target.com",
        "HD": "homedepot.com", "LOW": "lowes.com", "JPM": "jpmorganchase.com", "BAC": "bankofamerica.com",
        "WFC": "wellsfargo.com", "GS": "goldmansachs.com", "MS": "morganstanley.com", "V": "visa.com",
        "MA": "mastercard.com", "AXP": "americanexpress.com", "BRK.B": "berkshirehathaway.com",
        "BRK.A": "berkshirehathaway.com", "JNJ": "jnj.com", "PFE": "pfizer.com", "MRK": "merck.com",
        "ABBV": "abbvie.com", "LLY": "lilly.com", "UNH": "unitedhealthgroup.com", "TMO": "thermofisher.com",
        "XOM": "exxonmobil.com", "CVX": "chevron.com", "BA": "boeing.com", "CAT": "caterpillar.com",
        "GE": "ge.com", "F": "ford.com", "GM": "gm.com", "PG": "pg.com", "UPS": "ups.com",
        "FDX": "fedex.com", "DE": "deere.com", "MMM": "3m.com", "HON": "honeywell.com",
        "LMT": "lockheedmartin.com", "RTX": "rtx.com", "SPGI": "spglobal.com", "BLK": "blackrock.com",
        "C": "citigroup.com", "SCHW": "schwab.com", "COIN": "coinbase.com", "HOOD": "robinhood.com",
        "ASML": "asml.com", "TSM": "tsmc.com", "SAP": "sap.com", "TM": "toyota.com",
    ]

    private var isCash: Bool {
        let u = symbol.uppercased()
        return u == "$CASH" || u.contains("CASH") || symbol.contains("฿")
    }

    private var sources: [URL] {
        let s = symbol.trimmingCharacters(in: .whitespaces)
        guard !s.isEmpty else { return [] }
        var urls: [URL] = []
        if let domain = Self.brandDomains[s.uppercased()] {
            // Clearbit logos are accurate for mapped tickers; try first.
            if let u = URL(string: "https://logo.clearbit.com/\(domain)") { urls.append(u) }
            if let u = URL(string: "https://financialmodelingprep.com/image-stock/\(s).png") { urls.append(u) }
            if let u = URL(string: "https://www.google.com/s2/favicons?domain=\(domain)&sz=128") { urls.append(u) }
        } else {
            if let u = URL(string: "https://financialmodelingprep.com/image-stock/\(s).png") { urls.append(u) }
        }
        return urls
    }

    var body: some View {
        Group {
            if isCash {
                cashMonogram
            } else if let localImg = PlatformImage(named: symbol.lowercased()) {
                logoTile(Image(platformImage: localImg))
            } else if let logo = loader.image {
                logoTile(Image(platformImage: logo))
            } else if loader.resolved {
                monogram
            } else {
                RoundedRectangle(cornerRadius: size * 0.22).fill(.gray.opacity(0.15))
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.22))
        .task(id: symbol) {
            if !isCash && PlatformImage(named: symbol.lowercased()) == nil {
                await loader.load(symbol: symbol, sources: sources)
            }
        }
    }

    /// A logo on a white tile. The interior padding keeps full-bleed logos (e.g.
    /// the bundled Apple mark, which touches every edge) clear of the rounded
    /// corners, so they no longer look cropped. White fills the whole frame so
    /// the corners stay clean regardless of the logo's own margins.
    private func logoTile(_ image: Image) -> some View {
        image
            .resizable()
            .aspectRatio(contentMode: .fit)
            .padding(size * 0.16)
            .frame(width: size, height: size)
            .background(Color.white)
    }

    private var monogram: some View {
        let palette: [Color] = [.red, .orange, .yellow, .green, .mint, .teal, .cyan, .blue, .indigo, .purple, .pink, .brown]
        let color = palette[abs(hash(symbol)) % palette.count]
        return color.overlay(
            Text(symbol.prefix(1).uppercased())
                .font(.system(size: size * 0.5, weight: .bold)).foregroundStyle(.white))
    }

    private var cashMonogram: some View {
        let sym = symbol.contains("฿") || symbol.uppercased().contains("THB") ? "฿" : "$"
        return Color.gray.opacity(0.2).overlay(
            Text(sym).font(.system(size: size * 0.55, weight: .bold)).foregroundStyle(.primary))
    }

    private func hash(_ s: String) -> Int {
        var h = 0
        for scalar in s.unicodeScalars { h = Int(scalar.value) &+ ((h &<< 5) &- h) }
        return h
    }
}
