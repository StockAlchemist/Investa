import Foundation

/// One S&P 500 constituent returned by `GET /api/sp500/heatmap`.
/// Decoded tolerantly via `JSONValue` so missing/null fields never crash.
///
/// Unit convention, mirrored by `web_app/lib/api.ts`: `changePct` is percent
/// points (it comes straight off the quote), ratio-style fields are raw
/// numbers, and every other percentage — returns, growth, margins, yield,
/// float short — is a **fraction** (0.15 = 15%). Ratios expressed against
/// equity (`debtEquity`, `ltDebtEquity`) follow Yahoo and are percent points.
struct SP500HeatmapItem: Decodable, Sendable, Identifiable {
    let symbol: String
    let name: String
    let sector: String
    let subIndustry: String
    let price: Double
    let marketCap: Double?

    // Performance
    let changePct: Double?
    let weekChangePct: Double?
    let monthChangePct: Double?
    let mtdChangePct: Double?
    let threeMonthChangePct: Double?
    let sixMonthChangePct: Double?
    let ytdChangePct: Double?
    let oneYearChangePct: Double?
    let threeYearChangePct: Double?
    let fiveYearChangePct: Double?
    let tenYearChangePct: Double?
    /// Zero or below: the price cannot exceed its own 52-week high.
    let drawdown52w: Double?
    /// Zero or above: the price cannot fall below its own 52-week low.
    let gainFrom52wLow: Double?

    // Valuation
    let peRatio: Double?
    let forwardPE: Double?
    let pegRatio: Double?
    let psRatio: Double?
    let pbRatio: Double?
    let pFcf: Double?
    let evEbitda: Double?
    let evSales: Double?
    let dividendYield: Double?

    // Earnings & sales
    let epsTtm: Double?
    let epsQoQ: Double?
    let epsGrowth3y: Double?
    let epsGrowth5y: Double?
    let epsSurprise: Double?
    let salesTtm: Double?
    let salesQoQ: Double?
    let salesGrowth3y: Double?
    let salesGrowth5y: Double?

    // Profitability & balance sheet
    let roa: Double?
    let roe: Double?
    let roic: Double?
    let grossMargin: Double?
    let operatingMargin: Double?
    let netMargin: Double?
    let quickRatio: Double?
    let currentRatio: Double?
    let ltDebtEquity: Double?
    let debtEquity: Double?

    // Market & sentiment
    let relativeVolume: Double?
    let floatShort: Double?
    /// Yahoo consensus: 1 (strong buy) .. 5 (sell).
    let analystRecom: Double?
    /// Days until the next report; negative once it has happened.
    let earningsDays: Double?

    var id: String { symbol }

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode([String: JSONValue].self)
        func d(_ key: String) -> Double? { raw[key]?.doubleValue }

        symbol       = raw["symbol"]?.stringValue ?? ""
        name         = raw["name"]?.stringValue ?? ""
        sector       = raw["sector"]?.stringValue ?? "Unknown"
        subIndustry  = raw["sub_industry"]?.stringValue ?? "Unknown"
        price        = d("price") ?? 0
        marketCap    = d("market_cap")

        changePct           = d("change_pct")
        weekChangePct       = d("week_change_pct")
        monthChangePct      = d("month_change_pct")
        mtdChangePct        = d("mtd_change_pct")
        threeMonthChangePct = d("3m_change_pct")
        sixMonthChangePct   = d("6m_change_pct")
        ytdChangePct        = d("ytd_change_pct")
        oneYearChangePct    = d("1y_change_pct")
        threeYearChangePct  = d("3y_change_pct")
        fiveYearChangePct   = d("5y_change_pct")
        tenYearChangePct    = d("10y_change_pct")
        drawdown52w         = d("drawdown_52w")
        gainFrom52wLow      = d("gain_from_52w_low")

        peRatio       = d("pe_ratio")
        forwardPE     = d("forward_pe")
        pegRatio      = d("peg_ratio")
        psRatio       = d("ps_ratio")
        pbRatio       = d("pb_ratio")
        pFcf          = d("p_fcf")
        evEbitda      = d("ev_ebitda")
        evSales       = d("ev_sales")
        dividendYield = d("dividend_yield")

        epsTtm        = d("eps_ttm")
        epsQoQ        = d("eps_qoq")
        epsGrowth3y   = d("eps_growth_3y")
        epsGrowth5y   = d("eps_growth_5y")
        epsSurprise   = d("eps_surprise")
        salesTtm      = d("sales_ttm")
        salesQoQ      = d("sales_qoq")
        salesGrowth3y = d("sales_growth_3y")
        salesGrowth5y = d("sales_growth_5y")

        roa             = d("roa")
        roe             = d("roe")
        roic            = d("roic")
        grossMargin     = d("gross_margin")
        operatingMargin = d("operating_margin")
        netMargin       = d("net_margin")
        quickRatio      = d("quick_ratio")
        currentRatio    = d("current_ratio")
        ltDebtEquity    = d("lt_debt_equity")
        debtEquity      = d("debt_equity")

        relativeVolume = d("relative_volume")
        floatShort     = d("float_short")
        analystRecom   = d("analyst_recom")
        earningsDays   = d("earnings_days")
    }
}
