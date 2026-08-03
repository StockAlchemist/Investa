/**
 * Translate Investa's (Yahoo Finance) symbols into the `EXCHANGE:TICKER` form
 * TradingView's embedded chart expects.
 *
 * The two vendors disagree in three ways: Yahoo encodes the listing venue as a
 * dot suffix (`PTT.BK`) where TradingView uses a prefix (`SET:PTT`), Yahoo
 * writes share classes with a hyphen (`BRK-B`) where TradingView uses a dot,
 * and indices/FX/crypto have entirely unrelated tickers on each side.
 *
 * A miss here would silently show a *different company's* chart, so anything we
 * can't map confidently resolves to `null` and the caller falls back rather
 * than guessing.
 */

/** Yahoo listing suffix → TradingView exchange prefix. */
const SUFFIX_EXCHANGE: Record<string, string> = {
    // Asia-Pacific
    BK: 'SET',          // Stock Exchange of Thailand
    HK: 'HKEX',
    T: 'TSE',           // Tokyo
    SS: 'SSE',          // Shanghai
    SZ: 'SZSE',         // Shenzhen
    KS: 'KRX',          // KOSPI
    KQ: 'KRX',          // KOSDAQ
    TW: 'TWSE',
    TWO: 'TPEX',
    SI: 'SGX',
    KL: 'MYX',
    JK: 'IDX',
    NS: 'NSE',          // India
    BO: 'BSE',
    AX: 'ASX',
    NZ: 'NZX',
    // Europe
    L: 'LSE',
    DE: 'XETR',
    F: 'FWB',
    PA: 'EURONEXT',
    AS: 'EURONEXT',
    BR: 'EURONEXT',
    LS: 'EURONEXT',
    MI: 'MIL',
    MC: 'BME',
    SW: 'SIX',
    VI: 'VIE',
    ST: 'OMXSTO',
    HE: 'OMXHEX',
    CO: 'OMXCOP',
    OL: 'OSL',
    IC: 'OMXICE',
    IS: 'BIST',         // Istanbul
    AT: 'ATHEX',
    WA: 'GPW',
    // Americas / Africa / Middle East
    TO: 'TSX',
    V: 'TSXV',
    NE: 'NEO',
    MX: 'BMV',
    SA: 'BMFBOVESPA',
    JO: 'JSE',
    TA: 'TASE',
    SR: 'TADAWUL',
};

/** yfinance `info.exchange` code → TradingView exchange prefix (US venues). */
const EXCHANGE_CODE: Record<string, string> = {
    NMS: 'NASDAQ',      // NASDAQ Global Select
    NGM: 'NASDAQ',      // NASDAQ Global Market
    NCM: 'NASDAQ',      // NASDAQ Capital Market
    NAS: 'NASDAQ',
    NYQ: 'NYSE',
    NYS: 'NYSE',
    PCX: 'AMEX',        // NYSE Arca — TradingView files Arca listings under AMEX
    ASE: 'AMEX',        // NYSE American
    BTS: 'BATS',
    PNK: 'OTC',
    OTC: 'OTC',
};

/**
 * Yahoo index/FX/commodity tickers → their TradingView counterparts.
 *
 * The free embedded widget will not draw a licensed index feed: `SP:SPX`,
 * `DJ:DJI`, `NASDAQ:IXIC` and every `TVC:` index answer with "This symbol is
 * only available on TradingView" and an empty chart. What it does draw are the
 * freely redistributable index CFDs TradingView's own free widgets use, so
 * each index maps to the venue that actually renders — verified symbol by
 * symbol against the widget, not assumed.
 *
 * Those feeds carry the index's own scale and name (`FOREXCOM:SPXUSD` reads
 * "S&P 500 Index" at 7,538). The Nasdaq Composite is the one index with no
 * free feed at any venue, so it falls back to the ETF that tracks it —
 * a different scale, but the same underlying, and the widget's legend names it.
 * Where nothing free tracks the instrument at all (`^TNX`, `^SP500TR`,
 * `^SET.BK`) there is no entry: the caller falls back rather than showing a
 * chart that never draws.
 */
const SPECIAL_SYMBOL: Record<string, string> = {
    '^GSPC': 'FOREXCOM:SPXUSD',
    '^IXIC': 'NASDAQ:ONEQ',          // Fidelity Nasdaq Composite Index ETF
    '^NDX': 'FOREXCOM:NSXUSD',
    '^DJI': 'FOREXCOM:DJI',
    '^RUT': 'OANDA:US2000USD',
    '^VIX': 'CAPITALCOM:VIX',
    '^FTSE': 'FOREXCOM:UKXGBP',
    '^GDAXI': 'XETR:DAX',
    '^FCHI': 'OANDA:FR40EUR',
    '^STOXX50E': 'CAPITALCOM:EU50',
    '^N225': 'INDEX:NKY',
    '^HSI': 'CAPITALCOM:HK50',
    '^AXJO': 'ASX:XJO',
    'GC=F': 'TVC:GOLD',
    'SI=F': 'TVC:SILVER',
    'CL=F': 'TVC:USOIL',
    'NG=F': 'CAPITALCOM:NATURALGAS',
};

/**
 * Investa's benchmark display names and `/indices` map keys → Yahoo tickers.
 *
 * Indices reach the UI under two aliases that are neither Yahoo symbols:
 * `/indices` keys them by the backend's own codes (`config.INDICES_FOR_HEADER`:
 * `.DJI`, `IXIC`, `.INX`) and the graph modal names them the way
 * `config.BENCHMARK_MAPPING` does. Resolving to Yahoo first keeps the
 * TradingView tables above the single place a symbol is mapped.
 */
const BENCHMARK_YAHOO: Record<string, string> = {
    // `/indices` map keys
    '.DJI': '^DJI',
    'IXIC': '^IXIC',
    '.INX': '^GSPC',
    // Display names — INDEX_DISPLAY_NAMES and BENCHMARK_MAPPING
    'DOW': '^DJI',
    'DOW JONES': '^DJI',
    'NASDAQ': '^IXIC',
    'S&P 500': '^GSPC',
    'RUSSELL 2000': '^RUT',
    'S&P 500 TOTAL RETURN': '^SP500TR',
    'SPY (S&P 500 ETF)': 'SPY',
    'QQQ (NASDAQ 100 ETF)': 'QQQ',
    'DIA (DOW JONES ETF)': 'DIA',
};

/**
 * The Yahoo symbol behind a benchmark name or `/indices` key, or `null` when
 * the label isn't one we carry — the caller then has nothing to chart rather
 * than a guess at what "Nikkei" might have meant.
 */
export function benchmarkYahooSymbol(nameOrKey: string): string | null {
    return BENCHMARK_YAHOO[(nameOrKey || '').trim().toUpperCase()] ?? null;
}

/**
 * The TradingView symbol for `symbol`, or `null` when no confident mapping
 * exists (cash rows, unknown listing venues, futures we don't carry).
 *
 * @param exchange yfinance's `info.exchange` code, used to pin US listings to
 *                 the right venue. Optional — a bare US ticker resolves fine on
 *                 TradingView's side without it.
 */
export function toTradingViewSymbol(symbol: string, exchange?: string): string | null {
    const raw = (symbol || '').trim().toUpperCase();
    if (!raw || raw === '$CASH' || raw === 'CASH') return null;

    const special = SPECIAL_SYMBOL[raw];
    if (special) return special;

    // Crypto pairs: BTC-USD → CRYPTO:BTCUSD. Checked before the share-class
    // rewrite below, which also keys on the hyphen.
    const crypto = raw.match(/^([A-Z0-9]+)-(USD|USDT|EUR|BTC|ETH|THB)$/);
    if (crypto) return `CRYPTO:${crypto[1]}${crypto[2]}`;

    // FX: EURUSD=X (pair) and THB=X (implicit USD base).
    const fx = raw.match(/^([A-Z]{3,6})=X$/);
    if (fx) {
        const pair = fx[1].length === 3 ? `USD${fx[1]}` : fx[1];
        return `FX:${pair}`;
    }

    // Anything else carrying Yahoo's index caret or a futures suffix is outside
    // the tables above — don't guess at it.
    if (raw.startsWith('^') || raw.endsWith('=F')) return null;

    const dot = raw.lastIndexOf('.');
    if (dot > 0) {
        const suffix = raw.slice(dot + 1);
        const root = raw.slice(0, dot);
        const prefix = SUFFIX_EXCHANGE[suffix];
        // An unmapped suffix would otherwise fall through to `root` alone and
        // resolve against some unrelated US listing of the same name.
        if (!prefix) return null;
        return `${prefix}:${root.replace('-', '.')}`;
    }

    // US listing. Yahoo writes share classes as BRK-B, TradingView as BRK.B.
    const ticker = raw.replace('-', '.');
    const prefix = exchange ? EXCHANGE_CODE[exchange.toUpperCase()] : undefined;
    return prefix ? `${prefix}:${ticker}` : ticker;
}
