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

/** Yahoo index/FX/commodity tickers → their TradingView counterparts. */
const SPECIAL_SYMBOL: Record<string, string> = {
    '^GSPC': 'SP:SPX',
    '^SP500TR': 'SP:SPXTR',
    '^IXIC': 'NASDAQ:IXIC',
    '^NDX': 'NASDAQ:NDX',
    '^DJI': 'DJ:DJI',
    '^RUT': 'TVC:RUT',
    '^VIX': 'TVC:VIX',
    '^TNX': 'TVC:TNX',
    '^FTSE': 'TVC:UKX',
    '^GDAXI': 'XETR:DAX',
    '^FCHI': 'EURONEXT:PX1',
    '^STOXX50E': 'TVC:SX5E',
    '^N225': 'TVC:NI225',
    '^HSI': 'TVC:HSI',
    '^SET.BK': 'SET:SET',
    '^AXJO': 'ASX:XJO',
    'GC=F': 'COMEX:GC1!',
    'SI=F': 'COMEX:SI1!',
    'CL=F': 'NYMEX:CL1!',
    'NG=F': 'NYMEX:NG1!',
};

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
