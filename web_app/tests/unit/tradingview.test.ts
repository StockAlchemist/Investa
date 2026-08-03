import { describe, it, expect } from 'vitest';
import { benchmarkYahooSymbol, toTradingViewSymbol } from '../../lib/tradingview';

describe('toTradingViewSymbol', () => {
    it('pins a US listing to its venue when the exchange code is known', () => {
        expect(toTradingViewSymbol('AAPL', 'NMS')).toBe('NASDAQ:AAPL');
        expect(toTradingViewSymbol('JNJ', 'NYQ')).toBe('NYSE:JNJ');
        expect(toTradingViewSymbol('SPY', 'PCX')).toBe('AMEX:SPY');
    });

    it('falls back to a bare US ticker when the exchange is unknown or absent', () => {
        expect(toTradingViewSymbol('AAPL')).toBe('AAPL');
        expect(toTradingViewSymbol('AAPL', 'XXX')).toBe('AAPL');
    });

    it("rewrites Yahoo's hyphenated share classes to TradingView's dot form", () => {
        expect(toTradingViewSymbol('BRK-B', 'NYQ')).toBe('NYSE:BRK.B');
    });

    it('turns a Yahoo listing suffix into an exchange prefix', () => {
        expect(toTradingViewSymbol('PTT.BK')).toBe('SET:PTT');
        expect(toTradingViewSymbol('0700.HK')).toBe('HKEX:0700');
        expect(toTradingViewSymbol('SHOP.TO')).toBe('TSX:SHOP');
        expect(toTradingViewSymbol('SAP.DE')).toBe('XETR:SAP');
    });

    it('maps indices, FX and crypto onto their TradingView counterparts', () => {
        expect(toTradingViewSymbol('^GSPC')).toBe('FOREXCOM:SPXUSD');
        expect(toTradingViewSymbol('^GDAXI')).toBe('XETR:DAX');
        expect(toTradingViewSymbol('BTC-USD')).toBe('CRYPTO:BTCUSD');
        expect(toTradingViewSymbol('EURUSD=X')).toBe('FX:EURUSD');
        expect(toTradingViewSymbol('THB=X')).toBe('FX:USDTHB');
    });

    it('has no entry for an index the free widget refuses to draw', () => {
        // The licensed feeds (SP:SPX, TVC:*, SET:SET) answer "This symbol is
        // only available on TradingView" with an empty chart, so these resolve
        // to null and the caller falls back instead.
        expect(toTradingViewSymbol('^SP500TR')).toBeNull();
        expect(toTradingViewSymbol('^TNX')).toBeNull();
        expect(toTradingViewSymbol('^SET.BK')).toBeNull();
    });

    it('refuses to guess rather than resolving to the wrong company', () => {
        // An unmapped suffix must not fall through to the bare root, which would
        // land on an unrelated US listing of the same name.
        expect(toTradingViewSymbol('ABC.XYZ')).toBeNull();
        expect(toTradingViewSymbol('^SOMEINDEX')).toBeNull();
        expect(toTradingViewSymbol('ZC=F')).toBeNull();
        expect(toTradingViewSymbol('$CASH')).toBeNull();
        expect(toTradingViewSymbol('')).toBeNull();
    });
});

describe('benchmarkYahooSymbol', () => {
    it('resolves the /indices map keys the backend serves', () => {
        expect(benchmarkYahooSymbol('.DJI')).toBe('^DJI');
        expect(benchmarkYahooSymbol('IXIC')).toBe('^IXIC');
        expect(benchmarkYahooSymbol('.INX')).toBe('^GSPC');
    });

    it('resolves the benchmark names the graph modal uses', () => {
        expect(benchmarkYahooSymbol('S&P 500')).toBe('^GSPC');
        expect(benchmarkYahooSymbol('Dow Jones')).toBe('^DJI');
        expect(benchmarkYahooSymbol('Dow')).toBe('^DJI');
        expect(benchmarkYahooSymbol('nasdaq')).toBe('^IXIC');
        expect(benchmarkYahooSymbol('QQQ (Nasdaq 100 ETF)')).toBe('QQQ');
    });

    it('lands on a chart the free widget draws for the indices we show', () => {
        expect(toTradingViewSymbol(benchmarkYahooSymbol('.DJI')!)).toBe('FOREXCOM:DJI');
        expect(toTradingViewSymbol(benchmarkYahooSymbol('S&P 500')!)).toBe('FOREXCOM:SPXUSD');
        expect(toTradingViewSymbol(benchmarkYahooSymbol('IXIC')!)).toBe('NASDAQ:ONEQ');
        expect(toTradingViewSymbol(benchmarkYahooSymbol('Russell 2000')!)).toBe('OANDA:US2000USD');
    });

    it('returns null for a label we do not carry', () => {
        expect(benchmarkYahooSymbol('Nikkei 225')).toBeNull();
        expect(benchmarkYahooSymbol('')).toBeNull();
    });
});
