import { describe, it, expect } from 'vitest';
import { toTradingViewSymbol } from '../../lib/tradingview';

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
        expect(toTradingViewSymbol('^GSPC')).toBe('SP:SPX');
        expect(toTradingViewSymbol('^SET.BK')).toBe('SET:SET');
        expect(toTradingViewSymbol('BTC-USD')).toBe('CRYPTO:BTCUSD');
        expect(toTradingViewSymbol('EURUSD=X')).toBe('FX:EURUSD');
        expect(toTradingViewSymbol('THB=X')).toBe('FX:USDTHB');
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
