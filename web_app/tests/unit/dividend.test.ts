import { describe, it, expect } from 'vitest';
import { normalizeDividendYield, normalizeExpenseRatio } from '@/lib/dividend';

/**
 * Yahoo's `dividendYield` is percent-encoded on ~94% of records and
 * fraction-encoded on ~6%, with overlapping ranges. These tests pin the
 * corroboration-first behaviour that resolves the ambiguity, using real
 * shapes taken from the fundamentals cache.
 */
describe('normalizeDividendYield', () => {
    describe('with rate/price corroboration', () => {
        it('reads a percent-encoded value correctly (VICI)', () => {
            // dividendRate 1.80 at ~$26.83 => 6.71%. Raw 6.71 is a percent.
            const pct = normalizeDividendYield({
                rawYield: 6.71,
                dividendRate: 1.8,
                price: 26.83,
            });
            expect(pct).toBeCloseTo(6.71, 2);
        });

        it('reads a fraction-encoded value correctly (SCHD)', () => {
            // Same yield, opposite encoding: raw 0.033 is a fraction.
            const pct = normalizeDividendYield({
                rawYield: 0.033,
                dividendRate: 0.88,
                price: 26.67,
            });
            expect(pct).toBeCloseTo(3.3, 2);
        });

        it('resolves the ambiguous band a threshold cannot', () => {
            // 0.5 as a fraction is 50%; as a percent it is 0.5%. Only the
            // reference distinguishes them — and both cases must work.
            expect(normalizeDividendYield({ rawYield: 0.5, dividendRate: 0.05, price: 10 }))
                .toBeCloseTo(0.5, 4);   // 0.05/10 = 0.5% -> percent-encoded
            expect(normalizeDividendYield({ rawYield: 0.5, dividendRate: 5, price: 10 }))
                .toBeCloseTo(50, 4);    // 5/10 = 50%    -> fraction-encoded
        });

        it('handles a high yield that the old 0.12 cutoff broke', () => {
            const pct = normalizeDividendYield({
                rawYield: 0.15,
                dividendRate: 3.0,
                price: 20,
            });
            expect(pct).toBeCloseTo(15, 4);
        });

        it('tolerates a reference that is wrong by an order of magnitude', () => {
            // IDBOX is a monthly-paying bond fund, so `dividendRate` is the
            // monthly distribution and rate/price comes out 12x low
            // (0.029/10.41 = 0.28% against a true 3.03%). The choice still
            // lands correctly: the two readings are 100x apart, and a 12x
            // reference error cannot bridge that gap.
            const pct = normalizeDividendYield({
                rawYield: 3.03,
                dividendRate: 0.0290781,
                price: 10.41,
            });
            expect(pct).toBeCloseTo(3.03, 2);
        });

        it('falls back to trailingAnnualDividendYield when price is missing', () => {
            const pct = normalizeDividendYield({
                rawYield: 6.71,
                trailingYield: 0.0665,
            });
            expect(pct).toBeCloseTo(6.71, 2);
        });

        it('prefers rate/price over the trailing yield', () => {
            // Trailing is stale (post-cut); rate/price is forward looking.
            const pct = normalizeDividendYield({
                rawYield: 1.0,
                dividendRate: 1.0,
                price: 100,
                trailingYield: 0.04,
            });
            expect(pct).toBeCloseTo(1.0, 4);
        });
    });

    describe('idempotence', () => {
        it('does not double-scale a backend-normalised fraction', () => {
            // The backend already converts to a fraction; the client must not
            // then treat it as a percent, nor scale it twice.
            const once = normalizeDividendYield({ rawYield: 0.0671, dividendRate: 1.8, price: 26.83 });
            expect(once).toBeCloseTo(6.71, 2);
        });

        it('agrees on both encodings of the same security', () => {
            const asPercent = normalizeDividendYield({ rawYield: 6.71, dividendRate: 1.8, price: 26.83 });
            const asFraction = normalizeDividendYield({ rawYield: 0.0671, dividendRate: 1.8, price: 26.83 });
            expect(asPercent).toBeCloseTo(asFraction!, 6);
        });
    });

    describe('magnitude fallback (no corroborating signal)', () => {
        it('treats small values as fractions', () => {
            expect(normalizeDividendYield({ rawYield: 0.033 })).toBeCloseTo(3.3, 4);
            expect(normalizeDividendYield({ rawYield: 0.0007 })).toBeCloseTo(0.07, 4);
        });

        it('treats larger values as already-percent', () => {
            expect(normalizeDividendYield({ rawYield: 6.71 })).toBeCloseTo(6.71, 4);
            expect(normalizeDividendYield({ rawYield: 0.47 })).toBeCloseTo(0.47, 4);
        });
    });

    describe('missing and degenerate inputs', () => {
        it('returns null when there is no yield', () => {
            expect(normalizeDividendYield({})).toBeNull();
            expect(normalizeDividendYield({ rawYield: null })).toBeNull();
            expect(normalizeDividendYield({ rawYield: undefined })).toBeNull();
            expect(normalizeDividendYield({ rawYield: 0 })).toBeNull();
        });

        it('rejects non-finite values', () => {
            expect(normalizeDividendYield({ rawYield: NaN })).toBeNull();
            expect(normalizeDividendYield({ rawYield: Infinity })).toBeNull();
        });

        it('ignores an unusable reference rather than dividing by zero', () => {
            expect(normalizeDividendYield({ rawYield: 0.033, dividendRate: 0, price: 0 }))
                .toBeCloseTo(3.3, 4);
            expect(normalizeDividendYield({ rawYield: 0.033, dividendRate: 1, price: 0 }))
                .toBeCloseTo(3.3, 4);
        });

        it('does not treat a negative yield as usable', () => {
            expect(normalizeDividendYield({ rawYield: -1 })).toBeNull();
        });
    });
});

describe('normalizeExpenseRatio', () => {
    it('passes through percent-encoded ratios, which is the common case', () => {
        // Median of the cached corpus is 0.55 => a 0.55% fee, not 55%.
        expect(normalizeExpenseRatio(0.55)).toBeCloseTo(0.55, 4);
        expect(normalizeExpenseRatio(1.99)).toBeCloseTo(1.99, 4);
        expect(normalizeExpenseRatio(0.03)).toBeCloseTo(0.03, 4);
    });

    it('scales values too small to be a percent', () => {
        expect(normalizeExpenseRatio(0.0003)).toBeCloseTo(0.03, 4);
    });

    it('returns null for missing or non-positive input', () => {
        expect(normalizeExpenseRatio(null)).toBeNull();
        expect(normalizeExpenseRatio(undefined)).toBeNull();
        expect(normalizeExpenseRatio(0)).toBeNull();
        expect(normalizeExpenseRatio(NaN)).toBeNull();
    });
});
