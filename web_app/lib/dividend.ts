/**
 * Disambiguating Yahoo Finance's dividend-yield encoding.
 *
 * Yahoo delivers `dividendYield` in two different units depending on the
 * record, with no flag saying which:
 *
 *   VICI  -> dividendYield: 6.71     (percent:  6.71%)
 *   SCHD  -> dividendYield: 0.033    (fraction: 3.3%)
 *
 * The two ranges overlap badly. Measured over the 1,479 cached fundamentals
 * records that carry enough data to settle the question, 93.6% are
 * percent-encoded and 6.0% are fraction-encoded, with fraction values running
 * as high as 3.08 and percent values as low as 0.01. So *any* magnitude-only
 * threshold is guessing inside [0.01, 3.08]. Measured error rates for the
 * cutoffs this codebase has used:
 *
 *   > 0.12   ->  1.29% wrong
 *   > 0.30   ->  4.14% wrong   (the backend's old last-resort)
 *   > 1.00   -> 19.67% wrong
 *   best possible single threshold (0.069) -> 0.41% wrong, an irreducible floor
 *
 * The way out is to stop guessing from magnitude and corroborate against a
 * second signal instead: dividend rate over price, or Yahoo's
 * `trailingAnnualDividendYield`, which is always a fraction. Both readings of
 * the raw value are compared against that reference and the closer one wins.
 * One of the two is always ~100x off, so the choice is decisive rather than
 * probabilistic. A reference is available for 97.9% of records.
 *
 * Because the check re-derives the encoding from the data every time, it is
 * correct whether the caller hands it a raw Yahoo value or one the backend has
 * already normalised to a fraction — it cannot double-scale.
 */

/** Signals used to settle the encoding. Everything is optional; the more that
 *  is supplied, the less the result depends on the magnitude fallback. */
export interface DividendYieldSignals {
    /** Yahoo's `dividendYield`: a fraction on some records, a percent on others. */
    rawYield?: number | null;
    /** Forward annual dividend per share, in the same currency as `price`. */
    dividendRate?: number | null;
    /** Current share price. */
    price?: number | null;
    /** Yahoo's `trailingAnnualDividendYield` — always a fraction. */
    trailingYield?: number | null;
}

/**
 * Fallback cutoff, used only when no corroborating signal exists (~2% of
 * records). Below it, the fraction reading is the plausible one (0.05 -> 5%,
 * versus 0.05% which is vanishingly rare); above it the percent reading is
 * (0.5 -> 0.5%, versus a 50% yield). 0.10 sits near the empirically optimal
 * 0.069 without being fitted to the sample.
 */
const YIELD_FRACTION_CUTOFF = 0.10;

/**
 * Expense ratios have no corroborating signal, and their observed encoding is
 * different: across 105 cached records the median is 0.55 and the max 1.99,
 * i.e. overwhelmingly percent-encoded (a 0.55 *fraction* would be a 55% fee).
 * Only values below one basis point read more plausibly as a fraction — the
 * cheapest real funds charge about 0.02%.
 */
const EXPENSE_RATIO_FRACTION_CUTOFF = 0.01;

function positive(value: number | null | undefined): number | null {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null;
}

/**
 * Returns the dividend yield as a **percentage** (15 means 15%), or null when
 * there is no usable yield.
 */
export function normalizeDividendYield(signals: DividendYieldSignals): number | null {
    const raw = positive(signals.rawYield);
    if (raw === null) return null;

    const rate = positive(signals.dividendRate);
    const price = positive(signals.price);
    const trailing = positive(signals.trailingYield);

    // Reference yield, as a fraction. Rate/price is preferred: it is forward
    // looking and agrees with the projected-income figures elsewhere.
    const reference = rate !== null && price !== null ? rate / price : trailing;

    if (reference !== null) {
        // `raw` is either already a fraction, or a percent number 100x too big.
        // Compare in ratio space so the choice does not depend on the yield's
        // absolute size.
        const distance = (candidate: number) => Math.abs(Math.log(candidate / reference));
        const asFraction = raw;
        const asPercentEncoded = raw / 100;
        const chosen = distance(asFraction) <= distance(asPercentEncoded) ? asFraction : asPercentEncoded;
        return chosen * 100;
    }

    return raw <= YIELD_FRACTION_CUTOFF ? raw * 100 : raw;
}

/**
 * Returns a fund expense ratio as a **percentage** (0.55 means 0.55%), or null.
 */
export function normalizeExpenseRatio(raw: number | null | undefined): number | null {
    const value = positive(raw);
    if (value === null) return null;
    return value < EXPENSE_RATIO_FRACTION_CUTOFF ? value * 100 : value;
}
