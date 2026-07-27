import Foundation

/// Disambiguating Yahoo Finance's dividend-yield encoding.
///
/// Yahoo delivers `dividendYield` in two different units depending on the
/// record, with no flag saying which:
///
///     VICI  -> dividendYield: 6.71     (percent:  6.71%)
///     SCHD  -> dividendYield: 0.033    (fraction: 3.3%)
///
/// The two ranges overlap badly. Measured over the 1,479 cached fundamentals
/// records that carry enough data to settle the question, 93.6% are
/// percent-encoded and 6.0% are fraction-encoded, with fraction values running
/// as high as 3.08 and percent values as low as 0.01. So *any* magnitude-only
/// threshold is guessing inside [0.01, 3.08]. Measured error rates for the
/// cutoffs this codebase has used:
///
///     > 0.12   ->  1.29% wrong
///     > 0.30   ->  4.14% wrong   (the backend's old last-resort)
///     > 1.00   -> 19.67% wrong
///     best possible single threshold (0.069) -> 0.41% wrong, an irreducible floor
///
/// The way out is to stop guessing from magnitude and corroborate against a
/// second signal instead: dividend rate over price, or Yahoo's
/// `trailingAnnualDividendYield`, which is always a fraction. Both readings of
/// the raw value are compared against that reference and the closer one wins.
/// One of the two is always ~100x off, so the choice is decisive rather than
/// probabilistic. A reference is available for 97.9% of records.
///
/// Because the check re-derives the encoding from the data every time, it is
/// correct whether the caller hands it a raw Yahoo value or one the backend has
/// already normalised to a fraction — it cannot double-scale.
///
/// Kept in lockstep with `web_app/lib/dividend.ts`.
enum DividendYield {

    /// Fallback cutoff, used only when no corroborating signal exists (~2% of
    /// records). Below it the fraction reading is the plausible one (0.05 → 5%,
    /// versus 0.05% which is vanishingly rare); above it the percent reading is
    /// (0.5 → 0.5%, versus a 50% yield). 0.10 sits near the empirically optimal
    /// 0.069 without being fitted to the sample.
    static let fractionCutoff = 0.10

    /// Expense ratios have no corroborating signal, and their observed encoding
    /// differs: across 105 cached records the median is 0.55 and the max 1.99,
    /// i.e. overwhelmingly percent-encoded (a 0.55 *fraction* would be a 55%
    /// fee). Only values below one basis point read more plausibly as a
    /// fraction — the cheapest real funds charge about 0.02%.
    static let expenseRatioFractionCutoff = 0.01

    private static func positive(_ value: Double?) -> Double? {
        guard let value, value.isFinite, value > 0 else { return nil }
        return value
    }

    /// The dividend yield as a **percentage** (15 means 15%), or nil when there
    /// is no usable yield.
    ///
    /// - Parameters:
    ///   - rawYield: Yahoo's `dividendYield`; a fraction on some records, a percent on others.
    ///   - dividendRate: Forward annual dividend per share, in the same currency as `price`.
    ///   - price: Current share price.
    ///   - trailingYield: Yahoo's `trailingAnnualDividendYield` — always a fraction.
    static func normalize(
        rawYield: Double?,
        dividendRate: Double? = nil,
        price: Double? = nil,
        trailingYield: Double? = nil
    ) -> Double? {
        guard let raw = positive(rawYield) else { return nil }

        // Reference yield, as a fraction. Rate/price is preferred: it is forward
        // looking and agrees with the projected-income figures elsewhere.
        let reference: Double?
        if let rate = positive(dividendRate), let px = positive(price) {
            reference = rate / px
        } else {
            reference = positive(trailingYield)
        }

        if let reference {
            // `raw` is either already a fraction, or a percent number 100x too
            // big. Compare in ratio space so the choice does not depend on the
            // yield's absolute size.
            let distance = { (candidate: Double) in abs(log(candidate / reference)) }
            let asFraction = raw
            let asPercentEncoded = raw / 100.0
            let chosen = distance(asFraction) <= distance(asPercentEncoded) ? asFraction : asPercentEncoded
            return chosen * 100.0
        }

        return raw <= fractionCutoff ? raw * 100.0 : raw
    }

    /// A fund expense ratio as a **percentage** (0.55 means 0.55%), or nil.
    static func normalizeExpenseRatio(_ rawRatio: Double?) -> Double? {
        guard let value = positive(rawRatio) else { return nil }
        return value < expenseRatioFractionCutoff ? value * 100.0 : value
    }
}
