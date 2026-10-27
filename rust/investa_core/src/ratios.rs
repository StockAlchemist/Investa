//! The per-period arithmetic of `financial_ratios.calculate_key_ratios_timeseries`:
//! margins, returns, liquidity, solvency, ROIC, cash flow, growth and the
//! point-in-time valuation multiples, for every period of a company's
//! statements.
//!
//! Python resolves each statement lookup (the label, then its aliases, first
//! value present) once per frame, matches each period to its columns, finds
//! the price on the period end, and builds the result frame. A lookup that
//! found nothing is `None` here, as `_get_statement_value` returns it.
//!
//! Python truthiness is reproduced exactly: `a or b` returns `a` only when it
//! is a non-zero number, so a filed 0.0 falls through to the next label, and
//! `if x and ...` is false for None and for 0.0.

/// Input slots, one per statement lookup the original makes.
pub mod slot {
    // Income statement, this period.
    pub const TOTAL_REVENUE: usize = 0;
    pub const COST_OF_REVENUE: usize = 1;
    pub const GROSS_PROFIT: usize = 2;
    pub const OPERATING_INCOME: usize = 3;
    pub const OPERATING_REVENUE_OR_LOSS: usize = 4;
    pub const EBIT: usize = 5;
    pub const NET_INCOME: usize = 6;
    pub const NET_INCOME_CONTINUING: usize = 7;
    pub const INTEREST_EXPENSE: usize = 8;
    pub const PRETAX_INCOME: usize = 9;
    pub const TAX_PROVISION: usize = 10;
    pub const DILUTED_AVERAGE_SHARES: usize = 11;
    pub const DILUTED_EPS: usize = 12;
    pub const EBITDA_UPPER: usize = 13;
    pub const EBITDA: usize = 14;
    pub const NORMALIZED_EBITDA: usize = 15;
    pub const RECONCILED_DEPRECIATION: usize = 16;
    pub const DEPRECIATION_AND_AMORTIZATION_IS: usize = 17;
    // Balance sheet, this period.
    pub const STOCKHOLDERS_EQUITY: usize = 18;
    pub const TOTAL_STOCKHOLDER_EQUITY: usize = 19;
    pub const TOTAL_EQUITY_GROSS_MINORITY: usize = 20;
    pub const TOTAL_ASSETS: usize = 21;
    pub const CURRENT_ASSETS: usize = 22;
    pub const TOTAL_CURRENT_ASSETS: usize = 23;
    pub const CURRENT_LIABILITIES: usize = 24;
    pub const TOTAL_CURRENT_LIABILITIES: usize = 25;
    pub const INVENTORY: usize = 26;
    pub const TOTAL_LIABILITIES_NET_MINORITY: usize = 27;
    pub const TOTAL_LIAB: usize = 28;
    pub const TOTAL_LIABILITIES: usize = 29;
    pub const LONG_TERM_DEBT: usize = 30;
    pub const LONG_TERM_DEBT_AND_LEASES: usize = 31;
    pub const LONG_TERM_DEBT_NON_CURRENT: usize = 32;
    pub const TOTAL_DEBT: usize = 33;
    pub const CASH: usize = 34;
    pub const ORDINARY_SHARES: usize = 35;
    // Cash-flow statement, this period (None when it has no such column).
    pub const OPERATING_CASH_FLOW: usize = 36;
    pub const CAPITAL_EXPENDITURE: usize = 37;
    pub const COMMON_DIVIDEND_PAID: usize = 38;
    pub const CASH_DIVIDENDS_PAID: usize = 39;
    pub const DEPRECIATION_AND_AMORTIZATION_CF: usize = 40;
    pub const DEPRECIATION_AMORTIZATION_DEPLETION: usize = 41;
    pub const DEPRECIATION: usize = 42;
    // One year earlier.
    pub const PREV_TOTAL_STOCKHOLDER_EQUITY: usize = 43;
    pub const PREV_TOTAL_ASSETS: usize = 44;
    pub const PREV_TOTAL_REVENUE: usize = 45;
    pub const PREV_DILUTED_EPS: usize = 46;
    /// The price on or before the period end.
    pub const PRICE: usize = 47;
    pub const COUNT: usize = 48;
}

/// Output fields, in the original's key order (after "Period").
pub const FIELDS: [&str; 28] = [
    "Gross Profit Margin (%)",
    "Operating Margin (%)",
    "Net Profit Margin (%)",
    "Return on Equity (ROE) (%)",
    "Return on Assets (ROA) (%)",
    "Current Ratio",
    "Quick Ratio",
    "Debt-to-Equity Ratio",
    "Long-Term Debt to Equity",
    "Interest Coverage Ratio",
    "Asset Turnover",
    "Return on Invested Capital (ROIC) (%)",
    "Free Cash Flow Margin (%)",
    "Diluted Shares Outstanding",
    "Total Revenue",
    "Operating Income",
    "Net Income",
    "Diluted EPS",
    "Revenue Growth YoY (%)",
    "EPS Growth YoY (%)",
    "Price",
    "P/E Ratio",
    "P/S Ratio",
    "P/B Ratio",
    "P/FCF Ratio",
    "EV/EBITDA",
    "EV/Sales",
    "Dividend Yield (%)",
];

/// One period's flags: whether a year-earlier balance-sheet / income column
/// exists, and whether the cash-flow statement has this period.
#[derive(Debug, Clone, Copy, Default)]
pub struct Flags {
    pub prev_balance: bool,
    pub prev_income: bool,
    pub cashflow: bool,
}

/// Python's `if x`: a number other than zero (NaN counts, as in Python).
fn truthy(x: Option<f64>) -> bool {
    matches!(x, Some(v) if v != 0.0)
}

/// Python's `a or b`.
fn or(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    if truthy(a) { a } else { b }
}

/// `a or 0.0`.
fn or_zero(a: Option<f64>) -> f64 {
    if truthy(a) { a.unwrap_or(0.0) } else { 0.0 }
}

/// One period's ratios, in `FIELDS` order. `None` where the original stores
/// None (the level figures and the price); NaN where it stores `np.nan`.
pub fn period_ratios(input: &[Option<f64>; slot::COUNT], flags: Flags) -> [Option<f64>; 28] {
    use slot::*;
    let at = |k: usize| input[k];
    let nan = Some(f64::NAN);

    // 1. Profitability & operations
    let revenue = at(TOTAL_REVENUE);
    let cost_of_revenue = at(COST_OF_REVENUE);
    let mut gross_profit = at(GROSS_PROFIT);
    if gross_profit.is_none()
        && let (Some(r), Some(c)) = (revenue, cost_of_revenue)
    {
        gross_profit = Some(r - c);
    }
    let mut operating_income = or(at(OPERATING_INCOME), at(OPERATING_REVENUE_OR_LOSS));
    let ebit = at(EBIT);
    if operating_income.is_none() && ebit.is_some() {
        operating_income = ebit;
    }
    let net_income = at(NET_INCOME).or(at(NET_INCOME_CONTINUING));
    let total_equity = or(
        or(at(STOCKHOLDERS_EQUITY), at(TOTAL_STOCKHOLDER_EQUITY)),
        at(TOTAL_EQUITY_GROSS_MINORITY),
    );
    let total_assets = at(TOTAL_ASSETS);
    let (mut avg_equity, mut avg_assets) = (total_equity, total_assets);
    if flags.prev_balance {
        if let (Some(e), Some(p)) = (total_equity, at(PREV_TOTAL_STOCKHOLDER_EQUITY)) {
            avg_equity = Some((e + p) / 2.0);
        }
        if let (Some(a), Some(p)) = (total_assets, at(PREV_TOTAL_ASSETS)) {
            avg_assets = Some((a + p) / 2.0);
        }
    }
    let over_revenue = |x: Option<f64>| match (x, revenue) {
        (Some(x), Some(r)) if truthy(revenue) => Some((x / r) * 100.0),
        _ => nan,
    };
    let gross_margin = over_revenue(gross_profit);
    let operating_margin = over_revenue(operating_income);
    let net_margin = over_revenue(net_income);
    let roe = match (net_income, avg_equity) {
        (Some(n), Some(e)) if truthy(avg_equity) => Some((n / e) * 100.0),
        _ => nan,
    };
    let roa = match (net_income, avg_assets) {
        (Some(n), Some(a)) if truthy(avg_assets) => Some((n / a) * 100.0),
        _ => nan,
    };

    // 2. Liquidity
    let current_assets = or(at(CURRENT_ASSETS), at(TOTAL_CURRENT_ASSETS));
    let current_liabilities = or(at(CURRENT_LIABILITIES), at(TOTAL_CURRENT_LIABILITIES));
    let inventory = at(INVENTORY);
    let (current_ratio, quick_ratio) = match (current_assets, current_liabilities) {
        (Some(a), Some(l)) if truthy(current_liabilities) => {
            (Some(a / l), Some((a - or_zero(inventory)) / l))
        }
        _ => (nan, nan),
    };

    // 3. Solvency
    let total_liab = or(
        or(at(TOTAL_LIABILITIES_NET_MINORITY), at(TOTAL_LIAB)),
        at(TOTAL_LIABILITIES),
    );
    let over_equity = |x: Option<f64>| match (x, total_equity) {
        (Some(x), Some(e)) if truthy(total_equity) => Some(x / e),
        _ => nan,
    };
    let debt_to_equity = over_equity(total_liab);
    let lt_debt = or(
        or(at(LONG_TERM_DEBT), at(LONG_TERM_DEBT_AND_LEASES)),
        at(LONG_TERM_DEBT_NON_CURRENT),
    );
    let lt_debt_to_equity = over_equity(lt_debt);
    let interest_exp = at(INTEREST_EXPENSE).map(|v| if v < 0.0 { v.abs() } else { v });
    let interest_coverage = match (ebit, interest_exp) {
        (Some(e), Some(i)) if truthy(interest_exp) => Some(e / i),
        _ => nan,
    };

    // 4. Efficiency
    let asset_turnover = match (revenue, avg_assets) {
        (Some(r), Some(a)) if truthy(avg_assets) => Some(r / a),
        _ => nan,
    };

    // 5. ROIC
    let pretax = at(PRETAX_INCOME);
    let tax = at(TAX_PROVISION);
    let total_debt = at(TOTAL_DEBT);
    let cash = at(CASH);
    let mut roic = nan;
    if let (Some(p), Some(e)) = (pretax, total_equity)
        && truthy(total_equity)
    {
        let operating_earnings = p + or_zero(interest_exp);
        let mut tax_rate = match tax {
            Some(t) if truthy(pretax) && p > 0.0 => Some(t / p),
            _ => None,
        };
        if !matches!(tax_rate, Some(r) if (0.0..=0.6).contains(&r)) {
            tax_rate = Some(0.21);
        }
        let nopat = operating_earnings * (1.0 - tax_rate.unwrap_or(0.21));
        let invested = e + or_zero(total_debt) - or_zero(cash);
        if invested != 0.0 {
            let candidate = nopat / invested;
            if -2.0 < candidate && candidate < 3.0 {
                roic = Some(candidate * 100.0);
            }
        }
    }

    // 6. Cash flow & shares
    let mut fcf_margin = nan;
    let mut shares_out = nan;
    let (mut ocf, mut capex, mut div_paid) = (None, None, None);
    if flags.cashflow {
        ocf = at(OPERATING_CASH_FLOW);
        capex = at(CAPITAL_EXPENDITURE);
        div_paid = or(at(COMMON_DIVIDEND_PAID), at(CASH_DIVIDENDS_PAID));
        if let (Some(o), Some(c), Some(r)) = (ocf, capex, revenue)
            && truthy(revenue)
        {
            fcf_margin = Some(((o + c) / r) * 100.0);
        }
    }
    let shares = or(at(DILUTED_AVERAGE_SHARES), at(ORDINARY_SHARES));
    if truthy(shares) {
        shares_out = shares;
    }
    let shares_positive = matches!(shares, Some(s) if s != 0.0 && s > 0.0);

    // 7. Level figures & growth
    let mut eps = at(DILUTED_EPS);
    if eps.is_none()
        && let Some(n) = net_income
        && shares_positive
    {
        eps = Some(n / shares.unwrap_or(f64::NAN));
    }
    let mut revenue_growth = nan;
    let mut eps_growth = nan;
    if flags.prev_income {
        let prev_rev = at(PREV_TOTAL_REVENUE);
        if let (Some(p), Some(r)) = (prev_rev, revenue)
            && truthy(prev_rev)
            && p > 0.0
        {
            revenue_growth = Some(((r / p) - 1.0) * 100.0);
        }
        let prev_eps = at(PREV_DILUTED_EPS);
        if let (Some(p), Some(e)) = (prev_eps, eps)
            && truthy(prev_eps)
            && p > 0.0
            && e > 0.0
        {
            eps_growth = Some(((e / p) - 1.0) * 100.0);
        }
    }

    // 8. Valuation multiples at the period's price
    let price = at(PRICE);
    let (mut pe, mut ps, mut pb, mut pfcf, mut ev_ebitda, mut ev_sales, mut div_yield) =
        (nan, nan, nan, nan, nan, nan, nan);
    if let Some(px) = price
        && shares_positive
    {
        let s = shares.unwrap_or(f64::NAN);
        let mkt_cap = px * s;
        if matches!(eps, Some(e) if e != 0.0 && e > 0.0) {
            pe = Some(px / eps.unwrap_or(f64::NAN));
        } else if let Some(n) = net_income
            && n != 0.0
            && n > 0.0
        {
            pe = Some(mkt_cap / n);
        }
        if let Some(r) = revenue
            && r != 0.0
            && r > 0.0
        {
            ps = Some(mkt_cap / r);
        }
        if let Some(e) = total_equity
            && e != 0.0
            && e > 0.0
        {
            pb = Some(mkt_cap / e);
        }
        let fcf = match (ocf, capex) {
            (Some(o), Some(c)) => Some(o + c),
            _ => None,
        };
        if let Some(f) = fcf
            && f != 0.0
            && f > 0.0
        {
            pfcf = Some(mkt_cap / f);
        }
        let ev = mkt_cap + or_zero(total_debt) - or_zero(cash);
        if let Some(r) = revenue
            && r != 0.0
            && r > 0.0
            && ev > 0.0
        {
            ev_sales = Some(ev / r);
        }
        let mut ebitda = or(or(at(EBITDA_UPPER), at(EBITDA)), at(NORMALIZED_EBITDA));
        if ebitda.is_none() {
            let base = if ebit.is_some() {
                ebit
            } else {
                operating_income
            };
            if let Some(b) = base {
                let from_cashflow = |k: usize| if flags.cashflow { at(k) } else { None };
                let depr = or(
                    or(
                        or(
                            or(
                                at(RECONCILED_DEPRECIATION),
                                at(DEPRECIATION_AND_AMORTIZATION_IS),
                            ),
                            from_cashflow(DEPRECIATION_AND_AMORTIZATION_CF),
                        ),
                        from_cashflow(DEPRECIATION_AMORTIZATION_DEPLETION),
                    ),
                    from_cashflow(DEPRECIATION),
                );
                ebitda = Some(match depr {
                    Some(d) => b + d.abs(),
                    None => b,
                });
            }
        }
        if let Some(e) = ebitda
            && e != 0.0
            && e > 0.0
            && ev > 0.0
        {
            ev_ebitda = Some(ev / e);
        }
        if let Some(d) = div_paid
            && mkt_cap > 0.0
        {
            div_yield = Some((d.abs() / mkt_cap) * 100.0);
        }
    }

    [
        gross_margin,
        operating_margin,
        net_margin,
        roe,
        roa,
        current_ratio,
        quick_ratio,
        debt_to_equity,
        lt_debt_to_equity,
        interest_coverage,
        asset_turnover,
        roic,
        fcf_margin,
        shares_out,
        revenue,
        operating_income,
        net_income,
        eps,
        revenue_growth,
        eps_growth,
        price,
        pe,
        ps,
        pb,
        pfcf,
        ev_ebitda,
        ev_sales,
        div_yield,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(pairs: &[(usize, f64)]) -> [Option<f64>; slot::COUNT] {
        let mut a = [None; slot::COUNT];
        for &(k, v) in pairs {
            a[k] = Some(v);
        }
        a
    }

    #[test]
    fn a_filed_zero_falls_through_to_the_next_label() {
        use slot::*;
        let i = input(&[
            (TOTAL_REVENUE, 100.0),
            (OPERATING_INCOME, 0.0),
            (OPERATING_REVENUE_OR_LOSS, 20.0),
        ]);
        let out = period_ratios(&i, Flags::default());
        assert_eq!(out[1], Some(20.0)); // operating margin uses the fallback label
        assert_eq!(out[15], Some(20.0));
    }

    #[test]
    fn a_tax_rate_outside_the_band_is_replaced_by_21_percent() {
        use slot::*;
        let i = input(&[
            (PRETAX_INCOME, 100.0),
            (TAX_PROVISION, 90.0),
            (STOCKHOLDERS_EQUITY, 500.0),
        ]);
        let roic = period_ratios(&i, Flags::default())[11].unwrap();
        assert_eq!(roic, (100.0 * (1.0 - 0.21)) / 500.0 * 100.0);
    }

    #[test]
    fn levels_stay_none_and_ratios_nan_when_nothing_is_filed() {
        let out = period_ratios(&[None; slot::COUNT], Flags::default());
        assert!(out[0].unwrap().is_nan());
        assert_eq!(out[14], None);
        assert_eq!(out[20], None);
    }
}
