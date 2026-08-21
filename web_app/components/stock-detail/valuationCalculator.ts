/**
 * Client-side Valuation Calculation Engine.
 * Implements exact formulas matching Python backend (src/financial_ratios.py)
 * for real-time parameter tweaking and comparison.
 */

export type ValuationModelKey =
    | 'dcf'
    | 'dcfo'
    | 'dni'
    | 'ddm'
    | 'mean_pe'
    | 'peg'
    | 'mean_pb'
    | 'mean_ps'
    | 'psg'
    | 'graham'
    | 'lynch'
    | 'epv';

export interface ModelParamConfig {
    key: string;
    label: string;
    description: string;
    unit: 'percent' | 'currency' | 'multiple' | 'years' | 'number';
    step: number;
    min: number;
    max: number;
    isPercent?: boolean;
}

export const MODEL_PARAM_CONFIGS: Record<ValuationModelKey, ModelParamConfig[]> = {
    dcf: [
        {
            key: 'discount_rate',
            label: 'Discount Rate (WACC)',
            description: 'Discount rate to convert future cash flows to present value.',
            unit: 'percent',
            step: 0.25,
            min: 1.0,
            max: 30.0,
            isPercent: true,
        },
        {
            key: 'growth_rate',
            label: 'Growth Rate',
            description: 'Expected explicit annual growth of free cash flows.',
            unit: 'percent',
            step: 0.5,
            min: -20.0,
            max: 50.0,
            isPercent: true,
        },
        {
            key: 'terminal_growth_rate',
            label: 'Terminal Growth Rate',
            description: 'Long-term perpetual growth rate in mature stage.',
            unit: 'percent',
            step: 0.25,
            min: 0.0,
            max: 8.0,
            isPercent: true,
        },
        {
            key: 'projection_years',
            label: 'Projection Years',
            description: 'Number of explicit projection years.',
            unit: 'years',
            step: 1,
            min: 3,
            max: 25,
        },
        {
            key: 'base_fcf',
            label: 'Base Free Cash Flow',
            description: 'Starting annual Free Cash Flow value in native currency.',
            unit: 'currency',
            step: 1000000,
            min: 1,
            max: 1000000000000,
        },
    ],
    dcfo: [
        {
            key: 'cfo_per_share',
            label: 'Base CFO / Share',
            description: 'Starting Operating Cash Flow per share.',
            unit: 'currency',
            step: 0.1,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'growth_rate',
            label: 'CFO Growth Rate',
            description: 'Expected growth rate for Operating Cash Flow.',
            unit: 'percent',
            step: 0.5,
            min: -20.0,
            max: 40.0,
            isPercent: true,
        },
        {
            key: 'discount_rate',
            label: 'Discount Rate (WACC)',
            description: 'Cost of capital discount rate.',
            unit: 'percent',
            step: 0.25,
            min: 2.0,
            max: 30.0,
            isPercent: true,
        },
        {
            key: 'terminal_growth_rate',
            label: 'Terminal Growth Rate',
            description: 'Long-term sustainable CFO growth rate.',
            unit: 'percent',
            step: 0.25,
            min: 0.0,
            max: 8.0,
            isPercent: true,
        },
        {
            key: 'projection_years',
            label: 'Projection Years',
            description: 'Number of explicit forecast years.',
            unit: 'years',
            step: 1,
            min: 3,
            max: 25,
        },
    ],
    dni: [
        {
            key: 'base_eps',
            label: 'Base EPS',
            description: 'Starting normalized Net Income per share.',
            unit: 'currency',
            step: 0.1,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'growth_rate',
            label: 'Earnings Growth Rate',
            description: 'Expected growth rate for Net Income.',
            unit: 'percent',
            step: 0.5,
            min: -20.0,
            max: 40.0,
            isPercent: true,
        },
        {
            key: 'discount_rate',
            label: 'Cost of Equity',
            description: 'Cost of equity discount rate for financial firms.',
            unit: 'percent',
            step: 0.25,
            min: 2.0,
            max: 30.0,
            isPercent: true,
        },
        {
            key: 'terminal_growth_rate',
            label: 'Terminal Growth Rate',
            description: 'Long-term earnings growth rate.',
            unit: 'percent',
            step: 0.25,
            min: 0.0,
            max: 8.0,
            isPercent: true,
        },
        {
            key: 'projection_years',
            label: 'Projection Years',
            description: 'Number of explicit forecast years.',
            unit: 'years',
            step: 1,
            min: 3,
            max: 25,
        },
    ],
    ddm: [
        {
            key: 'base_dividend',
            label: 'Base Dividend / Share',
            description: 'Annual cash dividend rate per share.',
            unit: 'currency',
            step: 0.05,
            min: 0.01,
            max: 1000,
        },
        {
            key: 'growth_rate',
            label: 'Dividend Growth Rate',
            description: 'Expected growth rate for dividend payouts.',
            unit: 'percent',
            step: 0.25,
            min: -10.0,
            max: 30.0,
            isPercent: true,
        },
        {
            key: 'discount_rate',
            label: 'Cost of Equity (CAPM)',
            description: 'Required rate of return on equity.',
            unit: 'percent',
            step: 0.25,
            min: 2.0,
            max: 30.0,
            isPercent: true,
        },
        {
            key: 'terminal_growth_rate',
            label: 'Terminal Growth Rate',
            description: 'Perpetual long-term dividend growth rate.',
            unit: 'percent',
            step: 0.25,
            min: 0.0,
            max: 6.0,
            isPercent: true,
        },
        {
            key: 'projection_years',
            label: 'Projection Years',
            description: 'Number of explicit forecast years.',
            unit: 'years',
            step: 1,
            min: 3,
            max: 25,
        },
    ],
    mean_pe: [
        {
            key: 'eps',
            label: 'Trailing EPS',
            description: 'Trailing 12 months diluted earnings per share.',
            unit: 'currency',
            step: 0.1,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'applied_pe',
            label: 'Fair P/E Multiple',
            description: 'Target or historical mean P/E multiple.',
            unit: 'multiple',
            step: 0.5,
            min: 3.0,
            max: 80.0,
        },
    ],
    peg: [
        {
            key: 'eps',
            label: 'Trailing EPS',
            description: 'Trailing 12 months earnings per share.',
            unit: 'currency',
            step: 0.1,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'growth_rate_pct',
            label: 'Expected Growth Rate',
            description: 'Projected annual earnings growth rate (%).',
            unit: 'percent',
            step: 0.5,
            min: 1.0,
            max: 60.0,
        },
        {
            key: 'dividend_yield_pct',
            label: 'Dividend Yield',
            description: 'Annual dividend yield (%).',
            unit: 'percent',
            step: 0.1,
            min: 0.0,
            max: 20.0,
        },
        {
            key: 'target_peg',
            label: 'Target PEG Ratio',
            description: 'Fair value benchmark PEG ratio (default 1.0x).',
            unit: 'multiple',
            step: 0.1,
            min: 0.4,
            max: 3.0,
        },
    ],
    mean_pb: [
        {
            key: 'book_value_per_share',
            label: 'Book Value / Share',
            description: 'Total stockholders equity per share.',
            unit: 'currency',
            step: 0.25,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'applied_pb',
            label: 'Applied P/B Target',
            description: 'Target or benchmark Price-to-Book multiple.',
            unit: 'multiple',
            step: 0.05,
            min: 0.3,
            max: 20.0,
        },
    ],
    mean_ps: [
        {
            key: 'sales_per_share',
            label: 'Sales / Share',
            description: 'Total revenue per share.',
            unit: 'currency',
            step: 0.25,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'applied_ps',
            label: 'Applied P/S Multiple',
            description: 'Historical or benchmark Price-to-Sales multiple.',
            unit: 'multiple',
            step: 0.1,
            min: 0.2,
            max: 30.0,
        },
    ],
    psg: [
        {
            key: 'sales_per_share',
            label: 'Sales / Share',
            description: 'Total annual revenue per share.',
            unit: 'currency',
            step: 0.25,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'revenue_growth_pct',
            label: 'Revenue Growth Rate',
            description: 'Projected annual revenue growth rate (%).',
            unit: 'percent',
            step: 0.5,
            min: 1.0,
            max: 80.0,
        },
        {
            key: 'gross_margin_pct',
            label: 'Gross Margin',
            description: 'Gross profit as a percentage of revenue (%).',
            unit: 'percent',
            step: 0.5,
            min: 10.0,
            max: 99.0,
        },
        {
            key: 'target_psg',
            label: 'Target PSG Ratio',
            description: 'Target Price-to-Sales Growth ratio (default 1.0x).',
            unit: 'multiple',
            step: 0.1,
            min: 0.4,
            max: 3.0,
        },
    ],
    graham: [
        {
            key: 'eps',
            label: 'Trailing EPS',
            description: 'Earnings per share used as base in Graham formula.',
            unit: 'currency',
            step: 0.1,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'growth_rate_pct',
            label: 'Expected Growth (g)',
            description: '7-10 year expected annual growth rate (%).',
            unit: 'percent',
            step: 0.5,
            min: -5.0,
            max: 30.0,
        },
        {
            key: 'bond_yield_proxy',
            label: 'AAA Bond Yield (Y)',
            description: 'Current yield on AAA corporate bonds / 10Y Treasury (%).',
            unit: 'percent',
            step: 0.1,
            min: 2.0,
            max: 15.0,
        },
    ],
    lynch: [
        {
            key: 'eps',
            label: 'Trailing EPS',
            description: 'Earnings per share.',
            unit: 'currency',
            step: 0.1,
            min: 0.01,
            max: 10000,
        },
        {
            key: 'growth_rate_pct',
            label: 'Earnings Growth Rate',
            description: 'Long-term earnings growth rate (%).',
            unit: 'percent',
            step: 0.5,
            min: 1.0,
            max: 50.0,
        },
        {
            key: 'dividend_yield_pct',
            label: 'Dividend Yield',
            description: 'Dividend yield rate (%).',
            unit: 'percent',
            step: 0.1,
            min: 0.0,
            max: 20.0,
        },
    ],
    epv: [
        {
            key: 'normalized_ebit',
            label: 'Normalized Operating EBIT',
            description: 'Normalized steady-state operating income (EBIT).',
            unit: 'currency',
            step: 1000000,
            min: 1,
            max: 1000000000000,
        },
        {
            key: 'discount_rate',
            label: 'Cost of Capital',
            description: 'Weighted average cost of capital (WACC).',
            unit: 'percent',
            step: 0.25,
            min: 3.0,
            max: 30.0,
            isPercent: true,
        },
        {
            key: 'tax_rate',
            label: 'Effective Tax Rate',
            description: 'Estimated sustainable tax rate (%).',
            unit: 'percent',
            step: 0.5,
            min: 0.0,
            max: 50.0,
            isPercent: true,
        },
        {
            key: 'net_cash',
            label: 'Net Cash / (Net Debt)',
            description: 'Total Cash minus Total Debt.',
            unit: 'currency',
            step: 1000000,
            min: -1000000000000,
            max: 1000000000000,
        },
    ],
};

function clip(val: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, val));
}

export function calculateCustomModelValue(
    modelKey: ValuationModelKey,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    modelData: any,
    customValues: Record<string, number> = {}
): number | null {
    if (!modelData) return null;
    const baseParams = modelData.parameters !== undefined ? modelData.parameters : modelData;
    const p = { ...(baseParams || {}), ...customValues };

    try {
        switch (modelKey) {
            case 'dcf': {
                const baseFcf = p.base_fcf ?? 0;
                if (baseFcf <= 0) return null;
                const discountRate = clip(p.discount_rate ?? 0.09, 0.075, 0.30);
                const growthRate = p.growth_rate ?? 0.05;
                const appliedGrowth = clip(growthRate, -0.05, 0.30);
                const terminalGrowth = p.terminal_growth_rate ?? 0.02;
                const projectionYears = Math.round(p.projection_years ?? 10);
                const shares = p.shares_outstanding;
                const cash = p.total_cash ?? 0;
                const debt = p.total_debt ?? 0;

                let pvFcf = 0;
                let currentFcf = baseFcf;
                for (let y = 1; y <= projectionYears; y++) {
                    const fade = projectionYears > 1 ? (y - 1) / (projectionYears - 1) : 0;
                    const yearlyG = appliedGrowth - (appliedGrowth - terminalGrowth) * fade;
                    const nextFcf = currentFcf * (1 + yearlyG);
                    pvFcf += nextFcf / Math.pow(1 + discountRate, y - 0.5);
                    currentFcf = nextFcf;
                }

                const safeDiscount = Math.max(discountRate, terminalGrowth + 0.01);
                const terminalValue = (currentFcf * (1 + terminalGrowth)) / (safeDiscount - terminalGrowth);
                const pvTerminal = terminalValue / Math.pow(1 + discountRate, projectionYears);

                const enterpriseValue = pvFcf + pvTerminal;
                const equityValue = enterpriseValue + cash - debt;

                if (shares && shares > 0) {
                    const iv = equityValue / shares;
                    return iv > 0 && isFinite(iv) ? iv : null;
                }

                // Fallback: proportional scaling if shares are unavailable
                if (modelData.intrinsic_value && modelData.parameters?.base_fcf) {
                    const scale = baseFcf / modelData.parameters.base_fcf;
                    return modelData.intrinsic_value * scale;
                }
                return null;
            }

            case 'dcfo': {
                const cfoPerShare = p.cfo_per_share ?? 0;
                if (cfoPerShare <= 0) return null;
                const discountRate = clip(p.discount_rate ?? 0.09, 0.06, 0.25);
                const growthRate = p.growth_rate ?? 0.05;
                const appliedGrowth = clip(growthRate, -0.05, 0.20);
                const terminalGrowth = p.terminal_growth_rate ?? 0.02;
                const projectionYears = Math.round(p.projection_years ?? 10);
                const netCashPerShare = p.net_cash_per_share ?? 0;

                let pvCfos = 0;
                let currentCfo = cfoPerShare;
                for (let y = 1; y <= projectionYears; y++) {
                    const fade = projectionYears > 1 ? (y - 1) / (projectionYears - 1) : 0;
                    const yearlyG = appliedGrowth - (appliedGrowth - terminalGrowth) * fade;
                    const nextCfo = currentCfo * (1 + yearlyG);
                    pvCfos += nextCfo / Math.pow(1 + discountRate, y - 0.5);
                    currentCfo = nextCfo;
                }

                const safeDiscount = Math.max(discountRate, terminalGrowth + 0.01);
                const terminalCfo = (currentCfo * (1 + terminalGrowth)) / (safeDiscount - terminalGrowth);
                const pvTerminal = terminalCfo / Math.pow(1 + discountRate, projectionYears);

                const iv = pvCfos + pvTerminal + netCashPerShare;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'dni': {
                const baseEps = p.base_eps ?? 0;
                if (baseEps <= 0) return null;
                const discountRate = clip(p.discount_rate ?? 0.09, 0.06, 0.25);
                const growthRate = p.growth_rate ?? 0.05;
                const appliedGrowth = clip(growthRate, -0.05, 0.20);
                const terminalGrowth = p.terminal_growth_rate ?? 0.02;
                const projectionYears = Math.round(p.projection_years ?? 10);

                let pvEps = 0;
                let currentEps = baseEps;
                for (let y = 1; y <= projectionYears; y++) {
                    const fade = projectionYears > 1 ? (y - 1) / (projectionYears - 1) : 0;
                    const yearlyG = appliedGrowth - (appliedGrowth - terminalGrowth) * fade;
                    const nextEps = currentEps * (1 + yearlyG);
                    pvEps += nextEps / Math.pow(1 + discountRate, y - 0.5);
                    currentEps = nextEps;
                }

                const safeDiscount = Math.max(discountRate, terminalGrowth + 0.01);
                const terminalVal = (currentEps * (1 + terminalGrowth)) / (safeDiscount - terminalGrowth);
                const pvTerminal = terminalVal / Math.pow(1 + discountRate, projectionYears);

                const iv = pvEps + pvTerminal;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'ddm': {
                const baseDividend = p.base_dividend ?? 0;
                if (baseDividend <= 0) return null;
                const discountRate = clip(p.discount_rate ?? 0.09, 0.06, 0.25);
                const growthRate = p.growth_rate ?? 0.035;
                const appliedGrowth = clip(growthRate, -0.05, 0.15);
                const terminalGrowth = p.terminal_growth_rate ?? 0.02;
                const projectionYears = Math.round(p.projection_years ?? 10);

                let pvDivs = 0;
                let currentDiv = baseDividend;
                for (let y = 1; y <= projectionYears; y++) {
                    const fade = projectionYears > 1 ? (y - 1) / (projectionYears - 1) : 0;
                    const yearlyG = appliedGrowth - (appliedGrowth - terminalGrowth) * fade;
                    const nextDiv = currentDiv * (1 + yearlyG);
                    pvDivs += nextDiv / Math.pow(1 + discountRate, y - 0.5);
                    currentDiv = nextDiv;
                }

                const safeDiscount = Math.max(discountRate, terminalGrowth + 0.01);
                const terminalPrice = (currentDiv * (1 + terminalGrowth)) / (safeDiscount - terminalGrowth);
                const pvTerminal = terminalPrice / Math.pow(1 + discountRate, projectionYears);

                const iv = pvDivs + pvTerminal;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'mean_pe': {
                const eps = p.eps ?? 0;
                if (eps <= 0) return null;
                const appliedPe = clip(p.applied_pe ?? 17.5, 5.0, 45.0);
                const iv = eps * appliedPe;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'peg': {
                const eps = p.eps ?? 0;
                if (eps <= 0) return null;
                const growthRatePct = p.growth_rate_pct ?? p.growth_rate ?? 10.0;
                const divYieldPct = p.dividend_yield_pct ?? 0.0;
                const appliedGrowth = clip(growthRatePct, 3.0, 35.0);
                const targetPeg = clip(p.target_peg ?? 1.0, 0.5, 2.0);
                const fairPe = clip(targetPeg * (appliedGrowth + divYieldPct), 5.0, 35.0);
                const iv = eps * fairPe;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'mean_pb': {
                const bvps = p.book_value_per_share ?? 0;
                if (bvps <= 0) return null;
                const appliedPb = clip(p.applied_pb ?? 1.5, 0.4, 8.0);
                const iv = bvps * appliedPb;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'mean_ps': {
                const sps = p.sales_per_share ?? 0;
                if (sps <= 0) return null;
                const appliedPs = clip(p.applied_ps ?? 2.5, 0.4, 15.0);
                const iv = sps * appliedPs;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'psg': {
                const sps = p.sales_per_share ?? 0;
                if (sps <= 0) return null;
                const revGrowthPct = p.revenue_growth_pct ?? p.applied_growth_pct ?? 15.0;
                const appliedGrowth = clip(revGrowthPct, 5.0, 45.0);
                const grossMarginPct = p.gross_margin_pct ?? 50.0;
                const gm = clip(grossMarginPct / 100.0, 0.20, 0.95);
                const targetPsg = clip(p.target_psg ?? 1.0, 0.5, 2.0);
                const fairPs = clip((appliedGrowth / 10.0) * gm * targetPsg * 1.5, 0.5, 12.0);
                const iv = sps * fairPs;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'graham': {
                const eps = p.eps ?? 0;
                if (eps <= 0) return null;
                const growthRatePct = p.growth_rate_pct ?? p.applied_growth_pct ?? 10.0;
                const appliedGrowth = clip(growthRatePct, -5.0, 15.0);
                const bondYield = Math.max(p.bond_yield_proxy ?? 4.5, 3.5);
                const iv = (eps * (8.5 + 2 * appliedGrowth) * 4.4) / bondYield;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'lynch': {
                const eps = p.eps ?? 0;
                if (eps <= 0) return null;
                const growthRatePct = p.growth_rate_pct ?? 10.0;
                const divYieldPct = p.dividend_yield_pct ?? 0.0;
                const fairMultiplier = clip(growthRatePct + divYieldPct, 5.0, 25.0);
                const iv = eps * fairMultiplier;
                return iv > 0 && isFinite(iv) ? iv : null;
            }

            case 'epv': {
                const normEbit = p.normalized_ebit ?? 0;
                if (normEbit <= 0) return null;
                const discountRate = clip(p.discount_rate ?? 0.09, 0.06, 0.25);
                const taxRate = clip(p.tax_rate ?? 0.21, 0.0, 0.50);
                const netCash = p.net_cash ?? 0;
                const shares = p.shares;

                const nopat = normEbit * (1 - taxRate);
                const enterpriseValue = nopat / discountRate;
                const equityValue = enterpriseValue + netCash;

                if (shares && shares > 0) {
                    const iv = equityValue / shares;
                    return iv > 0 && isFinite(iv) ? iv : null;
                }

                if (modelData.intrinsic_value && modelData.parameters?.normalized_ebit) {
                    const scale = normEbit / modelData.parameters.normalized_ebit;
                    return modelData.intrinsic_value * scale;
                }
                return null;
            }

            default:
                return null;
        }
    } catch {
        return null;
    }
}

export function calculateBlendedScore(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    intrinsicValue: any,
    customOverrides: Record<string, Record<string, number>> = {},
    sector = ''
): {
    customAverage: number | null;
    customMarginOfSafety: number | null;
    customModelValues: Record<string, number | null>;
    hasAnyCustom: boolean;
} {
    if (!intrinsicValue || !intrinsicValue.models) {
        return {
            customAverage: null,
            customMarginOfSafety: null,
            customModelValues: {},
            hasAnyCustom: false,
        };
    }

    const currentPrice = intrinsicValue.current_price ?? null;
    const models = intrinsicValue.models;
    const customModelValues: Record<string, number | null> = {};
    let hasAnyCustom = false;

    // Calculate each model's custom value
    (Object.keys(models) as ValuationModelKey[]).forEach((key) => {
        const modelData = models[key];
        if (!modelData) return;
        const overrides = customOverrides[key];
        const defaultParams = modelData.parameters || {};
        const isActuallyModified = overrides && Object.keys(overrides).some((k) => {
            const def = defaultParams[k];
            return def !== undefined && Math.abs(overrides[k] - def) > 1e-5;
        });

        if (isActuallyModified) {
            hasAnyCustom = true;
            customModelValues[key] = calculateCustomModelValue(key, modelData, overrides);
        } else {
            customModelValues[key] = modelData.intrinsic_value ?? null;
        }
    });

    const isFinancial = ['financial', 'bank', 'insurance', 'real estate', 'utilities'].some((k) =>
        (sector || '').toLowerCase().includes(k)
    );

    // Weights matching backend
    const weights: Record<string, number> = isFinancial
        ? { ddm: 0.35, graham: 0.35, dcf: 0.3 }
        : customModelValues.ddm != null
          ? { dcf: 0.5, graham: 0.3, ddm: 0.2 }
          : { dcf: 0.6, graham: 0.4 };

    const contributions: { key: string; val: number; weight: number }[] = [];
    ['dcf', 'graham', 'ddm'].forEach((k) => {
        const val = customModelValues[k];
        if (val != null && val > 0 && isFinite(val)) {
            contributions.push({ key: k, val, weight: weights[k] || 0.33 });
        }
    });

    let customAverage: number | null = null;
    if (contributions.length > 0) {
        const totalW = contributions.reduce((acc, c) => acc + c.weight, 0);
        customAverage = contributions.reduce((acc, c) => acc + c.val * c.weight, 0) / totalW;
    } else {
        customAverage = intrinsicValue.average_intrinsic_value ?? null;
    }

    let customMarginOfSafety: number | null = null;
    if (customAverage != null && currentPrice && currentPrice > 0) {
        customMarginOfSafety = ((customAverage - currentPrice) / currentPrice) * 100;
    }

    return {
        customAverage,
        customMarginOfSafety,
        customModelValues,
        hasAnyCustom,
    };
}
