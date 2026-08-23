import { describe, it, expect } from 'vitest';
import {
    calculateCustomModelValue,
    calculateBlendedScore,
    MODEL_PARAM_CONFIGS,
} from '../../components/stock-detail/valuationCalculator';

describe('valuationCalculator', () => {
    describe('calculateCustomModelValue', () => {
        it('calculates DCF accurately with overrides', () => {
            const defaultParams = {
                discount_rate: 0.10,
                growth_rate: 0.12,
                terminal_growth_rate: 0.02,
                projection_years: 10,
                base_fcf: 100_000_000,
                total_cash: 20_000_000,
                total_debt: 10_000_000,
                shares_outstanding: 10_000_000,
            };

            const defaultVal = calculateCustomModelValue('dcf', defaultParams, {});
            expect(defaultVal).toBeGreaterThan(0);

            // Increase growth rate override
            const higherGrowthVal = calculateCustomModelValue('dcf', defaultParams, { growth_rate: 0.20 });
            expect(higherGrowthVal).toBeGreaterThan(defaultVal!);

            // Increase discount rate override (WACC) -> should reduce intrinsic value
            const higherWaccVal = calculateCustomModelValue('dcf', defaultParams, { discount_rate: 0.15 });
            expect(higherWaccVal).toBeLessThan(defaultVal!);
        });

        it('calculates Mean P/E accurately with overrides', () => {
            const defaultParams = {
                eps: 5.0,
                applied_pe: 20.0,
            };

            const defaultVal = calculateCustomModelValue('mean_pe', defaultParams, {});
            expect(defaultVal).toBe(100.0);

            const customVal = calculateCustomModelValue('mean_pe', defaultParams, { applied_pe: 25.0 });
            expect(customVal).toBe(125.0);
        });

        it('calculates Graham Revised Formula with overrides', () => {
            const defaultParams = {
                eps: 4.0,
                growth_rate_pct: 10.0,
                bond_yield_proxy: 4.4,
            };

            // Graham: (EPS * (8.5 + 2g) * 4.4) / Y
            // (4 * (8.5 + 20) * 4.4) / 4.4 = 4 * 28.5 = 114.0
            const val = calculateCustomModelValue('graham', defaultParams, {});
            expect(val).toBeCloseTo(114.0, 1);

            const customYield = calculateCustomModelValue('graham', defaultParams, { bond_yield_proxy: 5.5 });
            expect(customYield).toBeCloseTo((4 * 28.5 * 4.4) / 5.5, 1);
        });

        it('calculates Peter Lynch fair value', () => {
            const defaultParams = {
                eps: 6.0,
                growth_rate_pct: 12.0,
                dividend_yield_pct: 3.0,
            };

            // PEG=1.0: P/E = 12 + 3 = 15 -> 6 * 15 = 90.0
            const val = calculateCustomModelValue('lynch', defaultParams, {});
            expect(val).toBe(90.0);

            const customGrowth = calculateCustomModelValue('lynch', defaultParams, { growth_rate_pct: 18.0 });
            expect(customGrowth).toBe(6.0 * 21.0);
        });

        it('calculates EPV floor accurately', () => {
            const defaultParams = {
                normalized_ebit: 100_000_000,
                tax_rate: 0.20,
                discount_rate: 0.10,
                net_cash: 20_000_000,
                shares: 10_000_000,
            };

            // NOPAT = 100M * 0.8 = 80M. Value of firm = 80M / 0.10 + 20M = 820M. Per share = 82.0
            const val = calculateCustomModelValue('epv', defaultParams, {});
            expect(val).toBe(82.0);
        });

        it('calculates PSG with overrides', () => {
            const defaultParams = {
                sales_per_share: 50.0,
                revenue_growth_pct: 20.0,
                gross_margin_pct: 70.0,
                target_psg: 1.0,
            };

            const val = calculateCustomModelValue('psg', defaultParams, {});
            expect(val).toBeGreaterThan(0);

            const customPsg = calculateCustomModelValue('psg', defaultParams, { target_psg: 1.5 });
            expect(customPsg).toBeGreaterThan(val!);
        });
    });

    describe('calculateBlendedScore', () => {
        it('calculates composite blended score with general weights', () => {
            const mockIntrinsicValue = {
                current_price: 100,
                average_intrinsic_value: 120,
                margin_of_safety_pct: 20,
                models: {
                    dcf: {
                        intrinsic_value: 120,
                        parameters: {
                            discount_rate: 0.10,
                            growth_rate: 0.10,
                            terminal_growth_rate: 0.02,
                            projection_years: 10,
                            base_fcf: 10,
                            total_cash: 0,
                            total_debt: 0,
                            shares_outstanding: 1,
                        },
                    },
                    graham: {
                        intrinsic_value: 110,
                        parameters: {
                            eps: 4,
                            growth_rate_pct: 10,
                            bond_yield_proxy: 4.4,
                        },
                    },
                },
            };

            const res = calculateBlendedScore(mockIntrinsicValue, {
                dcf: { growth_rate: 0.15 },
            });

            expect(res.hasAnyCustom).toBe(true);
            expect(res.customModelValues.dcf).toBeGreaterThan(120);
            expect(res.customAverage).toBeGreaterThan(120);
            expect(res.customMarginOfSafety).toBeGreaterThan(20);
        });

        it('follows the weights the backend shipped, not a second copy of them', () => {
            // The backend derives weights from what the business is and from
            // each model's own Monte Carlo band; recomputing them here would
            // drift. A model with no weight must not vote even when it has a
            // value, and the relative weights must be honoured.
            const iv = {
                current_price: 100,
                average_intrinsic_value: 120,
                blend_profile: 'financial',
                model_weights: { dni: 0.7, graham: 0.3 },
                models: {
                    dni: { intrinsic_value: 200, parameters: { base_eps: 5 } },
                    graham: {
                        intrinsic_value: 100,
                        parameters: { eps: 4, growth_rate_pct: 10, bond_yield_proxy: 4.4 },
                    },
                    // Present, valued, and deliberately unweighted by the backend.
                    dcf: { intrinsic_value: 1000, parameters: { base_fcf: 10 } },
                },
            };

            const res = calculateBlendedScore(iv, { graham: { eps: 4.0000001 } });
            // 0.7 * 200 + 0.3 * 100 = 170, and nothing from the 1000 DCF.
            expect(res.customAverage).toBeCloseTo(170, 6);
        });

        it('clamps a custom blend to the same credible band as the backend', () => {
            const iv = {
                current_price: 10,
                average_intrinsic_value: 12,
                model_weights: { dcf: 1.0 },
                models: {
                    dcf: {
                        intrinsic_value: 12,
                        parameters: {
                            discount_rate: 0.1,
                            growth_rate: 0.05,
                            terminal_growth_rate: 0.02,
                            projection_years: 10,
                            base_fcf: 10,
                            total_cash: 0,
                            total_debt: 0,
                            shares_outstanding: 1,
                        },
                    },
                },
            };

            // Crank growth to the top of the band: unclamped this runs far past
            // 5x spot, which is a number the API would never have published.
            const res = calculateBlendedScore(iv, { dcf: { growth_rate: 0.25 } });
            expect(res.customModelValues.dcf!).toBeGreaterThan(50);
            expect(res.customAverage).toBeCloseTo(50, 6);
        });
    });

    describe('MODEL_PARAM_CONFIGS', () => {
        it('defines parameter configs for all 12 models', () => {
            const keys = Object.keys(MODEL_PARAM_CONFIGS);
            expect(keys).toContain('dcf');
            expect(keys).toContain('dcfo');
            expect(keys).toContain('dni');
            expect(keys).toContain('ddm');
            expect(keys).toContain('mean_pe');
            expect(keys).toContain('peg');
            expect(keys).toContain('mean_pb');
            expect(keys).toContain('mean_ps');
            expect(keys).toContain('psg');
            expect(keys).toContain('graham');
            expect(keys).toContain('lynch');
            expect(keys).toContain('epv');
            expect(keys.length).toBe(12);
        });
    });
});
