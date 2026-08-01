import { describe, it, expect } from 'vitest';
// Imported from the component under test — a local copy of these helpers would
// pass forever while the real ones drifted.
import {
    getScaledValue,
    formatMetric,
    formatScaled,
    signedDeviation,
    heatColor,
    heatColorScaled,
    metricDomain,
    tileTextColor,
    tileLuminance,
    HEAT_PALETTES,
    METRICS,
} from '@/components/SP500Heatmap';

const metric = (key: string) => {
    const m = METRICS.find(x => x.key === key);
    if (!m) throw new Error(`no metric ${key}`);
    return m;
};

describe('S&P 500 Heatmap metric scaling', () => {
    it('scales decimal performance returns (YTD, 1Y, 3Y, 5Y, 10Y)', () => {
        expect(getScaledValue(0.1542, 'ytd')).toBeCloseTo(15.42);
        expect(formatMetric(0.1542, 'pct', 'ytd')).toBe('+15.42%');
        expect(formatMetric(-0.082, 'pct', '1y')).toBe('-8.20%');
    });

    it('preserves 1D change, which the backend already sends in percent points', () => {
        expect(getScaledValue(1.5, 'day')).toBe(1.5);
        expect(formatMetric(1.5, 'pct', 'day')).toBe('+1.50%');
    });

    it('formats fundamental ratios without a percent sign', () => {
        expect(getScaledValue(25.4, 'pe')).toBe(25.4);
        expect(formatMetric(25.4, 'ratio', 'pe')).toBe('25.40');
        expect(formatMetric(1.45, 'ratio', 'peg')).toBe('1.45');
        expect(formatMetric(85.2, 'ratio', 'debt_equity')).toBe('85.20');
    });

    it('treats dividend yield as a fraction, as the backend now normalizes it', () => {
        // Regression: Yahoo reports `dividendYield` in percent for many symbols
        // (KO = 2.4 meaning 2.4%). The backend resolves it against
        // dividendRate/price and always ships a fraction, so 2.42% arrives as
        // 0.0242 and must render as +2.42%, not +242%.
        expect(getScaledValue(0.0242, 'yield')).toBeCloseTo(2.42);
        expect(formatMetric(0.0242, 'pct', 'yield')).toBe('+2.42%');
        expect(formatMetric(0.185, 'pct', 'roe')).toBe('+18.50%');
        expect(formatMetric(0.12, 'pct', 'eps_qoq')).toBe('+12.00%');
    });

    it('handles null and undefined metrics gracefully', () => {
        expect(formatMetric(null, 'pct', 'ytd')).toBe('n/a');
        expect(formatMetric(undefined, 'ratio', 'pe')).toBe('n/a');
        expect(getScaledValue(NaN, 'ytd')).toBeNull();
        expect(getScaledValue(Infinity, 'pe')).toBeNull();
    });

    it('formats already-scaled values for the legend', () => {
        expect(formatScaled(10, 'ratio')).toBe('10.00');
        expect(formatScaled(-30, 'pct')).toBe('-30.00%');
        expect(formatScaled(null, 'pct')).toBe('n/a');
    });
});

describe('S&P 500 Heatmap colour scale', () => {
    it('centres returns on zero: gains green, losses red', () => {
        expect(signedDeviation(5, metric('ytd'))).toBe(5);
        expect(signedDeviation(-5, metric('ytd'))).toBe(-5);
        expect(heatColor(0.5, 'ytd')).toBe('#30cc5a'); // +50% YTD, saturated green
        expect(heatColor(-0.5, 'ytd')).toBe('#f63538');
    });

    it('centres valuation ratios on a typical reading, not on zero', () => {
        // Regression: with a zero midpoint every positive P/E landed in the red
        // half, so the whole map rendered red and nothing was ever green.
        const pe = metric('pe');
        expect(pe.mid).toBeGreaterThan(0);
        expect(signedDeviation(10, pe)).toBeGreaterThan(0);  // cheap -> good
        expect(signedDeviation(40, pe)).toBeLessThan(0);     // expensive -> bad
        expect(heatColor(10, 'pe')).toBe('#30cc5a');
        expect(heatColor(40, 'pe')).toBe('#f63538');
    });

    it('keeps every inverted metric two-sided', () => {
        for (const m of METRICS.filter(x => x.inverted)) {
            const cheap = heatColorScaled(m.mid - m.clamp, m);
            const pricey = heatColorScaled(m.mid + m.clamp, m);
            expect(cheap, `${m.key} cheap end`).toBe('#30cc5a');
            expect(pricey, `${m.key} expensive end`).toBe('#f63538');
        }
    });

    it('paints the legend ends to match their labels', () => {
        // The gradient stop at the domain's low end must be the colour that
        // value really gets — for inverted metrics that is the green end, and
        // the label printed there is the low value.
        for (const m of METRICS.filter(x => x.scale !== 'sequential')) {
            const [low, high] = metricDomain(m);
            expect(heatColorScaled(low, m)).toBe(m.inverted ? '#30cc5a' : '#f63538');
            expect(heatColorScaled(high, m)).toBe(m.inverted ? '#f63538' : '#30cc5a');
        }
    });

    it('renders a missing value as neutral grey rather than a colour', () => {
        expect(heatColor(null, 'day')).toBe('#1a1a1a');
        expect(heatColorScaled(null, metric('pe'))).toBe('#1a1a1a');
    });
});

describe('S&P 500 Heatmap light theme', () => {
    const light = HEAT_PALETTES.light;
    const dark = HEAT_PALETTES.dark;

    const contrast = (a: string, b: string) => {
        const [hi, lo] = [tileLuminance(a), tileLuminance(b)].sort((x, y) => y - x);
        return (hi + 0.05) / (lo + 0.05);
    };
    /** Every fill the map can paint for a metric, sampled across its domain. */
    const sweep = (m: (typeof METRICS)[number], theme: 'dark' | 'light') => {
        const [low, high] = metricDomain(m);
        return Array.from({ length: 51 }, (_, i) => heatColorScaled(low + ((high - low) * i) / 50, m, theme));
    };

    it('defaults to the dark palette, so existing callers are unchanged', () => {
        for (const m of METRICS) {
            const [low, high] = metricDomain(m);
            expect(heatColorScaled(low, m)).toBe(heatColorScaled(low, m, 'dark'));
            expect(heatColorScaled(high, m)).toBe(heatColorScaled(high, m, 'dark'));
        }
        expect(heatColor(0.5, 'ytd', 'dark')).toBe('#30cc5a');
    });

    it('reads the same way up as the dark map', () => {
        // A tile must mean the same thing in either mode: same neutral point,
        // same side of the scale, red and green never swapped.
        for (const m of METRICS.filter(x => x.scale !== 'sequential')) {
            const [low, high] = metricDomain(m);
            expect(heatColorScaled(low, m, 'light'), `${m.key} low end`).toBe(m.inverted ? light.pos : light.neg);
            expect(heatColorScaled(high, m, 'light'), `${m.key} high end`).toBe(m.inverted ? light.neg : light.pos);
            expect(heatColorScaled(m.mid, m, 'light'), `${m.key} neutral`).toBe(light.mid);
        }
    });

    it('keeps one-sided metrics on the green ramp, pale to deep', () => {
        const y = metric('yield');
        for (const c of sweep(y, 'light')) {
            const [r, g] = /rgb\((\d+),(\d+),(\d+)\)/.exec(c)!.slice(1).map(Number);
            expect(g, `${c} should stay on the green ramp`).toBeGreaterThan(r);
        }
        // On paper "more" reads as darker, the opposite of the dark map.
        expect(tileLuminance(heatColorScaled(0, y, 'light')))
            .toBeGreaterThan(tileLuminance(heatColorScaled(5, y, 'light')));
        expect(tileLuminance(heatColorScaled(0, y, 'dark')))
            .toBeLessThan(tileLuminance(heatColorScaled(5, y, 'dark')));
    });

    it('never paints a tile the same colour as the surface it sits on', () => {
        // The 1px gaps are the grid: a tile that matches the surface would
        // dissolve into it.
        expect(light.surface).not.toBe(light.mid);
        expect(contrast(light.mid, light.surface)).toBeGreaterThan(1.3);
        expect(contrast(light.noData, light.surface)).toBeGreaterThan(1.05);
    });

    it('keeps a no-data tile distinguishable from a neutral one', () => {
        expect(heatColorScaled(null, metric('pe'), 'light')).toBe(light.noData);
        expect(heatColorScaled(null, metric('pe'), 'light')).not.toBe(light.mid);
        // At least as separable as the dark map, which ships '#1a1a1a' on black.
        expect(contrast(light.noData, light.mid))
            .toBeGreaterThanOrEqual(contrast(dark.noData, dark.mid));
    });

    it('keeps every tile label readable in either mode', () => {
        for (const theme of ['dark', 'light'] as const) {
            for (const m of METRICS) {
                for (const fill of sweep(m, theme)) {
                    const ratio = contrast(fill, tileTextColor(fill, theme));
                    expect(ratio, `${theme} ${m.key} label on ${fill}`).toBeGreaterThan(2.1);
                }
            }
        }
        // The light scale runs to near-white, so its labels have to flip; the
        // dark one never does.
        const lightInks = new Set(METRICS.flatMap(m => sweep(m, 'light')).map(c => tileTextColor(c, 'light')));
        expect(lightInks.size).toBe(2);
        const darkInks = new Set(METRICS.flatMap(m => sweep(m, 'dark')).map(c => tileTextColor(c, 'dark')));
        expect(Array.from(darkInks)).toEqual(['#ffffff']);
    });

    it('reads header text against its own header fill', () => {
        expect(contrast(light.headerText, light.sectorHeader)).toBeGreaterThan(4.5);
        expect(contrast(light.headerText, light.industryHeader)).toBeGreaterThan(4.5);
        expect(contrast(dark.headerText, dark.sectorHeader)).toBeGreaterThan(4.5);
        expect(contrast(dark.headerText, dark.industryHeader)).toBeGreaterThan(4.5);
    });
});

describe('S&P 500 Heatmap metric table', () => {
    it('has a unique key and a backing field for every metric', () => {
        const keys = METRICS.map(m => m.key);
        expect(new Set(keys).size, 'duplicate metric key').toBe(keys.length);
        for (const m of METRICS) {
            expect(m.field, `${m.key} has no field`).toBeTruthy();
            expect(m.clamp, `${m.key} clamp must be positive`).toBeGreaterThan(0);
            expect(Number.isFinite(m.mid), `${m.key} mid`).toBe(true);
        }
    });

    it('covers every metric offered in the selector groups', () => {
        const grouped = METRICS.filter(m =>
            ['Performance', 'Valuation', 'Earnings & Sales', 'Profitability', 'Market'].includes(m.group));
        expect(grouped.length).toBe(METRICS.length);
    });

    it('scales percent-formatted metrics as fractions, except the raw quote field', () => {
        // The backend sends `change_pct` in percent points and everything else
        // percentage-shaped as a fraction. A metric formatted 'pct' that is not
        // in the decimal set would render 100x too small (or too large).
        for (const m of METRICS.filter(x => x.format === 'pct')) {
            const scaled = getScaledValue(1, m.key);
            expect(scaled === 100 || m.key === 'day', `${m.key} fraction scaling`).toBe(true);
        }
        expect(getScaledValue(1.5, 'day')).toBe(1.5);
    });

    it('never puts an impossible value inside a metric domain', () => {
        // One-sided quantities must not have a domain that reaches past their
        // floor (or, for drawdown, past its ceiling of zero).
        expect(metricDomain(metric('yield'))[0]).toBe(0);
        expect(metricDomain(metric('up52'))[0]).toBe(0);
        expect(metricDomain(metric('sales_ttm'))[0]).toBe(0);
        expect(metricDomain(metric('dd52'))[1]).toBe(0);
    });

    it('formats each display type', () => {
        expect(formatScaled(5.66, 'dollar')).toBe('$5.66');
        expect(formatScaled(12, 'days')).toBe('12d');
        expect(formatScaled(-3, 'days')).toBe('3d ago');
        expect(formatScaled(1.48e11, 'cap')).toBe('$148.0B');
        expect(formatScaled(2.5, 'ratio')).toBe('2.50');
    });

    it('points inverted metrics the right way', () => {
        // Cheap, lightly shorted, well-rated and imminent all read green.
        expect(heatColor(7, 'fwd_pe')).toBe('#30cc5a');   // mid 17 - clamp 10
        expect(heatColor(27, 'fwd_pe')).toBe('#f63538');  // mid 17 + clamp 10
        expect(heatColor(0.0, 'float_short')).toBe('#30cc5a');   // no short interest
        expect(heatColor(0.15, 'float_short')).toBe('#f63538');  // 15% of float short
        expect(heatColor(1.0, 'analyst')).toBe('#30cc5a');       // strong buy
        expect(heatColor(4.0, 'analyst')).toBe('#f63538');       // sell
        expect(heatColor(0, 'earnings_days')).toBe('#30cc5a');   // reporting today
    });

    it('reads liquidity and volume the right way up', () => {
        // More cover is better; these are not inverted.
        expect(metric('quick_ratio').inverted).toBeFalsy();
        expect(heatColor(2.0, 'quick_ratio')).toBe('#30cc5a');
        expect(heatColor(0.1, 'current_ratio')).toBe('#f63538');
        expect(heatColor(2.0, 'rel_volume')).toBe('#30cc5a');
    });
});

describe('S&P 500 Heatmap one-sided metrics', () => {
    it('gives dividend yield a domain that starts at its real floor', () => {
        // Dividends over price cannot be negative — across the whole index the
        // observed minimum is exactly 0.00% — so a diverging scale would spend
        // half its range on values that cannot occur.
        const y = metric('yield');
        expect(y.scale).toBe('sequential');
        expect(metricDomain(y)).toEqual([0, 5]);
    });

    it('never paints a one-sided metric red', () => {
        // Sweep the domain and past both ends; nothing may land on the red arm.
        const y = metric('yield');
        for (let v = -2; v <= 8; v += 0.1) {
            const c = heatColorScaled(v, y);
            const [, r, g] = /rgb\((\d+),(\d+),(\d+)\)/.exec(c)!.map(Number);
            expect(g, `yield ${v.toFixed(1)}% should stay on the green ramp`).toBeGreaterThan(r);
        }
    });

    it('ramps monotonically from the floor colour to full green', () => {
        const y = metric('yield');
        expect(heatColorScaled(0, y)).toBe('rgb(22,84,51)');    // #165433
        expect(heatColorScaled(5, y)).toBe('rgb(48,204,90)');   // #30cc5a
        // Clipped beyond the top, not wrapped or darkened.
        expect(heatColorScaled(9, y)).toBe(heatColorScaled(5, y));
        const g = (v: number) => Number(/rgb\(\d+,(\d+),/.exec(heatColorScaled(v, y))![1]);
        expect(g(0)).toBeLessThan(g(1.4));   // index median
        expect(g(1.4)).toBeLessThan(g(3.9)); // 90th percentile
    });

    it('keeps a zero-yield tile distinguishable from a no-data tile', () => {
        expect(heatColorScaled(0, metric('yield'))).not.toBe('#1a1a1a');
    });

    it('leaves genuinely two-sided metrics diverging', () => {
        // Losses, negative growth and negative returns are all real.
        for (const key of ['day', 'ytd', '1y', 'roe', 'roa', 'eps_qoq']) {
            expect(metric(key).scale, key).not.toBe('sequential');
            expect(heatColorScaled(-999, metric(key)), key).toBe('#f63538');
        }
    });
});
