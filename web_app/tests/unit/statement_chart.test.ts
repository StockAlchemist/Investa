import { describe, expect, it } from 'vitest';
import {
    MAX_CHART_SERIES,
    SAME_SCALE_RATIO,
    defaultRange,
    formatStatementValue,
    groupBySharedScale,
    isFiniteNumber,
    periodAxisLabel,
    periodsInRange,
    pickDefaultMetrics,
    toggleSlot,
} from '@/lib/statement_chart';

describe('periodAxisLabel', () => {
    it('names a quarter by its month, so four columns a year are distinguishable', () => {
        expect(periodAxisLabel('2026-03-31', 'quarterly')).toBe('Mar 2026');
        expect(periodAxisLabel('2026-06-30', 'quarterly')).toBe('Jun 2026');
    });

    it('names an annual period by its year', () => {
        expect(periodAxisLabel('2026-09-28', 'annual')).toBe('2026');
    });

    // A filed period end is a calendar day, not an instant: parsing it in the
    // viewer's zone would label a Jan 1 fiscal close as the previous year for
    // anyone west of UTC.
    it('does not slide a January 1 period end back a year', () => {
        expect(periodAxisLabel('2022-01-01', 'quarterly')).toBe('Jan 2022');
        expect(periodAxisLabel('2022-01-01', 'annual')).toBe('2022');
    });
});

describe('formatStatementValue', () => {
    it('compacts statement magnitudes', () => {
        expect(formatStatementValue(94_036_000_000)).toBe('94.04B');
        expect(formatStatementValue(-2_500_000)).toBe('-2.5M');
    });

    it('leaves per-share figures readable', () => {
        expect(formatStatementValue(1.65)).toBe('1.65');
        expect(formatStatementValue(0)).toBe('0');
    });
});

describe('pickDefaultMetrics', () => {
    const row = (label: string, values: (number | null)[]) => ({ label, values });
    const full = [1, 2, 3, 4] as (number | null)[];

    it('opens on the statement’s preferred line items', () => {
        const rows = [row('Total Revenue', full), row('Net Income', full)];
        expect(pickDefaultMetrics(['Total Revenue', 'Net Income'], rows)).toEqual([
            'Total Revenue', 'Net Income',
        ]);
    });

    // Meta never tags gross profit, so it exists only where Yahoo derives it.
    it('skips a line item reported for only a few periods', () => {
        const rows = [
            row('Total Revenue', full),
            row('Gross Profit', [null, null, null, 4]),
            row('Net Income', full),
        ];
        expect(pickDefaultMetrics(['Total Revenue', 'Gross Profit', 'Net Income'], rows)).toEqual([
            'Total Revenue', 'Net Income',
        ]);
    });

    it('keeps a sparse item rather than charting nothing', () => {
        const rows = [row('Total Revenue', [null, null, null, 4])];
        expect(pickDefaultMetrics(['Total Revenue'], rows)).toEqual(['Total Revenue']);
    });

    it('ignores line items the statement does not carry', () => {
        expect(pickDefaultMetrics(['Free Cash Flow'], [row('Total Revenue', full)])).toEqual([]);
    });
});

describe('periodsInRange', () => {
    // A range is a span in years — the unit an investor thinks in — so the
    // column count differs between the two period types.
    it('counts four columns a year for quarters and one for years', () => {
        expect(periodsInRange('5y', 'quarterly')).toBe(20);
        expect(periodsInRange('5y', 'annual')).toBe(5);
        expect(periodsInRange('10y', 'quarterly')).toBe(40);
    });

    it('opens quarters on five years and years on ten', () => {
        expect(defaultRange('quarterly')).toBe('5y');
        expect(defaultRange('annual')).toBe('10y');
    });
});

describe('toggleSlot', () => {
    it('adds a line item to the first free colour slot', () => {
        expect(toggleSlot(['Total Revenue'], 'Net Income')).toEqual(['Total Revenue', 'Net Income']);
    });

    // The palette slot belongs to the line item, not to its rank in the chart.
    it('leaves a hole when a series is dropped, so survivors keep their colour', () => {
        const after = toggleSlot(['Total Revenue', 'Gross Profit', 'Net Income'], 'Total Revenue');
        expect(after).toEqual([null, 'Gross Profit', 'Net Income']);
        // Gross Profit is still slot 1 and Net Income still slot 2.
        expect(after.indexOf('Net Income')).toBe(2);
    });

    it('reuses a freed slot before growing', () => {
        expect(toggleSlot([null, 'Gross Profit'], 'Total Revenue')).toEqual(['Total Revenue', 'Gross Profit']);
    });

    it('refuses a fifth series rather than recolouring the four on screen', () => {
        const full = ['a', 'b', 'c', 'd'];
        expect(toggleSlot(full, 'e', MAX_CHART_SERIES)).toEqual(full);
    });
});

describe('groupBySharedScale', () => {
    const s = (label: string, maxAbs: number) => ({ label, maxAbs });

    it('keeps comparable magnitudes on one axis', () => {
        const groups = groupBySharedScale([s('Revenue', 400e9), s('Gross Profit', 180e9), s('Net Income', 100e9)]);
        expect(groups).toHaveLength(1);
        expect(groups[0].map(g => g.label)).toEqual(['Revenue', 'Gross Profit', 'Net Income']);
    });

    // Never a second y-axis: revenue and EPS get two charts instead.
    it('splits magnitudes that cannot honestly share a scale', () => {
        const groups = groupBySharedScale([s('Revenue', 400e9), s('Diluted EPS', 1.65)]);
        expect(groups).toHaveLength(2);
        expect(groups[0].map(g => g.label)).toEqual(['Revenue']);
        expect(groups[1].map(g => g.label)).toEqual(['Diluted EPS']);
    });

    it('groups on the leader, at the documented ratio', () => {
        const together = groupBySharedScale([s('big', SAME_SCALE_RATIO), s('small', 1)]);
        expect(together).toHaveLength(1);
        const apart = groupBySharedScale([s('big', SAME_SCALE_RATIO + 1), s('small', 1)]);
        expect(apart).toHaveLength(2);
    });

    it('does not let an all-zero row join a real one', () => {
        const groups = groupBySharedScale([s('Revenue', 400e9), s('Empty', 0)]);
        expect(groups).toHaveLength(2);
    });
});

describe('isFiniteNumber', () => {
    it('rejects the nulls a statement is full of', () => {
        expect(isFiniteNumber(null)).toBe(false);
        expect(isFiniteNumber(undefined)).toBe(false);
        expect(isFiniteNumber(NaN)).toBe(false);
        expect(isFiniteNumber(0)).toBe(true);
        expect(isFiniteNumber(-1.5)).toBe(true);
    });
});
