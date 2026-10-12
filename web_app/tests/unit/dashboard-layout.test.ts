import { describe, expect, it } from 'vitest';
import { originalDashboardLayout } from '@/lib/dashboard_layout';
import { INITIAL_VISIBLE_ITEMS } from '@/lib/dashboard_constants';

const redesign = ['portfolioHero', 'performanceGraph', 'annualTWR', 'totalReturn', 'cashBalance', 'ledgerAllocation', 'ledgerHoldings'];

describe('Original dashboard layout restoration', () => {
    it('restores the redesign default regardless of order', () => {
        expect(originalDashboardLayout(redesign)).toEqual(INITIAL_VISIBLE_ITEMS);
        expect(originalDashboardLayout([...redesign].reverse())).toEqual(INITIAL_VISIBLE_ITEMS);
    });
    it('retains custom selections and their order', () => {
        expect(originalDashboardLayout(['projection', 'cashBalance'])).toEqual(['projection', 'cashBalance']);
        expect(originalDashboardLayout(['projection', 'ledgerHoldings', 'cashBalance'])).toEqual(['projection', 'cashBalance']);
    });
    it('restores usable defaults when only removed widgets were selected', () => {
        expect(originalDashboardLayout(['ledgerHoldings', 'ledgerAllocation'])).toEqual(INITIAL_VISIBLE_ITEMS);
    });
});
