import { INITIAL_VISIBLE_ITEMS } from './dashboard_constants';

const REDESIGN_DEFAULT = ['portfolioHero', 'performanceGraph', 'annualTWR', 'totalReturn', 'cashBalance', 'ledgerAllocation', 'ledgerHoldings'];

/** Restore the original default from the redesign while preserving custom layouts. */
export function originalDashboardLayout(items: string[]): string[] {
    if (items.length === REDESIGN_DEFAULT.length && REDESIGN_DEFAULT.every(id => items.includes(id))) {
        return [...INITIAL_VISIBLE_ITEMS];
    }
    const original = items.filter(id => id !== 'ledgerAllocation' && id !== 'ledgerHoldings');
    return original.length ? original : [...INITIAL_VISIBLE_ITEMS];
}
