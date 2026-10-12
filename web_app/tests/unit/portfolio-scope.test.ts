import fixture from '../../../tests/fixtures/portfolio_contract.json';
import type { components } from '@/src/api/types';
import { describe, expect, it } from 'vitest';
import { portfolioKeys } from '@/lib/query_keys';
import { readView, writeView } from '@/lib/view_navigation';
import { submissionKey } from '@/lib/submission_keys';

describe('Portfolio selection and submissions', () => {
    it('accepts the shared backend/native response contract', () => {
        const summary: components['schemas']['SummaryResponse'] = fixture.summary;
        expect(summary.metrics?.market_value).toBe(125450);
        expect(summary.freshness?.calculated_at).toBe(1791792000);
    });
    it('shares identical requests regardless of account/benchmark order', () => {
        expect(portfolioKeys.holdings('alice', 'USD', ['B', 'A'])).toEqual(portfolioKeys.holdings('alice', 'USD', ['A', 'B']));
        expect(portfolioKeys.history('alice', 'USD', [], '1y', ['QQQ', 'SPY'], '1d'))
            .toEqual(portfolioKeys.history('alice', 'USD', [], '1y', ['SPY', 'QQQ'], '1d'));
        expect(portfolioKeys.holdings('alice', 'USD')).not.toEqual(portfolioKeys.holdings('alice', 'THB'));
        expect(portfolioKeys.holdings('alice', 'USD')).not.toEqual(portfolioKeys.holdings('bob', 'USD'));
    });

    it('round trips a shareable selection, including account names with spaces', () => {
        const selection = { section: 'allocation', currency: 'THB', accounts: ['Joint Account'], symbol: 'BRK-B' };
        expect(readView(writeView(selection))).toEqual(selection);
        expect(readView('?section=invalid&currency=javascript&symbol=<script>')).not.toHaveProperty('currency');
    });

    it('retains a receipt key after an ambiguous failure and releases it after success', async () => {
        const payload = { Symbol: 'AAPL', Quantity: 2 };
        const first = await submissionKey('/api/transactions', payload);
        const retry = await submissionKey('/api/transactions', payload);
        expect(retry.key).toBe(first.key);
        const changed = await submissionKey('/api/transactions', { ...payload, Quantity: 3 });
        expect(changed.key).not.toBe(first.key);
        retry.forget();
        expect((await submissionKey('/api/transactions', payload)).key).not.toBe(first.key);
    });
});
