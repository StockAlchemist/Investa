import { API_BASE_URL } from './api';

const accountsKey = (accounts?: string[]) => [...new Set(accounts ?? [])].sort();

export const portfolioKeys = {
    holdings: (user: string | undefined, currency: string, accounts?: string[], closed = false) =>
        ['holdings', user, currency, accountsKey(accounts), closed, API_BASE_URL] as const,
    history: (user: string | undefined, currency: string, accounts: string[] | undefined,
        period: string, benchmarks: string[], interval: string, from?: string, to?: string) =>
        ['history', user, currency, accountsKey(accounts), period, accountsKey(benchmarks), interval, from, to, API_BASE_URL] as const,
};
