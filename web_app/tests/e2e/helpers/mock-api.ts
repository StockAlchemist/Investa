import { Page } from '@playwright/test';
import {
    MOCK_USER,
    MOCK_SETTINGS,
    MOCK_SUMMARY_USD,
    MOCK_SUMMARY_EUR,
    MOCK_SUMMARY_THB,
    MOCK_HOLDINGS,
    MOCK_TRANSACTIONS,
    MOCK_AAPL_FUNDAMENTALS,
    MOCK_MSFT_FUNDAMENTALS,
    MOCK_AAPL_INTRINSIC_VALUE,
    MOCK_AAPL_TRACK_RECORD,
    MOCK_AAPL_FINANCIALS,
    MOCK_AAPL_RATIOS,
    MOCK_AAPL_ANALYSIS,
    MOCK_SCREENER_RESULTS,
    MOCK_BUFFETT_RUN,
    MOCK_BUFFETT_RANKINGS,
    MOCK_BUFFETT_EXCLUSIONS,
    MOCK_WATCHLISTS,
    MOCK_WATCHLIST_ITEMS,
} from './mock-data';
import type { Holding, PortfolioSummary, Settings, Transaction } from '@/lib/api';

export interface MockApiOptions {
    initialLoggedIn?: boolean;
    initialTransactions?: Transaction[];
    initialHoldings?: Holding[];
    initialSettings?: Settings;
    initialSummary?: PortfolioSummary;
}

export interface MockApiState {
    loggedIn: boolean;
    transactions: Transaction[];
    holdings: Holding[];
    settings: Settings;
    summary: PortfolioSummary;
}

function parseAccountsParam(url: URL): string[] {
    const raw = [...url.searchParams.getAll('accounts'), ...url.searchParams.getAll('accounts[]')];
    const result: string[] = [];
    for (const val of raw) {
        if (val.includes(',')) {
            result.push(...val.split(',').map(s => s.trim()));
        } else if (val) {
            result.push(val);
        }
    }
    return result;
}

export async function setupHermeticMockApi(page: Page, options: MockApiOptions = {}): Promise<MockApiState> {
    const state: MockApiState = {
        loggedIn: options.initialLoggedIn ?? true,
        transactions: JSON.parse(JSON.stringify(options.initialTransactions ?? MOCK_TRANSACTIONS)),
        holdings: JSON.parse(JSON.stringify(options.initialHoldings ?? MOCK_HOLDINGS)),
        settings: JSON.parse(JSON.stringify(options.initialSettings ?? MOCK_SETTINGS)),
        summary: JSON.parse(JSON.stringify(options.initialSummary ?? MOCK_SUMMARY_USD)),
    };

    // 1. Catch-all for API calls (FIRST, so subsequent specific routes take precedence)
    await page.route(/\/api\//, async (route) => {
        const method = route.request().method();
        if (method === 'OPTIONS') {
            return route.fulfill({ status: 200, body: '' });
        }
        return route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
    });

    // 2. Auth Routes
    await page.route(/\/api\/auth\/me(\/|\?|$)/, async (route) => {
        const method = route.request().method();
        if (method === 'PATCH') {
            const body = route.request().postDataJSON() || {};
            const updatedUser = { ...MOCK_USER, ...body };
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(updatedUser) });
        }
        if (state.loggedIn) {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_USER) });
        }
        return route.fulfill({
            status: 401,
            contentType: 'application/json',
            body: JSON.stringify({ detail: 'Could not validate credentials' }),
        });
    });

    await page.route(/\/api\/auth\/login(\/|\?|$)/, async (route) => {
        state.loggedIn = true;
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ access_token: 'fake-jwt-token-for-e2e', token_type: 'bearer' }),
        });
    });

    await page.route(/\/api\/auth\/logout(\/|\?|$)/, async (route) => {
        state.loggedIn = false;
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ status: 'ok', message: 'Logged out' }),
        });
    });

    // 3. Settings Route
    await page.route(/\/api\/settings(\/|\?|$)/, async (route) => {
        const method = route.request().method();
        if (method === 'PATCH' || method === 'POST') {
            const body = route.request().postDataJSON() || {};
            state.settings = { ...state.settings, ...body };
            return route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ status: 'ok', message: 'Settings updated' }),
            });
        }
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(state.settings),
        });
    });

    // 4. Summary Route (Handles currency and account filtering)
    await page.route(/\/api\/summary(\/|\?|$)/, async (route) => {
        const url = new URL(route.request().url());
        const currency = url.searchParams.get('currency') || 'USD';
        const accounts = parseAccountsParam(url);

        let baseSummary = MOCK_SUMMARY_USD;
        if (currency === 'EUR') baseSummary = MOCK_SUMMARY_EUR;
        else if (currency === 'THB') baseSummary = MOCK_SUMMARY_THB;

        const filteredSummary = JSON.parse(JSON.stringify(baseSummary));

        if (accounts.length > 0) {
            let subsetValue = 0;
            let subsetGain = 0;
            for (const acc of accounts) {
                const accMetrics = (filteredSummary.account_metrics as Record<string, { market_value?: number; unrealized_gain?: number }>)?.[acc];
                if (accMetrics) {
                    subsetValue += accMetrics.market_value || 0;
                    subsetGain += accMetrics.unrealized_gain || 0;
                }
            }
            if (filteredSummary.metrics) {
                filteredSummary.metrics.market_value = subsetValue;
                filteredSummary.metrics.unrealized_gain = subsetGain;
                filteredSummary.metrics.total_gain = subsetGain;
            }
        }

        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(filteredSummary),
        });
    });

    // 5. Holdings Route
    await page.route(/\/api\/holdings(\/|\?|$)/, async (route) => {
        const url = new URL(route.request().url());
        const accounts = parseAccountsParam(url);

        let currentHoldings = [...state.holdings];
        if (accounts.length > 0) {
            currentHoldings = currentHoldings.filter(h => h.Account && accounts.includes(h.Account));
        }

        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(currentHoldings),
        });
    });

    // 6. Transactions Route (GET, POST, PUT, DELETE)
    await page.route(/\/api\/transactions/, async (route) => {
        const method = route.request().method();
        const url = new URL(route.request().url());
        const matchId = url.pathname.match(/\/api\/transactions\/(\d+)/);
        const id = matchId ? parseInt(matchId[1], 10) : null;

        if (id !== null) {
            if (method === 'PUT') {
                const updatedData: Transaction = route.request().postDataJSON() || {};
                const idx = state.transactions.findIndex(t => t.id === id);
                if (idx !== -1) {
                    state.transactions[idx] = { ...state.transactions[idx], ...updatedData, id };
                }
                return route.fulfill({
                    status: 200,
                    contentType: 'application/json',
                    body: JSON.stringify({ status: 'ok', message: 'Transaction updated' }),
                });
            }

            if (method === 'DELETE') {
                state.transactions = state.transactions.filter(t => t.id !== id);
                return route.fulfill({
                    status: 200,
                    contentType: 'application/json',
                    body: JSON.stringify({ status: 'ok', message: 'Transaction deleted' }),
                });
            }
        }

        if (method === 'GET') {
            const accounts = parseAccountsParam(url);
            let list = [...state.transactions];
            if (accounts.length > 0) {
                list = list.filter(t => t.Account && accounts.includes(t.Account));
            }
            return route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify(list),
            });
        }

        if (method === 'POST') {
            const newTx: Transaction = route.request().postDataJSON() || {};
            const createdTx = {
                ...newTx,
                id: Date.now(),
            };
            state.transactions.unshift(createdTx);

            // Dynamically update holdings if it was a Buy
            if ((createdTx.Type || '').toLowerCase() === 'buy' && createdTx.Symbol) {
                const existing = state.holdings.find(h => h.Symbol === createdTx.Symbol && h.Account === createdTx.Account);
                if (existing) {
                    existing.Quantity += Number(createdTx.Quantity) || 0;
                    existing['Market Value'] = (Number(existing['Market Value']) || 0) + (Number(createdTx['Total Amount']) || 0);
                } else {
                    state.holdings.push({
                        Symbol: createdTx.Symbol,
                        Quantity: Number(createdTx.Quantity) || 0,
                        Account: createdTx.Account,
                        Price: Number(createdTx['Price/Share']) || 0,
                        'Cost Basis': Number(createdTx['Price/Share']) || 0,
                        'Market Value': Number(createdTx['Total Amount']) || 0,
                        'Unreal. Gain': 0,
                        'Unreal. Gain %': 0,
                        'Total Return %': 0,
                        'Day Change %': 0,
                        'Weight %': 5.0,
                    });
                }
            }

            return route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ status: 'ok', message: 'Transaction created successfully' }),
            });
        }

        return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });

    // 7. Stock Fundamentals & Modal Details
    await page.route(/\/api\/fundamentals(\/|\?|$)/, async (route) => {
        const url = route.request().url();
        if (url.includes('MSFT')) {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_MSFT_FUNDAMENTALS) });
        }
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_AAPL_FUNDAMENTALS) });
    });

    await page.route(/\/api\/intrinsic_value(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_AAPL_INTRINSIC_VALUE),
        });
    });

    await page.route(/\/api\/track-record(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_AAPL_TRACK_RECORD),
        });
    });

    await page.route(/\/api\/financials(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_AAPL_FINANCIALS),
        });
    });

    await page.route(/\/api\/ratios(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_AAPL_RATIOS),
        });
    });

    await page.route(/\/api\/stock-analysis(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_AAPL_ANALYSIS),
        });
    });

    await page.route(/\/api\/markets\/news(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([
                {
                    title: 'Tech Leaders Announce Strong Quarterly Results',
                    publisher: 'Financial Times',
                    link: 'https://example.com/news/1',
                    providerPublishTime: Date.now() - 3600000,
                    type: 'STORY',
                },
                {
                    title: 'Fed Holds Interest Rates Steady Amid Inflation Data',
                    publisher: 'Bloomberg',
                    link: 'https://example.com/news/2',
                    providerPublishTime: Date.now() - 7200000,
                    type: 'STORY',
                },
            ]),
        });
    });

    // 8. Screener Routes
    await page.route(/\/api\/screener\/review(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_AAPL_ANALYSIS),
        });
    });

    await page.route(/\/api\/screener\/narrative(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_SCREENER_RESULTS),
        });
    });

    await page.route(/\/api\/screener\/run(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_SCREENER_RESULTS),
        });
    });

    // 9. Buffett Rank Routes
    await page.route(/\/api\/buffett-rank\/latest(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_BUFFETT_RUN),
        });
    });

    await page.route(/\/api\/buffett-rank\/exclusions(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_BUFFETT_EXCLUSIONS),
        });
    });

    await page.route(/\/api\/buffett-rank(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_BUFFETT_RANKINGS),
        });
    });

    // 10. Watchlists
    await page.route(/\/api\/watchlists(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_WATCHLISTS),
        });
    });

    await page.route(/\/api\/watchlist(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_WATCHLIST_ITEMS),
        });
    });

    // 11. Dashboard & Auxiliary endpoints
    await page.route(/\/api\/market_status(\/|\?|$)/, async (route) => {
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ is_open: true }) });
    });

    await page.route(/\/api\/indices(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(MOCK_SUMMARY_USD.metrics?.indices || {}),
        });
    });

    await page.route(/\/api\/headline(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ metrics: MOCK_SUMMARY_USD.metrics }),
        });
    });

    await page.route(/\/api\/history(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([
                { date: '2026-01-01', value: 100000, twr: 0, drawdown: 0, 'S&P 500': 0 },
                { date: '2026-04-01', value: 112000, twr: 12.0, drawdown: -2.1, 'S&P 500': 8.5 },
                { date: '2026-08-01', value: 125450, twr: 25.45, drawdown: 0, 'S&P 500': 15.2 },
            ]),
        });
    });

    await page.route(/\/api\/risk_metrics(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                max_drawdown: -8.4,
                volatility_ann: 12.2,
                sharpe_ratio: 1.45,
                beta: 0.95,
                sortino_ratio: 2.1,
                var_95: 1.8,
            }),
        });
    });

    await page.route(/\/api\/dividend_calendar(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([]),
        });
    });

    await page.route(/\/api\/earnings_calendar(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([]),
        });
    });

    await page.route(/\/api\/dividends(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([]),
        });
    });

    await page.route(/\/api\/capital_gains(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify([]),
        });
    });

    await page.route(/\/api\/asset_change(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ performance: [], benchmarks: {} }),
        });
    });

    await page.route(/\/api\/projection(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ summary: {}, projections: [] }),
        });
    });

    await page.route(/\/api\/attribution(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ sectors: [], contributors: [] }),
        });
    });

    await page.route(/\/api\/portfolio\/health(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ score: 90, status: 'good' }),
        });
    });

    await page.route(/\/api\/income_projection(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ projected_monthly: [], projected_annual: 1850 }),
        });
    });

    await page.route(/\/api\/holding_returns(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({}),
        });
    });

    await page.route(/\/api\/strategies(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                strategies: [
                    {
                        id: 'buffett_top10',
                        name: 'Buffett Top 10 Quality',
                        summary: 'Top 10 quality-ranked companies held equally',
                        sleeves: { 'Core Quality': 1.0 },
                        backtest: { cagr: 18.2, max_drawdown: -14.5, sharpe: 1.25 },
                        risks: ['High valuation risk'],
                        is_default: true,
                        ranking: { quality_weight: 0.7, top_n: 10, max_per_sector: 3, sector_digits: 2, rebalance: 'monthly' },
                    },
                ],
                default: 'buffett_top10',
            }),
        });
    });

    await page.route(/\/api\/trend-signal(\/|\?|$)/, async (route) => {
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                advisory_only: true,
                signal_symbol: 'SPY',
                signal_name: 'S&P 500',
                market_timezone: 'America/New_York',
                state: 'in',
                sma_months: 10,
                decision_date: '2026-07-31',
                decision_close: 550.2,
                sma: 520.1,
                governs_month: '2026-08',
                provisional_state: 'in',
                provisional_sma: 522.4,
                latest_close: 555.0,
                latest_date: '2026-08-14',
                flip_close: 480.0,
                distance_pct: 15.6,
                would_flip: false,
                next_decision_date: '2026-08-31',
                history: [],
            }),
        });
    });

    return state;
}

export async function loginAsMockUser(page: Page, options: MockApiOptions = {}): Promise<MockApiState> {
    const state = await setupHermeticMockApi(page, { ...options, initialLoggedIn: true });
    await page.addInitScript((user) => {
        localStorage.setItem('investa_user', JSON.stringify(user));
    }, MOCK_USER);
    return state;
}
