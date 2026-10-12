/* Compare two production builds with identical API fixtures and fresh storage.
 * node scripts/benchmark-comparison.cjs OLD_URL NEW_URL OUTPUT_JSON [PAIRS]
 * Does not contact the real backend: every /api/ request is intercepted.
 */
/* eslint-disable @typescript-eslint/no-require-imports -- Standalone CommonJS benchmark runner. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const { chromium } = require('@playwright/test');

function loadFixture(file, imports = {}) {
    const fixtureModule = { exports: {} };
    const code = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
    }).outputText;
    vm.runInNewContext(code, {
        module: fixtureModule, exports: fixtureModule.exports,
        require: name => imports[name] || require(name),
        console, URL, Date, JSON, setTimeout,
    }, { filename: file });
    return fixtureModule.exports;
}

const fixtureDir = path.resolve(__dirname, '../tests/e2e/helpers');
const data = loadFixture(path.join(fixtureDir, 'mock-data.ts'));
const { loginAsMockUser } = loadFixture(path.join(fixtureDir, 'mock-api.ts'), { './mock-data': data });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const compact = ['portfolioHero', 'todayStrip', 'cashBalance', 'annualTWR', 'performanceGraph'];

function historyFor(url) {
    const period = url.searchParams.get('period') || 'all';
    const count = ({ '1d': 78, '5d': 130, '1m': 21, '3m': 63, '6m': 126, '1y': 252 })[period] || 1008;
    const start = Date.UTC(2026, 9, 9) - (count - 1) * 86400000;
    return Array.from({ length: count }, (_, i) => ({
        date: new Date(start + i * 86400000).toISOString().slice(0, 10),
        value: 100000 + 25450 * i / (count - 1),
        twr: 25.45 * i / (count - 1), drawdown: -2 * Math.abs(Math.sin(i / 40)),
        'S&P 500': 15.2 * i / (count - 1),
    }));
}

async function run(browser, url, savedCompact) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, serviceWorkers: 'block' });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await loginAsMockUser(page);
    if (savedCompact) await page.addInitScript(items => localStorage.setItem('investa_dashboard_visible_items', JSON.stringify(items)), compact);
    await page.route(/\/api\/summary\/headline(\?|$)/, route => route.fulfill({ json: {
        ...data.MOCK_SUMMARY_USD, freshness: { calculated_at: 1791760000, refreshing: false },
    } }));
    await page.route(/\/api\/history(\?|$)/, route => route.fulfill({ json: historyFor(new URL(route.request().url())) }));
    // Register last so this delay applies to all specific fixture routes.
    await page.route(/\/api\//, async route => { await delay(120); await route.fallback(); });
    const requests = [], responses = [];
    let pendingBodies = [];
    page.on('request', request => {
        if (request.url().includes('/api/') && request.method() === 'GET') requests.push(request.url());
    });
    page.on('response', response => {
        if (!response.url().includes('/api/') || response.request().method() !== 'GET') return;
        pendingBodies.push(response.body().then(body => responses.push({ url: response.url(), bytes: body.length, status: response.status() })).catch(() => {}));
    });
    const start = performance.now();
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    // Target the same hero in both versions. The old composition chart also
    // contains the full value, but shows it before the hero animation settles.
    await page.waitForFunction(() => {
        const figure = document.querySelector('section[aria-label="Portfolio value"] span.font-semibold.tabular-nums');
        return figure && Number(figure.textContent.replace(/[^0-9.]/g, '')) > 0;
    }, null, { timeout: 15000 });
    const firstValueMs = performance.now() - start;
    await page.locator('section[aria-label="Portfolio value"]').getByText('$125,450.00', { exact: true }).waitFor({ timeout: 15000 });
    const settledValueMs = performance.now() - start;
    await delay(Math.max(0, 10000 - (performance.now() - start)));
    await Promise.all(pendingBodies);
    const endpointCounts = {};
    for (const request of requests) {
        const endpoint = new URL(request).pathname.replace('/api/', '');
        endpointCounts[endpoint] = (endpointCounts[endpoint] || 0) + 1;
    }
    const result = {
        hero_first_value_ms: Math.round(firstValueMs), hero_settled_ms: Math.round(settledValueMs), get_requests: requests.length,
        response_body_bytes: responses.reduce((sum, response) => sum + response.bytes, 0),
        history_requests: requests.filter(request => new URL(request).pathname.endsWith('/history')).length,
        unique_history_urls: [...new Set(requests.filter(request => new URL(request).pathname.endsWith('/history')))].length,
        endpoint_counts: endpointCounts, page_errors: errors,
        failed_responses: responses.filter(response => response.status >= 400),
    };
    await context.close();
    return result;
}

async function main() {
    const [oldUrl, newUrl, output, rawPairs = '5'] = process.argv.slice(2);
    if (!oldUrl || !newUrl || !output) throw new Error('Supply OLD_URL NEW_URL OUTPUT_JSON [PAIRS]');
    const pairs = Number(rawPairs);
    const browser = await chromium.launch({ headless: true });
    const results = { api_delay_ms: 120, observation_ms: 10000, viewport: '1440x1000', pairs, default_dashboard: { old: [], new: [] }, saved_compact_layout: { old: [], new: [] } };
    try {
        // Warm server/runtime startup and filesystem caches without using a measured browser context.
        for (const url of [oldUrl, newUrl]) await run(browser, url, false);
        for (const [scenario, savedCompact] of [['default_dashboard', false], ['saved_compact_layout', true]]) {
            for (let i = 0; i < pairs; i++) {
                const order = i % 2 ? [['new', newUrl], ['old', oldUrl]] : [['old', oldUrl], ['new', newUrl]];
                for (const [version, url] of order) {
                    const result = await run(browser, url, savedCompact);
                    results[scenario][version].push(result);
                    fs.writeFileSync(output, JSON.stringify(results, null, 2) + '\n');
                    console.log(`${scenario} ${version} ${i + 1}/${pairs}: ${JSON.stringify(result)}`);
                }
            }
        }
    } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
