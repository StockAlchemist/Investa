import { test, expect, Page } from '@playwright/test';

/**
 * Hermetic auth flow tests: the backend is fully mocked via page.route, so
 * these run against `next dev` alone — no Python server, no real user data.
 */

const FAKE_TOKEN = 'fake-jwt-token-for-e2e';
const FAKE_USER = {
    id: 1,
    username: 'e2euser',
    alias: 'E2E User',
    is_active: true,
    created_at: '2026-01-01T00:00:00',
};

async function mockApi(page: Page) {
    // Catch-all FIRST (Playwright matches the most recently registered route
    // first, so specific mocks below take precedence).
    await page.route('**/api/**', async (route) => {
        const method = route.request().method();
        if (method === 'OPTIONS') {
            return route.fulfill({ status: 200, body: '' });
        }
        return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });
    await page.route('**/api/auth/me', (route) =>
        route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FAKE_USER) }),
    );
}

test.describe('Authentication', () => {
    test('unauthenticated visit to / redirects to the login page', async ({ page }) => {
        await mockApi(page);
        await page.goto('/');
        await page.waitForURL('**/login');
        await expect(page.getByLabel('Username')).toBeVisible();
        await expect(page.getByLabel('Password')).toBeVisible();
    });

    test('failed login shows the backend error message', async ({ page }) => {
        await mockApi(page);
        await page.route('**/api/auth/login', (route) =>
            route.fulfill({
                status: 401,
                contentType: 'application/json',
                body: JSON.stringify({ detail: 'Incorrect username or password' }),
            }),
        );

        await page.goto('/login');
        await page.getByLabel('Username').fill('wrong');
        await page.getByLabel('Password').fill('wrong');
        await page.getByRole('button', { name: /sign in|log ?in/i }).click();

        await expect(page.getByText('Incorrect username or password')).toBeVisible();
        await expect(page).toHaveURL(/login/);
    });

    test('rate-limited login surfaces the 429 message', async ({ page }) => {
        await mockApi(page);
        await page.route('**/api/auth/login', (route) =>
            route.fulfill({
                status: 429,
                contentType: 'application/json',
                body: JSON.stringify({ detail: 'Too many failed login attempts. Try again in 900 seconds.' }),
            }),
        );

        await page.goto('/login');
        await page.getByLabel('Username').fill('someone');
        await page.getByLabel('Password').fill('pw');
        await page.getByRole('button', { name: /sign in|log ?in/i }).click();

        await expect(page.getByText(/Too many failed login attempts/)).toBeVisible();
    });

    test('successful login lands on the dashboard', async ({ page }) => {
        await mockApi(page);
        await page.route('**/api/auth/login', (route) =>
            route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ access_token: FAKE_TOKEN, token_type: 'bearer' }),
            }),
        );

        await page.goto('/login');
        await page.getByLabel('Username').fill(FAKE_USER.username);
        await page.getByLabel('Password').fill('correct-password');
        await page.getByRole('button', { name: /sign in|log ?in/i }).click();

        // Login stores the token and navigates to the dashboard
        await page.waitForURL((url) => !url.pathname.includes('login'));
        const storedToken = await page.evaluate(() => localStorage.getItem('access_token'));
        expect(storedToken).toBe(FAKE_TOKEN);
    });

    test('expired token on a return visit logs out quietly to /login', async ({ page }) => {
        await mockApi(page);
        await page.route('**/api/auth/me', (route) =>
            route.fulfill({
                status: 401,
                contentType: 'application/json',
                body: JSON.stringify({ detail: 'Could not validate credentials' }),
            }),
        );

        // Seed a stale token before the app boots
        await page.addInitScript(() => {
            localStorage.setItem('access_token', 'stale-token');
        });

        const errors: string[] = [];
        page.on('console', (msg) => {
            if (msg.type() === 'error') errors.push(msg.text());
        });

        await page.goto('/');
        await page.waitForURL('**/login');

        // The stale token must be cleared, and the 401 handled without console.error
        const remaining = await page.evaluate(() => localStorage.getItem('access_token'));
        expect(remaining).toBeNull();
        expect(errors.filter((e) => e.includes('Failed to fetch user'))).toHaveLength(0);
    });
});
