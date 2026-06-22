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
    // Auth now lives in an httpOnly cookie, so the mock is stateful: /auth/me is
    // 401 (logged out) until /auth/login succeeds, mirroring the real cookie.
    let loggedIn = false;
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
        loggedIn
            ? route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FAKE_USER) })
            : route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ detail: 'Could not validate credentials' }) }),
    );
    // Default login succeeds and "sets the cookie" (flips the flag). Tests that
    // need a failed/rate-limited login override this route after mockApi().
    await page.route('**/api/auth/login', (route) => {
        loggedIn = true;
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ access_token: FAKE_TOKEN, token_type: 'bearer' }) });
    });
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
        await mockApi(page); // default login mock succeeds and flips the session on

        await page.goto('/login');
        await page.getByLabel('Username').fill(FAKE_USER.username);
        await page.getByLabel('Password').fill('correct-password');
        await page.getByRole('button', { name: /sign in|log ?in/i }).click();

        // Auth now lives in an httpOnly cookie (not JS-readable), so success is:
        // we leave the login page and the user profile is loaded/cached...
        await page.waitForURL((url) => !url.pathname.includes('login'));
        const cachedUser = await page.evaluate(() => localStorage.getItem('investa_user'));
        expect(cachedUser).toContain(FAKE_USER.username);
        // ...and the token is NOT persisted to localStorage anymore.
        const storedToken = await page.evaluate(() => localStorage.getItem('access_token'));
        expect(storedToken).toBeNull();
    });

    test('an invalid session on a return visit logs out quietly to /login', async ({ page }) => {
        await mockApi(page);
        await page.route('**/api/auth/me', (route) =>
            route.fulfill({
                status: 401,
                contentType: 'application/json',
                body: JSON.stringify({ detail: 'Could not validate credentials' }),
            }),
        );

        // Simulate a return visit: a cached profile is present (optimistic
        // restore), but the cookie is invalid (mocked /auth/me returns 401).
        await page.addInitScript((user) => {
            localStorage.setItem('investa_user', JSON.stringify(user));
        }, FAKE_USER);

        const errors: string[] = [];
        page.on('console', (msg) => {
            if (msg.type() === 'error') errors.push(msg.text());
        });

        await page.goto('/');
        await page.waitForURL('**/login');

        // The cached profile must be cleared, and the 401 handled without console.error
        const remaining = await page.evaluate(() => localStorage.getItem('investa_user'));
        expect(remaining).toBeNull();
        expect(errors.filter((e) => e.includes('Failed to fetch user'))).toHaveLength(0);
    });
});
