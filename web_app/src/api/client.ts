import createClient from 'openapi-fetch';
import type { paths } from './types';

// Detect the base URL similarly to how lib/api.ts did it.
const getApiBaseUrl = () => {
    let url = '';
    if (process.env.NEXT_PUBLIC_API_URL) {
        url = process.env.NEXT_PUBLIC_API_URL;
    } else if (typeof window !== 'undefined') {
        if (window.location.hostname.endsWith('ts.net')) {
            url = ''; // openapi paths already start with /api
        } else {
            url = `http://${window.location.hostname}:8000`;
        }
    } else {
        url = 'http://localhost:8000';
    }
    // Remove trailing /api since the generated paths already include it
    return url.replace(/\/api\/?$/, '');
};

// Create a globally typed API client using the generated paths.
// credentials:'include' sends the httpOnly auth cookie on every request (the
// token is no longer kept in JS-readable localStorage).
export const apiClient = createClient<paths>({
    baseUrl: getApiBaseUrl(),
    credentials: 'include',
});

// Middleware: broadcast expiry on 401 so AuthContext can log out (same contract
// authFetch in lib/api.ts had).
apiClient.use({
    onResponse({ request, response }) {
        if (
            response.status === 401 &&
            typeof window !== 'undefined' &&
            !request.url.includes('/auth/login')
        ) {
            window.dispatchEvent(new CustomEvent('auth:expired'));
        }
        return response;
    },
});
