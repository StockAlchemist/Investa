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

const getAuthHeaders = () => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
    return token ? { Authorization: `Bearer ${token}` } : {};
};

// Create a globally typed API client using the generated paths
export const apiClient = createClient<paths>({
    baseUrl: getApiBaseUrl(),
    // We can inject headers using a middleware
});

// Middleware to inject authentication token
apiClient.use({
    onRequest({ request }) {
        const headers = getAuthHeaders();
        if (headers.Authorization) {
            request.headers.set('Authorization', headers.Authorization);
        }
        return request;
    }
});
