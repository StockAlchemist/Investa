const API_BASE_URL = 'http://localhost:8000/api';

export interface PortfolioSummary {
    metrics: {
        'Total Value': number;
        'Day\'s G/L': number;
        'Day\'s G/L %': number;
        'Unrealized G/L': number;
        'Unrealized G/L %': number;
        'Realized G/L': number;
        'Dividends': number;
        'Cash Balance': number;
        [key: string]: number | null;
    } | null;
    account_metrics: Record<string, any> | null;
}

export interface Holding {
    Symbol: string;
    Quantity: number;
    'Price/Share': number;
    'Market Value': number;
    'Day\'s G/L': number;
    'Day\'s G/L %': number;
    'Unrealized G/L': number;
    'Unrealized G/L %': number;
    [key: string]: any;
}

export async function fetchSummary(currency: string = 'USD'): Promise<PortfolioSummary> {
    try {
        const res = await fetch(`${API_BASE_URL}/summary?currency=${currency}`, {
            cache: 'no-store', // Ensure fresh data
        });
        if (!res.ok) {
            throw new Error('Failed to fetch summary');
        }
        return res.json();
    } catch (error) {
        console.error('Error fetching summary:', error);
        return { metrics: null, account_metrics: null };
    }
}

export async function fetchHoldings(currency: string = 'USD'): Promise<Holding[]> {
    try {
        const res = await fetch(`${API_BASE_URL}/holdings?currency=${currency}`, {
            cache: 'no-store',
        });
        if (!res.ok) {
            throw new Error('Failed to fetch holdings');
        }
        return res.json();
    } catch (error) {
        console.error('Error fetching holdings:', error);
        return [];
    }
}
