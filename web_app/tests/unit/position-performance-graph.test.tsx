import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PositionPerformanceGraph } from '@/components/stock-detail/components/PositionPerformanceGraph';
import * as api from '@/lib/api';

vi.mock('@/lib/api', async () => {
    const actual = await vi.importActual<typeof import('@/lib/api')>('@/lib/api');
    return {
        ...actual,
        fetchStockPositionHistory: vi.fn(),
    };
});

describe('PositionPerformanceGraph', () => {
    const queryClient = new QueryClient({
        defaultOptions: {
            queries: {
                retry: false,
            },
        },
    });

    const mockPoints: api.StockPositionHistoryPoint[] = [
        {
            date: '2024-01-10',
            value: 15000.0,
            cost_basis: 15000.0,
            shares: 100.0,
            unrealized_gain: 0.0,
            unrealized_gain_pct: 0.0,
            return_pct: 0.0,
            'S&P 500': 0.0,
        },
        {
            date: '2024-06-15',
            value: 18000.0,
            cost_basis: 15000.0,
            shares: 100.0,
            unrealized_gain: 3000.0,
            unrealized_gain_pct: 20.0,
            return_pct: 20.0,
            'S&P 500': 5.0,
        },
    ];

    it('renders Value and Return mode toggle buttons', async () => {
        vi.mocked(api.fetchStockPositionHistory).mockResolvedValue(mockPoints);

        render(
            <QueryClientProvider client={queryClient}>
                <PositionPerformanceGraph symbol="AAPL" currency="USD" />
            </QueryClientProvider>
        );

        expect(screen.getByText('Position Performance History')).toBeInTheDocument();
        expect(screen.getByText('Value')).toBeInTheDocument();
        expect(screen.getByText('Return (%)')).toBeInTheDocument();
        expect(screen.getByText('1Y')).toBeInTheDocument();
        expect(screen.getByText('ALL')).toBeInTheDocument();
    });

    it('switches to Return (%) mode and shows benchmark toggles', async () => {
        vi.mocked(api.fetchStockPositionHistory).mockResolvedValue(mockPoints);

        render(
            <QueryClientProvider client={queryClient}>
                <PositionPerformanceGraph symbol="AAPL" currency="USD" />
            </QueryClientProvider>
        );

        const returnBtn = screen.getByText('Return (%)');
        fireEvent.click(returnBtn);

        expect(screen.getByText('Compare:')).toBeInTheDocument();
        expect(screen.getByText('S&P 500')).toBeInTheDocument();
        expect(screen.getByText('NASDAQ')).toBeInTheDocument();
        expect(screen.getByText('Dow Jones')).toBeInTheDocument();
    });
});
