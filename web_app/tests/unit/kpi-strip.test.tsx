import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import KpiStrip from '@/components/performance/KpiStrip';
import type { AssetChangeData } from '@/lib/api';

describe('KpiStrip', () => {
    const mockData: AssetChangeData = {
        M: [
            {
                Date: '2025-01-31',
                'Portfolio M-Return': 2.5,
                'S&P 500 M-Return': 1.5,
                'NASDAQ M-Return': 3.0,
            },
            {
                Date: '2025-02-28',
                'Portfolio M-Return': 3.0,
                'S&P 500 M-Return': 1.0,
                'NASDAQ M-Return': 2.0,
            },
        ],
        Y: [],
        W: [],
        D: [],
    };

    it('renders both vs S&P 500 and vs NASDAQ KPI tiles when provided in benchmarks', () => {
        render(
            <KpiStrip
                data={mockData}
                summary={null}
                benchmarks={['S&P 500', 'NASDAQ']}
            />
        );

        expect(screen.getByText('vs S&P 500')).toBeInTheDocument();
        expect(screen.getByText('vs NASDAQ')).toBeInTheDocument();

        // Portfolio compounded: (1.025 * 1.03 - 1) * 100 = 5.575%
        // S&P 500 compounded: (1.015 * 1.01 - 1) * 100 = 2.515%
        // vs S&P 500 = 5.575 - 2.515 = +3.06%
        expect(screen.getByText('+3.06%')).toBeInTheDocument();

        // NASDAQ compounded: (1.03 * 1.02 - 1) * 100 = 5.06%
        // vs NASDAQ = 5.575 - 5.06 = +0.52% (approx 0.515% -> 0.52%)
        expect(screen.getByText('+0.52%')).toBeInTheDocument();
    });

    it('defaults to S&P 500 and NASDAQ when benchmarks list is empty', () => {
        render(
            <KpiStrip
                data={mockData}
                summary={null}
                benchmarks={[]}
            />
        );

        expect(screen.getByText('vs S&P 500')).toBeInTheDocument();
        expect(screen.getByText('vs NASDAQ')).toBeInTheDocument();
    });

    it('renders all standard KPI tiles: YTD, 1Y, Win Rate, Best Month, Worst Month, Max DD', () => {
        render(
            <KpiStrip
                data={mockData}
                summary={null}
                benchmarks={['S&P 500', 'NASDAQ']}
            />
        );

        expect(screen.getByText('YTD')).toBeInTheDocument();
        expect(screen.getByText('1Y')).toBeInTheDocument();
        expect(screen.getByText('Win Rate')).toBeInTheDocument();
        expect(screen.getByText('Best Month')).toBeInTheDocument();
        expect(screen.getByText('Worst Month')).toBeInTheDocument();
        expect(screen.getByText('Max DD')).toBeInTheDocument();
    });
});
