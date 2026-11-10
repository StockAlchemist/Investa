import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import KpiStrip from '@/components/performance/KpiStrip';
import type { AssetChangeData, PortfolioSummary, RiskMetrics } from '@/lib/api';

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

    describe('YTD', () => {
        // dheematan's 2026 monthly TWR returns (Dec 2025 belongs to last year).
        // Summed they make +9.92%; compounded they make +7.25%, the 2026 annual
        // bar and the risk-metrics YTD.
        const months2026 = [2.907020011758177, -4.685547213290975, -5.789581156794565,
            17.91476013952742, 5.132544411435558, -7.451215008679368, 4.464400825611636,
            3.022301272299943, 2.1280774785926715, -10.032951548301172, 2.307572488182963];
        const ytdData: AssetChangeData = {
            M: [
                { Date: '2025-12-31', 'Portfolio M-Return': 0.020308379222711004 },
                ...months2026.map((v, i) => ({
                    Date: `2026-${String(i + 1).padStart(2, '0')}-28`,
                    'Portfolio M-Return': v,
                })),
            ],
            Y: [], W: [], D: [],
        };
        const ytdTile = () => screen.getByText('YTD').nextElementSibling?.textContent;

        afterEach(() => vi.useRealTimers());

        it('compounds the market year’s months instead of adding them', () => {
            vi.useFakeTimers({ toFake: ['Date'] });
            vi.setSystemTime(new Date('2026-11-10T12:00:00Z'));
            render(<KpiStrip data={ytdData} summary={null} benchmarks={[]} />);
            expect(ytdTile()).toBe('+7.25%');
        });

        it('takes the year from the market clock, not the device', () => {
            // 1 Jan 2027 06:00 in Bangkok is still 31 Dec 2026 in New York.
            vi.useFakeTimers({ toFake: ['Date'] });
            vi.setSystemTime(new Date('2026-12-31T23:00:00Z'));
            render(<KpiStrip data={ytdData} summary={null} benchmarks={[]} />);
            expect(ytdTile()).toBe('+7.25%');
        });

        it('prefers the summary YTD, then the same figure from risk metrics', () => {
            const summary = { metrics: { ytd_return: 7.264051225444201 } } as unknown as PortfolioSummary;
            const risk = { 'YTD Return': 0.05 } as RiskMetrics;
            render(<KpiStrip data={ytdData} summary={summary} riskMetrics={risk} benchmarks={[]} />);
            expect(ytdTile()).toBe('+7.26%');
        });

        it('takes the risk-metrics YTD when the summary has none', () => {
            const risk = { 'YTD Return': 0.07254306252507203 } as RiskMetrics;
            render(<KpiStrip data={ytdData} summary={null} riskMetrics={risk} benchmarks={[]} />);
            expect(ytdTile()).toBe('+7.25%');
        });
    });
});
