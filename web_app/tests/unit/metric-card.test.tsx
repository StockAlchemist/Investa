import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricCard } from '@/components/MetricCard';

describe('MetricCard', () => {
    it('renders title and a currency-formatted value', () => {
        render(<MetricCard title="Market Value" value={5021.43} isCurrency currency="USD" />);
        expect(screen.getByText('Market Value')).toBeInTheDocument();
        expect(screen.getByText('$5,021.43')).toBeInTheDocument();
    });

    it('renders a THB value with the ฿ symbol', () => {
        render(<MetricCard title="Cash" value={40000} isCurrency currency="THB" />);
        expect(screen.getByText(/฿40,000/)).toBeInTheDocument();
    });

    it('shows a skeleton while loading instead of the value', () => {
        const { container } = render(
            <MetricCard title="Market Value" value={1234} isCurrency isLoading />,
        );
        expect(screen.queryByText('$1,234.00')).not.toBeInTheDocument();
        expect(container.querySelector('[data-slot="skeleton"], .animate-pulse')).toBeTruthy();
    });

    it('renders string values verbatim', () => {
        render(<MetricCard title="Status" value="Open" />);
        expect(screen.getByText('Open')).toBeInTheDocument();
    });
});
