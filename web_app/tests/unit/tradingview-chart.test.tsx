import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TradingViewChart from '@/components/TradingViewChart';

// jsdom exposes no Fullscreen API, so these exercise the portalled-overlay
// fallback — the same path iOS Safari takes.
describe('TradingViewChart full screen', () => {
    afterEach(cleanup);

    it('toggles between the inline chart and a full-screen overlay', async () => {
        const user = userEvent.setup();
        render(<TradingViewChart symbol="AAPL" exchange="NMS" height={520} />);

        await user.click(screen.getByRole('button', { name: 'Full screen' }));
        const exit = screen.getByRole('button', { name: 'Exit full screen' });
        // Portalled clear of the surrounding card, whose backdrop-filter would
        // otherwise clip a fixed overlay back into it.
        expect(exit.closest('.fixed')?.parentElement).toBe(document.body);
        expect(document.body.style.overflow).toBe('hidden');

        await user.click(exit);
        expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();
        expect(document.body.style.overflow).not.toBe('hidden');
    });

    it('leaves full screen on Escape', async () => {
        const user = userEvent.setup();
        render(<TradingViewChart symbol="AAPL" exchange="NMS" />);

        await user.click(screen.getByRole('button', { name: 'Full screen' }));
        await user.keyboard('{Escape}');

        expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();
        expect(document.body.style.overflow).not.toBe('hidden');
    });

    it('offers no full-screen button when the symbol has no TradingView listing', () => {
        render(<TradingViewChart symbol="$CASH" />);
        expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });
});
