import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { exportToCSV } from '@/lib/export';

describe('exportToCSV', () => {
    let capturedBlob: Blob | null;
    let clickSpy: ReturnType<typeof vi.fn<() => void>>;

    beforeEach(() => {
        capturedBlob = null;
        clickSpy = vi.fn<() => void>();
        globalThis.URL.createObjectURL = vi.fn((blob: Blob) => {
            capturedBlob = blob;
            return 'blob:fake-url';
        });
        vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(clickSpy);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('does nothing for empty data', () => {
        exportToCSV([], 'empty.csv');
        expect(clickSpy).not.toHaveBeenCalled();
    });

    it('builds a CSV with headers from the first row and triggers a download', async () => {
        exportToCSV(
            [
                { Symbol: 'AAPL', Quantity: 10, Note: 'plain' },
                { Symbol: 'MSFT', Quantity: 5, Note: 'has, comma' },
            ],
            'holdings.csv',
        );

        expect(clickSpy).toHaveBeenCalledOnce();
        expect(capturedBlob).not.toBeNull();
        const text = await capturedBlob!.text();
        const lines = text.split('\n');
        expect(lines[0]).toBe('Symbol,Quantity,Note');
        expect(lines[1]).toBe('AAPL,10,plain');
        // values containing commas are quoted
        expect(lines[2]).toBe('MSFT,5,"has, comma"');
    });
});
