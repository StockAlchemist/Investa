import '@testing-library/jest-dom/vitest';

// recharts' ResponsiveContainer measures its parent via ResizeObserver,
// which jsdom doesn't implement.
class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver ?? (ResizeObserverStub as unknown as typeof ResizeObserver);
