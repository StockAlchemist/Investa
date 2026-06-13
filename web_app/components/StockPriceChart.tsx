import React, { useState, useMemo, useRef } from 'react';
import { useQuery, keepPreviousData } from '@tanstack/react-query';
import {
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Area,
    ReferenceLine,
    Bar,
    ComposedChart
} from 'recharts';
import PeriodSelector from './PeriodSelector';

import {
    fetchStockHistory,
    fetchTransactions,
    fetchDividends,
    fetchEarningsDates,
    fetchCapitalGains,
    Transaction,
    Dividend,
    EarningsDate,
    CapitalGain
} from '../lib/api';
import { formatCurrency } from '../lib/utils';

// --- Types ---
interface StockPriceChartProps {
    symbol: string;
    currency: string;
    benchmarks?: string[]; // Optional initial benchmarks
    avgCost?: number;
    hidePrice?: boolean;
    fxRate?: number;
    accounts?: string[]; // Account filter for transaction/dividend overlays
}

// Overlay event marker shapes
type EventKind = 'buy' | 'sell' | 'dividend' | 'earnings';

interface ChartEvent {
    kind: EventKind;
    timestamp: number; // snapped to a chart x value
    y: number;         // y-coordinate (price-axis) where the marker sits
    label: string;     // tooltip / title text
    gain?: number;     // sell only: realized gain/loss (display currency)
    gainPct?: number;  // sell only: realized gain/loss as % of cost basis
}

const EVENT_STYLES: Record<EventKind, { color: string; letter: string }> = {
    buy: { color: '#16a34a', letter: 'B' },       // green
    sell: { color: '#dc2626', letter: 'S' },      // red
    dividend: { color: '#d97706', letter: 'D' },  // amber
    earnings: { color: '#9333ea', letter: 'E' },  // purple
};

// Benchmarks selectable as return-% overlays. `name` matches the backend's
// BENCHMARK_MAPPING keys; `key` is the mapped Yahoo ticker the API returns as a column.
const BENCHMARKS = [
    { name: 'S&P 500', key: '^GSPC', color: '#f59e0b' },   // amber
    { name: 'NASDAQ', key: '^IXIC', color: '#8b5cf6' },    // purple
    { name: 'Dow Jones', key: '^DJI', color: '#0ea5e9' },  // sky
] as const;

interface CustomTooltipProps {
    active?: boolean;
    payload?: Array<{ value?: number; name?: string; color?: string; dataKey?: string | number; payload?: Record<string, unknown> }>;
    label?: string;
    view: 'price' | 'return';
    currency: string;
}

// --- Constants ---
// --- Helper Functions ---
const formatVolume = (val: number) => {
    if (val >= 1e9) return `${(val / 1e9).toFixed(2)}B`;
    if (val >= 1e6) return `${(val / 1e6).toFixed(2)}M`;
    if (val >= 1e3) return `${(val / 1e3).toFixed(2)}K`;
    return val.toString();
};

const calculateSMA = (data: Array<{ value: number; [k: string]: unknown }>, period: number) => {
    if (data.length < period) return [];
    // Calculate SMA
    const smaData = data.map((item, index, array) => {
        if (index < period - 1) return { ...item, sma: null };
        let sum = 0;
        for (let i = 0; i < period; i++) {
            sum += array[index - i].value;
        }
        return { ...item, sma: sum / period };
    });
    return smaData;
};

// Custom dot drawn for an overlay event. Rendered via a transparent <Line>'s `dot`
// callback so it inherits the same (reliable) axis positioning the price/SMA lines use.
// Recharts passes cx/cy (pixel coords) and the full data point as `payload`.
const EventDot = ({ cx, cy, payload, kind }: {
    cx?: number;
    cy?: number;
    payload?: Record<string, unknown>;
    kind: EventKind;
}) => {
    if (cx == null || cy == null || !payload) return null;
    // Only draw where this point actually carries an event of this kind.
    if (payload[`_evt_${kind}`] == null) return null;
    const style = EVENT_STYLES[kind];
    const r = 7;
    // Full event details are shown in the chart's hover tooltip (CustomTooltip).
    return (
        <g style={{ cursor: 'pointer' }}>
            <circle cx={cx} cy={cy} r={r} fill={style.color} stroke="#fff" strokeWidth={1.5} />
            <text x={cx} y={cy} dy={3.2} textAnchor="middle" fontSize={9} fontWeight={700} fill="#fff">
                {style.letter}
            </text>
        </g>
    );
};

export default function StockPriceChart({ symbol, currency, avgCost, hidePrice, fxRate = 1, accounts }: StockPriceChartProps) {
    const [view, setView] = useState<'price' | 'return'>('price');
    const [period, setPeriod] = useState('1y');
    const [showSMA50, setShowSMA50] = useState(false);
    const [showSMA200, setShowSMA200] = useState(false);

    // Event overlay toggles
    const [showBuys, setShowBuys] = useState(false);
    const [showSells, setShowSells] = useState(false);
    const [showDividends, setShowDividends] = useState(false);
    const [showEarnings, setShowEarnings] = useState(false);

    // Benchmark overlays (Return % view); stores BENCHMARK_MAPPING names.
    const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
    const toggleBenchmark = (name: string) =>
        setSelectedBenchmarks((prev) =>
            prev.includes(name) ? prev.filter((b) => b !== name) : [...prev, name]
        );

    const containerRef = useRef<HTMLDivElement>(null);

    // Determine interval base on period
    const interval = useMemo(() => {
        if (period === '1d') return '2m';
        if (period === '5d') return '15m';
        if (period === '1m') return '1d';
        if (period === '3m') return '1d';
        return '1d';
    }, [period]);

    // Determine Fetch Parameters (Fetch more data than shown for SMA)
    const fetchParams = useMemo(() => {
        let fetchPeriod = period;
        const fetchInterval = interval;

        // Map request to longer period for SMA buffer
        // Note: Backend supports: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        switch (period) {
            case '1d': fetchPeriod = '5d'; break;     // Need prev days for intraday SMA? 
            // Actually intraday SMA usually resets or needs 5d worth of minutes.
            case '5d': fetchPeriod = '1mo'; break;
            case '1m': fetchPeriod = '3mo'; break;
            case '3m': fetchPeriod = '6mo'; break;
            case '6m': fetchPeriod = '1y'; break;
            case 'ytd': fetchPeriod = '2y'; break;    // Safe buffer
            case '1y': fetchPeriod = '5y'; break;     // Increased to 5y to ensure clean 200 SMA
            case '3y': fetchPeriod = '5y'; break;
            case '5y': fetchPeriod = '10y'; break;
            case '10y': fetchPeriod = 'max'; break; // 10y -> max is best bet
            case 'max': fetchPeriod = 'max'; break;
        }

        return { period: fetchPeriod, interval: fetchInterval };
    }, [period, interval]);

    // Stable, sorted benchmark list for query identity.
    const benchmarkParam = useMemo(() => [...selectedBenchmarks].sort(), [selectedBenchmarks]);

    // Data Fetching
    const { data: rawData, isLoading } = useQuery({
        queryKey: ['stock_history', symbol, fetchParams.period, fetchParams.interval, benchmarkParam],
        queryFn: ({ signal }) => fetchStockHistory(symbol, fetchParams.period, fetchParams.interval, benchmarkParam, signal),
        placeholderData: keepPreviousData,
        staleTime: period === '1d' ? 60 * 1000 : 5 * 60 * 1000,
        refetchInterval: period === '1d' ? 60 * 1000 : false,
    });

    const data = useMemo(() => rawData || [], [rawData]);
    const isContinuous = period === '1d';

    // --- Overlay event data (only fetched when a toggle is enabled) ---
    const { data: transactions } = useQuery({
        queryKey: ['transactions', accounts],
        queryFn: ({ signal }) => fetchTransactions(accounts, signal),
        // Also needed by the dividend overlay to derive shares held (for yield).
        enabled: showBuys || showSells || showDividends,
        staleTime: 5 * 60 * 1000,
    });

    // Realized gains keyed by the originating sell transaction; powers the
    // proceeds / % gain shown in the sell marker's tooltip.
    const { data: capitalGainsData } = useQuery({
        queryKey: ['capital_gains', currency, accounts],
        queryFn: ({ signal }) => fetchCapitalGains(currency, accounts, undefined, undefined, signal),
        enabled: showSells,
        staleTime: 5 * 60 * 1000,
    });

    const { data: dividendsData } = useQuery({
        queryKey: ['dividends', currency, accounts],
        queryFn: ({ signal }) => fetchDividends(currency, accounts, signal),
        enabled: showDividends,
        staleTime: 5 * 60 * 1000,
    });

    const { data: earningsData } = useQuery({
        queryKey: ['earnings_dates', symbol],
        queryFn: ({ signal }) => fetchEarningsDates(symbol, signal),
        enabled: showEarnings,
        staleTime: 30 * 60 * 1000,
    });

    // Processing Data for Chart
    const chartedData = useMemo(() => {
        if (!data || data.length === 0) return [];

        let processed = data.map(d => ({
            ...d,
            timestamp: new Date(d.date).getTime(),
            value: d.value * fxRate,
        }));

        // Calculate SMAs on the FULL fetched dataset (including buffer)
        if (data.length >= 50) {
            const sma50 = calculateSMA(processed, 50);
            processed = processed.map((p, i) => ({ ...p, sma50: sma50[i].sma }));
        }

        if (data.length >= 200) {
            const sma200 = calculateSMA(processed, 200);
            processed = processed.map((p, i) => ({ ...p, sma200: sma200[i].sma }));
        }

        // NOW Filter down to the requested period for display
        // We calculate start cutoff based on the *User's Selected Period*
        const now = new Date();
        let cutoffTime = 0;

        // Simple cutoff logic
        switch (period) {
            case '1d':
                // For 1d, we just want today (or last trading day). 
                // However, our backend 1d logic usually returns strict 9:30-16:00 of *latest* day.
                // If we fetched 5d, we have 5 days. We want only the last day.
                // Let's take the date string of the last point and filter by that?
                if (processed.length > 0) {
                    const lastDate = new Date(processed[processed.length - 1].timestamp);
                    // Create midnight of that day in local or NY?
                    // Safe bet: Last 390 minutes (6.5 hours)?
                    // Better: Same Year-Month-Day as last point.
                    // Filter where date string starts with lastYMD
                    // But date object comparison is safer.
                    const startOfDay = new Date(lastDate);
                    startOfDay.setHours(0, 0, 0, 0);
                    cutoffTime = startOfDay.getTime();
                }
                break;
            case '5d': cutoffTime = now.getTime() - (5 * 24 * 60 * 60 * 1000); break;
            case '1m': cutoffTime = now.getTime() - (30 * 24 * 60 * 60 * 1000); break;
            case '3m': cutoffTime = now.getTime() - (90 * 24 * 60 * 60 * 1000); break;
            case '6m': cutoffTime = now.getTime() - (180 * 24 * 60 * 60 * 1000); break;
            case 'ytd':
                cutoffTime = new Date(now.getFullYear(), 0, 1).getTime();
                break;
            case '1y': cutoffTime = now.getTime() - (365 * 24 * 60 * 60 * 1000); break;
            case '3y': cutoffTime = now.getTime() - (3 * 365 * 24 * 60 * 60 * 1000); break;
            case '5y': cutoffTime = now.getTime() - (5 * 365 * 24 * 60 * 60 * 1000); break;
            case '10y': cutoffTime = now.getTime() - (10 * 365 * 24 * 60 * 60 * 1000); break;
            default: cutoffTime = 0; // max/all
        }

        // Apply visual filter
        if (period !== 'max') {
            processed = processed.filter(d => d.timestamp >= cutoffTime);
        }

        // Re-normalize Return % to the first VISIBLE point. The backend normalizes
        // both the stock and the benchmarks to the longer fetch window (used as an SMA
        // buffer), so without this the Return % view wouldn't start at 0% and benchmarks
        // wouldn't be comparable over the selected period.
        if (processed.length > 0) {
            const baseValue = processed[0].value;
            const benchBase: Record<string, number> = {};
            const first = processed[0] as unknown as Record<string, unknown>;
            for (const b of BENCHMARKS) {
                const b0 = first[b.key];
                if (typeof b0 === 'number') benchBase[b.key] = b0;
            }
            processed = processed.map((p) => {
                const next = { ...p };
                if (baseValue && baseValue > 0) {
                    next.return_pct = (p.value / baseValue - 1) * 100;
                }
                const pRec = p as unknown as Record<string, unknown>;
                const nextRec = next as unknown as Record<string, number>;
                for (const b of BENCHMARKS) {
                    const bt = pRec[b.key];
                    const b0 = benchBase[b.key];
                    // Convert "% since fetch start" back to a price ratio, then re-base
                    // it to the first visible point: (1+bt)/(1+b0) - 1.
                    if (typeof bt === 'number' && typeof b0 === 'number') {
                        nextRec[b.key] = ((1 + bt / 100) / (1 + b0 / 100) - 1) * 100;
                    }
                }
                return next;
            });
        }

        return processed;
    }, [data, period, fxRate]);

    // --- Build overlay event markers (snapped to visible chart points) ---
    // We attach event values directly onto the chart data points and render them via
    // transparent <Line>s, so markers inherit the same axis positioning as the price line
    // (ReferenceDot with an x on a category axis is unreliable in recharts).
    const { chartDataWithEvents, presentKinds } = useMemo(() => {
        const emptyKinds = { buy: false, sell: false, dividend: false, earnings: false };
        if (view !== 'price' || chartedData.length === 0) {
            return { chartDataWithEvents: chartedData, presentKinds: emptyKinds };
        }

        const points = chartedData; // ascending by timestamp
        const firstTs = points[0].timestamp;
        const lastTs = points[points.length - 1].timestamp;
        const dayMs = 24 * 60 * 60 * 1000;
        // Allow events landing on a non-trading day (weekend/holiday) to snap in,
        // but ignore events outside the visible window.
        const lowerBound = firstTs - 4 * dayMs;
        const upperBound = lastTs + 4 * dayMs;

        const snap = (eventMs: number) => {
            if (Number.isNaN(eventMs) || eventMs < lowerBound || eventMs > upperBound) return null;
            let best = points[0];
            let bestDiff = Math.abs(points[0].timestamp - eventMs);
            for (let i = 1; i < points.length; i++) {
                const diff = Math.abs(points[i].timestamp - eventMs);
                if (diff < bestDiff) {
                    bestDiff = diff;
                    best = points[i];
                }
            }
            return best;
        };

        // Aggregate multiple same-kind events that snap to the same point.
        const agg = new Map<string, ChartEvent>();

        // The price line is split-adjusted (Yahoo back-adjusts splits regardless of
        // dividend auto-adjust), but a transaction's stored Price/Share is the nominal
        // price at trade time. Divide by the cumulative ratio of any splits that
        // occurred AFTER the trade so the marker lands on the adjusted line.
        const splitEvents = transactions
            ? (transactions as Transaction[])
                .filter((t) => {
                    if (t.Symbol !== symbol) return false;
                    const ty = String(t.Type || '').toLowerCase();
                    return (ty === 'split' || ty === 'stock split') && Number(t['Split Ratio']) > 0;
                })
                .map((t) => ({ ts: new Date(t.Date).getTime(), ratio: Number(t['Split Ratio']) }))
            : [];
        const splitFactorAfter = (tradeTs: number) =>
            splitEvents.reduce((f, s) => (s.ts > tradeTs ? f * s.ratio : f), 1);
        const formatQty = (q: number) => (Number.isInteger(q) ? String(q) : String(Number(q.toFixed(4))));

        // Realized gain (proceeds / % gain) per originating sell transaction id.
        const cgByTxId = new Map<number, CapitalGain>();
        if (showSells && capitalGainsData) {
            for (const cg of capitalGainsData as CapitalGain[]) {
                if (cg.original_tx_id != null) cgByTxId.set(Number(cg.original_tx_id), cg);
            }
        }

        // Signed, split-adjusted share movements for this symbol, used to derive
        // shares held at a dividend date (so we can express a per-payment yield).
        // Quantities are adjusted to the price line's present-day basis, matching
        // pt.value, so `sharesHeld * pt.value` is the position's market value.
        const shareMoves: { ts: number; q: number }[] = [];
        if (transactions) {
            for (const t of transactions as Transaction[]) {
                if (t.Symbol !== symbol) continue;
                const type = String(t.Type || '').toLowerCase();
                const isBuy = type === 'buy' || type === 'buy to cover';
                const isSell = type === 'sell' || type === 'short sell';
                if (!isBuy && !isSell) continue;
                const tradeTs = new Date(t.Date).getTime();
                const qty = (Number(t.Quantity) || 0) * splitFactorAfter(tradeTs);
                shareMoves.push({ ts: tradeTs, q: isBuy ? qty : -qty });
            }
        }
        const sharesHeldAt = (ts: number) =>
            shareMoves.reduce((s, m) => (m.ts <= ts ? s + m.q : s), 0);

        // Running realized cost basis per sell point, so an aggregated gain % can
        // be derived from the summed gain/cost when several sells snap together.
        const sellCost = new Map<string, number>();

        if ((showBuys || showSells) && transactions) {
            for (const t of transactions as Transaction[]) {
                if (t.Symbol !== symbol) continue;
                const type = String(t.Type || '').toLowerCase();
                const isBuy = type === 'buy' || type === 'buy to cover';
                const isSell = type === 'sell' || type === 'short sell';
                if (!((isBuy && showBuys) || (isSell && showSells))) continue;

                const tradeTs = new Date(t.Date).getTime();
                const pt = snap(tradeTs);
                if (!pt) continue;

                const kind: EventKind = isBuy ? 'buy' : 'sell';
                const factor = splitFactorAfter(tradeTs);
                const qty = (Number(t.Quantity) || 0) * factor;
                const priceLocal = (Number(t['Price/Share']) || 0) / factor;
                const price = priceLocal > 0 ? priceLocal * fxRate : pt.value;
                const key = `${kind}:${pt.timestamp}`;

                // For a closing sell, carry the realized gain/loss for color-coded
                // display (proceeds are intentionally omitted from the label).
                let gain: number | undefined;
                let cost = 0;
                if (isSell) {
                    const cg = t.id != null ? cgByTxId.get(Number(t.id)) : undefined;
                    cost = cg ? Number(cg['Total Cost Basis (Display)']) || 0 : 0;
                    gain = cg ? Number(cg['Realized Gain (Display)']) || 0 : undefined;
                }

                const segment = `${formatQty(qty)} @ ${formatCurrency(price, currency)}`;
                const existing = agg.get(key);
                if (existing) {
                    existing.label = `${existing.label}, ${segment}`;
                    if (gain != null) existing.gain = (existing.gain ?? 0) + gain;
                } else {
                    agg.set(key, {
                        kind,
                        timestamp: pt.timestamp,
                        y: price,
                        label: `${isBuy ? 'Buy' : 'Sell'} ${segment}`,
                        gain,
                    });
                }
                if (isSell) {
                    sellCost.set(key, (sellCost.get(key) ?? 0) + cost);
                    const ev = agg.get(key)!;
                    const totalCost = sellCost.get(key)!;
                    ev.gainPct = totalCost > 0 && ev.gain != null
                        ? (ev.gain / totalCost) * 100
                        : undefined;
                }
            }
        }

        if (showDividends && dividendsData) {
            // Sum payments that snap to the same chart point, then express the
            // total as a yield against the position's market value at that date.
            const divTotals = new Map<string, number>();
            for (const d of dividendsData as Dividend[]) {
                if (d.Symbol !== symbol) continue;
                const divTs = new Date(d.Date).getTime();
                const pt = snap(divTs);
                if (!pt) continue;
                const amt = Number(d.DividendAmountDisplayCurrency) || 0;
                const key = `dividend:${pt.timestamp}`;
                const total = (divTotals.get(key) || 0) + amt;
                divTotals.set(key, total);

                const marketValue = sharesHeldAt(divTs) * pt.value;
                const yieldPct = marketValue > 0 ? (total / marketValue) * 100 : null;
                const label = `Dividend ${formatCurrency(total, currency)}`
                    + (yieldPct != null ? ` · ${yieldPct.toFixed(2)}% yield` : '');

                const existing = agg.get(key);
                if (existing) {
                    existing.label = label;
                } else {
                    agg.set(key, {
                        kind: 'dividend',
                        timestamp: pt.timestamp,
                        y: pt.value,
                        label,
                    });
                }
            }
        }

        if (showEarnings && earningsData) {
            for (const e of earningsData as EarningsDate[]) {
                const pt = snap(new Date(e.date).getTime());
                if (!pt) continue;
                const parts: string[] = ['Earnings'];
                if (e.eps_actual != null) parts.push(`EPS ${Number(e.eps_actual).toFixed(2)}`);
                else if (e.eps_estimate != null) parts.push(`Est. EPS ${Number(e.eps_estimate).toFixed(2)}`);
                if (e.surprise_pct != null) parts.push(`(${Number(e.surprise_pct) >= 0 ? '+' : ''}${Number(e.surprise_pct).toFixed(1)}%)`);
                agg.set(`earnings:${pt.timestamp}`, {
                    kind: 'earnings',
                    timestamp: pt.timestamp,
                    y: pt.value,
                    label: parts.join(' '),
                });
            }
        }

        // Index events by snapped timestamp so we can stamp them onto chart points.
        const byTs = new Map<number, Partial<Record<EventKind, ChartEvent>>>();
        const present = { ...emptyKinds };
        for (const ev of agg.values()) {
            present[ev.kind] = true;
            const slot = byTs.get(ev.timestamp) ?? {};
            slot[ev.kind] = ev;
            byTs.set(ev.timestamp, slot);
        }

        if (byTs.size === 0) {
            return { chartDataWithEvents: chartedData, presentKinds: present };
        }

        const merged = points.map((p) => {
            const slot = byTs.get(p.timestamp);
            if (!slot) return p;
            const extra: Record<string, number | string> = {};
            (Object.keys(slot) as EventKind[]).forEach((k) => {
                const ev = slot[k]!;
                extra[`_evt_${k}`] = ev.y;
                extra[`_evt_${k}_label`] = ev.label;
                if (k === 'sell' && ev.gain != null) {
                    extra['_evt_sell_gain'] = ev.gain;
                    if (ev.gainPct != null) extra['_evt_sell_gain_pct'] = ev.gainPct;
                }
            });
            return { ...p, ...extra };
        });

        return { chartDataWithEvents: merged, presentKinds: present };
    }, [view, chartedData, showBuys, showSells, showDividends, showEarnings, transactions, dividendsData, capitalGainsData, earningsData, symbol, currency, fxRate]);

    // Calculate Stats for Header (Based on Displayed Data)
    const stats = useMemo(() => {
        if (!chartedData || chartedData.length < 2) return null;
        const start = chartedData[0];
        const end = chartedData[chartedData.length - 1];

        const currentPrice = end.value;
        const startPrice = start.value;

        const change = currentPrice - startPrice;
        const changePct = startPrice !== 0 ? (change / startPrice) * 100 : 0;

        return {
            change,
            changePct,
            currentPrice
        };
    }, [chartedData]);

    const gradientOffset = useMemo(() => {
        if (!chartedData || chartedData.length === 0) return 0;

        const dataMax = Math.max(...chartedData.map((d) => d.return_pct));
        const dataMin = Math.min(...chartedData.map((d) => d.return_pct));

        if (dataMax <= 0) return 0;
        if (dataMin >= 0) return 1;

        return dataMax / (dataMax - dataMin);
    }, [chartedData]);

    // Domain Calculation (for X Axis)
    const xDomain = useMemo(() => {
        if (period === '1d' && chartedData.length > 0) {
            // Force strict 09:30 - 16:00 EST visual range
            try {
                // Construct 9:30 and 16:00 for *that* day
                // We'll use a heuristic since we can't easily construct Date from NY string in JS without library.
                // But we know the day. 
                // Let's assume the timestamp IS correct UTC.
                // We just need start/end timestamps.

                // Fallback: Use min/max of data if filtering is working correctly backend side.
                // If backend filters correctly, ['auto', 'auto'] is fine?
                // No, we want fixed range even if data starts at 09:31.

                // Let's rely on data extents + buffer or let auto handle it if backend is strict.
                return ['auto', 'auto'];
            } catch {
                return ['auto', 'auto'];
            }
        }
        return ['auto', 'auto'];
    }, [period, chartedData]);

    // Formatting Functions (EST Forced)
    const formatXAxis = (tickItem: number) => {
        const date = new Date(tickItem);
        if (period === '1d' || period === '5d') {
            return date.toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: '2-digit', minute: '2-digit', hour12: true });
        } else if (period === '1m') {
            return date.toLocaleDateString("en-US", { timeZone: "America/New_York", month: 'short', day: 'numeric' });
        } else {
            return date.toLocaleDateString("en-US", { timeZone: "America/New_York", month: 'short', day: 'numeric' });
        }
    };

    const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
        if (active && payload && payload.length) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any -- recharts tooltip point carries dynamic price/benchmark fields
            const dataPoint = payload[0].payload as any;
            const dateStr = new Date(dataPoint.date).toLocaleString("en-US", {
                timeZone: "America/New_York",
                weekday: 'short', month: 'short', day: 'numeric',
                hour: (period === '1d' || period === '5d') ? '2-digit' : undefined,
                minute: (period === '1d' || period === '5d') ? '2-digit' : undefined,
                year: (period !== '1d' && period !== '5d') ? 'numeric' : undefined
            });

            return (
                <div className="bg-background/98 backdrop-blur-2xl p-3 border border-border/60 shadow-2xl rounded-xl min-w-[240px] !opacity-100">
                    <p className="text-sm font-bold text-foreground mb-2 border-b border-border pb-1">
                        {dateStr}
                    </p>
                    <div className="space-y-1">
                        {/* Main Stock */}
                        <div className="flex items-center justify-between gap-4">
                            <span className="text-xs font-bold text-blue-500 uppercase">{symbol}</span>
                            <span className="text-sm font-bold text-foreground">
                                {view === 'price' ? formatCurrency(dataPoint.value, currency) :
                                    <span className={dataPoint.return_pct >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}>
                                        {dataPoint.return_pct.toFixed(2)}%
                                    </span>
                                }
                            </span>
                        </div>

                        {/* Volume */}
                        <div className="flex items-center justify-between gap-4">
                            <span className="text-xs text-muted-foreground uppercase">Volume</span>
                            <span className="text-sm text-foreground">{formatVolume(dataPoint.volume)}</span>
                        </div>

                        {/* SMAs */}
                        {view === 'price' && showSMA50 && dataPoint.sma50 != null && (
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-xs font-bold text-orange-500 uppercase">SMA 50</span>
                                <span className="text-sm font-medium text-foreground">{formatCurrency(dataPoint.sma50, currency)}</span>
                            </div>
                        )}
                        {view === 'price' && showSMA200 && dataPoint.sma200 != null && (
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-xs font-bold text-purple-600 uppercase">SMA 200</span>
                                <span className="text-sm font-medium text-foreground">{formatCurrency(dataPoint.sma200, currency)}</span>
                            </div>
                        )}

                        {/* Event indicators (price view): show details for any buy/sell/
                            dividend/earnings marker sitting on the hovered point. */}
                        {view === 'price' && (['buy', 'sell', 'dividend', 'earnings'] as EventKind[])
                            .filter((kind) => typeof dataPoint[`_evt_${kind}_label`] === 'string')
                            .map((kind) => {
                                const style = EVENT_STYLES[kind];
                                const gain = kind === 'sell' && typeof dataPoint['_evt_sell_gain'] === 'number'
                                    ? (dataPoint['_evt_sell_gain'] as number)
                                    : null;
                                const gainPct = kind === 'sell' && typeof dataPoint['_evt_sell_gain_pct'] === 'number'
                                    ? (dataPoint['_evt_sell_gain_pct'] as number)
                                    : null;
                                return (
                                    <div key={kind} className="flex items-center gap-2 pt-1.5 mt-1.5 border-t border-border/40">
                                        <span
                                            className="inline-flex items-center justify-center w-4 h-4 rounded-full text-[9px] font-bold text-white shrink-0"
                                            style={{ backgroundColor: style.color }}
                                        >
                                            {style.letter}
                                        </span>
                                        <span className="text-xs font-medium text-foreground">{dataPoint[`_evt_${kind}_label`]}</span>
                                        {gain != null && (
                                            <span className={`text-xs font-semibold ml-auto shrink-0 ${gain >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}>
                                                {gain >= 0 ? '+' : '−'}{formatCurrency(Math.abs(gain), currency)}
                                                {gainPct != null && ` (${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(2)}%)`}
                                            </span>
                                        )}
                                    </div>
                                );
                            })}

                        {/* Benchmarks (return view) */}
                        {view === 'return' && BENCHMARKS
                            .filter((b) => selectedBenchmarks.includes(b.name) && typeof dataPoint[b.key] === 'number')
                            .map((b) => (
                                <div key={b.key} className="flex items-center justify-between gap-4">
                                    <span className="text-xs font-bold uppercase" style={{ color: b.color }}>{b.name}</span>
                                    <span className={`text-sm font-medium ${dataPoint[b.key] >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400"}`}>
                                        {dataPoint[b.key].toFixed(2)}%
                                    </span>
                                </div>
                            ))}

                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div ref={containerRef} className="glass-card rounded-xl p-4 shadow-sm border mb-6 overflow-visible">
            {/* Header Layout (Matches PerformanceGraph) */}
            <div className="mb-6">
                <div className="flex flex-col items-start gap-4 md:flex-row md:justify-between md:items-center md:gap-0 mb-4">
                    {/* Price and Stats (Top Left) */}
                    {stats ? (
                        <div className="flex items-baseline gap-3">
                            {!hidePrice && (
                                <span className="text-3xl font-bold tracking-tight text-foreground">
                                    {formatCurrency(stats.currentPrice, currency)}
                                </span>
                            )}
                            <span className={`text-base font-medium ${stats.change >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-500'}`}>
                                {formatCurrency(stats.change, currency)} ({stats.changePct.toFixed(2)}%)
                            </span>
                        </div>
                    ) : (
                        <div className="h-9" />
                    )}

                    {/* Toggles (Top Right) */}
                    <div className="flex items-center gap-3 w-full md:w-auto justify-between md:justify-end">
                        <div className="flex items-center gap-2">
                            {view === 'price' && (
                                <div className="flex bg-secondary rounded-lg p-1 border border-border shrink-0 gap-1">
                                    <button
                                        onClick={() => setShowSMA50(!showSMA50)}
                                        className={`px-2 py-1 text-[10px] font-bold rounded-md transition-all ${showSMA50
                                            ? 'bg-orange-500 text-white shadow-sm'
                                            : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                            }`}
                                    >
                                        MA50
                                    </button>
                                    <button
                                        onClick={() => setShowSMA200(!showSMA200)}
                                        className={`px-2 py-1 text-[10px] font-bold rounded-md transition-all ${showSMA200
                                            ? 'bg-purple-600 text-white shadow-sm'
                                            : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                            }`}
                                    >
                                        MA200
                                    </button>
                                </div>
                            )}

                            <div className="flex bg-secondary rounded-lg p-1 border border-border shrink-0">
                                <button
                                    onClick={() => setView('price')}
                                    className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all ${view === 'price'
                                        ? 'bg-[#0097b2] text-white shadow-sm'
                                        : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                        }`}
                                >
                                    Price
                                </button>
                                <button
                                    onClick={() => setView('return')}
                                    className={`px-2 sm:px-3 py-1 text-xs sm:text-sm font-medium rounded-md transition-all ${view === 'return'
                                        ? 'bg-[#0097b2] text-white shadow-sm'
                                        : 'text-muted-foreground hover:text-foreground hover:bg-accent/10'
                                        }`}
                                >
                                    Return %
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Second Row: Period Selector */}
                <div className="w-full overflow-x-auto no-scrollbar pb-1 -mx-1 px-1">
                    <PeriodSelector selectedPeriod={period} onPeriodChange={setPeriod} />
                </div>

                {/* Third Row: Event Overlays (price view only) */}
                {view === 'price' && (
                    <div className="flex items-center gap-2 mt-3 flex-wrap">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mr-0.5">Overlays</span>
                        {([
                            { key: 'buy', label: 'Buys', active: showBuys, toggle: () => setShowBuys(v => !v) },
                            { key: 'sell', label: 'Sells', active: showSells, toggle: () => setShowSells(v => !v) },
                            { key: 'dividend', label: 'Dividends', active: showDividends, toggle: () => setShowDividends(v => !v) },
                            { key: 'earnings', label: 'Earnings', active: showEarnings, toggle: () => setShowEarnings(v => !v) },
                        ] as const).map(({ key, label, active, toggle }) => {
                            const color = EVENT_STYLES[key as EventKind].color;
                            return (
                                <button
                                    key={key}
                                    onClick={toggle}
                                    className={`flex items-center gap-1.5 px-2.5 py-1 text-[10px] font-bold rounded-full border transition-all ${active
                                        ? 'text-white shadow-sm border-transparent'
                                        : 'text-muted-foreground border-border bg-secondary hover:text-foreground hover:bg-accent/10'
                                        }`}
                                    style={active ? { backgroundColor: color } : undefined}
                                >
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: active ? '#fff' : color }} />
                                    {label}
                                </button>
                            );
                        })}
                    </div>
                )}

                {/* Third Row: Benchmark comparison (return view only) */}
                {view === 'return' && (
                    <div className="flex items-center gap-2 mt-3 flex-wrap">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mr-0.5">Benchmarks</span>
                        {BENCHMARKS.map((b) => {
                            const active = selectedBenchmarks.includes(b.name);
                            return (
                                <button
                                    key={b.key}
                                    onClick={() => toggleBenchmark(b.name)}
                                    className={`flex items-center gap-1.5 px-2.5 py-1 text-[10px] font-bold rounded-full border transition-all ${active
                                        ? 'text-white shadow-sm border-transparent'
                                        : 'text-muted-foreground border-border bg-secondary hover:text-foreground hover:bg-accent/10'
                                        }`}
                                    style={active ? { backgroundColor: b.color } : undefined}
                                >
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: active ? '#fff' : b.color }} />
                                    {b.name}
                                </button>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Chart Container */}
            <div className="h-[400px] w-full relative overflow-visible pb-4">
                {isLoading && (
                    <div className="absolute inset-0 bg-white/50 dark:bg-gray-800/50 flex items-center justify-center z-10 rounded-xl">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    </div>
                )}

                {chartedData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartDataWithEvents} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                            <defs>
                                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="splitColorFill" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset={gradientOffset} stopColor="#10b981" stopOpacity={0.15} />
                                    <stop offset={gradientOffset} stopColor="#ef4444" stopOpacity={0.15} />
                                </linearGradient>
                                <linearGradient id="splitColorStroke" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset={gradientOffset} stopColor="#10b981" stopOpacity={1} />
                                    <stop offset={gradientOffset} stopColor="#ef4444" stopOpacity={1} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.5} />
                            <XAxis
                                dataKey="timestamp"
                                domain={isContinuous ? xDomain : undefined}
                                type={isContinuous ? "number" : "category"}
                                scale={isContinuous ? "time" : undefined}
                                tickFormatter={formatXAxis}
                                tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                                axisLine={false}
                                tickLine={false}
                                minTickGap={40}
                            />
                            <YAxis
                                yAxisId="main"
                                tickFormatter={(val) => view === 'return' ? `${val.toFixed(1)}%` : new Intl.NumberFormat('en-US', { notation: "compact", maximumFractionDigits: 2 }).format(val)}
                                domain={['auto', 'auto']}
                                tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                                axisLine={false}
                                tickLine={false}
                                width={50}
                            />
                            <YAxis
                                yAxisId="vol"
                                orientation="right"
                                tickFormatter={() => ""}
                                axisLine={false}
                                tickLine={false}
                                width={0}
                                domain={[0, (dataMax: number) => dataMax * 5]} // Volume bars low
                            />

                            <Tooltip
                                wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                contentStyle={{ backgroundColor: 'transparent', border: 'none' }}
                                content={<CustomTooltip view={view} currency={currency} />}
                                cursor={{ stroke: 'var(--border)', strokeWidth: 1, strokeDasharray: '4 4' }}
                            />

                            <Bar dataKey="volume" yAxisId="vol" fill="#9ca3af" opacity={0.15} barSize={period === '1d' ? undefined : 6} />

                            {view === 'price' ? (
                                <>
                                    <Area
                                        yAxisId="main"
                                        type="monotone"
                                        dataKey="value"
                                        stroke="#2563eb"
                                        strokeWidth={2}
                                        fillOpacity={1}
                                        fill="url(#colorPrice)"
                                        activeDot={{ r: 4, strokeWidth: 0 }}
                                    />
                                    {showSMA50 && (
                                        <Line
                                            yAxisId="main"
                                            type="monotone"
                                            dataKey="sma50"
                                            stroke="#f97316" // Orange
                                            strokeWidth={1.5}
                                            dot={false}
                                            activeDot={{ r: 4 }}
                                            connectNulls
                                        />
                                    )}
                                    {showSMA200 && (
                                        <Line
                                            yAxisId="main"
                                            type="monotone"
                                            dataKey="sma200"
                                            stroke="#9333ea" // Purple
                                            strokeWidth={1.5}
                                            dot={false}
                                            activeDot={{ r: 4 }}
                                            connectNulls
                                        />
                                    )}
                                    {avgCost && avgCost > 0 && (
                                        <ReferenceLine
                                            yAxisId="main"
                                            y={avgCost}
                                            stroke="#64748b" // Slate 500
                                            strokeDasharray="5 5"
                                            strokeWidth={1.5}
                                            label={{
                                                value: `AVG COST: ${formatCurrency(avgCost, currency)}`,
                                                position: 'right',
                                                fill: '#64748b',
                                                fontSize: 10,
                                                fontWeight: 'bold',
                                                offset: 10
                                            }}
                                        />
                                    )}
                                    {(['buy', 'sell', 'dividend', 'earnings'] as EventKind[]).map((kind) => {
                                        const enabled =
                                            (kind === 'buy' && showBuys) ||
                                            (kind === 'sell' && showSells) ||
                                            (kind === 'dividend' && showDividends) ||
                                            (kind === 'earnings' && showEarnings);
                                        if (!enabled || !presentKinds[kind]) return null;
                                        return (
                                            <Line
                                                key={`evt-${kind}`}
                                                yAxisId="main"
                                                type="monotone"
                                                dataKey={`_evt_${kind}`}
                                                stroke="transparent"
                                                strokeWidth={0}
                                                isAnimationActive={false}
                                                legendType="none"
                                                connectNulls={false}
                                                activeDot={false}
                                                dot={(dotProps: { cx?: number; cy?: number; payload?: Record<string, unknown> }) => (
                                                    <EventDot cx={dotProps.cx} cy={dotProps.cy} payload={dotProps.payload} kind={kind} />
                                                )}
                                            />
                                        );
                                    })}
                                </>
                            ) : (
                                <>
                                    <Area
                                        yAxisId="main"
                                        type="monotone"
                                        dataKey="return_pct"
                                        stroke="url(#splitColorStroke)"
                                        fill="url(#splitColorFill)"
                                        strokeWidth={2}
                                        activeDot={{ r: 4, strokeWidth: 0 }}
                                    />
                                    {BENCHMARKS.filter((b) => selectedBenchmarks.includes(b.name)).map((b) => (
                                        <Line
                                            key={b.key}
                                            yAxisId="main"
                                            type="monotone"
                                            dataKey={b.key}
                                            stroke={b.color}
                                            strokeWidth={1.5}
                                            dot={false}
                                            activeDot={{ r: 4 }}
                                            connectNulls
                                            isAnimationActive={false}
                                        />
                                    ))}
                                </>
                            )}



                            {view === 'return' && <ReferenceLine y={0} yAxisId="main" stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />}
                        </ComposedChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="flex items-center justify-center h-full text-muted-foreground w-full">
                        {!isLoading ? "No data available." : ""}
                    </div>
                )}
            </div>
        </div >
    );
}
