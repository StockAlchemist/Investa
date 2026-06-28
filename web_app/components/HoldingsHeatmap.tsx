'use client';
import React, { useMemo, useState } from 'react';
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { Grid2x2 } from 'lucide-react';
import { Holding, fetchHoldingReturns, HoldingReturnPeriod, HoldingReturns } from '../lib/api';
import { formatCompactNumber, cn } from '../lib/utils';
import { useStockModal } from '@/context/StockModalContext';

interface HoldingsHeatmapProps {
    holdings: Holding[];
    currency: string;
}

// --- Metrics the heatmap can color by ----------------------------------------
// `clamp` is the ±% bound at which the color reaches full saturation; values
// beyond it are clipped (so one outlier doesn't wash out the rest).
type MetricSource = 'holding' | 'spark' | 'returns';
interface MetricDef {
    key: string;
    label: string;
    source: MetricSource;
    field?: string;              // holding column (source: 'holding')
    period?: HoldingReturnPeriod; // returns endpoint key (source: 'returns')
    clamp: number;
}

const METRICS: MetricDef[] = [
    { key: 'day', label: '1D', source: 'holding', field: 'Day Change %', clamp: 3 },
    { key: '7d', label: '7D', source: 'spark', clamp: 5 },
    { key: '1m', label: '1M', source: 'returns', period: '1m', clamp: 8 },
    { key: '3m', label: '3M', source: 'returns', period: '3m', clamp: 15 },
    { key: '6m', label: '6M', source: 'returns', period: '6m', clamp: 25 },
    { key: 'ytd', label: 'YTD', source: 'returns', period: 'ytd', clamp: 25 },
    { key: '1y', label: '1Y', source: 'returns', period: '1y', clamp: 40 },
    { key: 'unreal', label: 'Unreal.', source: 'holding', field: 'Unreal. Gain %', clamp: 40 },
    { key: 'total', label: 'Total', source: 'holding', field: 'Total Return %', clamp: 50 },
    { key: 'irr', label: 'IRR', source: 'holding', field: 'IRR (%)', clamp: 40 },
];

type GroupKey = 'Sector' | 'Account' | 'Country' | 'None';
const GROUPS: { key: GroupKey; label: string }[] = [
    { key: 'Sector', label: 'Sector' },
    { key: 'Account', label: 'Account' },
    { key: 'Country', label: 'Country' },
    { key: 'None', label: 'Flat' },
];

type SizeMode = 'value' | 'equal';

function isUnknown(v: unknown): boolean {
    if (v == null) return true;
    const s = String(v).trim().toUpperCase();
    return s === '' || s === '-' || s === 'NONE' || s === 'NULL' || s === 'UNKNOWN'
        || s.startsWith('N/A') || s.startsWith('UNKNOWN');
}

function groupOf(h: Holding, g: GroupKey): string {
    if (g === 'None') return 'All';
    let raw: unknown;
    if (g === 'Country') raw = (h['geography'] as string) || (h['Country'] as string);
    else raw = h[g];
    return isUnknown(raw) ? 'Unknown' : (raw as string);
}

// Diverging red→neutral→green color, centered at 0% and saturating at ±clamp.
// Bright, vivid tiles (Finviz-style) — extremes are a bright red / bright green;
// near 0% it eases toward a neutral grey.
function heatColor(v: number | null | undefined, clamp: number): string {
    if (v == null || !isFinite(v)) return 'hsl(220 6% 42%)'; // unknown → neutral grey
    const t = Math.max(-1, Math.min(1, v / clamp));
    const mag = Math.abs(t);
    const hue = t >= 0 ? 145 : 2;            // green up / red down
    const sat = 50 + 45 * mag;               // grey-ish near 0 → fully vivid at extremes
    const light = 40 + 8 * mag;              // brighter as magnitude grows
    return `hsl(${hue} ${sat.toFixed(0)}% ${light.toFixed(0)}%)`;
}

function metricValue(h: Holding, m: MetricDef, returns: HoldingReturns): number | null {
    if (m.source === 'holding') {
        const v = h[m.field as string];
        return typeof v === 'number' && isFinite(v) ? v : null;
    }
    if (m.source === 'spark') {
        const arr = h.sparkline_7d;
        if (Array.isArray(arr) && arr.length >= 2 && arr[0]) {
            return (arr[arr.length - 1] / arr[0] - 1) * 100;
        }
        return null;
    }
    // returns endpoint
    const r = returns[h.Symbol]?.[m.period as HoldingReturnPeriod];
    return typeof r === 'number' && isFinite(r) ? r : null;
}

interface Leaf {
    name: string;
    size: number;
    fill: string;
    metricVal: number | null;
    value: number;
    group: string;
    [key: string]: unknown;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function HeatCell(props: any) {
    const { x, y, width, height, name, fill, metricVal, depth } = props;
    if (depth === 0 || width <= 0 || height <= 0) return null;
    const hasVal = typeof metricVal === 'number' && isFinite(metricVal);
    const showLabel = width > 40 && height > 22 && !!name;
    const showPct = width > 52 && height > 36 && hasVal;
    return (
        <g>
            <rect
                x={x}
                y={y}
                width={width}
                height={height}
                rx={2}
                style={{ fill: fill || 'hsl(220 8% 26%)', stroke: 'var(--background)', strokeWidth: 1.5, cursor: 'pointer' }}
            />
            {showLabel && (
                <text x={x + width / 2} y={y + height / 2 - (showPct ? 6 : 0)} textAnchor="middle" fill="#fff" fontSize={11} fontWeight={800} className="pointer-events-none">
                    {name}
                </text>
            )}
            {showPct && (
                <text x={x + width / 2} y={y + height / 2 + 9} textAnchor="middle" fill="rgba(255,255,255,0.92)" fontSize={10} fontWeight={600} className="pointer-events-none tabular-nums">
                    {(metricVal as number) >= 0 ? '+' : ''}{(metricVal as number).toFixed(1)}%
                </text>
            )}
        </g>
    );
}

function HeatTooltip({ active, payload, currency, metricLabel, showGroup }: { active?: boolean; payload?: { payload: Leaf }[]; currency: string; metricLabel: string; showGroup: boolean }) {
    if (!active || !payload || !payload.length) return null;
    const d = payload[0].payload;
    const up = d.metricVal != null && d.metricVal >= 0;
    // Solid (not translucent) so the bright tile beneath never bleeds through and
    // washes out the text.
    return (
        <div
            className="text-popover-foreground px-3 py-2 rounded-xl border border-border shadow-2xl"
            style={{ backgroundColor: 'var(--menu-solid)' }}
        >
            <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: d.fill }} />
                <span className="font-bold text-sm">{d.name}</span>
                {showGroup && <span className="text-xs text-muted-foreground">{d.group}</span>}
            </div>
            <div className="mt-1 flex items-center gap-1.5 text-xs tabular-nums whitespace-nowrap">
                <span className="font-medium">{formatCompactNumber(d.value, currency)}</span>
                <span className="text-muted-foreground/50">·</span>
                <span className="text-muted-foreground">{metricLabel}</span>
                <span className={cn('font-bold', d.metricVal == null ? 'text-muted-foreground' : up ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400')}>
                    {d.metricVal == null ? 'n/a' : `${up ? '+' : ''}${d.metricVal.toFixed(2)}%`}
                </span>
            </div>
        </div>
    );
}

export default function HoldingsHeatmap({ holdings, currency }: HoldingsHeatmapProps) {
    const [metricKey, setMetricKey] = useState('day');
    const [group, setGroup] = useState<GroupKey>('Sector');
    const [sizeMode, setSizeMode] = useState<SizeMode>('value');
    const { openStockDetail } = useStockModal();

    const metric = METRICS.find(m => m.key === metricKey) || METRICS[0];

    const symbols = useMemo(
        () => Array.from(new Set(holdings.map(h => h.Symbol).filter(Boolean))).sort(),
        [holdings],
    );

    // Period returns power the 1M/3M/6M/1Y/YTD metrics. Fetched eagerly once
    // holdings load so switching metrics is instant.
    const { data: returns = {} } = useQuery({
        queryKey: ['holdingReturns', symbols],
        queryFn: ({ signal }) => fetchHoldingReturns(symbols, signal),
        enabled: symbols.length > 0,
        staleTime: 5 * 60 * 1000,
    });

    const mvKey = `Market Value (${currency})`;

    // Build per-group leaf lists (one symbol per leaf, aggregated across accounts
    // unless grouping by Account).
    const groups = useMemo(() => {
        const byGroup = new Map<string, Map<string, Leaf>>();
        for (const h of holdings) {
            const value = Math.max(0, (h[mvKey] as number) || 0);
            if (value <= 0) continue;
            const g = groupOf(h, group);
            // When grouping by account, the same symbol can appear in two accounts
            // as distinct leaves; otherwise aggregate by symbol within the group.
            const leafKey = h.Symbol;
            let leaves = byGroup.get(g);
            if (!leaves) { leaves = new Map(); byGroup.set(g, leaves); }
            const existing = leaves.get(leafKey);
            if (existing) {
                existing.value += value;
            } else {
                const mv = metricValue(h, metric, returns);
                leaves.set(leafKey, {
                    name: h.Symbol,
                    size: 0, // filled below per size mode
                    value,
                    metricVal: mv,
                    fill: heatColor(mv, metric.clamp),
                    group: g,
                });
            }
        }

        const result = Array.from(byGroup.entries()).map(([name, leafMap]) => {
            const leaves = Array.from(leafMap.values());
            leaves.forEach(l => { l.size = sizeMode === 'equal' ? 1 : l.value; });
            leaves.sort((a, b) => b.size - a.size);
            const totalValue = leaves.reduce((s, l) => s + l.value, 0);
            const weight = sizeMode === 'equal' ? leaves.length : totalValue;
            return { name, leaves, totalValue, weight };
        });
        result.sort((a, b) => b.weight - a.weight);
        return result;
    }, [holdings, group, metric, returns, sizeMode, mvKey]);

    const totalWeight = useMemo(() => groups.reduce((s, g) => s + g.weight, 0), [groups]);
    const totalValue = useMemo(() => groups.reduce((s, g) => s + g.totalValue, 0), [groups]);
    const holdingCount = useMemo(() => groups.reduce((s, g) => s + g.leaves.length, 0), [groups]);

    const onLeafClick = (node: unknown) => {
        const n = node as { name?: string } | undefined;
        if (n?.name) openStockDetail(n.name, currency);
    };

    const renderTreemap = (leaves: Leaf[], height: number | string) => (
        <div style={{ height }}>
            <ResponsiveContainer width="100%" height="100%">
                <Treemap
                    data={leaves}
                    dataKey="size"
                    stroke="var(--background)"
                    content={<HeatCell />}
                    isAnimationActive={false}
                    onClick={onLeafClick}
                >
                    <Tooltip
                        contentStyle={{ backgroundColor: 'transparent', border: 'none', boxShadow: 'none' }}
                        content={(p) => <HeatTooltip {...(p as object)} currency={currency} metricLabel={metric.label} showGroup={group !== 'None'} />}
                    />
                </Treemap>
            </ResponsiveContainer>
        </div>
    );

    // Legend gradient: red(-clamp) → neutral(0) → green(+clamp).
    const legendStops = [-1, -0.5, 0, 0.5, 1]
        .map(t => heatColor(t * metric.clamp, metric.clamp))
        .join(', ');

    return (
        <div className="metric-card p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-2">
                    <Grid2x2 className="w-3.5 h-3.5 text-indigo-500" />
                    <h3 className="section-label">Performance Heatmap</h3>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                    {/* Metric (period) selector */}
                    <select
                        value={metricKey}
                        onChange={e => setMetricKey(e.target.value)}
                        className="px-2.5 py-1 rounded-lg bg-secondary text-xs font-semibold text-foreground border border-border/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/40"
                        title="Color by"
                    >
                        {METRICS.map(m => (
                            <option key={m.key} value={m.key}>{m.label}</option>
                        ))}
                    </select>
                    {/* Grouping */}
                    <div className="inline-flex rounded-lg bg-secondary p-0.5">
                        {GROUPS.map(g => (
                            <button
                                key={g.key}
                                onClick={() => setGroup(g.key)}
                                className={cn(
                                    'px-2 py-1 rounded-md text-xs font-semibold transition-all',
                                    group === g.key ? 'bg-[#0097b2] text-white' : 'text-muted-foreground hover:text-foreground',
                                )}
                            >
                                {g.label}
                            </button>
                        ))}
                    </div>
                    {/* Sizing */}
                    <div className="inline-flex rounded-lg bg-secondary p-0.5">
                        {(['value', 'equal'] as SizeMode[]).map(s => (
                            <button
                                key={s}
                                onClick={() => setSizeMode(s)}
                                className={cn(
                                    'px-2 py-1 rounded-md text-xs font-semibold transition-all capitalize',
                                    sizeMode === s ? 'bg-indigo-500 text-white' : 'text-muted-foreground hover:text-foreground',
                                )}
                                title={s === 'value' ? 'Size tiles by position value' : 'Equal-size tiles'}
                            >
                                {s === 'value' ? 'Value' : 'Equal'}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {holdingCount === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-12">No holdings to map.</p>
            ) : group === 'None' ? (
                // Match PortfolioTreemap's height (h-80 = 320px) so the squarified
                // layout — which depends on the area's aspect ratio — is identical.
                renderTreemap(groups[0]?.leaves ?? [], 320)
            ) : (
                <div className="flex flex-col gap-1.5" style={{ height: 560 }}>
                    {groups.map(g => {
                        const frac = totalWeight > 0 ? g.weight / totalWeight : 0;
                        return (
                            <div
                                key={g.name}
                                className="relative rounded-lg overflow-hidden"
                                style={{ flexGrow: g.weight, flexBasis: 0, minHeight: 56 }}
                            >
                                {renderTreemap(g.leaves, '100%')}
                                <div className="absolute top-1 left-1.5 px-1.5 py-0.5 rounded bg-black/35 backdrop-blur-sm pointer-events-none">
                                    <span className="text-[10px] font-bold uppercase tracking-wide text-white/95">{g.name}</span>
                                    <span className="text-[10px] text-white/60 ml-1.5 tabular-nums">{(frac * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Legend + footer */}
            <div className="flex flex-wrap items-center justify-between gap-3 mt-3">
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-muted-foreground tabular-nums">−{metric.clamp}%</span>
                    <div className="h-2.5 w-40 rounded-full" style={{ background: `linear-gradient(to right, ${legendStops})` }} />
                    <span className="text-[10px] text-muted-foreground tabular-nums">+{metric.clamp}%</span>
                </div>
                <p className="text-[10px] text-muted-foreground/60 tabular-nums">
                    {holdingCount} holdings · {formatCompactNumber(totalValue, currency)} · click a tile for detail
                </p>
            </div>
        </div>
    );
}
