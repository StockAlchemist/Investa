'use client';
import React, { useMemo, useState } from 'react';
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';
import { LayoutGrid } from 'lucide-react';
import { Holding } from '../../lib/api';
import { formatCompactNumber, cn } from '../../lib/utils';
import { useStockModal } from '@/context/StockModalContext';

interface PortfolioTreemapProps {
    holdings: Holding[];
    currency: string;
}

type Dimension = 'Sector' | 'quoteType' | 'Country';

const DIMENSIONS: { key: Dimension; label: string }[] = [
    { key: 'Sector', label: 'Sector' },
    { key: 'quoteType', label: 'Asset Type' },
    { key: 'Country', label: 'Country' },
];

const PALETTE = [
    '#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ef4444',
    '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#84cc16',
    '#3b82f6', '#a855f7',
];

function isUnknown(v: unknown): boolean {
    if (v == null) return true;
    const s = String(v).trim().toUpperCase();
    return s === '' || s === '-' || s === 'NONE' || s === 'NULL' || s === 'UNKNOWN'
        || s.startsWith('N/A') || s.startsWith('UNKNOWN');
}

function groupOf(h: Holding, dim: Dimension): string {
    let raw: unknown;
    if (dim === 'Country') raw = (h['geography'] as string) || (h['Country'] as string);
    else raw = h[dim];
    return isUnknown(raw) ? 'Unknown' : (raw as string);
}

interface TreemapDatum {
    name: string;
    size: number;
    group: string;
    fill: string;
    pct: number;
    [key: string]: unknown;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function TreemapCell(props: any) {
    const { x, y, width, height, name, fill, pct, depth } = props;
    // Recharts renders a depth-0 root container node that lacks our leaf fields —
    // skip it so we only paint the holding tiles.
    if (depth === 0 || width <= 0 || height <= 0) return null;
    const hasPct = typeof pct === 'number' && isFinite(pct);
    const showLabel = width > 44 && height > 24 && !!name;
    const showPct = width > 60 && height > 38 && hasPct;
    return (
        <g>
            <rect
                x={x}
                y={y}
                width={width}
                height={height}
                rx={3}
                style={{ fill: fill || '#6366f1', stroke: 'var(--background)', strokeWidth: 2, cursor: 'pointer' }}
            />
            {showLabel && (
                <text x={x + 6} y={y + 16} fill="#fff" fontSize={11} fontWeight={700} className="pointer-events-none">
                    {name}
                </text>
            )}
            {showPct && (
                <text x={x + 6} y={y + 30} fill="rgba(255,255,255,0.8)" fontSize={10} className="pointer-events-none tabular-nums">
                    {pct.toFixed(1)}%
                </text>
            )}
        </g>
    );
}

export default function PortfolioTreemap({ holdings, currency }: PortfolioTreemapProps) {
    const [dim, setDim] = useState<Dimension>('Sector');
    const { openStockDetail } = useStockModal();

        const out: TreemapDatum[] = useMemo(() => {
            const mvKey = `Market Value (${currency})`;
            // Assign a stable color per group.
            const groupColors = new Map<string, string>();
            const orderedGroups: string[] = [];

            const rows = holdings
                .map(h => ({ h, value: Math.max(0, (h[mvKey] as number) || 0) }))
                .filter(r => r.value > 0);
            const tot = rows.reduce((s, r) => s + r.value, 0);

            // Order groups by total value desc for consistent coloring.
            const groupTotals = new Map<string, number>();
            for (const { h, value } of rows) {
                const g = groupOf(h, dim);
                groupTotals.set(g, (groupTotals.get(g) || 0) + value);
            }
            Array.from(groupTotals.entries())
                .sort((a, b) => b[1] - a[1])
                .forEach(([g], i) => {
                    groupColors.set(g, PALETTE[i % PALETTE.length]);
                    orderedGroups.push(g);
                });

            // Aggregate rows by symbol to prevent duplicate keys in Treemap
            const symbolMap = new Map<string, { h: Holding; value: number }>();
            for (const { h, value } of rows) {
                const existing = symbolMap.get(h.Symbol);
                if (existing) {
                    existing.value += value;
                } else {
                    symbolMap.set(h.Symbol, { h, value });
                }
            }
            const aggregatedRows = Array.from(symbolMap.values());

            const list: TreemapDatum[] = aggregatedRows
                .map(({ h, value }) => {
                    const g = groupOf(h, dim);
                    return {
                        name: h.Symbol,
                        size: value,
                        group: g,
                        fill: groupColors.get(g) || PALETTE[0],
                        pct: tot > 0 ? (value / tot) * 100 : 0,
                    };
                })
                .sort((a, b) => b.size - a.size);

            return list;
        }, [holdings, dim, currency]);

        const total = useMemo(() => {
            const mvKey = `Market Value (${currency})`;
            return holdings
                .map(h => Math.max(0, (h[mvKey] as number) || 0))
                .reduce((s, v) => s + v, 0);
        }, [holdings, currency]);

        const data = out;

    return (
        <div className="metric-card p-5">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <LayoutGrid className="w-3.5 h-3.5 text-indigo-500" />
                    <h3 className="section-label">Treemap</h3>
                </div>
                <div className="inline-flex rounded-lg bg-secondary p-0.5">
                    {DIMENSIONS.map(d => (
                        <button
                            key={d.key}
                            onClick={() => setDim(d.key)}
                            className={cn(
                                'px-2.5 py-1 rounded-md text-xs font-semibold transition-all',
                                dim === d.key ? 'bg-[#0097b2] text-white' : 'text-muted-foreground hover:text-foreground',
                            )}
                        >
                            {d.label}
                        </button>
                    ))}
                </div>
            </div>

            {data.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-12">No holdings to map.</p>
            ) : (
                <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        <Treemap
                            data={data}
                            dataKey="size"
                            stroke="var(--background)"
                            content={<TreemapCell />}
                            isAnimationActive={false}
                            onClick={(node: unknown) => {
                                const n = node as { name?: string } | undefined;
                                if (n?.name) openStockDetail(n.name, currency);
                            }}
                        >
                            <Tooltip
                                contentStyle={{ backgroundColor: 'transparent', border: 'none', boxShadow: 'none' }}
                                content={({ active, payload }) => {
                                    if (!active || !payload || !payload.length) return null;
                                    const d = payload[0].payload as TreemapDatum;
                                    return (
                                        <div className="bg-background/98 backdrop-blur-2xl p-3 rounded-xl border border-border/60 shadow-2xl">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="w-2 h-2 rounded-sm" style={{ backgroundColor: d.fill }} />
                                                <span className="font-bold text-foreground text-sm">{d.name}</span>
                                            </div>
                                            <div className="text-xs text-muted-foreground">{d.group}</div>
                                            <div className="text-xs tabular-nums mt-1">
                                                <span className="text-foreground font-medium">{formatCompactNumber(d.size, currency)}</span>
                                                <span className="text-muted-foreground/60 ml-1">· {d.pct.toFixed(1)}%</span>
                                            </div>
                                        </div>
                                    );
                                }}
                            />
                        </Treemap>
                    </ResponsiveContainer>
                </div>
            )}
            <p className="text-[10px] text-muted-foreground/60 mt-2 tabular-nums">
                {data.length} holdings · {formatCompactNumber(total, currency)} · click a tile for detail
            </p>
        </div>
    );
}
