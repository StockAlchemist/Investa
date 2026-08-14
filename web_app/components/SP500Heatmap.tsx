'use client';
import React, { useMemo, useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTheme } from 'next-themes';
import { Grid2x2, ChevronDown } from 'lucide-react';
import { fetchSP500Heatmap, type SP500HeatmapItem } from '../lib/api';
import { cn } from '../lib/utils';
import { useStockModal } from '@/context/StockModalContext';

// Metric catalogue, colour scale and value formatting live in lib/metrics.ts —
// the stock detail window reads the same definitions, and a reading must not
// change meaning between a tile and a row. Re-exported here so the map's own
// tests and callers keep one import.
export {
    METRICS,
    HEAT_PALETTES,
    getScaledValue,
    signedDeviation,
    metricDomain,
    metricTone,
    heatColor,
    heatColorScaled,
    tileLuminance,
    tileTextColor,
    formatCompactCap,
    formatMetric,
    formatScaled,
} from '../lib/metrics';
import {
    METRICS,
    HEAT_PALETTES,
    metricDomain,
    metricTone,
    heatColor,
    heatColorScaled,
    tileTextColor,
    formatCompactCap,
    formatMetric,
    formatScaled,
    type MetricGroup,
    type HeatTheme,
} from '../lib/metrics';
export type { MetricDef, HeatPalette, HeatTheme } from '../lib/metrics';

const METRIC_GROUPS: MetricGroup[] = ['Performance', 'Valuation', 'Earnings & Sales', 'Profitability', 'Market'];

type SizeMode = 'cap' | 'equal';


// ---------------------------------------------------------------------------
// Squarified Treemap Layout (Hierarchical)
// ---------------------------------------------------------------------------
interface TreemapNode {
    id: string;
    name: string;
    type: number; // 0=Root, 1=Sector, 2=Industry, 3=Company
    value: number;
    metricVal?: number | null;
    color?: string;
    /** Label ink for this tile, resolved against its own fill. */
    textColor?: string;
    item?: SP500HeatmapItem;
    children: TreemapNode[];
    frame: { x: number, y: number, width: number, height: number };
    /** False when the box was too short to afford a header, so the label is
     *  suppressed and the space goes to the constituents instead. */
    showHeader?: boolean;
}

const HEADER_HEIGHT: Record<number, number> = { 1: 22, 2: 16 };
/** Below this width the sub-industry tier costs more space than it conveys. */
const FLAT_LAYOUT_WIDTH = 640;
/** Fixed so the tooltip can be flipped away from the clipped map edges. */
const TOOLTIP_WIDTH = 240;
const TOOLTIP_HEIGHT = 78;

function squarifyFlat(nodes: TreemapNode[], rect: { x: number, y: number, width: number, height: number }) {
    if (nodes.length === 0 || rect.width <= 0 || rect.height <= 0) return;
    const total = nodes.reduce((s, n) => s + n.value, 0);
    if (total <= 0) return;

    let remaining = nodes.map((_, i) => i);
    let area = { ...rect };

    const layoutStrip = (indices: number[], vertical: boolean) => {
        const stripTotal = indices.reduce((s, i) => s + nodes[i].value, 0);
        const remainingTotal = remaining.reduce((s, i) => s + nodes[i].value, 0);
        const stripSize = vertical 
            ? area.width * (stripTotal / remainingTotal)
            : area.height * (stripTotal / remainingTotal);

        let offset = 0;
        for (const idx of indices) {
            const frac = nodes[idx].value / stripTotal;
            const length = vertical ? area.height : area.width;
            const span = length * frac;

            if (vertical) {
                nodes[idx].frame = { x: area.x, y: area.y + offset, width: stripSize, height: span };
            } else {
                nodes[idx].frame = { x: area.x + offset, y: area.y, width: span, height: stripSize };
            }
            offset += span;
        }

        if (vertical) {
            area = { x: area.x + stripSize, y: area.y, width: area.width - stripSize, height: area.height };
        } else {
            area = { x: area.x, y: area.y + stripSize, width: area.width, height: area.height - stripSize };
        }
    };

    let strip: number[] = [];
    const sortedIndices = [...remaining].sort((a, b) => nodes[b].value - nodes[a].value);

    const worstAspect = (indices: number[], vertical: boolean) => {
        const stripTotal = indices.reduce((s, i) => s + nodes[i].value, 0);
        const remainingTotal = remaining.reduce((s, i) => s + nodes[i].value, 0);
        if (remainingTotal <= 0 || stripTotal <= 0) return Infinity;
        const sideLen = vertical ? area.width * (stripTotal / remainingTotal) : area.height * (stripTotal / remainingTotal);
        if (sideLen <= 0) return Infinity;
        const length = vertical ? area.height : area.width;
        let worst = 0;
        for (const idx of indices) {
            const span = length * (nodes[idx].value / stripTotal);
            if (span > 0) {
                const aspect = Math.max(sideLen / span, span / sideLen);
                worst = Math.max(worst, aspect);
            } else {
                worst = Infinity;
            }
        }
        return worst;
    };

    for (const idx of sortedIndices) {
        const vertical = area.width > area.height;
        const currentWorst = strip.length === 0 ? Infinity : worstAspect(strip, vertical);
        const candidateWorst = worstAspect([...strip, idx], vertical);

        if (strip.length === 0 || candidateWorst <= currentWorst) {
            strip.push(idx);
        } else {
            layoutStrip(strip, vertical);
            remaining = remaining.filter(r => !strip.includes(r));
            strip = [idx];
        }
    }
    if (strip.length > 0) {
        layoutStrip(strip, area.width > area.height);
    }
}

function squarifyHierarchy(node: TreemapNode, rect: { x: number, y: number, width: number, height: number }) {
    node.frame = { ...rect };
    if (node.children.length === 0) return;

    const headerHeight = HEADER_HEIGHT[node.type] ?? 0;
    const sidePad = node.type === 1 ? 2 : (node.type === 2 ? 1 : 0);

    // A header must leave at least as much room for the constituents as it
    // takes for itself; otherwise it is dropped. Reserving it unconditionally
    // used to zero out every box shorter than the header, silently deleting
    // whole sub-industries from the map (worst on narrow viewports).
    const showHeader = headerHeight > 0 && rect.height >= headerHeight * 2;
    node.showHeader = showHeader;
    const topPad = showHeader ? headerHeight : 0;

    const innerRect = {
        x: rect.x + sidePad,
        y: rect.y + topPad,
        width: rect.width - sidePad * 2,
        height: rect.height - topPad - (showHeader ? sidePad : 0)
    };

    if (innerRect.width <= 0 || innerRect.height <= 0) {
        node.children.forEach(c => c.frame = { x: 0, y: 0, width: 0, height: 0 });
        return;
    }

    squarifyFlat(node.children, innerRect);

    node.children.forEach(child => {
        squarifyHierarchy(child, child.frame);
    });
}

function flattenNodes(node: TreemapNode): TreemapNode[] {
    let result = [node];
    for (const child of node.children) {
        result = result.concat(flattenNodes(child));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function SP500Heatmap() {
    const [metricKey, setMetricKey] = useState('day');
    const [sectorFilter, setSectorFilter] = useState('All');
    const [sizeMode, setSizeMode] = useState<SizeMode>('cap');
    const { openStockDetail } = useStockModal();
    const [containerElement, setContainerElement] = useState<HTMLDivElement | null>(null);
    const [containerSize, setContainerSize] = useState({ width: 0, height: 600 });
    // The server has no theme, so the first paint has to commit to one; staying
    // on the dark palette until mount keeps the markup stable and avoids a
    // hydration mismatch on ~500 inline tile colours.
    const { resolvedTheme } = useTheme();
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);
    const heatTheme: HeatTheme = mounted && resolvedTheme === 'light' ? 'light' : 'dark';
    const palette = HEAT_PALETTES[heatTheme];
    
    // Tooltip state
    const [hoverNode, setHoverNode] = useState<TreemapNode | null>(null);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

    useEffect(() => {
        if (!containerElement) return;
        const obs = new ResizeObserver(entries => {
            if (entries[0]) {
                const { width, height } = entries[0].contentRect;
                if (width > 0 && height > 0) {
                    setContainerSize({ width, height });
                }
            }
        });
        obs.observe(containerElement);
        return () => obs.disconnect();
    }, [containerElement]);

    const metric = METRICS.find(m => m.key === metricKey) || METRICS[0];

    const { data: rawData = [], isLoading } = useQuery({
        queryKey: ['sp500-heatmap'],
        queryFn: ({ signal }) => fetchSP500Heatmap(signal),
        staleTime: 5 * 60 * 1000,
        refetchOnWindowFocus: false,
    });

    const sectors = useMemo(() => {
        const s = new Set(rawData.map(d => d.sector).filter(Boolean));
        return ['All', ...Array.from(s).sort()];
    }, [rawData]);

    // On a phone the sub-industry tier eats more height in headers than it
    // repays in meaning, so the map collapses to sector -> company.
    const flatLayout = containerSize.width > 0 && containerSize.width < FLAT_LAYOUT_WIDTH;

    const rootNode = useMemo(() => {
        const filtered = sectorFilter === 'All' ? rawData : rawData.filter(d => d.sector === sectorFilter);
        const rootChildren: TreemapNode[] = [];
        let idCounter = 0;

        const companyNode = (item: SP500HeatmapItem): TreemapNode | null => {
            const mv = item.market_cap || 0;
            if (mv <= 0 && sizeMode === 'cap') return null;
            const val = typeof item[metric.field] === 'number' ? (item[metric.field] as number) : null;
            const color = heatColor(val, metric.key, heatTheme);
            return {
                id: `c_${idCounter++}`, name: item.symbol, type: 3,
                value: sizeMode === 'equal' ? 1 : mv, metricVal: val,
                color, textColor: tileTextColor(color, heatTheme),
                item, children: [], frame: { x: 0, y: 0, width: 0, height: 0 }
            };
        };

        const groupBy = <T,>(items: T[], key: (t: T) => string) => {
            const m = new Map<string, T[]>();
            for (const it of items) {
                const k = key(it);
                if (!k) continue;
                if (!m.has(k)) m.set(k, []);
                m.get(k)!.push(it);
            }
            return m;
        };

        // Drilled into one sector, the top tier becomes sub-industry.
        const topTier = groupBy(filtered, d => (sectorFilter !== 'All' ? d.sub_industry : d.sector));
        // With a sector selected the leaves already sit one level down, and the
        // narrow layout deliberately skips the sub-industry tier.
        const nestIndustries = sectorFilter === 'All' && !flatLayout;

        for (const [sectorName, sectorItems] of Array.from(topTier.entries())) {
            let sectorChildren: TreemapNode[] = [];

            if (!nestIndustries) {
                sectorChildren = sectorItems.map(companyNode).filter((n): n is TreemapNode => n !== null);
            } else {
                for (const [indName, indItems] of Array.from(groupBy(sectorItems, d => d.sub_industry || 'Other').entries())) {
                    const indChildren = indItems.map(companyNode).filter((n): n is TreemapNode => n !== null);
                    if (indChildren.length > 0) {
                        indChildren.sort((a, b) => b.value - a.value);
                        const weight = indChildren.reduce((s, c) => s + c.value, 0);
                        sectorChildren.push({
                            id: `i_${idCounter++}`, name: indName, type: 2, value: weight,
                            children: indChildren, frame: { x: 0, y: 0, width: 0, height: 0 }
                        });
                    }
                }
            }

            if (sectorChildren.length > 0) {
                sectorChildren.sort((a, b) => b.value - a.value);
                const weight = sectorChildren.reduce((s, c) => s + c.value, 0);
                rootChildren.push({
                    id: `s_${idCounter++}`, name: sectorName, type: 1, value: weight,
                    children: sectorChildren, frame: { x: 0, y: 0, width: 0, height: 0 }
                });
            }
        }

        rootChildren.sort((a, b) => b.value - a.value);
        const rootWeight = rootChildren.reduce((s, c) => s + c.value, 0);
        return {
            id: 'root', name: 'Root', type: 0, value: rootWeight, children: rootChildren,
            frame: { x: 0, y: 0, width: 0, height: 0 }
        };
    }, [rawData, sectorFilter, metric, sizeMode, flatLayout, heatTheme]);

    const layoutNodes = useMemo(() => {
        if (!containerSize.width || !containerSize.height || rootNode.children.length === 0) return [];
        // Clone the shells so the layout pass never writes frames back into the
        // memoized tree; `item` stays shared by reference (it is read-only).
        const clone = (n: TreemapNode): TreemapNode => ({ ...n, children: n.children.map(clone) });
        const root = clone(rootNode);
        squarifyHierarchy(root, { x: 0, y: 0, width: containerSize.width, height: containerSize.height });
        return flattenNodes(root).filter(n => n.type > 0 && n.frame.width > 0 && n.frame.height > 0);
    }, [rootNode, containerSize]);

    // Counted off the laid-out tree, so the footer can never claim more stocks
    // than the map actually draws.
    const shownCount = useMemo(() => layoutNodes.filter(n => n.type === 3).length, [layoutNodes]);

    // The legend spans exactly the metric's own domain, colouring each stop with
    // the value it actually represents — so the gradient and the end labels
    // agree for inverted metrics (low P/E green) and the one-sided ones start at
    // their real floor rather than at a negative that cannot occur.
    const [legendLow, legendHigh] = metricDomain(metric);
    const legendStops = [0, 0.25, 0.5, 0.75, 1]
        .map(t => heatColorScaled(legendLow + t * (legendHigh - legendLow), metric, heatTheme))
        .join(', ');

    return (
        <div>
            <div className="flex items-center gap-2 mb-4">
                <Grid2x2 className="w-5 h-5 text-muted-foreground" />
                <h2 className="text-2xl font-bold tracking-tight text-foreground">S&P 500 Heatmap</h2>
            </div>
            <div className="metric-card p-5">
                {/* Controls */}
                <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                    <div className="flex flex-wrap items-center gap-2">
                        {/* Metric selector */}
                        <select
                            value={metricKey}
                            onChange={e => setMetricKey(e.target.value)}
                            className="px-2.5 py-1 rounded-lg bg-secondary text-xs font-semibold text-foreground border border-border/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/40 appearance-none pr-6"
                            title="Select metric to display"
                            aria-label="Select metric to display"
                        >
                            {METRIC_GROUPS.map(g => (
                                <optgroup key={g} label={g}>
                                    {METRICS.filter(m => m.group === g).map(m => (
                                        <option key={m.key} value={m.key}>{m.label}</option>
                                    ))}
                                </optgroup>
                            ))}
                        </select>
                        {/* Sector filter */}
                        <div className="relative">
                            <select
                                value={sectorFilter}
                                onChange={e => setSectorFilter(e.target.value)}
                                className="px-2.5 py-1 rounded-lg bg-secondary text-xs font-semibold text-foreground border border-border/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/40 appearance-none pr-6"
                                title="Filter by sector"
                                aria-label="Filter by sector"
                            >
                                {sectors.map(s => (
                                    <option key={s} value={s}>{s === 'All' ? 'All Sectors' : s}</option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground pointer-events-none" />
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        {/* Sizing toggle */}
                        <div className="inline-flex rounded-lg bg-secondary p-0.5">
                            {(['cap', 'equal'] as SizeMode[]).map(s => (
                                <button
                                    key={s}
                                    onClick={() => setSizeMode(s)}
                                    className={cn(
                                        'px-2 py-1 rounded-md text-xs font-semibold transition-all',
                                        sizeMode === s ? 'bg-indigo-500 text-white' : 'text-muted-foreground hover:text-foreground',
                                    )}
                                    title={s === 'cap' ? 'Size tiles by market cap' : 'Equal-size tiles'}
                                >
                                    {s === 'cap' ? 'Mkt Cap' : 'Equal'}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Heatmap */}
                {isLoading ? (
                    <div className="flex items-center justify-center py-20">
                        <div className="flex flex-col items-center gap-3">
                            <div className="w-8 h-8 border-2 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
                            <p className="text-sm text-muted-foreground">Loading S&P 500 data…</p>
                        </div>
                    </div>
                ) : rootNode.children.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-12">No data available.</p>
                ) : (
                    <div
                        ref={setContainerElement}
                        className="relative w-full rounded-md overflow-hidden select-none"
                        style={{ height: 650, backgroundColor: palette.surface }}
                        onMouseMove={(e) => {
                            if (hoverNode && containerElement) {
                                const rect = containerElement.getBoundingClientRect();
                                if (rect) {
                                    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
                                }
                            }
                        }}
                        onMouseLeave={() => setHoverNode(null)}
                    >
                        {layoutNodes.map(node => {
                            if (node.type === 1) { // Sector
                                if (!node.showHeader) return null;
                                return (
                                    <div key={node.id} className="absolute pointer-events-none" style={{ left: node.frame.x, top: node.frame.y, width: node.frame.width, height: 22, backgroundColor: palette.sectorHeader }}>
                                        <div
                                            className="w-full h-full flex items-center px-2 text-[11px] font-bold tracking-wide truncate uppercase border-b"
                                            style={{ color: palette.headerText, borderColor: palette.headerRule }}
                                        >
                                            {node.name}
                                        </div>
                                    </div>
                                );
                            } else if (node.type === 2) { // Industry
                                if (!node.showHeader) return null;
                                return (
                                    <div key={node.id} className="absolute pointer-events-none" style={{ left: node.frame.x, top: node.frame.y, width: node.frame.width, height: 16, backgroundColor: palette.industryHeader }}>
                                        <div
                                            className="w-full h-full flex items-center px-1.5 text-[9px] font-semibold tracking-tight truncate uppercase border-b opacity-90"
                                            style={{ color: palette.headerText, borderColor: palette.headerRule }}
                                        >
                                            {node.name}
                                        </div>
                                    </div>
                                );
                            } else if (node.type === 3) { // Company
                                const w = Math.max(0, node.frame.width - 1);
                                const h = Math.max(0, node.frame.height - 1);
                                return (
                                    <div 
                                        key={node.id} 
                                        className="absolute cursor-pointer flex flex-col items-center justify-center overflow-hidden transition-opacity hover:opacity-85"
                                        style={{ left: node.frame.x, top: node.frame.y, width: w, height: h, backgroundColor: node.color || palette.mid }}
                                        onClick={() => { if (node.name) openStockDetail(node.name, 'USD'); }}
                                        onMouseEnter={(e) => {
                                            if (containerElement) {
                                                const rect = containerElement.getBoundingClientRect();
                                                if (rect) {
                                                    setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
                                                    setHoverNode(node);
                                                }
                                            }
                                        }}
                                    >
                                        {w > 32 && h > 18 && (
                                            <span className="font-heavy truncate px-1" style={{ fontSize: w > 60 ? 11 : 8, fontWeight: 800, color: node.textColor }}>
                                                {node.name}
                                            </span>
                                        )}
                                        {w > 44 && h > 30 && node.metricVal != null && (
                                            <span className="font-semibold tabular-nums truncate px-1 opacity-90" style={{ fontSize: w > 60 ? 9 : 7, color: node.textColor }}>
                                                {formatMetric(node.metricVal, metric.format, metric.key)}
                                            </span>
                                        )}
                                    </div>
                                );
                            }
                            return null;
                        })}

                        {/* Tooltip Overlay */}
                        {hoverNode && hoverNode.item && (
                            <div
                                className="absolute pointer-events-none z-10 text-popover-foreground px-3 py-2 rounded-xl border border-border shadow-2xl"
                                style={{
                                    backgroundColor: 'var(--menu-solid)',
                                    // Flip near the edges: the map clips its overflow, so a
                                    // tooltip that runs past the right/bottom is cut in half.
                                    left: Math.max(0, Math.min(mousePos.x + 15, containerSize.width - TOOLTIP_WIDTH)),
                                    top: mousePos.y + 15 + TOOLTIP_HEIGHT > containerSize.height
                                        ? Math.max(0, mousePos.y - TOOLTIP_HEIGHT - 10)
                                        : mousePos.y + 15,
                                    width: TOOLTIP_WIDTH
                                }}
                            >
                                <div className="flex items-center gap-2">
                                    <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: hoverNode.color }} />
                                    <span className="font-bold text-sm">{hoverNode.name}</span>
                                    <span className="text-xs text-muted-foreground truncate max-w-[140px]">{hoverNode.item.sector}</span>
                                </div>
                                <div className="mt-1 flex items-center gap-1.5 text-xs tabular-nums whitespace-nowrap flex-wrap">
                                    <span className="font-medium">${hoverNode.item.price?.toFixed(2) || '0.00'}</span>
                                    <span className="text-muted-foreground/50">·</span>
                                    <span className="text-muted-foreground">{formatCompactCap(hoverNode.item.market_cap || 0)}</span>
                                    <span className="text-muted-foreground/50">·</span>
                                    <span className="text-muted-foreground">{metric.label}</span>
                                    <span className={cn('font-bold', metricTone(hoverNode.metricVal, metric))}>
                                        {formatMetric(hoverNode.metricVal, metric.format, metric.key)}
                                    </span>
                                </div>
                                <div className="text-[10px] text-muted-foreground/70 mt-1 truncate max-w-[220px]">{hoverNode.item.sub_industry}</div>
                            </div>
                        )}
                    </div>
                )}

                {/* Legend + footer */}
                <div className="flex flex-wrap items-center justify-between gap-3 mt-3">
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] text-muted-foreground tabular-nums">{formatScaled(legendLow, metric.format)}</span>
                        <div className="h-2.5 w-40 rounded-full" style={{ background: `linear-gradient(to right, ${legendStops})` }} />
                        <span className="text-[10px] text-muted-foreground tabular-nums">{formatScaled(legendHigh, metric.format)}</span>
                    </div>
                    <p className="text-[10px] text-muted-foreground/60 tabular-nums">
                        {shownCount} of {rawData.length} stocks · {sectorFilter === 'All' ? `${rootNode.children.length} sectors` : `${rootNode.children.length} sub-industries`} · click a tile for detail
                    </p>
                </div>
            </div>
        </div>
    );
}
