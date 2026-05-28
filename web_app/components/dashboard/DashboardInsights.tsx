'use client';
import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { useQuery } from '@tanstack/react-query';
import { Lightbulb, Hourglass, AlertTriangle, Gem, Sparkles, ChevronRight, X } from 'lucide-react';
import { Holding, fetchSettings } from '../../lib/api';
import { cn, formatCurrency } from '../../lib/utils';

interface DashboardInsightsProps {
    holdings: Holding[];
    currency: string;
}

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const ONE_YEAR_DAYS = 365;
const RIPENING_WINDOW_DAYS = 30;
const DRIFT_ALERT_PCT = 10;
const MOS_SIGNIFICANT = 10; // margin of safety > 10% counts as "significantly undervalued"

function isUnknown(v: unknown): boolean {
    if (v == null) return true;
    const s = String(v).trim().toUpperCase();
    return s === '' || s === '-' || s === 'NONE' || s === 'NULL' || s === 'UNKNOWN'
        || s.startsWith('N/A') || s.startsWith('UNKNOWN');
}

type InsightKind = 'ripening' | 'drift' | 'undervalued';

interface RipeningLot {
    symbol: string;
    account?: string;
    date: string;
    daysRemaining: number;
    quantity: number;
    gain: number;
}

interface DriftBucket {
    dim: string;
    bucket: string;
    currentPct: number;
    targetPct: number;
    drift: number;
}

interface UndervaluedHolding {
    symbol: string;
    account?: string;
    mos: number;
    intrinsic?: number;
    marketValue?: number;
}

interface Insight {
    kind: InsightKind;
    tone: 'pos' | 'warn' | 'alert' | 'neutral';
    icon: React.ComponentType<{ className?: string }>;
    title: string;
    sub?: string;
}

interface InsightDetails {
    ripening: RipeningLot[];
    drift: DriftBucket[];
    undervalued: UndervaluedHolding[];
}

export default function DashboardInsights({ holdings, currency }: DashboardInsightsProps) {
    const settingsQuery = useQuery({ queryKey: ['settings'], queryFn: fetchSettings, staleTime: 5 * 60 * 1000 });
    const targets = settingsQuery.data?.target_allocation;
    // Freeze "now" at mount — re-renders shouldn't shift the ripening window mid-session.
    const [now] = useState<number>(() => Date.now());
    const [openKind, setOpenKind] = useState<InsightKind | 'all' | null>(null);

    const { insights, details } = useMemo<{ insights: Insight[]; details: InsightDetails }>(() => {
        const out: Insight[] = [];
        const det: InsightDetails = { ripening: [], drift: [], undervalued: [] };
        const mvKey = `Market Value (${currency})`;

        // 1) Lots ripening to long-term within 30 days, with a positive gain.
        for (const h of holdings) {
            for (const lot of h.lots || []) {
                if (!lot.Date) continue;
                const lotMs = new Date(lot.Date).getTime();
                if (isNaN(lotMs)) continue;
                const heldDays = (now - lotMs) / ONE_DAY_MS;
                const remaining = ONE_YEAR_DAYS - heldDays;
                const gain = (lot['Unreal. Gain'] as number) || 0;
                if (remaining > 0 && remaining <= RIPENING_WINDOW_DAYS && gain > 0) {
                    det.ripening.push({
                        symbol: h.Symbol,
                        account: h.Account,
                        date: lot.Date,
                        daysRemaining: Math.ceil(remaining),
                        quantity: lot.Quantity,
                        gain,
                    });
                }
            }
        }
        det.ripening.sort((a, b) => a.daysRemaining - b.daysRemaining);
        if (det.ripening.length > 0) {
            const n = det.ripening.length;
            out.push({
                kind: 'ripening',
                tone: 'warn',
                icon: Hourglass,
                title: `${n} ${n === 1 ? 'lot ripens' : 'lots ripen'} to long-term in the next 30 days`,
                sub: 'Holding past the 1-year mark unlocks the LTCG rate.',
            });
        }

        // 2) Drift breaches: any bucket > DRIFT_ALERT_PCT off its target.
        if (targets) {
            type Dim = { key: 'quoteType' | 'sector' | 'country'; label: string };
            const dims: Dim[] = [
                { key: 'quoteType', label: 'Asset Type' },
                { key: 'sector', label: 'Sector' },
                { key: 'country', label: 'Country' },
            ];
            for (const dim of dims) {
                const t = (targets as Record<string, Record<string, number>>)[dim.key];
                if (!t || Object.keys(t).length === 0) continue;

                const agg: Record<string, number> = {};
                for (const h of holdings) {
                    const v = Math.max(0, (h[mvKey] as number) || 0);
                    const raw = dim.key === 'country'
                        ? ((h['geography'] as string) || (h['Country'] as string))
                        : dim.key === 'sector'
                            ? h['Sector']
                            : h['quoteType'];
                    const cat = isUnknown(raw) ? 'Unknown' : (raw as string);
                    agg[cat] = (agg[cat] || 0) + v;
                }
                const tot = Object.values(agg).reduce((s, v) => s + v, 0);
                if (tot <= 0) continue;

                for (const [bucket, targetPct] of Object.entries(t)) {
                    const currentPct = ((agg[bucket] || 0) / tot) * 100;
                    const drift = currentPct - targetPct;
                    if (Math.abs(drift) >= DRIFT_ALERT_PCT) {
                        det.drift.push({ dim: dim.label, bucket, currentPct, targetPct, drift });
                    }
                }
            }
            det.drift.sort((a, b) => Math.abs(b.drift) - Math.abs(a.drift));
            if (det.drift.length > 0) {
                const worst = det.drift[0];
                const more = det.drift.length - 1;
                out.push({
                    kind: 'drift',
                    tone: 'alert',
                    icon: AlertTriangle,
                    title: `${worst.bucket} (${worst.dim}) ${worst.drift > 0 ? '+' : ''}${worst.drift.toFixed(1)}% off target`,
                    sub: more > 0
                        ? `${more} more bucket${more === 1 ? '' : 's'} also breached — tap for the full list.`
                        : 'See the Portfolio tab to rebalance.',
                });
            }
        }

        // 3) Significantly undervalued holdings (margin of safety > 10%).
        for (const h of holdings) {
            const mos = h.margin_of_safety;
            if (typeof mos === 'number' && mos > MOS_SIGNIFICANT) {
                det.undervalued.push({
                    symbol: h.Symbol,
                    account: h.Account,
                    mos,
                    intrinsic: h.intrinsic_value,
                    marketValue: (h[mvKey] as number) || undefined,
                });
            }
        }
        det.undervalued.sort((a, b) => b.mos - a.mos);
        if (det.undervalued.length > 0) {
            const n = det.undervalued.length;
            out.push({
                kind: 'undervalued',
                tone: 'pos',
                icon: Gem,
                title: `${n} ${n === 1 ? 'holding trades' : 'holdings trade'} below fair value`,
                sub: 'Margin of safety > 10% on your latest screen.',
            });
        }

        return { insights: out, details: det };
    }, [holdings, targets, currency, now]);

    const hasAny = insights.length > 0;

    return (
        <>
            <div
                role={hasAny ? 'button' : undefined}
                tabIndex={hasAny ? 0 : undefined}
                onClick={hasAny ? () => setOpenKind('all') : undefined}
                onKeyDown={hasAny ? (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setOpenKind('all');
                    }
                } : undefined}
                className={cn(
                    'metric-card p-5 h-full',
                    hasAny && 'cursor-pointer transition-all hover:border-amber-500/40 hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/50'
                )}
            >
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                        <Lightbulb className="w-3.5 h-3.5 text-amber-500" />
                        <h3 className="section-label">Insights</h3>
                    </div>
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold">
                        {insights.length} item{insights.length === 1 ? '' : 's'}
                    </span>
                </div>

                {!hasAny ? (
                    <div className="flex flex-col items-center justify-center py-6 text-center gap-1">
                        <Sparkles className="w-5 h-5 text-muted-foreground/40" />
                        <p className="text-sm text-muted-foreground">Nothing to flag today.</p>
                        <p className="text-[11px] text-muted-foreground/60">No ripening lots, drift breaches, or new value buys.</p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        {insights.map((ins) => {
                            const Icon = ins.icon;
                            const tone = ins.tone === 'pos'
                                ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10'
                                : ins.tone === 'alert'
                                    ? 'text-red-600 dark:text-red-400 bg-red-500/10'
                                    : ins.tone === 'warn'
                                        ? 'text-amber-600 dark:text-amber-400 bg-amber-500/10'
                                        : 'text-foreground bg-muted/30';
                            return (
                                <div
                                    key={ins.kind}
                                    role="button"
                                    tabIndex={0}
                                    onClick={(e) => { e.stopPropagation(); setOpenKind(ins.kind); }}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' || e.key === ' ') {
                                            e.preventDefault();
                                            e.stopPropagation();
                                            setOpenKind(ins.kind);
                                        }
                                    }}
                                    className="group flex items-start gap-3 -mx-2 px-2 py-1.5 rounded-md hover:bg-muted/40 transition-colors cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/40"
                                >
                                    <span className={cn('flex items-center justify-center w-7 h-7 rounded-md shrink-0 mt-0.5', tone)}>
                                        <Icon className="w-3.5 h-3.5" />
                                    </span>
                                    <div className="min-w-0 flex-1">
                                        <p className="text-sm font-semibold text-foreground leading-snug">{ins.title}</p>
                                        {ins.sub && (
                                            <p className="text-[11px] text-muted-foreground/70 leading-snug mt-0.5">{ins.sub}</p>
                                        )}
                                    </div>
                                    <ChevronRight className="w-4 h-4 text-muted-foreground/40 group-hover:text-muted-foreground shrink-0 mt-1 transition-colors" />
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {openKind && (
                <InsightsDetailModal
                    open={openKind}
                    onClose={() => setOpenKind(null)}
                    details={details}
                    insights={insights}
                    currency={currency}
                />
            )}
        </>
    );
}

interface InsightsDetailModalProps {
    open: InsightKind | 'all';
    onClose: () => void;
    details: InsightDetails;
    insights: Insight[];
    currency: string;
}

function InsightsDetailModal({ open, onClose, details, insights, currency }: InsightsDetailModalProps) {
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
        document.addEventListener('keydown', onKey);
        return () => document.removeEventListener('keydown', onKey);
    }, [onClose]);

    if (typeof document === 'undefined') return null;

    const showRipening = open === 'all' || open === 'ripening';
    const showDrift = open === 'all' || open === 'drift';
    const showUndervalued = open === 'all' || open === 'undervalued';

    return createPortal(
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in" onClick={onClose} />
            <div
                style={{ backgroundColor: 'var(--menu-solid)' }}
                className="relative w-full max-w-2xl max-h-[85vh] rounded-3xl flex flex-col overflow-hidden animate-in zoom-in-95 slide-in-from-bottom-10 duration-300 shadow-2xl"
            >
                <div className="px-6 pt-6 pb-4 flex items-start justify-between border-b border-border/40">
                    <div className="flex items-center gap-3">
                        <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center">
                            <Lightbulb className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h2 className="text-xl font-black tracking-tight text-foreground">Insight Details</h2>
                            <p className="text-xs text-muted-foreground mt-0.5">
                                {open === 'all'
                                    ? `${insights.length} signal${insights.length === 1 ? '' : 's'} flagged across your portfolio`
                                    : 'Underlying records behind this signal'}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors text-muted-foreground hover:text-foreground"
                        aria-label="Close"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
                    {showRipening && details.ripening.length > 0 && (
                        <DetailSection
                            icon={Hourglass}
                            tone="warn"
                            title="Lots Ripening to Long-Term"
                            description="Selling these lots after the date below qualifies for the long-term capital gains rate."
                        >
                            <div className="overflow-hidden rounded-xl border border-border/40">
                                <table className="w-full text-sm">
                                    <thead className="bg-muted/30 text-[10px] uppercase tracking-widest text-muted-foreground font-bold">
                                        <tr>
                                            <th className="text-left px-3 py-2">Symbol</th>
                                            <th className="text-left px-3 py-2">Acquired</th>
                                            <th className="text-right px-3 py-2">Qty</th>
                                            <th className="text-right px-3 py-2">Unrealized</th>
                                            <th className="text-right px-3 py-2">Days Left</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-border/30">
                                        {details.ripening.map((lot, i) => (
                                            <tr key={i} className="hover:bg-muted/20">
                                                <td className="px-3 py-2 font-semibold text-foreground">
                                                    {lot.symbol}
                                                    {lot.account && (
                                                        <span className="text-[10px] text-muted-foreground/70 ml-1.5">{lot.account}</span>
                                                    )}
                                                </td>
                                                <td className="px-3 py-2 text-muted-foreground tabular-nums">
                                                    {new Date(lot.date).toLocaleDateString()}
                                                </td>
                                                <td className="px-3 py-2 text-right tabular-nums">{lot.quantity.toLocaleString()}</td>
                                                <td className="px-3 py-2 text-right tabular-nums text-emerald-600 dark:text-emerald-400 font-semibold">
                                                    +{formatCurrency(lot.gain, currency)}
                                                </td>
                                                <td className="px-3 py-2 text-right tabular-nums font-bold text-amber-600 dark:text-amber-400">
                                                    {lot.daysRemaining}d
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </DetailSection>
                    )}

                    {showDrift && details.drift.length > 0 && (
                        <DetailSection
                            icon={AlertTriangle}
                            tone="alert"
                            title="Allocation Drift"
                            description={`Buckets that have drifted ${DRIFT_ALERT_PCT}% or more from their target weight.`}
                        >
                            <div className="space-y-2">
                                {details.drift.map((d, i) => {
                                    const overweight = d.drift > 0;
                                    const pctBar = Math.min(100, Math.abs(d.drift) * 3); // visual scale
                                    return (
                                        <div key={i} className="rounded-xl border border-border/40 p-3">
                                            <div className="flex items-baseline justify-between gap-3 mb-2">
                                                <div className="min-w-0">
                                                    <p className="text-sm font-bold text-foreground truncate">{d.bucket}</p>
                                                    <p className="text-[10px] uppercase tracking-widest text-muted-foreground/70 font-semibold">{d.dim}</p>
                                                </div>
                                                <span className={cn(
                                                    'text-sm font-black tabular-nums shrink-0',
                                                    overweight ? 'text-red-600 dark:text-red-400' : 'text-amber-600 dark:text-amber-400'
                                                )}>
                                                    {overweight ? '+' : ''}{d.drift.toFixed(1)}%
                                                </span>
                                            </div>
                                            <div className="flex items-center justify-between text-[11px] text-muted-foreground tabular-nums mb-1.5">
                                                <span>Current <span className="font-semibold text-foreground">{d.currentPct.toFixed(1)}%</span></span>
                                                <span>Target <span className="font-semibold text-foreground">{d.targetPct.toFixed(1)}%</span></span>
                                            </div>
                                            <div className="h-1.5 bg-muted/40 rounded-full overflow-hidden">
                                                <div
                                                    className={cn('h-full rounded-full', overweight ? 'bg-red-500/70' : 'bg-amber-500/70')}
                                                    style={{ width: `${pctBar}%` }}
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </DetailSection>
                    )}

                    {showUndervalued && details.undervalued.length > 0 && (
                        <DetailSection
                            icon={Gem}
                            tone="pos"
                            title="Trading Below Fair Value"
                            description={`Holdings with a margin of safety greater than ${MOS_SIGNIFICANT}% on the latest screen.`}
                        >
                            <div className="overflow-hidden rounded-xl border border-border/40">
                                <table className="w-full text-sm">
                                    <thead className="bg-muted/30 text-[10px] uppercase tracking-widest text-muted-foreground font-bold">
                                        <tr>
                                            <th className="text-left px-3 py-2">Symbol</th>
                                            <th className="text-right px-3 py-2">Intrinsic</th>
                                            <th className="text-right px-3 py-2">Position</th>
                                            <th className="text-right px-3 py-2">Margin of Safety</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-border/30">
                                        {details.undervalued.map((u, i) => (
                                            <tr key={i} className="hover:bg-muted/20">
                                                <td className="px-3 py-2 font-semibold text-foreground">
                                                    {u.symbol}
                                                    {u.account && (
                                                        <span className="text-[10px] text-muted-foreground/70 ml-1.5">{u.account}</span>
                                                    )}
                                                </td>
                                                <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                                                    {u.intrinsic != null ? formatCurrency(u.intrinsic, currency) : '—'}
                                                </td>
                                                <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                                                    {u.marketValue != null ? formatCurrency(u.marketValue, currency) : '—'}
                                                </td>
                                                <td className="px-3 py-2 text-right tabular-nums font-bold text-emerald-600 dark:text-emerald-400">
                                                    {u.mos.toFixed(1)}%
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </DetailSection>
                    )}

                    {open !== 'all'
                        && ((open === 'ripening' && details.ripening.length === 0)
                            || (open === 'drift' && details.drift.length === 0)
                            || (open === 'undervalued' && details.undervalued.length === 0)) && (
                            <div className="flex flex-col items-center justify-center py-10 text-center gap-2">
                                <Sparkles className="w-6 h-6 text-muted-foreground/40" />
                                <p className="text-sm text-muted-foreground">No records to show.</p>
                            </div>
                        )}
                </div>
            </div>
        </div>,
        document.body
    );
}

interface DetailSectionProps {
    icon: React.ComponentType<{ className?: string }>;
    tone: 'pos' | 'warn' | 'alert' | 'neutral';
    title: string;
    description?: string;
    children: React.ReactNode;
}

function DetailSection({ icon: Icon, tone, title, description, children }: DetailSectionProps) {
    const toneClass = tone === 'pos'
        ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10'
        : tone === 'alert'
            ? 'text-red-600 dark:text-red-400 bg-red-500/10'
            : tone === 'warn'
                ? 'text-amber-600 dark:text-amber-400 bg-amber-500/10'
                : 'text-foreground bg-muted/30';
    return (
        <section>
            <div className="flex items-start gap-3 mb-3">
                <span className={cn('flex items-center justify-center w-8 h-8 rounded-lg shrink-0', toneClass)}>
                    <Icon className="w-4 h-4" />
                </span>
                <div className="min-w-0">
                    <h3 className="text-sm font-bold text-foreground">{title}</h3>
                    {description && <p className="text-[11px] text-muted-foreground/80 mt-0.5">{description}</p>}
                </div>
            </div>
            {children}
        </section>
    );
}
