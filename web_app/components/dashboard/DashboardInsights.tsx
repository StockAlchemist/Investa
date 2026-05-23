'use client';
import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Lightbulb, Hourglass, AlertTriangle, Gem, Sparkles } from 'lucide-react';
import { Holding, fetchSettings } from '../../lib/api';
import { cn } from '../../lib/utils';

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

interface Insight {
    tone: 'pos' | 'warn' | 'alert' | 'neutral';
    icon: React.ComponentType<{ className?: string }>;
    title: string;
    sub?: string;
}

export default function DashboardInsights({ holdings, currency }: DashboardInsightsProps) {
    const settingsQuery = useQuery({ queryKey: ['settings'], queryFn: fetchSettings, staleTime: 5 * 60 * 1000 });
    const targets = settingsQuery.data?.target_allocation;
    // Freeze "now" at mount — re-renders shouldn't shift the ripening window mid-session.
    const [now] = useState<number>(() => Date.now());

    const insights = useMemo<Insight[]>(() => {
        const out: Insight[] = [];
        const mvKey = `Market Value (${currency})`;

        // 1) Lots ripening to long-term within 30 days, with a positive gain.
        let ripening = 0;
        for (const h of holdings) {
            for (const lot of h.lots || []) {
                if (!lot.Date) continue;
                const lotMs = new Date(lot.Date).getTime();
                if (isNaN(lotMs)) continue;
                const heldDays = (now - lotMs) / ONE_DAY_MS;
                const remaining = ONE_YEAR_DAYS - heldDays;
                const gain = (lot['Unreal. Gain'] as number) || 0;
                if (remaining > 0 && remaining <= RIPENING_WINDOW_DAYS && gain > 0) {
                    ripening += 1;
                }
            }
        }
        if (ripening > 0) {
            out.push({
                tone: 'warn',
                icon: Hourglass,
                title: `${ripening} ${ripening === 1 ? 'lot ripens' : 'lots ripen'} to long-term in the next 30 days`,
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
            let worst: { dim: string; bucket: string; drift: number } | null = null;
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
                        if (!worst || Math.abs(drift) > Math.abs(worst.drift)) {
                            worst = { dim: dim.label, bucket, drift };
                        }
                    }
                }
            }
            if (worst) {
                out.push({
                    tone: 'alert',
                    icon: AlertTriangle,
                    title: `${worst.bucket} (${worst.dim}) ${worst.drift > 0 ? '+' : ''}${worst.drift.toFixed(1)}% off target`,
                    sub: 'See the Portfolio tab to rebalance.',
                });
            }
        }

        // 3) Significantly undervalued holdings (margin of safety > 10%).
        const undervalued = holdings.filter(h => {
            const mos = h.margin_of_safety;
            return typeof mos === 'number' && mos > MOS_SIGNIFICANT;
        });
        if (undervalued.length > 0) {
            out.push({
                tone: 'pos',
                icon: Gem,
                title: `${undervalued.length} ${undervalued.length === 1 ? 'holding trades' : 'holdings trade'} below fair value`,
                sub: 'Margin of safety > 10% on your latest screen.',
            });
        }

        return out;
    }, [holdings, targets, currency, now]);

    return (
        <div className="metric-card p-5 h-full">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Lightbulb className="w-3.5 h-3.5 text-amber-500" />
                    <h3 className="section-label">Insights</h3>
                </div>
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold">
                    {insights.length} item{insights.length === 1 ? '' : 's'}
                </span>
            </div>

            {insights.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-6 text-center gap-1">
                    <Sparkles className="w-5 h-5 text-muted-foreground/40" />
                    <p className="text-sm text-muted-foreground">Nothing to flag today.</p>
                    <p className="text-[11px] text-muted-foreground/60">No ripening lots, drift breaches, or new value buys.</p>
                </div>
            ) : (
                <div className="space-y-2">
                    {insights.map((ins, i) => {
                        const Icon = ins.icon;
                        const tone = ins.tone === 'pos'
                            ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10'
                            : ins.tone === 'alert'
                                ? 'text-red-600 dark:text-red-400 bg-red-500/10'
                                : ins.tone === 'warn'
                                    ? 'text-amber-600 dark:text-amber-400 bg-amber-500/10'
                                    : 'text-foreground bg-muted/30';
                        return (
                            <div key={i} className="flex items-start gap-3">
                                <span className={cn('flex items-center justify-center w-7 h-7 rounded-md shrink-0 mt-0.5', tone)}>
                                    <Icon className="w-3.5 h-3.5" />
                                </span>
                                <div className="min-w-0">
                                    <p className="text-sm font-semibold text-foreground leading-snug">{ins.title}</p>
                                    {ins.sub && (
                                        <p className="text-[11px] text-muted-foreground/70 leading-snug mt-0.5">{ins.sub}</p>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
