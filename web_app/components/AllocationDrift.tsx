'use client';
import React, { useEffect, useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Pencil, Check, X, AlertTriangle, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { Holding, fetchSettings, updateSettings } from '../lib/api';
import { cn } from '../lib/utils';

interface AllocationDriftProps {
    holdings: Holding[];
    currency: string;
    /**
     * Which holding field to bucket by. Currently used by Asset Type ("quoteType")
     * and Sector. Country uses a separate path because of the geography override.
     */
    bucketKey: keyof Holding | 'Sector' | 'quoteType';
    title: string;
    /**
     * Local-storage key — used as a cold-start cache before the server settings
     * load, and as a fallback for users without a backend session.
     */
    storageKey: string;
    /**
     * Settings-side bucket key. Used as the outer key in target_allocation
     * (e.g. "quoteType", "sector"). Falls back to bucketKey if omitted.
     */
    settingsBucket?: string;
}

const DRIFT_WARN_PCT = 5;
const DRIFT_ALERT_PCT = 10;

function loadLocalTargets(key: string): Record<string, number> {
    if (typeof window === 'undefined') return {};
    try {
        const raw = window.localStorage.getItem(key);
        return raw ? JSON.parse(raw) : {};
    } catch {
        return {};
    }
}

function saveLocalTargets(key: string, targets: Record<string, number>) {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(key, JSON.stringify(targets));
}

export default function AllocationDrift({
    holdings,
    currency,
    bucketKey,
    title,
    storageKey,
    settingsBucket,
}: AllocationDriftProps) {
    const bucket = settingsBucket ?? (bucketKey as string);
    const queryClient = useQueryClient();

    const settingsQuery = useQuery({
        queryKey: ['settings'],
        queryFn: fetchSettings,
        staleTime: 5 * 60 * 1000,
    });

    const settingsMutation = useMutation({
        mutationFn: updateSettings,
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['settings'] }),
    });

    // Render-time source of truth: server settings if loaded, else localStorage cache.
    const targets: Record<string, number> = useMemo(() => {
        const serverMap = (settingsQuery.data?.target_allocation as Record<string, Record<string, number>> | undefined)?.[bucket];
        if (serverMap && Object.keys(serverMap).length > 0) return serverMap;
        return loadLocalTargets(storageKey);
    }, [settingsQuery.data, bucket, storageKey]);

    // Keep localStorage warm so a reload before the server settings query
    // resolves still renders the same numbers.
    useEffect(() => {
        const serverMap = (settingsQuery.data?.target_allocation as Record<string, Record<string, number>> | undefined)?.[bucket];
        if (serverMap) saveLocalTargets(storageKey, serverMap);
    }, [settingsQuery.data, bucket, storageKey]);

    const [editing, setEditing] = useState(false);
    const [draft, setDraft] = useState<Record<string, string>>({});

    const mvKey = `Market Value (${currency})`;

    // Current allocation: aggregate holdings by the chosen bucket key.
    const { rows, total } = useMemo(() => {
        const agg: Record<string, number> = {};
        for (const h of holdings) {
            const v = (h[mvKey] as number) || 0;
            const raw = h[bucketKey] as unknown;
            const cat = !raw || (raw as string).startsWith('N/A') || (raw as string).startsWith('Unknown')
                ? 'Unknown'
                : (raw as string);
            agg[cat] = (agg[cat] || 0) + v;
        }
        const tot = Object.values(agg).reduce((s, v) => s + v, 0);
        // Build rows: every bucket that exists in either current or target
        const allBuckets = new Set([...Object.keys(agg), ...Object.keys(targets)]);
        const r = Array.from(allBuckets).map(b => {
            const current = tot > 0 ? (agg[b] || 0) / tot * 100 : 0;
            const target = targets[b] ?? 0;
            return { bucket: b, current, target, drift: current - target };
        });
        // Sort by current desc, then target desc
        r.sort((a, b) => b.current - a.current || b.target - a.target);
        return { rows: r, total: tot };
    }, [holdings, targets, bucketKey, mvKey]);

    const targetSum = Object.values(targets).reduce((s, v) => s + v, 0);

    const startEdit = () => {
        // Seed draft with current targets, but include every visible bucket
        // so the user can fill in 0% rows.
        const seed: Record<string, string> = {};
        for (const r of rows) seed[r.bucket] = String(targets[r.bucket] ?? '');
        setDraft(seed);
        setEditing(true);
    };

    const cancelEdit = () => {
        setEditing(false);
        setDraft({});
    };

    const commitEdit = () => {
        const next: Record<string, number> = {};
        for (const [k, v] of Object.entries(draft)) {
            const n = parseFloat(v);
            if (!isNaN(n) && n > 0) next[k] = n;
        }
        // Persist to backend (single settings call merges per bucket).
        const existing = (settingsQuery.data?.target_allocation as Record<string, Record<string, number>> | undefined) ?? {};
        const merged: Record<string, Record<string, number>> = { ...existing, [bucket]: next };
        settingsMutation.mutate({ target_allocation: merged });
        // Optimistically update the local cache so the UI doesn't flicker
        // while the mutation round-trips.
        saveLocalTargets(storageKey, next);
        setEditing(false);
        setDraft({});
    };

    const draftSum = Object.values(draft).reduce((s, v) => s + (parseFloat(v) || 0), 0);

    return (
        <div className="metric-card card-shine p-5">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="section-label">{title}</h3>
                    {targetSum > 0 && !editing && (
                        <p className="text-[10px] text-muted-foreground mt-0.5 tabular-nums">
                            Targets sum to {targetSum.toFixed(1)}%
                            {Math.abs(targetSum - 100) > 0.5 && (
                                <span className="text-amber-500"> (not 100%)</span>
                            )}
                        </p>
                    )}
                </div>
                {!editing ? (
                    <button
                        onClick={startEdit}
                        className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-muted rounded-md transition-colors"
                    >
                        <Pencil className="w-3 h-3" />
                        {targetSum > 0 ? 'Edit targets' : 'Set targets'}
                    </button>
                ) : (
                    <div className="flex items-center gap-1.5">
                        <span className={cn(
                            'text-[11px] font-bold tabular-nums px-2',
                            Math.abs(draftSum - 100) < 0.5 ? 'text-emerald-500' : 'text-amber-500',
                        )}>
                            Σ {draftSum.toFixed(1)}%
                        </span>
                        <button
                            onClick={commitEdit}
                            className="p-1.5 rounded-md bg-emerald-500/15 text-emerald-600 hover:bg-emerald-500/25 transition-colors"
                            aria-label="Save targets"
                        >
                            <Check className="w-3.5 h-3.5" />
                        </button>
                        <button
                            onClick={cancelEdit}
                            className="p-1.5 rounded-md bg-muted text-muted-foreground hover:bg-muted/80 transition-colors"
                            aria-label="Cancel"
                        >
                            <X className="w-3.5 h-3.5" />
                        </button>
                    </div>
                )}
            </div>

            {total === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-6">No holdings to bucket.</p>
            ) : targetSum === 0 && !editing ? (
                <p className="text-sm text-muted-foreground py-6 text-center">
                    Set target % per bucket to see drift from your plan.
                </p>
            ) : (
                <div className="space-y-2">
                    {rows.map(r => {
                        const drift = r.drift;
                        const absDrift = Math.abs(drift);
                        const isAlert = absDrift >= DRIFT_ALERT_PCT && r.target > 0;
                        const isWarn = !isAlert && absDrift >= DRIFT_WARN_PCT && r.target > 0;
                        return (
                            <div key={r.bucket} className="grid grid-cols-[1fr_auto] gap-3 items-center">
                                <div>
                                    <div className="flex items-center justify-between gap-2 mb-1">
                                        <span className="text-xs font-medium text-foreground truncate">{r.bucket}</span>
                                        <span className="text-[11px] tabular-nums text-muted-foreground shrink-0">
                                            {r.current.toFixed(1)}% / {r.target.toFixed(1)}%
                                        </span>
                                    </div>
                                    {/* Stacked progress bar: current bar with target marker */}
                                    <div className="relative h-2 bg-muted rounded-full overflow-hidden">
                                        <div
                                            className="absolute inset-y-0 left-0 bg-[#0097b2] rounded-full transition-all"
                                            style={{ width: `${Math.min(100, r.current)}%` }}
                                        />
                                        {r.target > 0 && (
                                            <div
                                                className="absolute inset-y-0 w-0.5 bg-foreground/70"
                                                style={{ left: `${Math.min(100, r.target)}%` }}
                                            />
                                        )}
                                    </div>
                                </div>
                                {editing ? (
                                    <input
                                        type="number"
                                        step="0.1"
                                        min="0"
                                        max="100"
                                        value={draft[r.bucket] ?? ''}
                                        onChange={e => setDraft(d => ({ ...d, [r.bucket]: e.target.value }))}
                                        className="w-14 text-xs px-1.5 py-1 bg-secondary text-foreground rounded border border-transparent focus:border-[#0097b2] focus:outline-none text-right tabular-nums"
                                        placeholder="0"
                                    />
                                ) : (
                                    <span
                                        className={cn(
                                            'flex items-center gap-0.5 text-[11px] font-bold tabular-nums w-14 justify-end',
                                            r.target === 0
                                                ? 'text-muted-foreground/60'
                                                : isAlert
                                                    ? 'text-red-500'
                                                    : isWarn
                                                        ? 'text-amber-500'
                                                        : 'text-emerald-500',
                                        )}
                                    >
                                        {isAlert && <AlertTriangle className="w-3 h-3" />}
                                        {drift > 0
                                            ? <ArrowUpRight className="w-3 h-3" />
                                            : drift < 0
                                                ? <ArrowDownRight className="w-3 h-3" />
                                                : null}
                                        {drift > 0 ? '+' : ''}{drift.toFixed(1)}
                                    </span>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
