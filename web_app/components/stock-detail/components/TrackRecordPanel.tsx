import React from 'react';
import { Shield, AlertCircle, History } from 'lucide-react';
import { TrackRecord } from '../../../lib/api';
import { cn } from '../../../lib/utils';

function fiscalPeriodYear(iso: string): string {
    return iso.slice(0, 4);
}

export const TrackRecordPanel: React.FC<{ record: TrackRecord }> = ({ record }) => {
    const span = record.first_period && record.latest_period
        ? `${fiscalPeriodYear(record.first_period)}–${fiscalPeriodYear(record.latest_period)}`
        : null;

    return (
        <div className="bg-muted rounded-2xl p-6 space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                        <Shield className="w-4 h-4 text-indigo-500" />
                        Track Record
                    </h4>
                    <p className="text-xs text-muted-foreground mt-1">
                        {record.period_count} years of SEC filings{span ? ` (${span})` : ''}
                        {' · '}measured over the last {record.window_years}
                        {record.model !== 'generic' ? ` · ${record.model} model` : ''}
                    </p>
                </div>
                {record.rank?.rank != null && (
                    <div className="text-right">
                        <div className="text-2xl font-bold tabular-nums">#{record.rank.rank}</div>
                        <div className="text-[10px] text-muted-foreground uppercase tracking-wider">Buffett rank</div>
                    </div>
                )}
            </div>

            {record.gate_failures.length > 0 && (
                <div className="flex items-start gap-2 text-xs bg-amber-500/10 text-amber-600 dark:text-amber-400 rounded-xl px-3 py-2">
                    <AlertCircle className="w-4 h-4 flex-shrink-0 mt-px" />
                    <span>
                        Not eligible for the ranking:{' '}
                        {record.gate_failures.map(reason => reason.replace(/_/g, ' ')).join(', ')}
                    </span>
                </div>
            )}

            {record.valuation_bands && record.valuation_bands.length > 0 && (
                <div className="bg-card rounded-xl px-4 py-3">
                    <h5 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-3">
                        Against its own history
                    </h5>
                    <div className="space-y-4">
                        {record.valuation_bands.map(band => {
                            const span = Math.max(band.high - band.low, 1e-9);
                            const at = (v: number) => `${Math.min(100, Math.max(0, ((v - band.low) / span) * 100))}%`;
                            return (
                                <div key={band.metric}>
                                    <div className="flex items-baseline justify-between gap-3 text-sm mb-1.5">
                                        <span className="text-muted-foreground">{band.label}</span>
                                        <span className="tabular-nums">
                                            <span className="font-semibold">{band.display}</span>
                                            <span className="text-muted-foreground text-xs"> vs {band.median_display} median</span>
                                        </span>
                                    </div>
                                    <div className="relative h-2 rounded-full bg-muted">
                                        <div
                                            className="absolute h-2 rounded-full bg-indigo-500/25"
                                            style={{ left: at(band.p25), width: `calc(${at(band.p75)} - ${at(band.p25)})` }}
                                        />
                                        <div className="absolute h-2 w-px bg-muted-foreground/60" style={{ left: at(band.median) }} />
                                        <div
                                            className="absolute -top-0.5 h-3 w-1 rounded-sm bg-indigo-500"
                                            style={{ left: at(band.current) }}
                                            title={`${band.display} — ${band.percentile.toFixed(0)}th percentile of ${band.observations} years`}
                                        />
                                    </div>
                                    <div className="text-[11px] text-muted-foreground mt-1">
                                        {band.summary} ({band.observations} years)
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {record.stress?.some(w => w.covered) && (
                <div className="bg-card rounded-xl px-4 py-3">
                    <h5 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-2">
                        In a downturn
                    </h5>
                    <div className="space-y-2">
                        {record.stress.map(window => (
                            <div key={window.key} className="flex flex-col sm:flex-row sm:items-baseline gap-x-3 gap-y-1 text-sm">
                                <span className="text-muted-foreground whitespace-nowrap sm:w-40 shrink-0">{window.label}</span>
                                {window.covered ? (
                                    <span className="flex flex-wrap gap-x-4 gap-y-1">
                                        {window.items.map(item => (
                                            <span key={item.metric} className="whitespace-nowrap">
                                                {item.label}{' '}
                                                <span className={cn(
                                                    "font-medium tabular-nums",
                                                    item.change_pct < 0 ? "text-down" : "text-up"
                                                )}>{item.display}</span>
                                                <span className="text-muted-foreground text-xs">
                                                    {' '}({item.recovery_display ?? 'no fall'})
                                                </span>
                                            </span>
                                        ))}
                                    </span>
                                ) : (
                                    <span className="text-muted-foreground/60 italic">not filing then</span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {record.revisions && record.revisions.count > 0 && (
                <details className="bg-card rounded-xl px-4 py-3 group">
                    <summary className="cursor-pointer text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2 select-none">
                        <History className="w-3.5 h-3.5" />
                        {record.revisions.count} figure{record.revisions.count === 1 ? '' : 's'} revised after first reporting
                    </summary>
                    <p className="text-[11px] text-muted-foreground mt-2 mb-3">
                        Later filings changed these. Usually a retrospectively adopted accounting
                        standard or a reclassification — the size and the gap are what matter.
                    </p>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <tbody>
                                {record.revisions.items.map(item => (
                                    <tr key={`${item.concept}-${item.period_end}`} className="border-t border-border/40">
                                        <td className="py-1.5 pr-3 text-muted-foreground whitespace-nowrap">{item.label}</td>
                                        <td className="py-1.5 pr-3 text-muted-foreground tabular-nums whitespace-nowrap">{fiscalPeriodYear(item.period_end)}</td>
                                        <td className="py-1.5 pr-3 tabular-nums whitespace-nowrap">{item.display}</td>
                                        <td className={cn(
                                            "py-1.5 pr-3 text-right font-medium tabular-nums whitespace-nowrap",
                                            item.change_pct < 0 ? "text-down" : "text-up"
                                        )}>{item.change_display}</td>
                                        <td className="py-1.5 text-right text-[11px] text-muted-foreground whitespace-nowrap">
                                            {fiscalPeriodYear(item.first_filed)} → {fiscalPeriodYear(item.restated_filed)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {record.revisions.count > record.revisions.items.length && (
                        <p className="text-[11px] text-muted-foreground mt-2">
                            Showing the {record.revisions.items.length} largest of {record.revisions.count}.
                        </p>
                    )}
                </details>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {record.groups.map(group => (
                    <div key={group.key} className="bg-card rounded-xl p-4">
                        <div className="flex items-baseline justify-between gap-2 mb-3">
                            <h5 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{group.title}</h5>
                            {record.rank?.pillars?.[group.key] != null && (
                                <span className="text-xs font-semibold tabular-nums text-muted-foreground">
                                    {(record.rank.pillars[group.key] as number).toFixed(0)}
                                </span>
                            )}
                        </div>
                        <dl className="space-y-2">
                            {group.items.map(item => (
                                <div key={item.key} className="flex items-baseline justify-between gap-3 text-sm">
                                    <dt className="text-muted-foreground">{item.label}</dt>
                                    <dd
                                        className={cn(
                                            "font-medium tabular-nums text-right whitespace-nowrap",
                                            item.display ? "text-foreground" : "text-muted-foreground/60"
                                        )}
                                        title={item.note ?? undefined}
                                    >
                                        {item.display ?? (item.note ? 'n/a' : '—')}
                                    </dd>
                                </div>
                            ))}
                        </dl>
                    </div>
                ))}
            </div>
        </div>
    );
};
