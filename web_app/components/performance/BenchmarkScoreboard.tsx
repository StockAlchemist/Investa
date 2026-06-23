'use client';
import React from 'react';
import { Scale } from 'lucide-react';
import { BenchmarkStat } from '../../lib/api';
import { cn } from '../../lib/utils';

interface BenchmarkScoreboardProps {
    // alpha/beta/R²/TE/IR/excess are computed server-side (/benchmark_scoreboard)
    // so the web and native clients share one correctly-annualized source.
    data: BenchmarkStat[] | null;
    isLoading?: boolean;
}

export default function BenchmarkScoreboard({ data, isLoading }: BenchmarkScoreboardProps) {
    const rows = data ?? [];
    const num = (v: number, digits = 2) => `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`;
    const tone = (v: number) => v >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400';

    return (
        <div className="metric-card p-5 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-cyan-500 opacity-80" />
            <div className="flex items-center gap-2 mb-4">
                <Scale className="w-3.5 h-3.5 text-cyan-500" />
                <h3 className="section-label">Vs Benchmark</h3>
            </div>

            {isLoading ? (
                <div className="h-32 animate-pulse bg-muted/30 rounded-lg" />
            ) : rows.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                    Not enough history to compute risk-adjusted stats.
                </p>
            ) : (
                <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                        <thead>
                            <tr className="text-[10px] uppercase tracking-wider text-muted-foreground/70 border-b border-border/50">
                                <th className="py-1.5 pr-3 text-left font-semibold">Benchmark</th>
                                <th className="py-1.5 px-2 text-right font-semibold" title="Annualized Jensen's alpha">α</th>
                                <th className="py-1.5 px-2 text-right font-semibold" title="Beta vs benchmark">β</th>
                                <th className="py-1.5 px-2 text-right font-semibold" title="R-squared (fit)">R²</th>
                                <th className="py-1.5 px-2 text-right font-semibold" title="Annualized tracking error">TE</th>
                                <th className="py-1.5 px-2 text-right font-semibold" title="Information ratio">IR</th>
                                <th className="py-1.5 pl-2 text-right font-semibold" title="Cumulative excess return">Excess</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map(r => (
                                <tr key={r.name} className="border-b border-border/30 last:border-0 hover:bg-muted/30">
                                    <td className="py-2 pr-3 font-bold text-foreground truncate max-w-[120px]">{r.name}</td>
                                    <td className={cn('py-2 px-2 text-right tabular-nums font-semibold', tone(r.alpha))}>{num(r.alpha)}%</td>
                                    <td className="py-2 px-2 text-right tabular-nums text-foreground">{r.beta.toFixed(2)}</td>
                                    <td className="py-2 px-2 text-right tabular-nums text-muted-foreground">{r.r2.toFixed(2)}</td>
                                    <td className="py-2 px-2 text-right tabular-nums text-muted-foreground">{r.tracking_error.toFixed(1)}%</td>
                                    <td className={cn('py-2 px-2 text-right tabular-nums font-semibold', tone(r.information_ratio))}>{num(r.information_ratio)}</td>
                                    <td className={cn('py-2 pl-2 text-right tabular-nums font-semibold', tone(r.excess_return))}>{num(r.excess_return, 1)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <p className="text-[10px] text-muted-foreground/50 mt-3 leading-relaxed">
                        α = annualized excess vs beta-adjusted benchmark · β = sensitivity · R² = correlation² · TE = annualized tracking error · IR = excess ÷ TE.
                    </p>
                </div>
            )}
        </div>
    );
}
