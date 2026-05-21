'use client';
import React, { useMemo } from 'react';
import { Scale } from 'lucide-react';
import { PerformanceData } from '../../lib/api';
import { cn } from '../../lib/utils';

interface BenchmarkScoreboardProps {
    history: PerformanceData[] | null;
    isLoading?: boolean;
}

const RESERVED_KEYS = new Set(['date', 'value', 'twr', 'drawdown', 'fx_rate', 'abs_gain', 'abs_roi', 'cum_flow']);
const PERIODS_PER_YEAR = 252; // daily series

interface BenchStats {
    name: string;
    alpha: number;   // annualized %
    beta: number;
    r2: number;      // 0-1
    te: number;      // annualized tracking error %
    ir: number;      // information ratio
    excess: number;  // cumulative excess return over the window, %
}

// Convert a cumulative-TWR series (percent) into period simple returns.
function twrToReturns(cumTwr: (number | undefined)[]): (number | null)[] {
    const out: (number | null)[] = [];
    for (let i = 1; i < cumTwr.length; i++) {
        const prev = cumTwr[i - 1];
        const cur = cumTwr[i];
        if (typeof prev !== 'number' || typeof cur !== 'number') { out.push(null); continue; }
        const fPrev = 1 + prev / 100;
        const fCur = 1 + cur / 100;
        out.push(fPrev !== 0 ? fCur / fPrev - 1 : null);
    }
    return out;
}

function mean(xs: number[]): number {
    return xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : 0;
}

export default function BenchmarkScoreboard({ history, isLoading }: BenchmarkScoreboardProps) {
    const rows = useMemo<BenchStats[]>(() => {
        const data = history ?? [];
        if (data.length < 20) return [];

        const benchKeys = Object.keys(data[0]).filter(k => !RESERVED_KEYS.has(k));
        const portReturns = twrToReturns(data.map(d => d.twr));

        const results: BenchStats[] = [];
        for (const key of benchKeys) {
            const benchReturns = twrToReturns(data.map(d => d[key] as number | undefined));

            // Align pairs where both returns exist.
            const rp: number[] = [];
            const rb: number[] = [];
            for (let i = 0; i < portReturns.length; i++) {
                const a = portReturns[i];
                const b = benchReturns[i];
                if (a != null && b != null) { rp.push(a); rb.push(b); }
            }
            if (rp.length < 20) continue;

            const mp = mean(rp);
            const mb = mean(rb);
            let cov = 0, varB = 0, varP = 0;
            for (let i = 0; i < rp.length; i++) {
                cov += (rp[i] - mp) * (rb[i] - mb);
                varB += (rb[i] - mb) ** 2;
                varP += (rp[i] - mp) ** 2;
            }
            cov /= rp.length; varB /= rp.length; varP /= rp.length;

            const beta = varB > 0 ? cov / varB : 0;
            const alphaDaily = mp - beta * mb;
            const alpha = alphaDaily * PERIODS_PER_YEAR * 100;
            const corr = (varP > 0 && varB > 0) ? cov / Math.sqrt(varP * varB) : 0;
            const r2 = corr * corr;

            const diffs = rp.map((v, i) => v - rb[i]);
            const mDiff = mean(diffs);
            const teDaily = Math.sqrt(mean(diffs.map(d => (d - mDiff) ** 2)));
            const te = teDaily * Math.sqrt(PERIODS_PER_YEAR) * 100;
            const ir = teDaily > 0 ? (mDiff * PERIODS_PER_YEAR) / (teDaily * Math.sqrt(PERIODS_PER_YEAR)) : 0;

            // Cumulative excess over the window from the final cumulative TWRs.
            const lastPort = data[data.length - 1].twr;
            const lastBench = data[data.length - 1][key] as number | undefined;
            const excess = (typeof lastPort === 'number' && typeof lastBench === 'number') ? lastPort - lastBench : 0;

            results.push({ name: key, alpha, beta, r2, te, ir, excess });
        }
        return results;
    }, [history]);

    const num = (v: number, digits = 2) => `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`;
    const tone = (v: number) => v >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400';

    return (
        <div className="metric-card p-5 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-cyan-500 opacity-80" />
            <div className="flex items-center gap-2 mb-4">
                <Scale className="w-3.5 h-3.5 text-cyan-500" />
                <h3 className="section-label">Vs Benchmark (1Y)</h3>
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
                                <th className="py-1.5 pl-2 text-right font-semibold" title="Cumulative excess return over 1Y">Excess</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map(r => (
                                <tr key={r.name} className="border-b border-border/30 last:border-0 hover:bg-muted/30">
                                    <td className="py-2 pr-3 font-bold text-foreground truncate max-w-[120px]">{r.name}</td>
                                    <td className={cn('py-2 px-2 text-right tabular-nums font-semibold', tone(r.alpha))}>{num(r.alpha)}%</td>
                                    <td className="py-2 px-2 text-right tabular-nums text-foreground">{r.beta.toFixed(2)}</td>
                                    <td className="py-2 px-2 text-right tabular-nums text-muted-foreground">{r.r2.toFixed(2)}</td>
                                    <td className="py-2 px-2 text-right tabular-nums text-muted-foreground">{r.te.toFixed(1)}%</td>
                                    <td className={cn('py-2 px-2 text-right tabular-nums font-semibold', tone(r.ir))}>{num(r.ir)}</td>
                                    <td className={cn('py-2 pl-2 text-right tabular-nums font-semibold', tone(r.excess))}>{num(r.excess, 1)}%</td>
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
