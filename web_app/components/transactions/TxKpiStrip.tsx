'use client';
import React, { useMemo } from 'react';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { Transaction } from '../../lib/api';
import { cn } from '../../lib/utils';

interface TxKpiStripProps {
    transactions: Transaction[];
    /**
     * The user's selected display currency. Used only to sort the per-currency
     * cards (preferred currency first). Sums are never converted — currencies
     * are reported separately because we have no per-transaction FX.
     */
    preferredCurrency?: string;
}

// Compact currency formatter — returns just the number; the currency code is
// rendered separately in the card chrome.
function formatAmount(value: number): string {
    const abs = Math.abs(value);
    if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
    if (abs >= 10_000)    return `${(value / 1_000).toFixed(1)}K`;
    if (abs >= 100)       return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

interface CurrencyBucket {
    count: number;
    inflow: number;
    outflow: number;
    fees: number;
    tax: number;
    traded: number;
}

interface CurrencyRow extends CurrencyBucket {
    currency: string;
    netFlow: number;
}

interface ActivityCounts {
    total: number;
    buy: number;
    sell: number;
    dividend: number;
    interest: number;
    deposit: number;
    withdrawal: number;
    tax: number;
    fees: number;
    other: number;
}

export default function TxKpiStrip({ transactions, preferredCurrency }: TxKpiStripProps) {
    const { counts, rows } = useMemo(() => {
        const counts: ActivityCounts = {
            total: 0, buy: 0, sell: 0, dividend: 0, interest: 0,
            deposit: 0, withdrawal: 0, tax: 0, fees: 0, other: 0,
        };

        // Per-currency ledgers — never combined; we have no per-transaction FX.
        const byCurrency = new Map<string, CurrencyBucket>();
        const getCcy = (c: string): CurrencyBucket => {
            if (!byCurrency.has(c)) {
                byCurrency.set(c, { count: 0, inflow: 0, outflow: 0, fees: 0, tax: 0, traded: 0 });
            }
            return byCurrency.get(c)!;
        };

        for (const tx of transactions) {
            counts.total += 1;
            const t = (tx.Type || '').toLowerCase();
            const ccy = (tx['Local Currency'] || 'USD').toUpperCase();
            const amount = Math.abs(tx['Total Amount'] || 0);
            const fee = Math.abs(tx.Commission || 0);

            if (t === 'buy') counts.buy += 1;
            else if (t === 'sell') counts.sell += 1;
            else if (t === 'dividend') counts.dividend += 1;
            else if (t === 'interest') counts.interest += 1;
            else if (t === 'deposit') counts.deposit += 1;
            else if (t === 'withdrawal') counts.withdrawal += 1;
            else if (t === 'tax') counts.tax += 1;
            else if (t === 'fees') counts.fees += 1;
            else counts.other += 1;

            // Cash flow: only non-trade money movements; buys/sells are
            // self-canceling against their cash leg.
            const bucket = getCcy(ccy);
            bucket.count += 1;
            if (t === 'deposit' || t === 'dividend' || t === 'interest') {
                bucket.inflow += amount;
            } else if (t === 'withdrawal') {
                bucket.outflow += amount;
            } else if (t === 'tax') {
                bucket.tax += amount;
                bucket.outflow += amount;
            } else if (t === 'fees') {
                bucket.fees += amount;
                bucket.outflow += amount;
            } else if (t === 'buy' || t === 'sell') {
                bucket.traded += amount;
            }
            if (fee > 0 && t !== 'fees') bucket.fees += fee;
        }

        // Only surface currencies that have cash-side activity (skip those that
        // only appear in cash-neutral buy/sell pairs).
        const rows: CurrencyRow[] = Array.from(byCurrency.entries())
            .map(([currency, b]) => ({ currency, netFlow: b.inflow - b.outflow, ...b }))
            .filter(r => Math.abs(r.netFlow) > 0.001 || r.fees > 0.001 || r.tax > 0.001
                       || r.inflow > 0.001 || r.outflow > 0.001);

        // Sort: preferred currency first, then most-active by count.
        const preferred = preferredCurrency?.toUpperCase();
        rows.sort((a, b) => {
            if (preferred) {
                if (a.currency === preferred && b.currency !== preferred) return -1;
                if (b.currency === preferred && a.currency !== preferred) return 1;
            }
            return b.count - a.count;
        });

        return { counts, rows };
    }, [transactions, preferredCurrency]);

    const tradeCount = counts.buy + counts.sell;
    const incomeCount = counts.dividend + counts.interest;
    const cashEventCount = counts.deposit + counts.withdrawal;

    return (
        <div className="metric-card p-4 space-y-4">
            {/* Activity counts — spread across the card width */}
            <div className="flex flex-wrap justify-between items-baseline gap-x-4 gap-y-1.5">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">
                    Activity
                </span>
                <span className="inline-flex items-baseline gap-1.5">
                    <span className="text-base font-bold tabular-nums text-foreground">
                        {counts.total.toLocaleString()}
                    </span>
                    <span className="text-[11px] text-muted-foreground">transactions</span>
                </span>
                {tradeCount > 0 && (
                    <span className="inline-flex items-baseline gap-1.5">
                        <span className="text-sm font-bold tabular-nums text-foreground">
                            {counts.buy.toLocaleString()}
                        </span>
                        <span className="text-[11px] text-muted-foreground">buys</span>
                        <span className="text-muted-foreground/40">/</span>
                        <span className="text-sm font-bold tabular-nums text-foreground">
                            {counts.sell.toLocaleString()}
                        </span>
                        <span className="text-[11px] text-muted-foreground">sells</span>
                    </span>
                )}
                {incomeCount > 0 && (
                    <span className="inline-flex items-baseline gap-1.5">
                        <span className="text-sm font-bold tabular-nums text-emerald-600 dark:text-emerald-400">
                            {counts.dividend.toLocaleString()}
                        </span>
                        <span className="text-[11px] text-muted-foreground">div</span>
                        {counts.interest > 0 && (
                            <>
                                <span className="text-muted-foreground/40">·</span>
                                <span className="text-sm font-bold tabular-nums text-emerald-600 dark:text-emerald-400">
                                    {counts.interest.toLocaleString()}
                                </span>
                                <span className="text-[11px] text-muted-foreground">int</span>
                            </>
                        )}
                    </span>
                )}
                {cashEventCount > 0 && (
                    <span className="inline-flex items-baseline gap-1.5">
                        <span className="text-sm font-bold tabular-nums text-foreground">
                            {cashEventCount.toLocaleString()}
                        </span>
                        <span className="text-[11px] text-muted-foreground">cash flows</span>
                    </span>
                )}
            </div>

            {/* Per-currency cards — auto-fit grid so cards stretch to fill the
                full row width regardless of how many currencies are present. */}
            {rows.length > 0 && (
                <div className="grid gap-3 grid-cols-[repeat(auto-fit,minmax(260px,1fr))]">
                    {rows.map(row => {
                        const positive = row.netFlow >= 0;
                        const netTone = positive
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : 'text-red-600 dark:text-red-400';
                        const NetArrow = positive ? ArrowDownRight : ArrowUpRight;
                        return (
                            <div
                                key={row.currency}
                                className="relative rounded-lg border border-border/60 bg-card/40 p-4"
                            >
                                {/* Currency tag */}
                                <span className="absolute top-3 right-3 text-[10px] uppercase tracking-widest font-bold bg-muted/70 text-foreground px-2 py-0.5 rounded">
                                    {row.currency}
                                </span>

                                {/* Hero: net cash flow */}
                                <div className="pr-14">
                                    <div className={cn('inline-flex items-center gap-1.5 text-2xl font-bold tabular-nums leading-none', netTone)}>
                                        <NetArrow className="w-5 h-5 opacity-80 shrink-0" />
                                        <span>
                                            {positive ? '+' : '−'}{formatAmount(Math.abs(row.netFlow))}
                                        </span>
                                    </div>
                                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold mt-1">
                                        net cash flow
                                    </div>
                                </div>

                                {/* In / Out */}
                                <div className="grid grid-cols-2 gap-3 mt-4 pt-3 border-t border-border/40">
                                    <div>
                                        <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold mb-0.5">In</div>
                                        <div className={cn(
                                            'text-sm font-bold tabular-nums',
                                            row.inflow > 0 ? 'text-foreground' : 'text-muted-foreground/40',
                                        )}>
                                            {row.inflow > 0 ? formatAmount(row.inflow) : '—'}
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold mb-0.5">Out</div>
                                        <div className={cn(
                                            'text-sm font-bold tabular-nums',
                                            row.outflow > 0 ? 'text-foreground' : 'text-muted-foreground/40',
                                        )}>
                                            {row.outflow > 0 ? formatAmount(row.outflow) : '—'}
                                        </div>
                                    </div>
                                </div>

                                {/* Fees / Tax */}
                                <div className="grid grid-cols-2 gap-3 mt-3 pt-2 border-t border-border/40">
                                    <div className="flex items-baseline gap-1.5">
                                        <span className="text-[9px] uppercase tracking-wider text-muted-foreground/60 font-semibold">Fees</span>
                                        <span className={cn(
                                            'text-xs font-bold tabular-nums',
                                            row.fees > 0 ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground/40',
                                        )}>
                                            {row.fees > 0 ? formatAmount(row.fees) : '—'}
                                        </span>
                                    </div>
                                    <div className="flex items-baseline justify-end gap-1.5">
                                        <span className="text-[9px] uppercase tracking-wider text-muted-foreground/60 font-semibold">Tax</span>
                                        <span className={cn(
                                            'text-xs font-bold tabular-nums',
                                            row.tax > 0 ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground/40',
                                        )}>
                                            {row.tax > 0 ? formatAmount(row.tax) : '—'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
