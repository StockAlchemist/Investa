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
        <div className="metric-card p-4">
            {/* Activity counts — one line of inline stats, wrapping on a phone */}
            <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1.5">
                <span className="section-label">Activity</span>
                <Count value={counts.total} label="transactions" />
                {tradeCount > 0 && (
                    <span className="inline-flex items-baseline gap-1.5">
                        <Count value={counts.buy} label="buys" />
                        <span className="text-muted-foreground/40">/</span>
                        <Count value={counts.sell} label="sells" />
                    </span>
                )}
                {incomeCount > 0 && (
                    <span className="inline-flex items-baseline gap-1.5">
                        <Count value={counts.dividend} label="div" tone="text-up" />
                        {counts.interest > 0 && (
                            <>
                                <span className="text-muted-foreground/40">·</span>
                                <Count value={counts.interest} label="int" tone="text-up" />
                            </>
                        )}
                    </span>
                )}
                {cashEventCount > 0 && <Count value={cashEventCount} label="cash flows" />}
            </div>

            {/* Per-currency ledger — one row per currency, columns shared so the
                figures line up. Each row is its own grid with the same track
                template, so the header and every row agree. Below `sm` a row
                stacks: currency + net on one line, the four figures under it. */}
            {rows.length > 0 && (
                <div className="mt-3 border-t border-border">
                    <div className={cn(ROW_GRID, 'hidden sm:grid pt-3 pb-1.5 text-[11px] text-muted-foreground')}>
                        <span>Currency</span>
                        <span className="text-right">Net cash flow</span>
                        <span className="text-right">In</span>
                        <span className="text-right">Out</span>
                        <span className="text-right">Fees</span>
                        <span className="text-right">Tax</span>
                    </div>
                    <div className="divide-y divide-border">
                        {rows.map(row => {
                            const positive = row.netFlow >= 0;
                            const NetArrow = positive ? ArrowDownRight : ArrowUpRight;
                            return (
                                <div
                                    key={row.currency}
                                    className={cn(ROW_GRID, 'grid-cols-4 gap-y-2 py-3 sm:py-2.5 sm:items-center')}
                                >
                                    <div className="col-span-4 flex items-center justify-between gap-3 sm:contents">
                                        <span className="justify-self-start text-[10px] uppercase tracking-widest font-semibold bg-muted text-foreground px-1.5 py-0.5 rounded">
                                            {row.currency}
                                        </span>
                                        <span className={cn(
                                            'inline-flex items-center justify-end gap-1 whitespace-nowrap text-lg font-semibold tabular-nums leading-none',
                                            positive ? 'text-up' : 'text-down',
                                        )}>
                                            <NetArrow className="w-4 h-4 opacity-80 shrink-0" />
                                            {positive ? '+' : '−'}{formatAmount(Math.abs(row.netFlow))}
                                        </span>
                                    </div>
                                    <Figure label="In" value={row.inflow} />
                                    <Figure label="Out" value={row.outflow} />
                                    <Figure label="Fees" value={row.fees} tone="text-warn-ink" />
                                    <Figure label="Tax" value={row.tax} tone="text-warn-ink" />
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}

// Currency tag, net figure, then In / Out / Fees / Tax. Net gets the widest
// track because it carries the arrow and the larger type.
const ROW_GRID = 'grid gap-x-4 sm:grid-cols-[4rem_minmax(0,1.4fr)_repeat(4,minmax(0,1fr))]';

function Count({ value, label, tone = 'text-foreground' }: { value: number; label: string; tone?: string }) {
    return (
        <span className="inline-flex items-baseline gap-1.5 whitespace-nowrap">
            <span className={cn('text-sm font-semibold tabular-nums', tone)}>{value.toLocaleString()}</span>
            <span className="text-[11px] text-muted-foreground">{label}</span>
        </span>
    );
}

// One figure cell. The label only shows on the stacked (phone) layout — on the
// wide layout the header row names the column.
function Figure({ label, value, tone = 'text-foreground' }: { label: string; value: number; tone?: string }) {
    const has = value > 0.001;
    return (
        <div className="min-w-0 sm:text-right">
            <div className="section-label sm:hidden">{label}</div>
            <div className={cn(
                'text-sm font-medium tabular-nums whitespace-nowrap',
                has ? tone : 'text-muted-foreground/40',
            )}>
                {has ? formatAmount(value) : '—'}
            </div>
        </div>
    );
}
