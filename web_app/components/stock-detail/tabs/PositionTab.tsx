'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Wallet,
    TrendingUp,
    DollarSign,
    Hash,
    Tag,
    PieChart as PieChartIcon,
    Activity as LucideActivity,
    Clock,
    Scale,
    ShieldAlert,
    CheckCircle2,
    Calendar,
    ArrowUpRight,
    ArrowDownRight,
} from 'lucide-react';
import { fetchStockPosition, StockPositionData } from '../../../lib/api';
import { formatCurrency, cn } from '../../../lib/utils';
import { Skeleton } from '../../ui/skeleton';

interface PositionTabProps {
    symbol: string;
    currency: string;
    accounts?: string[];
}

function PositionKpiCard({
    icon: Icon,
    label,
    value,
    subValue,
    subValueColor,
    valueColor,
    iconColor = "text-indigo-500",
    bgTint,
}: {
    icon: React.ElementType;
    label: string;
    value: string;
    subValue?: string;
    subValueColor?: string;
    valueColor?: string;
    iconColor?: string;
    bgTint?: string;
}) {
    return (
        <div className={cn(
            "bg-muted/60 hover:bg-muted/80 p-3.5 rounded-2xl flex flex-col justify-between gap-1.5 transition-all relative overflow-hidden border border-border/40 min-w-0 shadow-xs",
            bgTint
        )}>
            {/* Header: Icon + Label */}
            <div className="flex items-center gap-2 min-w-0">
                <div className={cn("p-1.5 rounded-lg bg-card/80 shrink-0 shadow-2xs", iconColor)}>
                    <Icon className="w-3.5 h-3.5" />
                </div>
                <span className="text-[10.5px] font-semibold text-muted-foreground uppercase tracking-wider truncate" title={label}>
                    {label}
                </span>
            </div>

            {/* Value + SubValue */}
            <div className="flex items-baseline justify-between gap-1.5 flex-wrap min-w-0 pt-0.5">
                <span 
                    className={cn("text-sm sm:text-base lg:text-lg font-bold tracking-tight tabular-nums truncate", valueColor || "text-foreground")}
                    title={value}
                >
                    {value}
                </span>
                {subValue && (
                    <span className={cn("text-xs font-semibold tabular-nums whitespace-nowrap shrink-0", subValueColor)}>
                        {subValue}
                    </span>
                )}
            </div>
        </div>
    );
}

export const PositionTab: React.FC<PositionTabProps> = ({
    symbol,
    currency,
    accounts = [],
}) => {
    const { data, isLoading, error } = useQuery<StockPositionData>({
        queryKey: ['stock-position', symbol, currency, accounts],
        queryFn: () => fetchStockPosition(symbol, currency, accounts),
        staleTime: 60 * 1000,
    });

    if (isLoading) {
        return (
            <div className="space-y-6 animate-pulse">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Array.from({ length: 4 }).map((_, i) => (
                        <Skeleton key={i} className="h-24 rounded-2xl" />
                    ))}
                </div>
                <Skeleton className="h-36 rounded-2xl" />
                <Skeleton className="h-64 rounded-2xl" />
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="bg-destructive/10 text-destructive p-6 rounded-2xl flex items-center gap-3">
                <ShieldAlert className="w-6 h-6 shrink-0" />
                <div>
                    <h4 className="font-semibold">Unable to load position data</h4>
                    <p className="text-sm opacity-90">
                        {error instanceof Error ? error.message : 'An error occurred while fetching position data.'}
                    </p>
                </div>
            </div>
        );
    }

    if (!data.has_position) {
        return (
            <div className="bg-muted/40 rounded-2xl p-8 text-center space-y-3">
                <div className="w-12 h-12 rounded-full bg-indigo-500/10 text-indigo-500 mx-auto flex items-center justify-center">
                    <Wallet className="w-6 h-6" />
                </div>
                <h3 className="text-base font-semibold">No Position in {symbol}</h3>
                <p className="text-sm text-muted-foreground max-w-md mx-auto">
                    You currently have no recorded buy or sell transactions for {symbol} in the selected accounts.
                </p>
            </div>
        );
    }

    const { summary, returns, open_lots = [], closed_trades = [] } = data;

    const qty = summary?.quantity || 0;
    const mktVal = summary?.market_value || 0;
    const avgCost = summary?.avg_cost_price || 0;
    const costBasis = summary?.cost_basis || 0;
    const unrealGain = returns?.unrealized_gain || 0;
    const unrealGainPct = returns?.unrealized_gain_pct || 0;
    const realGain = returns?.realized_gain || 0;
    const totalGain = returns?.total_gain || 0;
    const totalReturnPct = returns?.total_return_pct || 0;
    const divs = returns?.lifetime_dividends || 0;
    const irr = returns?.irr_pct;
    const yoc = returns?.yield_on_cost_pct;
    const mktYield = returns?.market_yield_pct;
    const fxGain = returns?.fx_gain_loss || 0;
    const fxGainPct = returns?.fx_gain_loss_pct || 0;
    const commissions = returns?.commissions || 0;
    const taxes = returns?.withholding_taxes || 0;

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* 1. Top Position KPI Grid */}
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <h3 className="text-base font-semibold flex items-center gap-2">
                        <Wallet className="w-4 h-4 text-indigo-500" />
                        Position Overview
                    </h3>
                    {summary?.portfolio_weight_pct != null && summary.portfolio_weight_pct > 0 && (
                        <div className="text-xs font-semibold px-2.5 py-1 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
                            {summary.portfolio_weight_pct.toFixed(2)}% of Portfolio
                        </div>
                    )}
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                    <PositionKpiCard
                        label="Shares Held"
                        value={qty.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                        icon={Hash}
                        iconColor="text-indigo-500"
                    />
                    <PositionKpiCard
                        label="Avg Cost"
                        value={formatCurrency(avgCost, currency)}
                        icon={Tag}
                        iconColor="text-slate-500"
                    />
                    <PositionKpiCard
                        label="Market Value"
                        value={formatCurrency(mktVal, currency)}
                        icon={PieChartIcon}
                        iconColor="text-indigo-500"
                    />
                    <PositionKpiCard
                        label="Cost Basis"
                        value={formatCurrency(costBasis, currency)}
                        icon={Scale}
                        iconColor="text-slate-500"
                    />
                    <PositionKpiCard
                        label="Unrealized G/L"
                        value={formatCurrency(unrealGain, currency)}
                        subValue={unrealGainPct === Infinity ? '∞' : `${unrealGainPct >= 0 ? '+' : ''}${unrealGainPct.toFixed(2)}%`}
                        subValueColor={unrealGainPct >= 0 ? 'text-emerald-500' : 'text-rose-500'}
                        valueColor={unrealGain >= 0 ? 'text-emerald-500' : 'text-rose-500'}
                        icon={LucideActivity}
                        iconColor={unrealGain >= 0 ? 'text-emerald-500' : 'text-rose-500'}
                        bgTint={unrealGain >= 0 ? 'bg-emerald-500/5 hover:bg-emerald-500/10' : 'bg-rose-500/5 hover:bg-rose-500/10'}
                    />
                    <PositionKpiCard
                        label="Total Return"
                        value={formatCurrency(totalGain, currency)}
                        subValue={totalReturnPct === Infinity ? '∞' : `${totalReturnPct >= 0 ? '+' : ''}${totalReturnPct.toFixed(2)}%`}
                        subValueColor={totalReturnPct >= 0 ? 'text-emerald-500' : 'text-rose-500'}
                        valueColor={totalGain >= 0 ? 'text-emerald-500' : 'text-rose-500'}
                        icon={TrendingUp}
                        iconColor={totalReturnPct >= 0 ? 'text-emerald-500' : 'text-rose-500'}
                        bgTint={totalReturnPct >= 0 ? 'bg-emerald-500/5 hover:bg-emerald-500/10' : 'bg-rose-500/5 hover:bg-rose-500/10'}
                    />
                    <PositionKpiCard
                        label="IRR (Annualized)"
                        value={irr != null ? `${irr >= 0 ? '+' : ''}${irr.toFixed(2)}%` : '—'}
                        icon={TrendingUp}
                        iconColor={irr != null && irr >= 0 ? 'text-emerald-500' : irr != null ? 'text-rose-500' : 'text-slate-500'}
                        valueColor={irr != null && irr >= 0 ? 'text-emerald-500' : irr != null ? 'text-rose-500' : undefined}
                    />
                    <PositionKpiCard
                        label="Yield on Cost"
                        value={yoc != null ? `${yoc.toFixed(2)}%` : '—'}
                        subValue={mktYield != null ? `Mkt: ${mktYield.toFixed(2)}%` : undefined}
                        icon={DollarSign}
                        iconColor="text-amber-500"
                    />
                    <PositionKpiCard
                        label="Lifetime Dividends"
                        value={formatCurrency(divs, currency)}
                        icon={DollarSign}
                        iconColor="text-amber-500"
                    />
                    <PositionKpiCard
                        label="Realized G/L"
                        value={formatCurrency(realGain, currency)}
                        icon={CheckCircle2}
                        iconColor={realGain >= 0 ? 'text-emerald-500' : 'text-rose-500'}
                        valueColor={realGain >= 0 ? 'text-emerald-500' : 'text-rose-500'}
                    />
                </div>
            </div>

            {/* 2. Return Attribution Waterfall Card */}
            <div className="bg-muted/50 border border-border/50 rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Scale className="w-4 h-4 text-indigo-500" />
                        Return Attribution Breakdown
                    </h3>
                    <span className="text-xs font-semibold text-muted-foreground">
                        Total Economic Gain: {formatCurrency(totalGain, currency)} ({totalReturnPct >= 0 ? '+' : ''}{totalReturnPct.toFixed(2)}%)
                    </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                    <div className="p-3 rounded-xl bg-background border border-border/40 space-y-1">
                        <div className="text-muted-foreground flex items-center justify-between">
                            <span>Capital Appreciation</span>
                            {unrealGain + realGain >= 0 ? (
                                <ArrowUpRight className="w-3.5 h-3.5 text-emerald-500" />
                            ) : (
                                <ArrowDownRight className="w-3.5 h-3.5 text-rose-500" />
                            )}
                        </div>
                        <div className="font-semibold text-sm">
                            {formatCurrency(unrealGain + realGain, currency)}
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                            Unreal: {formatCurrency(unrealGain, currency)} · Real: {formatCurrency(realGain, currency)}
                        </div>
                    </div>

                    <div className="p-3 rounded-xl bg-background border border-border/40 space-y-1">
                        <div className="text-muted-foreground flex items-center justify-between">
                            <span>Dividend Income</span>
                            <DollarSign className="w-3.5 h-3.5 text-amber-500" />
                        </div>
                        <div className="font-semibold text-sm text-emerald-600 dark:text-emerald-400">
                            +{formatCurrency(divs, currency)}
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                            YoC: {yoc != null ? `${yoc.toFixed(2)}%` : '—'}
                        </div>
                    </div>

                    <div className="p-3 rounded-xl bg-background border border-border/40 space-y-1">
                        <div className="text-muted-foreground flex items-center justify-between">
                            <span>Currency (FX) Impact</span>
                            <LucideActivity className="w-3.5 h-3.5 text-indigo-500" />
                        </div>
                        <div className={`font-semibold text-sm ${fxGain >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                            {fxGain >= 0 ? '+' : ''}{formatCurrency(fxGain, currency)}
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                            {fxGainPct >= 0 ? '+' : ''}{fxGainPct.toFixed(2)}% on cost
                        </div>
                    </div>

                    <div className="p-3 rounded-xl bg-background border border-border/40 space-y-1">
                        <div className="text-muted-foreground flex items-center justify-between">
                            <span>Fees & Tax Friction</span>
                            <ShieldAlert className="w-3.5 h-3.5 text-rose-500" />
                        </div>
                        <div className="font-semibold text-sm text-rose-600 dark:text-rose-400">
                            -{formatCurrency(commissions + taxes, currency)}
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                            Fees: {formatCurrency(commissions, currency)} · Tax: {formatCurrency(taxes, currency)}
                        </div>
                    </div>
                </div>
            </div>

            {/* 3. Open FIFO Lots Table */}
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <h3 className="text-base font-semibold flex items-center gap-2">
                        <Clock className="w-4 h-4 text-indigo-500" />
                        Open FIFO Tax Lots ({open_lots.length})
                    </h3>
                    <span className="text-xs text-muted-foreground">
                        {open_lots.reduce((acc, l) => acc + l.quantity, 0).toFixed(4)} total shares
                    </span>
                </div>

                {open_lots.length > 0 ? (
                    <div className="overflow-x-auto rounded-2xl border border-border/50">
                        <table className="w-full text-xs text-left">
                            <thead className="bg-muted/70 text-muted-foreground uppercase text-[10px] tracking-wider font-semibold border-b border-border/50">
                                <tr>
                                    <th className="py-3 px-4 whitespace-nowrap">Purchase Date</th>
                                    <th className="py-3 px-4 whitespace-nowrap">Account</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Shares</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Unit Cost</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Cost Basis</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Current Value</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Unrealized G/L</th>
                                    <th className="py-3 px-4 text-center whitespace-nowrap">Holding Period</th>
                                    <th className="py-3 px-4 text-center whitespace-nowrap">Term</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/40">
                                {open_lots.map((lot, idx) => (
                                    <tr key={lot.lot_id || idx} className="hover:bg-muted/30 transition-colors">
                                        <td className="py-3 px-4 font-medium whitespace-nowrap">
                                            <div className="inline-flex items-center gap-1.5 whitespace-nowrap">
                                                <Calendar className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                                                <span className="tabular-nums">{lot.date}</span>
                                            </div>
                                        </td>
                                        <td className="py-3 px-4 text-muted-foreground whitespace-nowrap">{lot.account}</td>
                                        <td className="py-3 px-4 text-right font-medium tabular-nums whitespace-nowrap">
                                            {lot.quantity.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                                        </td>
                                        <td className="py-3 px-4 text-right tabular-nums whitespace-nowrap">
                                            {formatCurrency(lot.cost_per_share_local, data.local_currency)}
                                        </td>
                                        <td className="py-3 px-4 text-right tabular-nums whitespace-nowrap">
                                            {formatCurrency(lot.cost_basis_display, currency)}
                                        </td>
                                        <td className="py-3 px-4 text-right font-semibold tabular-nums whitespace-nowrap">
                                            {formatCurrency(lot.market_value_display, currency)}
                                        </td>
                                        <td className="py-3 px-4 text-right tabular-nums whitespace-nowrap">
                                            <span className={lot.unrealized_gain_display >= 0 ? 'text-emerald-600 dark:text-emerald-400 font-semibold' : 'text-rose-600 dark:text-rose-400 font-semibold'}>
                                                {lot.unrealized_gain_display >= 0 ? '+' : ''}{formatCurrency(lot.unrealized_gain_display, currency)}
                                                <span className="text-[10px] ml-1 font-normal opacity-85">
                                                    ({lot.unrealized_gain_pct >= 0 ? '+' : ''}{lot.unrealized_gain_pct.toFixed(2)}%)
                                                </span>
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-center tabular-nums text-muted-foreground whitespace-nowrap">
                                            {lot.holding_period_days} days
                                        </td>
                                        <td className="py-3 px-4 text-center whitespace-nowrap">
                                            <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider ${
                                                lot.tax_term === 'long_term'
                                                    ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20'
                                                    : 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20'
                                            }`}>
                                                {lot.tax_term === 'long_term' ? 'Long-Term' : 'Short-Term'}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <p className="text-xs text-muted-foreground">No open lots (all positions closed).</p>
                )}
            </div>

            {/* 4. Closed Trades & Realized Gains Ledger */}
            {closed_trades.length > 0 && (
                <div className="space-y-3">
                    <h3 className="text-base font-semibold flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-indigo-500" />
                        Closed Trades & Realized Sells ({closed_trades.length})
                    </h3>

                    <div className="overflow-x-auto rounded-2xl border border-border/50">
                        <table className="w-full text-xs text-left">
                            <thead className="bg-muted/70 text-muted-foreground uppercase text-[10px] tracking-wider font-semibold border-b border-border/50">
                                <tr>
                                    <th className="py-3 px-4 whitespace-nowrap">Sale Date</th>
                                    <th className="py-3 px-4 whitespace-nowrap">Account</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Shares Sold</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Sale Price</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Net Proceeds</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Cost Basis</th>
                                    <th className="py-3 px-4 text-right whitespace-nowrap">Realized Gain / Loss</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/40">
                                {closed_trades.map((trade, idx) => (
                                    <tr key={trade.original_tx_id || idx} className="hover:bg-muted/30 transition-colors">
                                        <td className="py-3 px-4 font-medium whitespace-nowrap">
                                            <div className="inline-flex items-center gap-1.5 whitespace-nowrap">
                                                <Calendar className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                                                <span className="tabular-nums">{trade.sell_date}</span>
                                            </div>
                                        </td>
                                        <td className="py-3 px-4 text-muted-foreground whitespace-nowrap">{trade.account}</td>
                                        <td className="py-3 px-4 text-right tabular-nums font-medium whitespace-nowrap">
                                            {trade.quantity_sold.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                                        </td>
                                        <td className="py-3 px-4 text-right tabular-nums whitespace-nowrap">
                                            {formatCurrency(trade.sale_price, data.local_currency)}
                                        </td>
                                        <td className="py-3 px-4 text-right tabular-nums font-semibold whitespace-nowrap">
                                            {formatCurrency(trade.proceeds_display, currency)}
                                        </td>
                                        <td className="py-3 px-4 text-right tabular-nums text-muted-foreground whitespace-nowrap">
                                            {formatCurrency(trade.cost_basis_display, currency)}
                                        </td>
                                        <td className="py-3 px-4 text-right tabular-nums whitespace-nowrap">
                                            <span className={trade.realized_gain_display >= 0 ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-rose-600 dark:text-rose-400 font-bold'}>
                                                {trade.realized_gain_display >= 0 ? '+' : ''}{formatCurrency(trade.realized_gain_display, currency)}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
};
