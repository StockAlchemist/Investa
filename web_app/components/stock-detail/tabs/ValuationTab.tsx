import React, { useState } from 'react';
import { useTheme } from 'next-themes';
import {
    TrendingUp,
    Scale,
    AlertCircle,
    Info,
    CheckCircle2,
    BarChart3,
    X,
    HelpCircle
} from 'lucide-react';
import {
    ResponsiveContainer,
    AreaChart,
    Area,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    ReferenceLine
} from 'recharts';
import { Badge } from '../../ui/badge';
import { cn, formatCurrency, formatPercent as formatPercentShared } from '../../../lib/utils';

interface ValuationTabProps {
    symbol: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- intrinsicValue payload
    intrinsicValue: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- fundamentals payload
    fundamentals: any;
    currency: string;
    fxRate: number;
}

const VALUATION_INFO = {
    discount_rate: {
        description: "The rate used to discount future cash flows to their present value. High rate = lower valuation.",
        default: "Calculated WACC (~10%)"
    },
    growth_rate: {
        description: "Expected annual growth of cash flows during the projection years.",
        default: "Historical CAGR"
    },
    terminal_growth: {
        description: "Long-term growth rate after the projection period (stable stage).",
        default: "2.0%"
    },
    projection_years: {
        description: "Number of years to forecast explicit free cash flows.",
        default: "5 Years"
    },
    base_fcf: {
        description: "The starting free cash flow value for DCF projections.",
        default: "Latest TTM FCF"
    },
    eps: {
        description: "Earnings per share used as the base for the Graham Formula.",
        default: "TTM EPS"
    },
    graham_growth: {
        description: "Expected annual growth (g) used in Graham's Formula.",
        default: "Historical CAGR"
    },
    fcf_margin: {
        description: "Free Cash Flow as a percentage of revenue, used to normalize future cash flow projections if current FCF is an outlier.",
        default: "5-Year Average"
    },
    bond_yield: {
        description: "Current yield on high-quality bonds (proxy for risk-free rate).",
        default: "10Y Treasury (~4.5%)"
    }
};

const ParamItem = ({ label, value, info, className }: { label: string, value: React.ReactNode, info?: { description: string, default: string }, className?: string }) => (
    <div>
        <div className="flex items-center gap-1 mb-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold">{label}</p>
            {info && (
                <div className="group relative">
                    <HelpCircle className="w-2.5 h-2.5 text-muted-foreground/50 hover:text-indigo-500 cursor-help" />
                    <div className="absolute bottom-full left-0 mb-2 w-48 p-3 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[10px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100]">
                        {info.description}
                        <div className="mt-1 pt-1 font-bold text-indigo-600 dark:text-indigo-400">Default: {info.default}</div>
                    </div>
                </div>
            )}
        </div>
        <p className={cn("text-sm font-semibold", className)}>{value ?? '-'}</p>
    </div>
);

export const ValuationTab: React.FC<ValuationTabProps> = ({
    intrinsicValue,
    fundamentals,
    currency,
    fxRate
}) => {
    const [viewingDistribution, setViewingDistribution] = useState<'dcf' | 'graham' | null>(null);
    const { resolvedTheme } = useTheme();
    const isDarkMode = resolvedTheme === 'dark';

    if (!intrinsicValue) return null;
    const { models, average_intrinsic_value, margin_of_safety_pct, current_price } = intrinsicValue;
    const status = intrinsicValue.valuation_status;
    const hasValue = average_intrinsic_value !== null && average_intrinsic_value !== undefined;
    const isRefusal = status === "ineligible" || status === "no_model";

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Summary Header */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-muted p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                    <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">
                        {status === "nav" ? "Net Asset Value" : "Blended Intrinsic Value"}
                    </p>
                    {hasValue ? (
                        <p className="text-3xl font-bold text-indigo-500">{formatCurrency((average_intrinsic_value ?? 0) * fxRate, currency)}</p>
                    ) : (
                        <p className="text-2xl font-bold text-muted-foreground">Not valued</p>
                    )}
                    {hasValue && intrinsicValue?.range && (
                        <p className="text-xs text-muted-foreground mt-2 font-medium">
                            Range: {formatCurrency((intrinsicValue.range.bear ?? 0) * fxRate, currency)} - {formatCurrency((intrinsicValue.range.bull ?? 0) * fxRate, currency)}
                        </p>
                    )}
                    {hasValue && intrinsicValue.earnings_power_floor !== undefined && (
                        <p className="text-xs text-muted-foreground mt-2 font-medium">
                            No-growth floor: {formatCurrency(intrinsicValue.earnings_power_floor * fxRate, currency)}
                        </p>
                    )}
                </div>
                <div className="bg-muted p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                    <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Current Price</p>
                    <p className="text-3xl font-bold">{formatCurrency((current_price ?? 0) * fxRate, currency)}</p>
                </div>
                <div className={cn(
                    "p-6 rounded-2xl flex flex-col items-center justify-center text-center transition-all",
                    !hasValue
                        ? "bg-muted"
                        : (margin_of_safety_pct || 0) > 0
                            ? "bg-emerald-500/10 dark:bg-emerald-500/5"
                            : "bg-rose-500/10 dark:bg-rose-500/5"
                )}>
                    <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Margin of Safety</p>
                    {hasValue ? (
                        <p className={cn(
                            "text-3xl font-bold tracking-tight",
                            (margin_of_safety_pct || 0) > 0 ? "text-emerald-500" : "text-rose-500"
                        )}>
                            {(margin_of_safety_pct || 0).toFixed(2)}%
                        </p>
                    ) : (
                        <p className="text-2xl font-bold text-muted-foreground">&mdash;</p>
                    )}
                </div>
            </div>

            {intrinsicValue.valuation_note && (
                <div className={cn(
                    "p-4 rounded-2xl flex items-start gap-3 animate-in fade-in slide-in-from-top-2 duration-500",
                    isRefusal ? "bg-muted" : "bg-amber-500/10"
                )}>
                    <AlertCircle className={cn("w-5 h-5 shrink-0 mt-0.5", isRefusal ? "text-muted-foreground" : "text-amber-500")} />
                    <div className="flex flex-col gap-1">
                        <p className={cn(
                            "text-xs font-bold uppercase tracking-wider",
                            isRefusal ? "text-muted-foreground" : "text-amber-600 dark:text-amber-400"
                        )}>
                            {status === "no_model" ? "Cannot be valued"
                                : status === "ineligible" ? "Not eligible for valuation"
                                    : status === "clamped" ? "Output outside credible range"
                                        : status === "low_confidence" ? "Models disagree"
                                            : "Valuation note"}
                        </p>
                        <p className={cn(
                            "text-sm leading-relaxed italic",
                            isRefusal ? "text-muted-foreground" : "text-amber-700 dark:text-amber-500"
                        )}>{intrinsicValue.valuation_note}</p>
                    </div>
                </div>
            )}

            {/* Models Detail */}
            {!models.dcf.parameters && !models.graham.parameters ? (
                <div className="bg-muted rounded-2xl p-8 text-center animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="w-16 h-16 bg-indigo-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
                        <Info className="w-8 h-8 text-indigo-500" />
                    </div>
                    <h3 className="text-xl font-bold mb-3">Why standard models aren&apos;t shown?</h3>
                    <p className="text-muted-foreground text-sm leading-relaxed max-w-xl mx-auto mb-6">
                        Traditional valuation methods like <strong>Discounted Cash Flow (DCF)</strong> and <strong>Graham&apos;s Formula</strong> rely on free cash flow and earnings growth, which are company-specific metrics.
                        <br /><br />
                        For <strong>ETFs and Mutual Funds</strong>, the intrinsic value is best represented by the <strong>Net Asset Value (NAV)</strong>, which is the total value of the fund&apos;s assets minus its liabilities, divided by the number of outstanding shares.
                    </p>
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-background rounded-full text-xs font-medium text-foreground">
                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                        Using Industry Standard NAV Valuation
                    </div>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* DCF Model */}
                    <div className="bg-muted rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-semibold flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-emerald-500" />
                                {models.dcf.model}
                            </h3>
                            {models.dcf.intrinsic_value !== undefined && (
                                <div className="flex flex-col items-end">
                                    <Badge className="bg-emerald-500/20 text-emerald-500 border-none">
                                        {formatCurrency((models.dcf.intrinsic_value ?? 0) * fxRate, currency)}
                                    </Badge>
                                    {models.dcf.model !== 'DCF' && (
                                        <span className="text-[9px] text-muted-foreground mt-1">
                                            (Fallback Used)
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                        {models.dcf.error ? (
                            <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.dcf.error}</p>
                        ) : !models.dcf.parameters ? (
                            <div className="flex flex-col items-center justify-center py-10 text-center opacity-50">
                                <Info className="w-8 h-8 mb-2" />
                                <p className="text-sm">Not applicable for this asset type.</p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <ParamItem
                                        label="Discount Rate (WACC)"
                                        value={formatPercentShared(models.dcf.parameters.discount_rate ?? 0)}
                                        info={VALUATION_INFO.discount_rate}
                                    />
                                    <ParamItem
                                        label="Growth Rate"
                                        value={formatPercentShared(models.dcf.parameters.growth_rate ?? 0)}
                                        info={VALUATION_INFO.growth_rate}
                                    />
                                    {models.dcf.parameters.applied_growth !== undefined && models.dcf.parameters.applied_growth !== models.dcf.parameters.growth_rate && (
                                        <ParamItem
                                            label="Applied Growth"
                                            value={formatPercentShared(models.dcf.parameters.applied_growth ?? 0)}
                                            className="text-cyan-500 font-bold"
                                            info={{ description: "The growth rate actually used in the projection after applying physical reality caps.", default: "100% Max" }}
                                        />
                                    )}
                                    <ParamItem
                                        label="Terminal Growth"
                                        value={formatPercentShared(models.dcf.parameters.terminal_growth_rate ?? 0)}
                                        info={VALUATION_INFO.terminal_growth}
                                    />
                                    <ParamItem
                                        label="Projection Years"
                                        value={models.dcf.parameters.projection_years}
                                        info={VALUATION_INFO.projection_years}
                                    />
                                    {models.dcf.parameters.note && (
                                        <div className="col-span-2 text-xs text-muted-foreground italic bg-secondary/30 p-2 rounded">
                                            Note: {models.dcf.parameters.note}
                                        </div>
                                    )}
                                </div>
                                <div className="pt-4">
                                    <ParamItem
                                        label="Base Free Cash Flow"
                                        value={formatCurrency((models.dcf.parameters.base_fcf ?? 0) * fxRate, currency)}
                                        info={VALUATION_INFO.base_fcf}
                                    />
                                    {models.dcf.parameters.fcf_margin && (
                                        <ParamItem
                                            label="Est. FCF Margin"
                                            value={formatPercentShared(models.dcf.parameters.fcf_margin ?? 0)}
                                            info={VALUATION_INFO.fcf_margin}
                                        />
                                    )}
                                </div>
                                {models.dcf.mc && (
                                    <div className="pt-4">
                                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold mb-3">Probabilistic Scenarios (Monte Carlo)</p>
                                        <div className="grid grid-cols-3 gap-2">
                                            <div
                                                className="bg-rose-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-rose-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('dcf')}
                                            >
                                                <p className="text-[10px] text-rose-500 font-bold uppercase mb-1">Bear (10th)</p>
                                                <p className="text-sm font-bold">{formatCurrency((models.dcf.mc.bear ?? 0) * fxRate, currency)}</p>
                                            </div>
                                            <div
                                                className="bg-indigo-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-indigo-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('dcf')}
                                            >
                                                <p className="text-[10px] text-indigo-500 font-bold uppercase mb-1">Median (50th)</p>
                                                <p className="text-sm font-bold">{formatCurrency((models.dcf.mc.base ?? 0) * fxRate, currency)}</p>
                                            </div>
                                            <div
                                                className="bg-emerald-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-emerald-500/10 transition-colors group/mc"
                                                onClick={() => setViewingDistribution('dcf')}
                                            >
                                                <p className="text-[10px] text-emerald-500 font-bold uppercase mb-1">Bull (90th)</p>
                                                <p className="text-sm font-bold">{formatCurrency((models.dcf.mc.bull ?? 0) * fxRate, currency)}</p>
                                            </div>
                                        </div>
                                        <p className="text-[9px] text-muted-foreground mt-2 text-center opacity-50 group-hover/mc:opacity-100 transition-opacity">Click to view distribution</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Graham Model */}
                    <div className="bg-muted rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-semibold flex items-center gap-2">
                                <Scale className="w-5 h-5 text-amber-500" />
                                {models.graham.model}
                            </h3>
                            {models.graham.intrinsic_value !== undefined && (
                                <div className="flex flex-col items-end">
                                    <Badge className="bg-amber-500/20 text-amber-500 border-none">
                                        {formatCurrency((models.graham.intrinsic_value ?? 0) * fxRate, currency)}
                                    </Badge>
                                    {models.graham.model !== "Graham's Revised Formula" && (
                                        <span className="text-[9px] text-muted-foreground mt-1">
                                            (Fallback Used)
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                        {models.graham.error ? (
                            <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.graham.error}</p>
                        ) : !models.graham.parameters ? (
                            <div className="flex flex-col items-center justify-center py-10 text-center opacity-50">
                                <Info className="w-8 h-8 mb-2" />
                                <p className="text-sm">Not applicable for this asset type.</p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <ParamItem
                                        label="Trailing EPS"
                                        value={(models.graham.parameters.eps || 0).toFixed(2)}
                                        info={VALUATION_INFO.eps}
                                    />
                                    <ParamItem
                                        label="Growth Rate (g)"
                                        value={`${(models.graham.parameters.growth_rate_pct ?? 0).toFixed(2)}%`}
                                        info={VALUATION_INFO.graham_growth}
                                    />
                                    {models.graham.parameters.applied_growth_pct !== undefined && models.graham.parameters.applied_growth_pct !== models.graham.parameters.growth_rate_pct && (
                                        <ParamItem
                                            label="Applied Growth"
                                            value={`${(models.graham.parameters.applied_growth_pct || 0).toFixed(2)}%`}
                                            className="text-amber-500 font-bold"
                                            info={{ description: "The growth rate actually used in the formula after applying stability caps.", default: "30% Max" }}
                                        />
                                    )}
                                    <ParamItem
                                        label="Bond Yield (Y)"
                                        value={`${(models.graham.parameters.bond_yield_proxy || 0).toFixed(2)}%`}
                                        info={VALUATION_INFO.bond_yield}
                                    />
                                    {models.graham.parameters.note && (
                                        <div className="col-span-2 text-xs text-amber-600 dark:text-amber-400 bg-amber-500/10 p-3 rounded-xl flex items-start gap-2">
                                            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                                            <span>{models.graham.parameters.note}</span>
                                        </div>
                                    )}
                                </div>
                                <div className="mt-4 p-4 bg-secondary/5 rounded-xl flex flex-col items-center select-none overflow-visible">
                                    <div className="flex items-center gap-2">
                                        <div className="flex items-baseline gap-1">
                                            <div className="group relative">
                                                <span className="text-lg font-bold text-foreground cursor-help decoration-dotted underline-offset-4 hover:text-cyan-500 transition-colors">V</span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                    Intrinsic Value
                                                </div>
                                            </div>
                                            <span className="text-lg font-light opacity-30">=</span>
                                        </div>
                                        <div className="flex items-center gap-1.5 px-4">
                                            <div className="group relative">
                                                <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">EPS</span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                    Trailing 12-Month Earnings Per Share
                                                </div>
                                            </div>
                                            <span className="text-[9px] opacity-40">×</span>
                                            <div className="group relative">
                                                <span className="px-1.5 py-0.5 bg-secondary/30 rounded-md text-[10px] font-bold text-foreground cursor-help transition-colors">
                                                    8.5 + 2G
                                                </span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2.5 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-relaxed text-center font-medium">
                                                    <div className="mb-0.5"><span className="font-bold">8.5</span>: Base P/E for zero growth</div>
                                                    <div><span className="font-bold">G</span>: Expected long-term growth rate</div>
                                                </div>
                                            </div>
                                            <span className="text-[9px] opacity-40">×</span>
                                            <div className="group relative">
                                                <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">4.4</span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-40 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                    Average yield of high-grade corporate bonds in 1962
                                                </div>
                                            </div>
                                            <span className="text-[10px] opacity-40 mx-0.5">/</span>
                                            <div className="group relative">
                                                <span className="text-xs font-bold text-foreground cursor-help hover:text-cyan-500 transition-colors">Y</span>
                                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-44 p-2 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[9px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] leading-tight text-center font-medium">
                                                    <div className="font-bold mb-1">Y = {models.graham.parameters?.bond_yield_proxy || '4.5'}%</div>
                                                    Current yield on AAA corporate bonds
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {models.graham.mc && (
                                    <div className="mt-8 pt-4">
                                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold mb-3">Probabilistic Scenarios (Monte Carlo)</p>
                                        <div className="grid grid-cols-3 gap-2">
                                            <div
                                                className="bg-rose-500/5 relative overflow-hidden p-2 rounded-lg text-center cursor-pointer hover:bg-rose-500/10 transition-all group/mc"
                                                onClick={() => setViewingDistribution('graham')}
                                            >
                                                <div className="absolute -top-12 -right-12 w-24 h-24 blur-[30px] opacity-10 group-hover/mc:opacity-20 transition-opacity pointer-events-none bg-rose-500" />
                                                <p className="text-[10px] text-rose-500 font-bold uppercase mb-1 relative z-10">Bear (10th)</p>
                                                <p className="text-sm font-bold relative z-10">{formatCurrency(models.graham.mc.bear * fxRate, currency)}</p>
                                            </div>
                                            <div
                                                className="bg-amber-500/5 relative overflow-hidden p-2 rounded-lg text-center cursor-pointer hover:bg-amber-500/10 transition-all group/mc shadow-none"
                                                onClick={() => setViewingDistribution('graham')}
                                            >
                                                <div className="absolute -top-12 -right-12 w-24 h-24 blur-[30px] opacity-10 group-hover/mc:opacity-20 transition-opacity pointer-events-none bg-amber-500" />
                                                <p className="text-[10px] text-amber-500 font-bold uppercase mb-1 relative z-10">Median (50th)</p>
                                                <p className="text-sm font-bold relative z-10">{formatCurrency(models.graham.mc.base * fxRate, currency)}</p>
                                            </div>
                                            <div
                                                className="bg-emerald-500/5 relative overflow-hidden p-2 rounded-lg text-center cursor-pointer hover:bg-emerald-500/10 transition-all group/mc"
                                                onClick={() => setViewingDistribution('graham')}
                                            >
                                                <div className="absolute -top-12 -right-12 w-24 h-24 blur-[30px] opacity-10 group-hover/mc:opacity-20 transition-opacity pointer-events-none bg-emerald-500" />
                                                <p className="text-[10px] text-emerald-500 font-bold uppercase mb-1 relative z-10">Bull (90th)</p>
                                                <p className="text-sm font-bold relative z-10">{formatCurrency(models.graham.mc.bull * fxRate, currency)}</p>
                                            </div>
                                        </div>
                                        <p className="text-[9px] text-muted-foreground mt-2 text-center opacity-50 group-hover/mc:opacity-100 transition-opacity">Click to view distribution</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}

            <div className="bg-secondary/20 rounded-2xl p-6 italic text-sm text-muted-foreground text-center">
                Note: Intrinsic value calculations are estimates based on various assumptions.
                Actual stock performance may vary significantly.
            </div>

            {/* Distribution Modal */}
            {viewingDistribution && intrinsicValue && (
                <div className={cn(
                    "fixed inset-0 z-[110] flex items-center justify-center p-4 backdrop-blur-sm animate-in fade-in duration-300",
                    isDarkMode ? "bg-black/60" : "bg-slate-500/20"
                )}>
                    <div className={cn(
                        "w-full max-w-2xl rounded-3xl overflow-hidden animate-in zoom-in-95 duration-300",
                        isDarkMode ? "bg-slate-900 text-white" : "bg-white text-slate-900"
                    )}>
                        <div className={cn(
                            "p-6 flex items-center justify-between",
                            isDarkMode ? "bg-muted/30" : "bg-slate-50/50"
                        )}>
                            <div>
                                <h3 className="text-xl font-bold flex items-center gap-2 text-inherit">
                                    <BarChart3 className="w-5 h-5 text-indigo-500" />
                                    {viewingDistribution === 'dcf' ? 'DCF' : 'Graham'} Probabilistic Distribution
                                </h3>
                                <p className={isDarkMode ? "text-slate-400 text-sm" : "text-slate-500 text-sm"}>Monte Carlo Simulation (10,000 iterations)</p>
                            </div>
                            <button
                                onClick={() => setViewingDistribution(null)}
                                className={cn(
                                    "p-2 rounded-full transition-colors cursor-pointer",
                                    isDarkMode ? "hover:bg-slate-800" : "hover:bg-slate-100"
                                )}
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="p-6">
                            <div className="h-[300px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    {(() => {
                                        const modelKey = viewingDistribution;
                                        if (!modelKey) return null;

                                        const model = intrinsicValue.models[modelKey];
                                        const mc = model?.mc;
                                        const rawHistogram = mc?.histogram || [];

                                        if (rawHistogram.length === 0 || !mc) return null;

                                        const histogram = rawHistogram.map((h: { price: number; count: number }) => ({ ...h, price: h.price * fxRate }));
                                        const scaledMc = {
                                            bear: (mc.bear ?? 0) * fxRate,
                                            base: (mc.base ?? 0) * fxRate,
                                            bull: (mc.bull ?? 0) * fxRate
                                        };

                                        const minPrice = histogram[0].price;
                                        const maxPrice = histogram[histogram.length - 1].price;
                                        const currentPriceVal = (intrinsicValue.current_price || fundamentals?.regularMarketPrice || 0) * fxRate;

                                        const domainMin = Math.min(minPrice, currentPriceVal > 0 ? currentPriceVal * 0.95 : minPrice);
                                        const domainMax = Math.max(maxPrice, currentPriceVal > 0 ? currentPriceVal * 1.05 : maxPrice);

                                        const range = maxPrice - minPrice;
                                        const bearOffset = Math.max(0, Math.min(100, ((scaledMc.bear - minPrice) / range) * 100));
                                        const bullOffset = Math.max(0, Math.min(100, ((scaledMc.bull - minPrice) / range) * 100));

                                        return (
                                            <AreaChart
                                                data={histogram}
                                                margin={{ top: 35, right: 20, left: 20, bottom: 0 }}
                                            >
                                                <defs>
                                                    <linearGradient id="colorBellFill" x1="0" y1="0" x2="1" y2="0">
                                                        <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.4} />
                                                        <stop offset={`${bearOffset}%`} stopColor="#f43f5e" stopOpacity={0.4} />
                                                        <stop offset={`${bearOffset}%`} stopColor="#06b6d4" stopOpacity={0.4} />
                                                        <stop offset={`${bullOffset}%`} stopColor="#06b6d4" stopOpacity={0.4} />
                                                        <stop offset={`${bullOffset}%`} stopColor="#10b981" stopOpacity={0.4} />
                                                        <stop offset="100%" stopColor="#10b981" stopOpacity={0.4} />
                                                    </linearGradient>
                                                    <linearGradient id="colorBellFade" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="white" stopOpacity={1} />
                                                        <stop offset="95%" stopColor="white" stopOpacity={0} />
                                                    </linearGradient>
                                                    <mask id="bellMask">
                                                        <rect x="0" y="0" width="100%" height="100%" fill="url(#colorBellFade)" />
                                                    </mask>
                                                </defs>
                                                <CartesianGrid
                                                    strokeDasharray="3 3"
                                                    vertical={false}
                                                    stroke={isDarkMode ? "#334155" : "#e2e8f0"}
                                                    opacity={isDarkMode ? 0.3 : 0.8}
                                                />
                                                <XAxis
                                                    dataKey="price"
                                                    type="number"
                                                    domain={[domainMin, domainMax]}
                                                    tickFormatter={(val) => formatCurrency(val, currency)}
                                                    fontSize={10}
                                                    tickLine={false}
                                                    axisLine={false}
                                                    minTickGap={30}
                                                    stroke={isDarkMode ? "#94a3b8" : "#64748b"}
                                                />
                                                <YAxis hide />
                                                <Tooltip
                                                    wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                                    content={({ active, payload }) => {
                                                        if (active && payload && payload.length) {
                                                            const count = Number(payload[0].value);
                                                            const probability = (count / 10000) * 100;
                                                            return (
                                                                <div className="p-3 rounded-xl outline-none scale-105 transition-transform bg-white/95 dark:bg-slate-950/95 backdrop-blur-md">
                                                                    <p className={cn(
                                                                        "text-[10px] uppercase font-bold mb-1 tracking-wider",
                                                                        isDarkMode ? "text-slate-500" : "text-slate-400"
                                                                    )}>Estimated Value</p>
                                                                    <p className={cn(
                                                                        "text-lg font-black",
                                                                        isDarkMode ? "text-white" : "text-slate-900"
                                                                    )}>{formatCurrency(payload[0].payload.price, currency)}</p>

                                                                    <div className="flex flex-col gap-1 mt-3 pt-2">
                                                                        <div className="flex items-center justify-between gap-4">
                                                                            <div className="flex items-center gap-1.5">
                                                                                <div className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                                                                                <span className={cn("text-[10px] font-bold uppercase", isDarkMode ? "text-slate-400" : "text-slate-500")}>Probability</span>
                                                                            </div>
                                                                            <span className="text-[10px] font-black text-indigo-500">{probability.toFixed(2)}%</span>
                                                                        </div>
                                                                        <div className="flex items-center justify-between gap-4">
                                                                            <div className="flex items-center gap-1.5">
                                                                                <div className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                                                                                <span className={cn("text-[10px] font-bold uppercase", isDarkMode ? "text-slate-400" : "text-slate-500")}>Frequency</span>
                                                                            </div>
                                                                            <span className={cn("text-[10px] font-black", isDarkMode ? "text-slate-300" : "text-slate-700")}>{count.toLocaleString()} Iterations</span>
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            );
                                                        }
                                                        return null;
                                                    }}
                                                />
                                                <Area
                                                    type="basis"
                                                    dataKey="count"
                                                    stroke="#06b6d4"
                                                    strokeWidth={4}
                                                    fill="url(#colorBellFill)"
                                                    mask="url(#bellMask)"
                                                    animationDuration={1500}
                                                />
                                                {scaledMc && (
                                                    <>
                                                        <ReferenceLine
                                                            x={scaledMc.bear}
                                                            stroke="#f43f5e"
                                                            strokeDasharray="4 4"
                                                            strokeWidth={2}
                                                            label={{ value: 'BEAR', position: 'top', fill: '#f43f5e', fontSize: 9, fontWeight: '900' }}
                                                        />
                                                        <ReferenceLine
                                                            x={scaledMc.base}
                                                            stroke="#06b6d4"
                                                            strokeDasharray="4 4"
                                                            strokeWidth={2}
                                                            label={{ value: 'MEDIAN', position: 'top', fill: '#06b6d4', fontSize: 9, fontWeight: '900' }}
                                                        />
                                                        <ReferenceLine
                                                            x={scaledMc.bull}
                                                            stroke="#10b981"
                                                            strokeDasharray="4 4"
                                                            strokeWidth={2}
                                                            label={{ value: 'BULL', position: 'top', fill: '#10b981', fontSize: 9, fontWeight: '900' }}
                                                        />
                                                    </>
                                                )}
                                                {currentPriceVal > 0 && (
                                                    <ReferenceLine
                                                        x={currentPriceVal}
                                                        stroke={isDarkMode ? "#cbd5e1" : "#475569"}
                                                        strokeWidth={2}
                                                        strokeDasharray="3 3"
                                                        label={{
                                                            value: 'CURRENT',
                                                            position: 'top',
                                                            fill: isDarkMode ? "#cbd5e1" : "#475569",
                                                            fontSize: 9,
                                                            fontWeight: '900',
                                                            dy: -12
                                                        }}
                                                    />
                                                )}
                                            </AreaChart>
                                        );
                                    })()}
                                </ResponsiveContainer>
                            </div>
                            <div className={cn(
                                "mt-6 flex items-center justify-between text-[10px] font-bold uppercase tracking-widest p-4 rounded-2xl",
                                isDarkMode ? "bg-slate-800/50 text-slate-400" : "bg-slate-50 text-slate-500"
                            )}>
                                {(() => {
                                    const modelKey = viewingDistribution;
                                    if (!modelKey) return null;
                                    const mc = intrinsicValue.models[modelKey]?.mc;
                                    if (!mc) return null;

                                    return (
                                        <>
                                            <div className="flex items-center gap-2">
                                                <div className="w-2.5 h-2.5 rounded-full bg-rose-500" /> <span className="hidden sm:inline">Bear: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency((mc.bear ?? 0) * fxRate, currency)}</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <div className="w-2.5 h-2.5 rounded-full bg-indigo-500" /> <span className="hidden sm:inline">Median: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency((mc.base ?? 0) * fxRate, currency)}</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> <span className="hidden sm:inline">Bull: </span><span className={isDarkMode ? "text-white" : "text-slate-900"}>{formatCurrency((mc.bull ?? 0) * fxRate, currency)}</span>
                                            </div>
                                        </>
                                    );
                                })()}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
