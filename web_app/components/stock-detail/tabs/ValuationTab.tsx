import React, { useState, useMemo } from 'react';
import { useTheme } from 'next-themes';
import {
    TrendingUp,
    Scale,
    AlertCircle,
    Info,
    BarChart3,
    X,
    HelpCircle,
    Coins,
    Calculator,
    Anchor,
    DollarSign,
    Sparkles,
    Building2,
    Zap,
    Percent,
    Layers,
    BookOpen,
    SlidersHorizontal,
    RotateCcw,
    CheckCircle2,
    AlertTriangle,
    ChevronDown,
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
import ValuationComparisonChart from '../ValuationComparisonChart';
import type { IntrinsicValueResponse, IntrinsicValueModel } from '@/lib/api';
import {
    ValuationModelKey,
    MODEL_PARAM_CONFIGS,
    calculateBlendedScore,
} from '../valuationCalculator';

interface ValuationTabProps {
    symbol: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- intrinsicValue payload
    intrinsicValue: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- fundamentals payload
    fundamentals: any;
    currency: string;
    fxRate: number;
}

const VALUATION_INFO: Record<string, { description: string; default: string }> = {
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
        default: "10 Years"
    },
    base_fcf: {
        description: "The starting free cash flow value for DCF projections.",
        default: "Normalized Cycle FCF"
    },
    base_cfo: {
        description: "Starting Operating Cash Flow (CFO/OCF) for Cash from Operations DCF projections.",
        default: "TTM Operating Cash Flow"
    },
    base_dividend: {
        description: "Annual dividend per share used as starting payout for DDM projections.",
        default: "Latest Dividend Rate"
    },
    dividend_growth: {
        description: "Expected long-term growth rate for dividend payouts.",
        default: "Historical Dividend CAGR"
    },
    payout_ratio: {
        description: "Proportion of net earnings paid out to shareholders in dividends.",
        default: "< 100% Sustainable"
    },
    lynch_multiplier: {
        description: "Fair P/E multiplier = Growth Rate (%) + Dividend Yield (%) where PEG = 1.0.",
        default: "PEG = 1.0 (5x - 25x)"
    },
    peg_target: {
        description: "Target PEG multiple where Fair P/E = Growth Rate * Target PEG (Fair value at 1.0).",
        default: "1.0x"
    },
    mean_pe: {
        description: "Historical 5-year average P/E multiple used to value steady earnings.",
        default: "5Y Historical Mean P/E"
    },
    mean_pb: {
        description: "Price-to-Book benchmark (1.2x-1.4x for banks, 1.0x for REIT NAV, 5Y mean for cyclicals).",
        default: "1.30x Bank Benchmark"
    },
    mean_ps: {
        description: "Historical 5-year average Price-to-Sales multiple applied to revenue per share.",
        default: "5Y Historical Mean P/S"
    },
    psg_multiplier: {
        description: "Price-to-Sales Growth multiplier scaling revenue growth rate by gross margin.",
        default: "PSG = 1.0 Fair Value"
    },
    eps: {
        description: "Earnings per share used as the base for valuation formulas.",
        default: "TTM EPS"
    },
    graham_growth: {
        description: "Expected annual growth (g) used in Graham's Formula.",
        default: "Historical CAGR"
    },
    fcf_margin: {
        description: "Free Cash Flow as a percentage of revenue, used to normalize future cash flow projections if current FCF is an outlier.",
        default: "Through-Cycle Margin"
    },
    bond_yield: {
        description: "Current yield on high-quality bonds (proxy for risk-free rate).",
        default: "10Y Treasury (~4.5%)"
    }
};

const ParamItem = ({
    label,
    value,
    info,
    isCustom,
    defaultValue,
    className
}: {
    label: string;
    value: React.ReactNode;
    info?: { description: string; default: string };
    isCustom?: boolean;
    defaultValue?: React.ReactNode;
    className?: string;
}) => (
    <div>
        <div className="flex items-center gap-1 mb-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold flex items-center gap-1">
                {label}
                {isCustom && (
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-500 inline-block" title="Custom parameter" />
                )}
            </p>
            {info && (
                <div className="group relative">
                    <HelpCircle className="w-2.5 h-2.5 text-muted-foreground/50 hover:text-indigo-500 cursor-help" />
                    <div className="absolute bottom-full left-0 mb-2 w-52 p-3 bg-white dark:bg-[#1e293b] text-slate-900 dark:text-white text-[10px] rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-[100] shadow-xl border border-slate-200 dark:border-slate-800">
                        {info.description}
                        <div className="mt-1 pt-1 font-bold text-indigo-600 dark:text-indigo-400">Default: {info.default}</div>
                    </div>
                </div>
            )}
        </div>
        <p className={cn("text-sm font-semibold", isCustom && "text-amber-600 dark:text-amber-400", className)}>
            {value ?? '-'}
        </p>
        {isCustom && defaultValue !== undefined && (
            <p className="text-[10px] text-muted-foreground">Default: {defaultValue}</p>
        )}
    </div>
);

const LimitationCallout = ({
    bestSuitedFor,
    keyCaveats,
    whenToUse,
    keyLimitation,
}: {
    bestSuitedFor?: string;
    keyCaveats?: string;
    whenToUse?: string;
    keyLimitation?: string;
}) => {
    const [expanded, setExpanded] = useState(false);
    const suited = bestSuitedFor || whenToUse;
    const caveats = keyCaveats || keyLimitation;
    if (!suited && !caveats) return null;
    return (
        <div className="bg-secondary/40 border border-border/40 rounded-xl text-[11px] mt-4">
            <button
                type="button"
                onClick={() => setExpanded(v => !v)}
                aria-expanded={expanded}
                className="w-full flex items-center justify-between gap-2 px-3 py-2 text-left"
            >
                <span className="flex items-center gap-1.5 font-bold uppercase tracking-wide text-[10px] text-muted-foreground">
                    <Info className="w-3 h-3" />
                    Best Suited For &amp; Key Caveats
                </span>
                <ChevronDown className={cn("w-3 h-3 text-muted-foreground transition-transform", expanded && "rotate-180")} />
            </button>
            {expanded && (
                <div className="px-3 pb-3 space-y-2">
                    {suited && (
                        <div className="space-y-0.5">
                            <div className="flex items-center gap-1 font-bold uppercase tracking-wide text-[10px] text-emerald-600 dark:text-emerald-400">
                                <CheckCircle2 className="w-3 h-3" />
                                Best Suited For
                            </div>
                            <p className="text-muted-foreground leading-tight">{suited}</p>
                        </div>
                    )}
                    {suited && caveats && <div className="border-t border-border/40" />}
                    {caveats && (
                        <div className="space-y-0.5">
                            <div className="flex items-center gap-1 font-bold uppercase tracking-wide text-[10px] text-amber-600 dark:text-amber-400">
                                <AlertTriangle className="w-3 h-3" />
                                Key Caveats
                            </div>
                            <p className="text-muted-foreground leading-tight">{caveats}</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

const MODEL_TITLES: Record<string, string> = {
    dcf: 'Discounted Free Cash Flow (DCF)',
    dcfo: 'Discounted Cash from Operations (D-CFO)',
    dni: 'Discounted Net Income (D-NI)',
    mean_pe: 'Mean P/E Ratio Valuation',
    peg: 'PEG Ratio Fair Value',
    mean_pb: 'Mean P/B Ratio Valuation',
    mean_ps: 'Mean P/S Ratio Valuation',
    psg: 'Price-to-Sales Growth (PSG)',
    graham: "Benjamin Graham Revised Formula",
    ddm: 'Dividend Discount Model (DDM)',
    lynch: 'Peter Lynch Fair Value',
    epv: 'Earnings Power Value (EPV Floor)',
};

const MonteCarloPillRow: React.FC<{
    mc?: { bear?: number | null; base?: number | null; bull?: number | null } | null;
    fxRate: number;
    currency: string;
    onOpenModal?: () => void;
}> = ({ mc, fxRate, currency, onOpenModal }) => {
    if (!mc || mc.bear == null || mc.bull == null) return null;
    return (
        <div className="pt-2">
            <div className="grid grid-cols-3 gap-2">
                <div
                    className="bg-rose-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-rose-500/10 transition-colors"
                    onClick={onOpenModal}
                    title="Click to view full distribution"
                >
                    <p className="text-[10px] text-rose-500 font-bold uppercase mb-0.5">Bear (10th)</p>
                    <p className="text-xs font-bold">{formatCurrency((mc.bear ?? 0) * fxRate, currency)}</p>
                </div>
                <div
                    className="bg-indigo-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-indigo-500/10 transition-colors"
                    onClick={onOpenModal}
                    title="Click to view full distribution"
                >
                    <p className="text-[10px] text-indigo-500 font-bold uppercase mb-0.5">Median (50th)</p>
                    <p className="text-xs font-bold">{formatCurrency((mc.base ?? 0) * fxRate, currency)}</p>
                </div>
                <div
                    className="bg-emerald-500/5 p-2 rounded-lg text-center cursor-pointer hover:bg-emerald-500/10 transition-colors"
                    onClick={onOpenModal}
                    title="Click to view full distribution"
                >
                    <p className="text-[10px] text-emerald-500 font-bold uppercase mb-0.5">Bull (90th)</p>
                    <p className="text-xs font-bold">{formatCurrency((mc.bull ?? 0) * fxRate, currency)}</p>
                </div>
            </div>
        </div>
    );
};

interface ParameterEditorProps {
    modelKey: ValuationModelKey;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    defaultParams: Record<string, any>;
    customValues: Record<string, number>;
    onChange: (key: string, val: number | undefined) => void;
    onReset: () => void;
    currency: string;
    fxRate: number;
}

const ParameterEditor: React.FC<ParameterEditorProps> = ({
    modelKey,
    defaultParams,
    customValues,
    onChange,
    onReset,
    currency,
    fxRate,
}) => {
    const configs = MODEL_PARAM_CONFIGS[modelKey] || [];
    const hasModifications = Object.keys(customValues).some((k) => {
        const def = defaultParams?.[k];
        return def !== undefined && Math.abs(customValues[k] - def) > 1e-5;
    });

    return (
        <div className="mt-4 p-4 rounded-xl bg-background/80 border border-border/70 space-y-4 animate-in fade-in duration-200">
            <div className="flex items-center justify-between border-b border-border/40 pb-2">
                <div className="flex items-center gap-1.5 text-xs font-bold text-foreground">
                    <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-500" />
                    <span>Edit Model Parameters</span>
                </div>
                {hasModifications && (
                    <button
                        onClick={onReset}
                        className="text-[10px] font-bold text-muted-foreground hover:text-foreground flex items-center gap-1 px-2 py-0.5 rounded bg-muted/60 hover:bg-muted transition-colors cursor-pointer"
                    >
                        <RotateCcw className="w-2.5 h-2.5" /> Reset Card Defaults
                    </button>
                )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {configs.map((cfg) => {
                    const rawDefault = defaultParams?.[cfg.key];
                    const rawCustom = customValues[cfg.key];
                    const defaultVal = cfg.isPercent
                        ? (rawDefault != null ? Number((rawDefault * 100).toFixed(2)) : undefined)
                        : (rawDefault != null ? Number(rawDefault) : undefined);

                    const isCustom = rawCustom !== undefined && rawDefault !== undefined && Math.abs(rawCustom - rawDefault) > 1e-5;
                    const activeVal = isCustom
                        ? (cfg.isPercent ? Number((rawCustom * 100).toFixed(2)) : rawCustom)
                        : defaultVal;

                    const formattedDefault = defaultVal != null
                        ? (cfg.unit === 'percent'
                            ? `${defaultVal}%`
                            : cfg.unit === 'currency'
                                ? formatCurrency(defaultVal * fxRate, currency)
                                : cfg.unit === 'multiple'
                                    ? `${defaultVal}x`
                                    : `${defaultVal}`)
                        : '-';

                    return (
                        <div key={cfg.key} className="space-y-1 bg-secondary/30 p-2.5 rounded-lg border border-border/30">
                            <div className="flex items-center justify-between text-[11px]">
                                <span className="font-semibold text-muted-foreground truncate">{cfg.label}</span>
                                {isCustom && (
                                    <button
                                        onClick={() => {
                                            onChange(cfg.key, undefined);
                                        }}
                                        title="Revert to default"
                                        className="text-[9px] text-amber-500 font-bold hover:underline cursor-pointer"
                                    >
                                        Revert
                                    </button>
                                )}
                            </div>

                            <div className="flex items-center gap-1.5">
                                <input
                                    type="number"
                                    step={cfg.step}
                                    min={cfg.min}
                                    max={cfg.max}
                                    value={activeVal ?? ''}
                                    onChange={(e) => {
                                        const num = parseFloat(e.target.value);
                                        if (!isNaN(num)) {
                                            const computedVal = cfg.isPercent ? num / 100 : num;
                                            onChange(cfg.key, computedVal);
                                        }
                                    }}
                                    className={cn(
                                        "w-full px-2.5 py-1 text-xs rounded-md border font-semibold tabular-nums bg-background focus:outline-none focus:ring-1 focus:ring-indigo-500",
                                        isCustom ? "border-amber-500/60 text-amber-600 dark:text-amber-400 bg-amber-500/5" : "border-border"
                                    )}
                                />
                                <span className="text-[10px] text-muted-foreground font-bold shrink-0">
                                    {cfg.unit === 'percent' ? '%' : cfg.unit === 'multiple' ? 'x' : cfg.unit === 'years' ? 'yrs' : ''}
                                </span>
                            </div>

                            <div className="flex items-center justify-between text-[9px] text-muted-foreground pt-0.5">
                                <span>Default: {formattedDefault}</span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

/**
 * The parenthetical the backend appends to a disagreement note — e.g.
 * "(dcf=392.89, graham=671.56, ddm=14.48)". Those figures are in the company's
 * native currency, so strip them and re-render the same models below in whatever
 * currency the rest of the tab is showing. If the backend wording ever changes,
 * the pattern stops matching and the note simply renders as it was sent.
 */
const MODEL_DETAIL_PATTERN = /\s*\((?:dcf|graham|ddm)=-?[\d.]+(?:,\s*(?:dcf|graham|ddm)=-?[\d.]+)*\)/gi;

/**
 * A blended estimate the backend does not fully stand behind. Two statuses carry
 * that doubt: `low_confidence` (the contributing models sit further apart than
 * the blend is large) and `clamped` (raw output was so far from spot that it was
 * pulled back into the credible band, so the headline is not the number the
 * models actually produced). The macOS and iOS clients surface both; the web app
 * used to drop them and show the headline with no caveat at all.
 */
const ValuationCaveatBanner: React.FC<{
    intrinsicValue: IntrinsicValueResponse;
    currency: string;
    fxRate: number;
}> = ({ intrinsicValue, currency, fxRate }) => {
    const status = intrinsicValue.valuation_status;
    const note = intrinsicValue.valuation_note;
    if (!note || (status !== 'low_confidence' && status !== 'clamped')) return null;

    // Clamped is the stronger of the two claims — the displayed value was altered,
    // not merely averaged over noisy models — so it gets the louder colour.
    const isClamped = status === 'clamped';
    const Icon = isClamped ? AlertCircle : AlertTriangle;

    const models = (intrinsicValue.models ?? {}) as Record<string, IntrinsicValueModel | undefined>;
    const contributions = Object.keys(intrinsicValue.model_weights ?? {})
        .map((key) => ({ key, value: models[key]?.intrinsic_value }))
        .filter((c): c is { key: string; value: number } => typeof c.value === 'number' && Number.isFinite(c.value));

    return (
        <div className={cn(
            "rounded-2xl border p-5 flex items-start gap-3",
            isClamped ? "bg-rose-500/10 border-rose-500/30" : "bg-amber-500/10 border-amber-500/30"
        )}>
            <Icon className={cn("w-5 h-5 shrink-0 mt-0.5", isClamped ? "text-rose-500" : "text-amber-500")} />
            <div className="space-y-2 min-w-0">
                <p className={cn(
                    "text-xs font-bold uppercase tracking-wider",
                    isClamped ? "text-rose-600 dark:text-rose-400" : "text-amber-600 dark:text-amber-400"
                )}>
                    {isClamped ? "Output outside credible range" : "Models disagree"}
                </p>
                <p className="text-sm text-muted-foreground leading-relaxed">
                    {note.replace(MODEL_DETAIL_PATTERN, '')}
                </p>
                {contributions.length > 1 && (
                    <div className="flex items-center gap-1.5 flex-wrap text-[10px]">
                        {contributions.map(({ key, value }) => (
                            <span key={key} className="px-2 py-0.5 bg-background/60 rounded font-semibold tabular-nums">
                                <span className="uppercase text-muted-foreground">{key}</span>{' '}
                                {formatCurrency(value * fxRate, currency)}
                            </span>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

/**
 * How much the backend stands behind the number, as a bar. Confidence is
 * continuous — the models' own Monte Carlo bands, how far apart they landed,
 * and how many of them there were — so it is shown as a level rather than the
 * old pass/fail that read as fine at 99% disagreement and alarming at 101%.
 */
const ConfidenceMeter: React.FC<{ confidence?: number }> = ({ confidence }) => {
    if (typeof confidence !== 'number' || !Number.isFinite(confidence)) return null;
    const pct = Math.max(0, Math.min(1, confidence));
    const tone = pct >= 0.66 ? 'bg-emerald-500' : pct >= 0.4 ? 'bg-amber-500' : 'bg-rose-500';
    return (
        <div className="mt-3 w-full max-w-[190px]">
            <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
                <span className="uppercase font-semibold tracking-wider">Confidence</span>
                <span className="font-bold tabular-nums">{(pct * 100).toFixed(0)}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-secondary/70 overflow-hidden">
                <div className={cn('h-full rounded-full transition-all', tone)} style={{ width: `${pct * 100}%` }} />
            </div>
        </div>
    );
};

const BLEND_PROFILE_LABEL: Record<string, string> = {
    financial: 'Financial — valued on discounted net income, not free cash flow',
    reit: 'REIT — valued on cash from operations and the distribution, since net income is charged with non-cash depreciation',
    operating: 'Operating company — valued on discounted free cash flow',
};

/**
 * The composition of the blend: which models were held out and why, and the
 * floors that travel beside the estimate rather than inside it. A model can be
 * excluded because it does not describe this business (a DCF of a bank) or
 * because it prices only part of it (a DDM of a company that retains most of
 * its earnings) — in both cases the number is still worth seeing, just not
 * worth averaging in.
 */
const BlendComposition: React.FC<{
    intrinsicValue: IntrinsicValueResponse;
    currency: string;
    fxRate: number;
}> = ({ intrinsicValue, currency, fxRate }) => {
    const exclusions = Object.entries(intrinsicValue.blend_exclusions ?? {});
    const floors = [
        { label: 'Earnings power floor', hint: 'Current earnings, no growth', value: intrinsicValue.earnings_power_floor },
        { label: 'Dividend-only value', hint: 'What the dividend stream alone is worth', value: intrinsicValue.dividend_discount_floor },
    ].filter((f) => typeof f.value === 'number' && Number.isFinite(f.value));
    const profile = intrinsicValue.blend_profile;
    if (!exclusions.length && !floors.length && !profile) return null;

    return (
        <div className="rounded-2xl border border-border/60 bg-muted/40 p-5 space-y-3">
            <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-muted-foreground" />
                <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">How this blend was built</p>
            </div>
            {profile && BLEND_PROFILE_LABEL[profile] && (
                <p className="text-sm text-muted-foreground leading-relaxed">{BLEND_PROFILE_LABEL[profile]}.</p>
            )}
            {floors.length > 0 && (
                <div className="flex flex-wrap gap-2">
                    {floors.map((f) => (
                        <div key={f.label} className="px-3 py-2 rounded-xl bg-background/70 border border-border/50">
                            <p className="text-[10px] uppercase font-bold text-muted-foreground">{f.label}</p>
                            <p className="text-sm font-bold tabular-nums">{formatCurrency((f.value as number) * fxRate, currency)}</p>
                            <p className="text-[10px] text-muted-foreground">{f.hint}</p>
                        </div>
                    ))}
                </div>
            )}
            {exclusions.length > 0 && (
                <ul className="space-y-1.5">
                    {exclusions.map(([key, reason]) => (
                        <li key={key} className="text-xs text-muted-foreground flex gap-2">
                            <span className="uppercase font-bold text-foreground/70 shrink-0">{key}</span>
                            <span className="leading-relaxed">held out — {reason}</span>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

/**
 * The interquartile range of the multiples this company has actually traded at.
 * The median alone reads as an opinion; "usually 12.4x-19.1x over 15 years" is
 * the record it was taken from, and it is what tells the reader whether today's
 * multiple is unusual or ordinary.
 */
const TradedRangeItem: React.FC<{
    params: Record<string, unknown> | undefined;
    digits?: number;
}> = ({ params, digits = 1 }) => {
    const p25 = params?.multiple_p25 as number | undefined;
    const p75 = params?.multiple_p75 as number | undefined;
    const n = params?.multiple_observations as number | undefined;
    if (typeof p25 !== 'number' || typeof p75 !== 'number' || !n) return null;
    return (
        <ParamItem
            label="Usually Traded At"
            value={`${p25.toFixed(digits)}x – ${p75.toFixed(digits)}x (${n}y)`}
        />
    );
};

/** Selector sentinels: show only the backend's best-fit method, or every available model. */
const BEST_FIT = '__best_fit__';
const ALL_METHODS = '__all__';

type ModelCategory = 'cash_earnings' | 'multiples_growth' | 'floors_relative';

const MODEL_CATEGORY_LABELS: Record<ModelCategory, string> = {
    cash_earnings: 'Cash Flow & Earnings',
    multiples_growth: 'Multiples & Growth',
    floors_relative: 'Floors & Relative',
};

/** Every model the tab can render, in display order, with its selector label. */
const MODEL_CATALOG: { key: ValuationModelKey; label: string; category: ModelCategory }[] = [
    { key: 'dcf', label: 'Discounted Free Cash Flow (DCF)', category: 'cash_earnings' },
    { key: 'dcfo', label: 'Discounted Cash from Operations (D-CFO)', category: 'cash_earnings' },
    { key: 'dni', label: 'Discounted Net Income (D-NI)', category: 'cash_earnings' },
    { key: 'ddm', label: 'Dividend Discount Model (DDM)', category: 'cash_earnings' },
    { key: 'mean_pe', label: 'Mean P/E Ratio', category: 'multiples_growth' },
    { key: 'peg', label: 'PEG Ratio Fair Value', category: 'multiples_growth' },
    { key: 'mean_pb', label: 'Mean P/B Ratio', category: 'multiples_growth' },
    { key: 'mean_ps', label: 'Mean P/S Ratio', category: 'multiples_growth' },
    { key: 'psg', label: 'Price-to-Sales Growth (PSG)', category: 'multiples_growth' },
    { key: 'graham', label: "Graham's Formula", category: 'floors_relative' },
    { key: 'lynch', label: 'Peter Lynch Fair Value', category: 'floors_relative' },
    { key: 'epv', label: 'Earnings Power Value (EPV Floor)', category: 'floors_relative' },
];

export const ValuationTab: React.FC<ValuationTabProps> = ({
    symbol,
    intrinsicValue,
    fundamentals,
    currency,
    fxRate
}) => {
    const [methodFilter, setMethodFilter] = useState<string>(BEST_FIT);
    const [viewingDistribution, setViewingDistribution] = useState<string | null>(null);
    const [customOverrides, setCustomOverrides] = useState<Record<string, Record<string, number>>>({});
    const [editingModelKeys, setEditingModelKeys] = useState<Set<string>>(new Set());

    const { resolvedTheme } = useTheme();
    const isDarkMode = resolvedTheme === 'dark';

    // Live calculation of blended score & model custom values
    const { customAverage, customMarginOfSafety, customModelValues, hasAnyCustom } = useMemo(() => {
        return calculateBlendedScore(intrinsicValue, customOverrides, fundamentals?.sector);
    }, [intrinsicValue, customOverrides, fundamentals?.sector]);

    if (!intrinsicValue) return null;
    const { models, average_intrinsic_value, margin_of_safety_pct, current_price, recommended_method } = intrinsicValue;
    const status = intrinsicValue.valuation_status;
    const hasDefaultValue = average_intrinsic_value !== null && average_intrinsic_value !== undefined;
    const isRefusal = status === "ineligible" || status === "no_model";

    // Effective active blended value & margin of safety
    const displayAverage = hasAnyCustom && customAverage != null ? customAverage : average_intrinsic_value;
    const displayMos = hasAnyCustom && customMarginOfSafety != null ? customMarginOfSafety : margin_of_safety_pct;
    const hasDisplayValue = displayAverage !== null && displayAverage !== undefined;

    // --- Method selector -------------------------------------------------
    // Only models the backend actually returned are offered; the best-fit
    // method is the default view so the tab opens on one card, not twelve.
    const availableModels = MODEL_CATALOG.filter((m) => models?.[m.key]);
    const recommendedKey = recommended_method?.method_key;
    const bestFitKey = availableModels.some((m) => m.key === recommendedKey) ? recommendedKey : undefined;
    // A selection can go stale when the user switches to a stock lacking that model.
    const effectiveFilter =
        methodFilter !== BEST_FIT && methodFilter !== ALL_METHODS && !availableModels.some((m) => m.key === methodFilter)
            ? BEST_FIT
            : methodFilter;
    const showModel = (key: ValuationModelKey) => {
        if (effectiveFilter === ALL_METHODS) return true;
        if (effectiveFilter === BEST_FIT) return bestFitKey ? key === bestFitKey : true;
        return key === effectiveFilter;
    };
    const visibleCount = availableModels.filter((m) => showModel(m.key)).length;

    const toggleEditing = (key: string) => {
        setEditingModelKeys((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    const handleParamChange = (modelKey: string, paramKey: string, val: number | undefined) => {
        setCustomOverrides((prev) => {
            const modelOverrides = { ...(prev[modelKey] || {}) };
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const rawDefault = (models as Record<string, any>)?.[modelKey]?.parameters?.[paramKey];
            if (val === undefined || (rawDefault !== undefined && Math.abs(val - rawDefault) < 1e-5)) {
                delete modelOverrides[paramKey];
            } else {
                modelOverrides[paramKey] = val;
            }

            const next = { ...prev };
            if (Object.keys(modelOverrides).length === 0) {
                delete next[modelKey];
            } else {
                next[modelKey] = modelOverrides;
            }
            return next;
        });
    };

    const handleResetModel = (modelKey: string) => {
        setCustomOverrides((prev) => {
            const next = { ...prev };
            delete next[modelKey];
            return next;
        });
    };

    const handleResetAll = () => {
        setCustomOverrides({});
        setEditingModelKeys(new Set());
    };

    const renderCardHeader = (
        title: string,
        icon: React.ReactNode,
        modelKey: ValuationModelKey,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        modelData: any,
        primaryBadge?: string
    ) => {
        const isEditing = editingModelKeys.has(modelKey);
        const customVal = customModelValues[modelKey];
        const defaultVal = modelData?.intrinsic_value;
        const overrides = customOverrides[modelKey];
        const defaultParams = modelData?.parameters || {};
        const isCustom = Boolean(overrides && Object.keys(overrides).some((k) => {
            const def = defaultParams[k];
            return def !== undefined && Math.abs(overrides[k] - def) > 1e-5;
        }));
        const activeVal = isCustom ? customVal : defaultVal;

        const diffPct = isCustom && defaultVal && activeVal && Math.abs(activeVal - defaultVal) > 0.001
            ? ((activeVal - defaultVal) / defaultVal) * 100
            : null;

        return (
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-2">
                    {icon}
                    <div>
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                            {title}
                            {primaryBadge && (
                                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded">
                                    {primaryBadge}
                                </span>
                            )}
                            {isCustom && (
                                <span className="text-[9px] font-extrabold uppercase px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                                    Custom
                                </span>
                            )}
                        </h3>
                    </div>
                </div>

                <div className="flex items-center gap-2 self-end sm:self-auto flex-wrap">
                    {/* Value Badge & Comparison */}
                    {activeVal !== undefined && activeVal !== null && (
                        <div className="flex flex-col items-end">
                            <Badge className={cn(
                                "border-none text-xs font-bold",
                                isCustom ? "bg-amber-500/20 text-amber-600 dark:text-amber-400" : "bg-emerald-500/20 text-emerald-500"
                            )}>
                                {formatCurrency(activeVal * fxRate, currency)}
                            </Badge>
                            {isCustom && defaultVal != null && (
                                <div className="text-[10px] text-muted-foreground flex items-center gap-1 mt-0.5">
                                    <span>Def: {formatCurrency(defaultVal * fxRate, currency)}</span>
                                    {diffPct != null && (
                                        <span className={cn("font-bold", diffPct >= 0 ? "text-emerald-500" : "text-rose-500")}>
                                            ({diffPct >= 0 ? '+' : ''}{diffPct.toFixed(1)}%)
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Edit Parameters Toggle Button */}
                    <button
                        onClick={() => toggleEditing(modelKey)}
                        title="Customize parameters"
                        className={cn(
                            "flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-lg border transition-all cursor-pointer",
                            isEditing
                                ? "bg-indigo-600 text-white border-indigo-600 shadow-sm"
                                : "bg-background/80 hover:bg-muted text-muted-foreground hover:text-foreground border-border/70"
                        )}
                    >
                        <SlidersHorizontal className="w-3 h-3" />
                        <span>{isEditing ? 'Done' : 'Edit'}</span>
                    </button>

                    {/* Reset Button */}
                    {isCustom && (
                        <button
                            onClick={() => handleResetModel(modelKey)}
                            title="Reset to default values"
                            className="p-1 rounded-lg bg-muted/70 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                        >
                            <RotateCcw className="w-3 h-3" />
                        </button>
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Custom Valuation Alert Bar */}
            {hasAnyCustom && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-4 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
                    <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-amber-500 shrink-0" />
                        <span className="font-semibold text-amber-600 dark:text-amber-400">
                            Custom parameters active — Intrinsic values and composite score are recalculated in real time.
                        </span>
                    </div>
                    <button
                        onClick={handleResetAll}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-amber-500 text-white font-bold text-xs hover:bg-amber-600 transition-colors shadow-xs cursor-pointer whitespace-nowrap"
                    >
                        <RotateCcw className="w-3 h-3" /> Reset All to Defaults
                    </button>
                </div>
            )}

            {/* Summary Header */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className={cn(
                    "p-6 rounded-2xl flex flex-col items-center justify-center text-center transition-all",
                    hasAnyCustom ? "bg-amber-500/5 border border-amber-500/30" : "bg-muted"
                )}>
                    <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2 flex items-center gap-1">
                        {status === "nav" ? "Net Asset Value" : (hasAnyCustom ? "Custom Blended Value" : "Blended Intrinsic Value")}
                        {hasAnyCustom && <Sparkles className="w-3 h-3 text-amber-500" />}
                    </p>
                    {hasDisplayValue ? (
                        <p className={cn("text-3xl font-bold", hasAnyCustom ? "text-amber-600 dark:text-amber-400" : "text-indigo-500")}>
                            {formatCurrency((displayAverage ?? 0) * fxRate, currency)}
                        </p>
                    ) : (
                        <p className="text-2xl font-bold text-muted-foreground">Not valued</p>
                    )}
                    {hasAnyCustom && hasDefaultValue && (
                        <div className="mt-1 text-[11px] text-muted-foreground">
                            Default: {formatCurrency((average_intrinsic_value ?? 0) * fxRate, currency)}{' '}
                            <span className={cn("font-bold", (displayAverage! - average_intrinsic_value) >= 0 ? "text-emerald-500" : "text-rose-500")}>
                                ({(displayAverage! - average_intrinsic_value) >= 0 ? '+' : ''}
                                {(((displayAverage! - average_intrinsic_value) / average_intrinsic_value) * 100).toFixed(1)}%)
                            </span>
                        </div>
                    )}
                    {hasDisplayValue && current_price && (
                        <div className="mt-1 text-xs text-muted-foreground">
                            Spot: {formatCurrency((current_price ?? 0) * fxRate, currency)}
                        </div>
                    )}
                </div>

                <div className={cn(
                    "p-6 rounded-2xl flex flex-col items-center justify-center text-center transition-all",
                    hasAnyCustom ? "bg-amber-500/5 border border-amber-500/30" : "bg-muted"
                )}>
                    <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Margin of Safety</p>
                    {displayMos !== null && displayMos !== undefined ? (
                        <p className={cn(
                            "text-3xl font-bold",
                            displayMos > 0 ? "text-emerald-500" : "text-rose-500"
                        )}>
                            {displayMos > 0 ? "+" : ""}{displayMos.toFixed(1)}%
                        </p>
                    ) : (
                        <p className="text-2xl font-bold text-muted-foreground">—</p>
                    )}
                    {hasAnyCustom && margin_of_safety_pct != null && (
                        <div className="mt-1 text-[11px] text-muted-foreground">
                            Default MOS: {margin_of_safety_pct > 0 ? '+' : ''}{margin_of_safety_pct.toFixed(1)}%
                        </div>
                    )}
                    <span className="text-[10px] text-muted-foreground mt-1">
                        {displayMos !== null && displayMos !== undefined
                            ? (displayMos > 0 ? "Undervalued vs Market" : "Overvalued vs Market")
                            : "No estimate available"}
                    </span>
                </div>

                <div className="bg-muted p-6 rounded-2xl flex flex-col items-center justify-center text-center">
                    <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">Confidence & Range</p>
                    {intrinsicValue.range && hasDefaultValue ? (
                        <>
                            <p className="text-sm font-semibold mb-1">
                                {formatCurrency((intrinsicValue.range.bear ?? 0) * fxRate, currency)} — {formatCurrency((intrinsicValue.range.bull ?? 0) * fxRate, currency)}
                            </p>
                            <span className="text-[10px] text-muted-foreground">Bear (10th) to Bull (90th), from the blended models</span>
                        </>
                    ) : (
                        <p className="text-sm text-muted-foreground">Range unavailable</p>
                    )}
                    <ConfidenceMeter confidence={intrinsicValue.valuation_confidence} />
                    {intrinsicValue.model_weights && Object.keys(intrinsicValue.model_weights).length > 0 && (
                        <div className="mt-2 flex items-center gap-1.5 flex-wrap justify-center text-[10px] text-muted-foreground">
                            {Object.entries(intrinsicValue.model_weights).map(([k, w]) => (
                                <span key={k} className="px-1.5 py-0.5 bg-secondary/50 rounded uppercase font-semibold">
                                    {k}: {((w as number) * 100).toFixed(0)}%
                                </span>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Blend caveats the backend attaches to the headline number. */}
            <ValuationCaveatBanner intrinsicValue={intrinsicValue} currency={currency} fxRate={fxRate} />

            {/* What was left out of the blend, and the floors that sit beside it. */}
            <BlendComposition intrinsicValue={intrinsicValue} currency={currency} fxRate={fxRate} />

            {/* Recommended Method Banner */}
            {recommended_method && recommended_method.method_key !== 'none' && (
                <div className="bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10 border border-indigo-500/30 rounded-2xl p-6 relative overflow-hidden">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div className="space-y-1">
                            <div className="flex items-center gap-2">
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold bg-indigo-500 text-white shadow-sm">
                                    <Sparkles className="w-3 h-3" /> Best-Fit Valuation Method
                                </span>
                                <h4 className="text-base font-bold">{recommended_method.name}</h4>
                            </div>
                            <p className="text-xs text-muted-foreground">{recommended_method.rationale}</p>
                        </div>
                        {recommended_method.intrinsic_value != null && (
                            <div className="flex items-center gap-3 bg-background/80 backdrop-blur-sm px-4 py-3 rounded-xl border border-border/50">
                                <div>
                                    <p className="text-[10px] uppercase font-bold text-muted-foreground">Fair Value</p>
                                    <p className="text-lg font-black text-indigo-500">{formatCurrency(recommended_method.intrinsic_value * fxRate, currency)}</p>
                                </div>
                                {current_price && (
                                    <div className="text-right pl-3 border-l border-border/50">
                                        <p className="text-[10px] uppercase font-bold text-muted-foreground">Upside</p>
                                        <p className={cn(
                                            "text-sm font-bold",
                                            ((recommended_method.intrinsic_value - current_price) / current_price) >= 0 ? "text-emerald-500" : "text-rose-500"
                                        )}>
                                            {((recommended_method.intrinsic_value - current_price) / current_price) >= 0 ? "+" : ""}
                                            {(((recommended_method.intrinsic_value - current_price) / current_price) * 100).toFixed(1)}%
                                        </p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4 pt-3 border-t border-indigo-500/20 text-xs">
                        <div>
                            <span className="font-semibold text-emerald-600 dark:text-emerald-400">Best Suited For: </span>
                            <span className="text-muted-foreground">{recommended_method.best_suited_for || recommended_method.when_to_use}</span>
                        </div>
                        <div>
                            <span className="font-semibold text-amber-600 dark:text-amber-400">Key Caveats: </span>
                            <span className="text-muted-foreground">{recommended_method.key_caveats || recommended_method.key_limitation}</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Intrinsic Value Comparison Spectrum Chart */}
            <ValuationComparisonChart
                symbol={symbol}
                intrinsicValue={intrinsicValue}
                currency={currency}
                fxRate={fxRate}
                recommendedMethod={recommended_method}
                customModelValues={customModelValues}
                customBlendedValue={hasAnyCustom ? customAverage : undefined}
            />

            {/* Method selector — defaults to the best-fit model */}
            <div className="flex flex-wrap items-center gap-2">
                <label htmlFor="valuation-method-select" className="text-xs font-semibold text-muted-foreground">
                    Valuation method
                </label>
                <select
                    id="valuation-method-select"
                    value={effectiveFilter}
                    onChange={(e) => setMethodFilter(e.target.value)}
                    className="px-2.5 py-1.5 rounded-lg bg-secondary text-xs font-semibold text-foreground border border-border/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/40 cursor-pointer"
                    aria-label="Valuation method"
                >
                    <option value={BEST_FIT}>
                        {bestFitKey
                            ? `Best Fit — ${MODEL_CATALOG.find((m) => m.key === bestFitKey)?.label}`
                            : 'Best Fit (none selected)'}
                    </option>
                    <option value={ALL_METHODS}>All Methods ({availableModels.length})</option>
                    {(Object.keys(MODEL_CATEGORY_LABELS) as ModelCategory[]).map((cat) => {
                        const group = availableModels.filter((m) => m.category === cat);
                        if (group.length === 0) return null;
                        return (
                            <optgroup key={cat} label={MODEL_CATEGORY_LABELS[cat]}>
                                {group.map((m) => (
                                    <option key={m.key} value={m.key}>{m.label}</option>
                                ))}
                            </optgroup>
                        );
                    })}
                </select>
                {effectiveFilter === BEST_FIT && bestFitKey && (
                    <span className="text-[11px] text-muted-foreground">
                        Showing the best-fit model only — switch above to compare others.
                    </span>
                )}
            </div>

            {/* Models Breakdown */}
            {isRefusal ? (
                <div className="bg-muted/40 rounded-2xl p-8 text-center max-w-xl mx-auto space-y-3">
                    <AlertCircle className="w-10 h-10 text-amber-500 mx-auto opacity-80" />
                    <h3 className="text-base font-semibold">Valuation Unavailable</h3>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                        {intrinsicValue.valuation_note || "The fundamental data for this asset does not support a defensible valuation."}
                    </p>
                </div>
            ) : (
                <div className={cn("grid grid-cols-1 gap-8", visibleCount > 1 && "md:grid-cols-2")}>
                    {/* 1. DCF Model (Primary Method) */}
                    {showModel('dcf') && models.dcf && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'dcf' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.dcf && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                models.dcf.model || 'Discounted Free Cash Flow',
                                <TrendingUp className="w-5 h-5 text-emerald-500" />,
                                'dcf',
                                models.dcf,
                                'Primary'
                            )}
                            {models.dcf.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.dcf.error}</p>
                            ) : !models.dcf.parameters ? (
                                <div className="flex flex-col items-center justify-center py-10 text-center opacity-50">
                                    <Info className="w-8 h-8 mb-2" />
                                    <p className="text-sm">Not applicable for this asset type.</p>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    {editingModelKeys.has('dcf') ? (
                                        <ParameterEditor
                                            modelKey="dcf"
                                            defaultParams={models.dcf.parameters}
                                            customValues={customOverrides.dcf || {}}
                                            onChange={(k, v) => handleParamChange('dcf', k, v)}
                                            onReset={() => handleResetModel('dcf')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="Discount Rate (WACC)"
                                                value={formatPercentShared(customOverrides.dcf?.discount_rate ?? models.dcf.parameters.discount_rate ?? 0)}
                                                isCustom={customOverrides.dcf?.discount_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dcf.parameters.discount_rate ?? 0)}
                                                info={VALUATION_INFO.discount_rate}
                                            />
                                            <ParamItem
                                                label="Growth Rate"
                                                value={formatPercentShared(customOverrides.dcf?.growth_rate ?? models.dcf.parameters.growth_rate ?? 0)}
                                                isCustom={customOverrides.dcf?.growth_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dcf.parameters.growth_rate ?? 0)}
                                                info={VALUATION_INFO.growth_rate}
                                            />
                                            <ParamItem
                                                label="Base FCF"
                                                value={formatCurrency((customOverrides.dcf?.base_fcf ?? models.dcf.parameters.base_fcf ?? 0) * fxRate, currency)}
                                                isCustom={customOverrides.dcf?.base_fcf !== undefined}
                                                defaultValue={formatCurrency((models.dcf.parameters.base_fcf ?? 0) * fxRate, currency)}
                                                info={VALUATION_INFO.base_fcf}
                                            />
                                            <ParamItem
                                                label="Terminal Growth"
                                                value={formatPercentShared(customOverrides.dcf?.terminal_growth_rate ?? models.dcf.parameters.terminal_growth_rate ?? 0)}
                                                isCustom={customOverrides.dcf?.terminal_growth_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dcf.parameters.terminal_growth_rate ?? 0)}
                                                info={VALUATION_INFO.terminal_growth}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Cash-generative companies with steady, predictable Free Cash Flow (Operating Cash Flow minus Capital Expenditures)."
                                        keyCaveats="Highly sensitive to growth and discount rate (WACC) inputs; unsuitable for cyclical, negative-FCF, or lumpy CapEx businesses."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.dcf.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('dcf')}
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {/* 2. Discounted Cash from Operations (D-CFO) */}
                    {showModel('dcfo') && models.dcfo && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'dcfo' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.dcfo && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'Discounted Cash from Operations',
                                <DollarSign className="w-5 h-5 text-teal-500" />,
                                'dcfo',
                                models.dcfo
                            )}
                            {models.dcfo.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.dcfo.error}</p>
                            ) : models.dcfo.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('dcfo') ? (
                                        <ParameterEditor
                                            modelKey="dcfo"
                                            defaultParams={models.dcfo.parameters}
                                            customValues={customOverrides.dcfo || {}}
                                            onChange={(k, v) => handleParamChange('dcfo', k, v)}
                                            onReset={() => handleResetModel('dcfo')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="Base CFO / Share"
                                                value={formatCurrency((customOverrides.dcfo?.cfo_per_share ?? models.dcfo.parameters.cfo_per_share ?? 0) * fxRate, currency)}
                                                isCustom={customOverrides.dcfo?.cfo_per_share !== undefined}
                                                defaultValue={formatCurrency((models.dcfo.parameters.cfo_per_share ?? 0) * fxRate, currency)}
                                                info={VALUATION_INFO.base_cfo}
                                            />
                                            <ParamItem
                                                label="CFO Growth Rate"
                                                value={formatPercentShared(customOverrides.dcfo?.growth_rate ?? models.dcfo.parameters.growth_rate ?? 0)}
                                                isCustom={customOverrides.dcfo?.growth_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dcfo.parameters.growth_rate ?? 0)}
                                                info={VALUATION_INFO.growth_rate}
                                            />
                                            <ParamItem
                                                label="Discount Rate (WACC)"
                                                value={formatPercentShared(customOverrides.dcfo?.discount_rate ?? models.dcfo.parameters.discount_rate ?? 0)}
                                                isCustom={customOverrides.dcfo?.discount_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dcfo.parameters.discount_rate ?? 0)}
                                                info={VALUATION_INFO.discount_rate}
                                            />
                                            <ParamItem
                                                label="Terminal Growth"
                                                value={formatPercentShared(customOverrides.dcfo?.terminal_growth_rate ?? models.dcfo.parameters.terminal_growth_rate ?? 0.02)}
                                                isCustom={customOverrides.dcfo?.terminal_growth_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dcfo.parameters.terminal_growth_rate ?? 0.02)}
                                                info={VALUATION_INFO.terminal_growth}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Companies with consistent operating cash flow but erratic or heavy multi-year CapEx cycles (e.g., telecom, infrastructure, logistics)."
                                        keyCaveats="Excludes ongoing reinvestment needs (CapEx), risking overvaluation for capital-intensive companies that require heavy sustaining capital."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.dcfo.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('dcfo')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 3. Discounted Net Income (D-NI) */}
                    {showModel('dni') && models.dni && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'dni' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.dni && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'Discounted Net Income (D-NI)',
                                <Building2 className="w-5 h-5 text-blue-500" />,
                                'dni',
                                models.dni
                            )}
                            {models.dni.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.dni.error}</p>
                            ) : models.dni.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('dni') ? (
                                        <ParameterEditor
                                            modelKey="dni"
                                            defaultParams={models.dni.parameters}
                                            customValues={customOverrides.dni || {}}
                                            onChange={(k, v) => handleParamChange('dni', k, v)}
                                            onReset={() => handleResetModel('dni')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="Base EPS"
                                                value={(customOverrides.dni?.base_eps ?? models.dni.parameters.base_eps ?? 0).toFixed(2)}
                                                isCustom={customOverrides.dni?.base_eps !== undefined}
                                                defaultValue={(models.dni.parameters.base_eps ?? 0).toFixed(2)}
                                                info={VALUATION_INFO.eps}
                                            />
                                            <ParamItem
                                                label="Net Income Growth"
                                                value={formatPercentShared(customOverrides.dni?.growth_rate ?? models.dni.parameters.growth_rate ?? 0)}
                                                isCustom={customOverrides.dni?.growth_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dni.parameters.growth_rate ?? 0)}
                                                info={VALUATION_INFO.growth_rate}
                                            />
                                            <ParamItem
                                                label="Cost of Equity"
                                                value={formatPercentShared(customOverrides.dni?.discount_rate ?? models.dni.parameters.discount_rate ?? 0)}
                                                isCustom={customOverrides.dni?.discount_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dni.parameters.discount_rate ?? 0)}
                                                info={VALUATION_INFO.discount_rate}
                                            />
                                            <ParamItem
                                                label="Terminal Growth"
                                                value={formatPercentShared(customOverrides.dni?.terminal_growth_rate ?? models.dni.parameters.terminal_growth_rate ?? 0.02)}
                                                isCustom={customOverrides.dni?.terminal_growth_rate !== undefined}
                                                defaultValue={formatPercentShared(models.dni.parameters.terminal_growth_rate ?? 0.02)}
                                                info={VALUATION_INFO.terminal_growth}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Financial institutions (Banks, Insurance, Brokers, Asset Managers) where cash flow lines are distorted by financial leverage and regulatory capital."
                                        keyCaveats="Net Income is vulnerable to non-recurring items (NRI) and accounting choices, and does not capture working capital or cash conversion drag."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.dni.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('dni')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 4. Multi-Stage Dividend Discount Model (DDM) */}
                    {showModel('ddm') && models.ddm && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'ddm' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.ddm && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                models.ddm.model || 'Dividend Discount Model',
                                <Coins className="w-5 h-5 text-purple-500" />,
                                'ddm',
                                models.ddm
                            )}
                            {models.ddm.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.ddm.error}</p>
                            ) : models.ddm.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('ddm') ? (
                                        <ParameterEditor
                                            modelKey="ddm"
                                            defaultParams={models.ddm.parameters}
                                            customValues={customOverrides.ddm || {}}
                                            onChange={(k, v) => handleParamChange('ddm', k, v)}
                                            onReset={() => handleResetModel('ddm')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="Base Dividend"
                                                value={formatCurrency((customOverrides.ddm?.base_dividend ?? models.ddm.parameters.base_dividend ?? 0) * fxRate, currency)}
                                                isCustom={customOverrides.ddm?.base_dividend !== undefined}
                                                defaultValue={formatCurrency((models.ddm.parameters.base_dividend ?? 0) * fxRate, currency)}
                                                info={VALUATION_INFO.base_dividend}
                                            />
                                            <ParamItem
                                                label="Dividend Yield"
                                                value={models.ddm.parameters.dividend_yield_pct != null ? `${models.ddm.parameters.dividend_yield_pct.toFixed(2)}%` : '-'}
                                            />
                                            <ParamItem
                                                label="Dividend Growth Rate"
                                                value={formatPercentShared(customOverrides.ddm?.growth_rate ?? models.ddm.parameters.growth_rate ?? 0)}
                                                isCustom={customOverrides.ddm?.growth_rate !== undefined}
                                                defaultValue={formatPercentShared(models.ddm.parameters.growth_rate ?? 0)}
                                                info={VALUATION_INFO.dividend_growth}
                                            />
                                            <ParamItem
                                                label="Cost of Equity (CAPM)"
                                                value={formatPercentShared(customOverrides.ddm?.discount_rate ?? models.ddm.parameters.discount_rate ?? 0)}
                                                isCustom={customOverrides.ddm?.discount_rate !== undefined}
                                                defaultValue={formatPercentShared(models.ddm.parameters.discount_rate ?? 0)}
                                                info={VALUATION_INFO.discount_rate}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Mature dividend payers and utilities with long track records of consistent dividend growth and sustainable payout ratios (<100%)."
                                        keyCaveats="Only reflects value returned as direct dividends; entirely unsuited for non-dividend payers and ignores share repurchases or cash retained."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.ddm.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('ddm')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 5. Mean P/E Ratio */}
                    {showModel('mean_pe') && models.mean_pe && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'mean_pe' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.mean_pe && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'Mean P/E Ratio',
                                <Percent className="w-5 h-5 text-indigo-500" />,
                                'mean_pe',
                                models.mean_pe
                            )}
                            {models.mean_pe.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.mean_pe.error}</p>
                            ) : models.mean_pe.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('mean_pe') ? (
                                        <ParameterEditor
                                            modelKey="mean_pe"
                                            defaultParams={models.mean_pe.parameters}
                                            customValues={customOverrides.mean_pe || {}}
                                            onChange={(k, v) => handleParamChange('mean_pe', k, v)}
                                            onReset={() => handleResetModel('mean_pe')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="TTM EPS"
                                                value={(customOverrides.mean_pe?.eps ?? models.mean_pe.parameters.eps ?? 0).toFixed(2)}
                                                isCustom={customOverrides.mean_pe?.eps !== undefined}
                                                defaultValue={(models.mean_pe.parameters.eps ?? 0).toFixed(2)}
                                                info={VALUATION_INFO.eps}
                                            />
                                            <ParamItem
                                                label="Mean P/E Multiple"
                                                value={`${(customOverrides.mean_pe?.applied_pe ?? models.mean_pe.parameters.applied_pe ?? 0).toFixed(1)}x`}
                                                isCustom={customOverrides.mean_pe?.applied_pe !== undefined}
                                                defaultValue={`${(models.mean_pe.parameters.applied_pe ?? 0).toFixed(1)}x`}
                                                info={VALUATION_INFO.mean_pe}
                                            />
                                            <ParamItem
                                                label="Multiple Source"
                                                value={models.mean_pe.parameters.pe_source || 'Historical Baseline'}
                                            />
                                            <TradedRangeItem params={models.mean_pe.parameters} digits={1} />
                                            <ParamItem
                                                label="Fair Multiplier"
                                                value={`${(customOverrides.mean_pe?.applied_pe ?? models.mean_pe.parameters.applied_pe ?? 0).toFixed(1)}x EPS`}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Mature, profitable companies with stable earnings predictability and an established historical valuation multiple baseline."
                                        keyCaveats="Ignores future earnings growth rates and margin trajectory; easily distorted by one-off non-operating gains or restructuring charges."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.mean_pe.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('mean_pe')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 6. PEG Ratio Fair Value */}
                    {showModel('peg') && models.peg && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'peg' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.peg && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'PEG Ratio Fair Value',
                                <Zap className="w-5 h-5 text-amber-500" />,
                                'peg',
                                models.peg
                            )}
                            {models.peg.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.peg.error}</p>
                            ) : models.peg.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('peg') ? (
                                        <ParameterEditor
                                            modelKey="peg"
                                            defaultParams={models.peg.parameters}
                                            customValues={customOverrides.peg || {}}
                                            onChange={(k, v) => handleParamChange('peg', k, v)}
                                            onReset={() => handleResetModel('peg')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="TTM EPS"
                                                value={(customOverrides.peg?.eps ?? models.peg.parameters.eps ?? 0).toFixed(2)}
                                                isCustom={customOverrides.peg?.eps !== undefined}
                                                defaultValue={(models.peg.parameters.eps ?? 0).toFixed(2)}
                                                info={VALUATION_INFO.eps}
                                            />
                                            <ParamItem
                                                label="Growth Rate"
                                                value={`${(customOverrides.peg?.growth_rate_pct ?? models.peg.parameters.growth_rate_pct ?? 0).toFixed(1)}%`}
                                                isCustom={customOverrides.peg?.growth_rate_pct !== undefined}
                                                defaultValue={`${(models.peg.parameters.growth_rate_pct ?? 0).toFixed(1)}%`}
                                                info={VALUATION_INFO.growth_rate}
                                            />
                                            <ParamItem
                                                label="Target PEG"
                                                value={`${(customOverrides.peg?.target_peg ?? models.peg.parameters.target_peg ?? 1.0).toFixed(1)}x`}
                                                isCustom={customOverrides.peg?.target_peg !== undefined}
                                                defaultValue={`${(models.peg.parameters.target_peg ?? 1.0).toFixed(1)}x`}
                                                info={VALUATION_INFO.peg_target}
                                            />
                                            <ParamItem
                                                label="Fair P/E Multiplier"
                                                value={`${((customOverrides.peg?.target_peg ?? models.peg.parameters.target_peg ?? 1.0) * ((customOverrides.peg?.growth_rate_pct ?? models.peg.parameters.growth_rate_pct ?? 0) + (models.peg.parameters.dividend_yield_pct ?? 0))).toFixed(1)}x`}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Profitable growth companies with positive, expanding earnings where growth rate directly anchors the fair earnings multiple."
                                        keyCaveats="Assumes earnings growth is linear and sustainable; vulnerable to short-term earnings volatility and ignores balance sheet debt burden."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.peg.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('peg')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 7. Mean P/B Ratio */}
                    {showModel('mean_pb') && models.mean_pb && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'mean_pb' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.mean_pb && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'Mean P/B Ratio',
                                <BookOpen className="w-5 h-5 text-orange-500" />,
                                'mean_pb',
                                models.mean_pb
                            )}
                            {models.mean_pb.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.mean_pb.error}</p>
                            ) : models.mean_pb.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('mean_pb') ? (
                                        <ParameterEditor
                                            modelKey="mean_pb"
                                            defaultParams={models.mean_pb.parameters}
                                            customValues={customOverrides.mean_pb || {}}
                                            onChange={(k, v) => handleParamChange('mean_pb', k, v)}
                                            onReset={() => handleResetModel('mean_pb')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="Book Value / Share"
                                                value={formatCurrency((customOverrides.mean_pb?.book_value_per_share ?? models.mean_pb.parameters.book_value_per_share ?? 0) * fxRate, currency)}
                                                isCustom={customOverrides.mean_pb?.book_value_per_share !== undefined}
                                                defaultValue={formatCurrency((models.mean_pb.parameters.book_value_per_share ?? 0) * fxRate, currency)}
                                            />
                                            <ParamItem
                                                label="Applied P/B Target"
                                                value={`${(customOverrides.mean_pb?.applied_pb ?? models.mean_pb.parameters.applied_pb ?? 0).toFixed(2)}x`}
                                                isCustom={customOverrides.mean_pb?.applied_pb !== undefined}
                                                defaultValue={`${(models.mean_pb.parameters.applied_pb ?? 0).toFixed(2)}x`}
                                                info={VALUATION_INFO.mean_pb}
                                            />
                                            <ParamItem
                                                label="Benchmark Source"
                                                value={models.mean_pb.parameters.pb_source || 'Industry Standard'}
                                            />
                                            <TradedRangeItem params={models.mean_pb.parameters} digits={2} />
                                            <ParamItem
                                                label="Sector Classification"
                                                value={models.mean_pb.parameters.sector || 'Financial/Asset'}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Asset-heavy businesses, Banks (1.2–1.4x benchmark), REITs (Price/NAV), and property developers whose assets are marked to market."
                                        keyCaveats="Understates high-ROE, asset-light, and tech businesses with valuable off-balance-sheet intangible assets or intellectual property."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.mean_pb.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('mean_pb')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 8. Mean P/S Ratio */}
                    {showModel('mean_ps') && models.mean_ps && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'mean_ps' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.mean_ps && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'Mean P/S Ratio',
                                <Layers className="w-5 h-5 text-rose-500" />,
                                'mean_ps',
                                models.mean_ps
                            )}
                            {models.mean_ps.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.mean_ps.error}</p>
                            ) : models.mean_ps.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('mean_ps') ? (
                                        <ParameterEditor
                                            modelKey="mean_ps"
                                            defaultParams={models.mean_ps.parameters}
                                            customValues={customOverrides.mean_ps || {}}
                                            onChange={(k, v) => handleParamChange('mean_ps', k, v)}
                                            onReset={() => handleResetModel('mean_ps')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="Sales / Share"
                                                value={formatCurrency((customOverrides.mean_ps?.sales_per_share ?? models.mean_ps.parameters.sales_per_share ?? 0) * fxRate, currency)}
                                                isCustom={customOverrides.mean_ps?.sales_per_share !== undefined}
                                                defaultValue={formatCurrency((models.mean_ps.parameters.sales_per_share ?? 0) * fxRate, currency)}
                                            />
                                            <ParamItem
                                                label="Mean P/S Multiple"
                                                value={`${(customOverrides.mean_ps?.applied_ps ?? models.mean_ps.parameters.applied_ps ?? 0).toFixed(2)}x`}
                                                isCustom={customOverrides.mean_ps?.applied_ps !== undefined}
                                                defaultValue={`${(models.mean_ps.parameters.applied_ps ?? 0).toFixed(2)}x`}
                                                info={VALUATION_INFO.mean_ps}
                                            />
                                            <ParamItem
                                                label="Multiple Source"
                                                value={models.mean_ps.parameters.ps_source || 'Historical Mean'}
                                            />
                                            <TradedRangeItem params={models.mean_ps.parameters} digits={2} />
                                            <ParamItem
                                                label="Fair Multiplier"
                                                value={`${(customOverrides.mean_ps?.applied_ps ?? models.mean_ps.parameters.applied_ps ?? 0).toFixed(2)}x Sales`}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Early-stage or cyclical growth companies not yet consistently profitable, where top-line revenue reflects commercial traction."
                                        keyCaveats="Ignores profit margins and cash burn entirely; a business can grow revenue rapidly while accumulating severe cash flow deficits."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.mean_ps.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('mean_ps')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 9. Price-to-Sales Growth (PSG) */}
                    {showModel('psg') && models.psg && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'psg' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.psg && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'Price-to-Sales Growth (PSG)',
                                <Sparkles className="w-5 h-5 text-fuchsia-500" />,
                                'psg',
                                models.psg
                            )}
                            {models.psg.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.psg.error}</p>
                            ) : models.psg.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('psg') ? (
                                        <ParameterEditor
                                            modelKey="psg"
                                            defaultParams={models.psg.parameters}
                                            customValues={customOverrides.psg || {}}
                                            onChange={(k, v) => handleParamChange('psg', k, v)}
                                            onReset={() => handleResetModel('psg')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="Sales / Share"
                                                value={formatCurrency((customOverrides.psg?.sales_per_share ?? models.psg.parameters.sales_per_share ?? 0) * fxRate, currency)}
                                                isCustom={customOverrides.psg?.sales_per_share !== undefined}
                                                defaultValue={formatCurrency((models.psg.parameters.sales_per_share ?? 0) * fxRate, currency)}
                                            />
                                            <ParamItem
                                                label="Revenue Growth"
                                                value={`${(customOverrides.psg?.revenue_growth_pct ?? models.psg.parameters.applied_growth_pct ?? 0).toFixed(1)}%`}
                                                isCustom={customOverrides.psg?.revenue_growth_pct !== undefined}
                                                defaultValue={`${(models.psg.parameters.applied_growth_pct ?? 0).toFixed(1)}%`}
                                                info={VALUATION_INFO.growth_rate}
                                            />
                                            <ParamItem
                                                label="Gross Margin"
                                                value={`${(customOverrides.psg?.gross_margin_pct ?? models.psg.parameters.gross_margin_pct ?? 0).toFixed(1)}%`}
                                                isCustom={customOverrides.psg?.gross_margin_pct !== undefined}
                                                defaultValue={`${(models.psg.parameters.gross_margin_pct ?? 0).toFixed(1)}%`}
                                            />
                                            <ParamItem
                                                label="Target PSG Ratio"
                                                value={`${(customOverrides.psg?.target_psg ?? models.psg.parameters.target_psg ?? 1.0).toFixed(1)}x`}
                                                isCustom={customOverrides.psg?.target_psg !== undefined}
                                                defaultValue={`${(models.psg.parameters.target_psg ?? 1.0).toFixed(1)}x`}
                                                info={VALUATION_INFO.psg_multiplier}
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="High-growth, unprofitable software and tech businesses, scaling top-line revenue growth weighted by gross margin quality."
                                        keyCaveats="Assumes rapid revenue expansion will eventually achieve profitable operating leverage; breaks down quickly if revenue growth decelerates."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.psg.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('psg')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 10. Benjamin Graham's Revised Formula */}
                    {showModel('graham') && models.graham && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'graham' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.graham && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                models.graham.model || "Graham's Formula",
                                <Scale className="w-5 h-5 text-amber-500" />,
                                'graham',
                                models.graham
                            )}
                            {models.graham.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-4 rounded-xl">{models.graham.error}</p>
                            ) : models.graham.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('graham') ? (
                                        <ParameterEditor
                                            modelKey="graham"
                                            defaultParams={models.graham.parameters}
                                            customValues={customOverrides.graham || {}}
                                            onChange={(k, v) => handleParamChange('graham', k, v)}
                                            onReset={() => handleResetModel('graham')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-4">
                                            <ParamItem
                                                label="Trailing EPS"
                                                value={(customOverrides.graham?.eps ?? models.graham.parameters.eps ?? 0).toFixed(2)}
                                                isCustom={customOverrides.graham?.eps !== undefined}
                                                defaultValue={(models.graham.parameters.eps ?? 0).toFixed(2)}
                                                info={VALUATION_INFO.eps}
                                            />
                                            <ParamItem
                                                label="Growth Rate (g)"
                                                value={`${(customOverrides.graham?.growth_rate_pct ?? models.graham.parameters.growth_rate_pct ?? 0).toFixed(2)}%`}
                                                isCustom={customOverrides.graham?.growth_rate_pct !== undefined}
                                                defaultValue={`${(models.graham.parameters.growth_rate_pct ?? 0).toFixed(2)}%`}
                                                info={VALUATION_INFO.graham_growth}
                                            />
                                            <ParamItem
                                                label="Bond Yield (Y)"
                                                value={`${(customOverrides.graham?.bond_yield_proxy ?? models.graham.parameters.bond_yield_proxy ?? 0).toFixed(2)}%`}
                                                isCustom={customOverrides.graham?.bond_yield_proxy !== undefined}
                                                defaultValue={`${(models.graham.parameters.bond_yield_proxy ?? 0).toFixed(2)}%`}
                                                info={VALUATION_INFO.bond_yield}
                                            />
                                            {models.graham.parameters.graham_number && (
                                                <ParamItem
                                                    label="Graham Number"
                                                    value={formatCurrency(models.graham.parameters.graham_number * fxRate, currency)}
                                                    className="text-amber-400 font-semibold"
                                                    info={{ description: "Classical Graham Number = sqrt(22.5 * EPS * BVPS)", default: "Conservative Screen" }}
                                                />
                                            )}
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Defensive value screening comparing EPS and moderate growth against prevailing AAA corporate bond yield opportunity cost."
                                        keyCaveats="Formula multiplier is aggressive if growth inputs are elevated; relies on normalized EPS and requires stability growth caps."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.graham.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('graham')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 11. Peter Lynch Fair Value */}
                    {showModel('lynch') && models.lynch && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'lynch' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.lynch && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'Peter Lynch Fair Value',
                                <Calculator className="w-5 h-5 text-cyan-500" />,
                                'lynch',
                                models.lynch
                            )}
                            {models.lynch.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-3 rounded-xl">{models.lynch.error}</p>
                            ) : models.lynch.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('lynch') ? (
                                        <ParameterEditor
                                            modelKey="lynch"
                                            defaultParams={models.lynch.parameters}
                                            customValues={customOverrides.lynch || {}}
                                            onChange={(k, v) => handleParamChange('lynch', k, v)}
                                            onReset={() => handleResetModel('lynch')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-3 text-xs">
                                            <ParamItem
                                                label="EPS"
                                                value={(customOverrides.lynch?.eps ?? models.lynch.parameters.eps ?? 0).toFixed(2)}
                                                isCustom={customOverrides.lynch?.eps !== undefined}
                                                defaultValue={(models.lynch.parameters.eps ?? 0).toFixed(2)}
                                                info={VALUATION_INFO.eps}
                                            />
                                            <ParamItem
                                                label="Growth + Yield"
                                                value={`${((customOverrides.lynch?.growth_rate_pct ?? models.lynch.parameters.growth_rate_pct ?? 0) + (customOverrides.lynch?.dividend_yield_pct ?? models.lynch.parameters.dividend_yield_pct ?? 0)).toFixed(1)}%`}
                                                isCustom={customOverrides.lynch?.growth_rate_pct !== undefined || customOverrides.lynch?.dividend_yield_pct !== undefined}
                                                defaultValue={`${((models.lynch.parameters.growth_rate_pct ?? 0) + (models.lynch.parameters.dividend_yield_pct ?? 0)).toFixed(1)}%`}
                                                info={VALUATION_INFO.lynch_multiplier}
                                            />
                                            <ParamItem
                                                label="Fair P/E Multiplier"
                                                value={`${Math.min(Math.max((customOverrides.lynch?.growth_rate_pct ?? models.lynch.parameters.growth_rate_pct ?? 0) + (customOverrides.lynch?.dividend_yield_pct ?? models.lynch.parameters.dividend_yield_pct ?? 0), 5.0), 25.0).toFixed(1)}x`}
                                                className="text-cyan-400 font-bold"
                                            />
                                            <ParamItem
                                                label="Principle"
                                                value="PEG = 1.0 Fair Value"
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Fast rule-of-thumb valuation equating fair P/E multiple to expected earnings growth rate plus dividend yield (PEG=1.0 benchmark)."
                                        keyCaveats="Heuristic benchmark that does not account for cost of capital, multi-stage growth decay, or balance sheet solvency."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.lynch.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('lynch')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}

                    {/* 12. Earnings Power Value (EPV Floor) */}
                    {showModel('epv') && models.epv && (
                        <div className={cn(
                            "bg-muted rounded-2xl p-6 transition-all",
                            recommended_method?.method_key === 'epv' && "ring-2 ring-indigo-500/50 shadow-lg shadow-indigo-500/5",
                            customOverrides.epv && "border border-amber-500/40"
                        )}>
                            {renderCardHeader(
                                'Earnings Power Value (EPV Floor)',
                                <Anchor className="w-5 h-5 text-sky-400" />,
                                'epv',
                                models.epv
                            )}
                            {models.epv.error ? (
                                <p className="text-sm text-destructive bg-destructive/5 p-3 rounded-xl">{models.epv.error}</p>
                            ) : models.epv.parameters ? (
                                <div className="space-y-4">
                                    {editingModelKeys.has('epv') ? (
                                        <ParameterEditor
                                            modelKey="epv"
                                            defaultParams={models.epv.parameters}
                                            customValues={customOverrides.epv || {}}
                                            onChange={(k, v) => handleParamChange('epv', k, v)}
                                            onReset={() => handleResetModel('epv')}
                                            currency={currency}
                                            fxRate={fxRate}
                                        />
                                    ) : (
                                        <div className="grid grid-cols-2 gap-3 text-xs">
                                            <ParamItem
                                                label="Normalized NOPAT"
                                                value={formatCurrency(((customOverrides.epv?.normalized_ebit ?? models.epv.parameters.normalized_ebit ?? 0) * (1 - (customOverrides.epv?.tax_rate ?? models.epv.parameters.tax_rate ?? 0.21))) * fxRate, currency)}
                                                isCustom={customOverrides.epv?.normalized_ebit !== undefined || customOverrides.epv?.tax_rate !== undefined}
                                                defaultValue={formatCurrency((models.epv.parameters.nopat ?? 0) * fxRate, currency)}
                                            />
                                            <ParamItem
                                                label="Cost of Capital"
                                                value={formatPercentShared(customOverrides.epv?.discount_rate ?? models.epv.parameters.discount_rate ?? 0)}
                                                isCustom={customOverrides.epv?.discount_rate !== undefined}
                                                defaultValue={formatPercentShared(models.epv.parameters.discount_rate ?? 0)}
                                            />
                                            <ParamItem
                                                label="Net Cash Added"
                                                value={formatCurrency((customOverrides.epv?.net_cash ?? models.epv.parameters.net_cash ?? 0) * fxRate, currency)}
                                                isCustom={customOverrides.epv?.net_cash !== undefined}
                                                defaultValue={formatCurrency((models.epv.parameters.net_cash ?? 0) * fxRate, currency)}
                                            />
                                            <ParamItem
                                                label="Growth Assumption"
                                                value="0.0% (Zero Growth)"
                                            />
                                        </div>
                                    )}
                                    <LimitationCallout
                                        bestSuitedFor="Conservative valuation of normalized sustainable operating earnings in perpetuity assuming zero future growth."
                                        keyCaveats="Strictly a no-growth baseline floor; gives zero credit to value-accretive capital reinvestment or growth opportunities."
                                    />
                                    <MonteCarloPillRow
                                        mc={models.epv.mc}
                                        fxRate={fxRate}
                                        currency={currency}
                                        onOpenModal={() => setViewingDistribution('epv')}
                                    />
                                </div>
                            ) : null}
                        </div>
                    )}
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
                                    {MODEL_TITLES[viewingDistribution] || viewingDistribution.toUpperCase()} Probabilistic Distribution
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

                                        const model = intrinsicValue.models[modelKey as keyof typeof intrinsicValue.models];
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

                                        return (
                                            <AreaChart
                                                data={histogram}
                                                margin={{ top: 35, right: 20, left: 20, bottom: 0 }}
                                            >
                                                <defs>
                                                    <linearGradient id="colorBellFill" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                                                        <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                                                    </linearGradient>
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
                                                            return (
                                                                <div className="p-3 rounded-xl outline-none scale-105 transition-transform bg-white/95 dark:bg-slate-950/95 backdrop-blur-md">
                                                                    <p className="text-[10px] uppercase font-bold mb-1 text-slate-400">Estimated Value</p>
                                                                    <p className="text-lg font-black text-slate-900 dark:text-white">{formatCurrency(payload[0].payload.price, currency)}</p>
                                                                    <p className="text-[10px] font-black text-indigo-500">{((Number(payload[0].value) / 10000) * 100).toFixed(2)}% Probability</p>
                                                                </div>
                                                            );
                                                        }
                                                        return null;
                                                    }}
                                                />
                                                <Area
                                                    type="monotone"
                                                    dataKey="count"
                                                    stroke="#6366f1"
                                                    strokeWidth={2}
                                                    fill="url(#colorBellFill)"
                                                    isAnimationActive={false}
                                                />
                                                <ReferenceLine
                                                    x={scaledMc.bear}
                                                    stroke="#f43f5e"
                                                    strokeDasharray="3 3"
                                                    label={{ value: 'Bear', position: 'top', fill: '#f43f5e', fontSize: 10, fontWeight: 'bold' }}
                                                />
                                                <ReferenceLine
                                                    x={scaledMc.base}
                                                    stroke="#06b6d4"
                                                    strokeDasharray="3 3"
                                                    label={{ value: 'Median', position: 'top', fill: '#06b6d4', fontSize: 10, fontWeight: 'bold' }}
                                                />
                                                <ReferenceLine
                                                    x={scaledMc.bull}
                                                    stroke="#10b981"
                                                    strokeDasharray="3 3"
                                                    label={{ value: 'Bull', position: 'top', fill: '#10b981', fontSize: 10, fontWeight: 'bold' }}
                                                />
                                                {currentPriceVal > 0 && (
                                                    <ReferenceLine
                                                        x={currentPriceVal}
                                                        stroke="#a855f7"
                                                        strokeWidth={2}
                                                        label={{ value: 'Current Price', position: 'bottom', fill: '#a855f7', fontSize: 10, fontWeight: 'bold' }}
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
                                    const mc = intrinsicValue.models[modelKey as keyof typeof intrinsicValue.models]?.mc;
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
