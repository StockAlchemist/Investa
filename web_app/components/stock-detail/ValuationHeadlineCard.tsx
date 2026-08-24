import React from 'react';
import { Anchor, Layers, Sparkles } from 'lucide-react';
import { cn, formatCurrency } from '../../lib/utils';
import type { IntrinsicValueResponse } from '@/lib/api';

interface ValuationHeadlineCardProps {
    intrinsicValue: IntrinsicValueResponse;
    /** Blended value actually on screen — the custom one when parameters are edited. */
    displayAverage?: number | null;
    displayMos?: number | null;
    hasAnyCustom: boolean;
    currency: string;
    fxRate: number;
}

/** The short form of the blend profile; the composition card below spells it out. */
const PROFILE_TAG: Record<string, string> = {
    financial: 'Financial',
    reit: 'REIT',
    operating: 'Operating company',
};

const isPlottable = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0;

/**
 * Price and intrinsic value on one scale, with the models' bear-to-bull band
 * behind them and the gap between the two — the margin of safety — drawn as a
 * solid bar through the middle of it.
 *
 * The domain is padded past the outermost value so that no marker lands on the
 * very edge of the track, where its label would have nowhere to sit.
 */
const ValueLine: React.FC<{
    value: number;
    price: number;
    bear?: number | null;
    bull?: number | null;
    accentBar: string;
    accentText: string;
    gapBar: string;
    currency: string;
    fxRate: number;
}> = ({ value, price, bear, bull, accentBar, accentText, gapBar, currency, fxRate }) => {
    const hasBand = isPlottable(bear) && isPlottable(bull) && bull > bear;
    const points = [value, price, ...(hasBand ? [bear as number, bull as number] : [])];
    const rawLo = Math.min(...points);
    const rawHi = Math.max(...points);
    const pad = (rawHi - rawLo) * 0.08 || Math.abs(rawHi) * 0.08 || 1;
    const lo = rawLo - pad;
    const hi = rawHi + pad;
    const pos = (v: number) => (hi > lo ? ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * 100 : 50);

    const xValue = pos(value);
    const xPrice = pos(price);
    const bandStart = hasBand ? pos(bear as number) : 0;
    const bandWidth = hasBand ? pos(bull as number) - bandStart : 0;
    return (
        <div className={cn('relative px-4 sm:px-8 pt-8', hasBand ? 'pb-[4.5rem] sm:pb-14' : 'pb-11')}>
            <div className="relative h-2.5 rounded-full bg-secondary">
                {hasBand && (
                    <div
                        className={cn('absolute inset-y-0 rounded-full opacity-25', accentBar)}
                        style={{ left: `${bandStart}%`, width: `${Math.max(bandWidth, 0.5)}%` }}
                    />
                )}
                {/* The gap runs through the middle of the band so both stay readable. */}
                <div
                    className={cn('absolute top-1/2 -translate-y-1/2 h-1 rounded-full', gapBar)}
                    style={{
                        left: `${Math.min(xValue, xPrice)}%`,
                        width: `${Math.max(Math.abs(xValue - xPrice), 0.5)}%`,
                    }}
                />

                {/* Markers */}
                <div
                    className={cn(
                        'absolute top-1/2 w-4 h-4 rounded-full shadow-sm ring-2 ring-white dark:ring-slate-950',
                        accentBar
                    )}
                    style={{ left: `${xValue}%`, transform: 'translate(-50%, -50%)' }}
                />
                <div
                    className="absolute top-1/2 w-4 h-4 rounded-full shadow-sm bg-slate-900 dark:bg-white ring-2 ring-white dark:ring-slate-950"
                    style={{ left: `${xPrice}%`, transform: 'translate(-50%, -50%)' }}
                />

                {/* Intrinsic value above the track, price below — they never collide. */}
                <div
                    className="absolute bottom-full mb-2.5 whitespace-nowrap"
                    style={{ left: `${xValue}%`, transform: 'translateX(-50%)' }}
                >
                    <p className={cn('text-[10px] font-extrabold uppercase tracking-wider', accentText)}>
                        Intrinsic value
                    </p>
                </div>
                <div
                    className="absolute top-full mt-2 text-center whitespace-nowrap"
                    style={{ left: `${xPrice}%`, transform: 'translateX(-50%)' }}
                >
                    <p className="text-[13px] font-bold tabular-nums">{formatCurrency(price * fxRate, currency)}</p>
                    <p className="text-[9px] font-extrabold uppercase tracking-wider text-muted-foreground">Price</p>
                </div>

            </div>

            {hasBand && (
                <div className="absolute left-4 sm:left-auto sm:right-8 bottom-3 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    <span className={cn('inline-block w-4 h-1.5 rounded-full opacity-25', accentBar)} />
                    <span className="font-semibold uppercase tracking-wider">Model range</span>
                    <span className="font-bold tabular-nums">
                        {formatCurrency((bear as number) * fxRate, currency)} – {formatCurrency((bull as number) * fxRate, currency)}
                    </span>
                </div>
            )}
        </div>
    );
};

/** A labelled fact that sits under the headline number without competing with it. */
const FactChip: React.FC<{ icon: React.ReactNode; label: string; value: string }> = ({ icon, label, value }) => (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-muted border border-border/60">
        <span className="text-muted-foreground">{icon}</span>
        <span className="text-[10px] font-extrabold uppercase tracking-wider text-muted-foreground">{label}</span>
        <span className="text-[11px] font-bold tabular-nums">{value}</span>
    </span>
);

/**
 * How much the backend stands behind the number, as a bar. Confidence is
 * continuous — the models' own Monte Carlo bands, how far apart they landed,
 * and how many of them there were — so it is shown as a level rather than the
 * old pass/fail that read as fine at 99% disagreement and alarming at 101%.
 */
const ConfidenceChip: React.FC<{ confidence: number }> = ({ confidence }) => {
    const pct = Math.max(0, Math.min(1, confidence));
    const tone = pct >= 0.66 ? 'bg-emerald-500' : pct >= 0.4 ? 'bg-amber-500' : 'bg-rose-500';
    const text = pct >= 0.66 ? 'text-emerald-500' : pct >= 0.4 ? 'text-amber-500' : 'text-rose-500';
    return (
        <span
            className="inline-flex items-center gap-2 px-2.5 py-1.5 rounded-full bg-muted border border-border/60"
            aria-label={`Valuation confidence ${Math.round(pct * 100)} percent`}
        >
            <span className="text-[10px] font-extrabold uppercase tracking-wider text-muted-foreground">Confidence</span>
            <span className="h-[5px] w-[54px] rounded-full bg-secondary/80 overflow-hidden">
                <span className={cn('block h-full rounded-full transition-all', tone)} style={{ width: `${pct * 100}%` }} />
            </span>
            <span className={cn('text-[11px] font-bold tabular-nums', text)}>{Math.round(pct * 100)}%</span>
        </span>
    );
};

/**
 * The valuation headline: the blended intrinsic value, the margin of safety,
 * and — the part three separate stat cards could never show — where today's
 * price actually sits inside the range the models produced.
 *
 * The scale is the whole point. A reader looking at "$502.40" beside "$341.83"
 * has to do the arithmetic and then guess how much of the gap is signal; a
 * reader looking at the price sitting near the bottom of a wide bear-to-bull
 * band can see both at once.
 *
 * Mirrors `ValuationHeadlineCard` in
 * `macos_app/Investa/Features/StockDetail/StockValuationTabView.swift`.
 */
const ValuationHeadlineCard: React.FC<ValuationHeadlineCardProps> = ({
    intrinsicValue,
    displayAverage,
    displayMos,
    hasAnyCustom,
    currency,
    fxRate,
}) => {
    const value = displayAverage ?? null;
    const mos = displayMos ?? null;
    const price = intrinsicValue.current_price;
    const defaultValue = intrinsicValue.average_intrinsic_value;

    // Custom parameters recolor the card amber, the way the alert bar and the
    // edited model cards already do, so an edited number never reads as the
    // backend's own.
    const accentText = hasAnyCustom ? 'text-amber-600 dark:text-amber-400' : 'text-indigo-500';
    const accentBar = hasAnyCustom ? 'bg-amber-500' : 'bg-indigo-500';
    const mosPositive = mos !== null && mos >= 0;
    const mosText = mos === null ? 'text-muted-foreground' : mosPositive ? 'text-emerald-500' : 'text-rose-500';
    const mosBg = mos === null ? 'bg-muted' : mosPositive ? 'bg-emerald-500/10' : 'bg-rose-500/10';
    const gapBar = mos === null ? 'bg-secondary' : mosPositive ? 'bg-emerald-500' : 'bg-rose-500';

    const valueLabel =
        intrinsicValue.valuation_status === 'nav'
            ? 'Net Asset Value'
            : hasAnyCustom
              ? 'Custom Blended Value'
              : 'Blended Intrinsic Value';

    const profileTag = intrinsicValue.blend_profile ? PROFILE_TAG[intrinsicValue.blend_profile] : undefined;
    const customDiffPct =
        hasAnyCustom && isPlottable(defaultValue) && value !== null ? ((value - defaultValue) / defaultValue) * 100 : null;

    const floor = intrinsicValue.earnings_power_floor;
    const modelCount = Object.keys(intrinsicValue.model_weights ?? {}).length;
    const confidence = intrinsicValue.valuation_confidence;

    return (
        <div
            data-testid="valuation-headline"
            className={cn(
                'rounded-2xl border bg-card shadow-xs p-5 sm:p-6',
                hasAnyCustom ? 'border-amber-500/40' : 'border-border/70'
            )}
        >
            <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
                <div className="min-w-0">
                    <p className="flex items-center gap-1.5 text-[10px] font-extrabold uppercase tracking-[1.5px] text-muted-foreground">
                        {valueLabel}
                        {hasAnyCustom && <Sparkles className="w-3 h-3 text-amber-500" />}
                    </p>
                    <p className={cn('mt-1 text-4xl sm:text-5xl font-bold tabular-nums', value === null ? 'text-muted-foreground' : accentText)}>
                        {value === null ? 'Not valued' : formatCurrency(value * fxRate, currency)}
                    </p>
                    {customDiffPct !== null && isPlottable(defaultValue) ? (
                        <p className="mt-1.5 text-xs font-semibold text-muted-foreground tabular-nums">
                            Default {formatCurrency(defaultValue * fxRate, currency)}{' '}
                            <span className={customDiffPct >= 0 ? 'text-emerald-500' : 'text-rose-500'}>
                                {customDiffPct >= 0 ? '+' : ''}
                                {customDiffPct.toFixed(1)}%
                            </span>
                        </p>
                    ) : (
                        profileTag && <p className="mt-1.5 text-xs font-medium text-muted-foreground">{profileTag}</p>
                    )}
                </div>

                <div className={cn('rounded-2xl px-4 py-3 shrink-0 sm:text-right', mosBg)}>
                    <p className="text-[10px] font-extrabold uppercase tracking-[1.5px] text-muted-foreground">
                        Margin of Safety
                    </p>
                    <p className={cn('mt-0.5 text-2xl sm:text-3xl font-bold tabular-nums', mosText)}>
                        {mos === null ? '—' : `${mos >= 0 ? '+' : ''}${mos.toFixed(1)}%`}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                        {mos === null
                            ? 'No estimate available'
                            : mosPositive
                              ? 'Undervalued vs market'
                              : 'Overvalued vs market'}
                    </p>
                </div>
            </div>

            {isPlottable(value) && isPlottable(price) && (
                <ValueLine
                    value={value}
                    price={price}
                    bear={hasAnyCustom ? null : intrinsicValue.range?.bear}
                    bull={hasAnyCustom ? null : intrinsicValue.range?.bull}
                    accentBar={accentBar}
                    accentText={accentText}
                    gapBar={gapBar}
                    currency={currency}
                    fxRate={fxRate}
                />
            )}

            {/* The qualifiers that belong beside the number rather than inside it. */}
            <div className="flex flex-wrap items-center gap-2">
                {!hasAnyCustom && typeof confidence === 'number' && Number.isFinite(confidence) && (
                    <ConfidenceChip confidence={confidence} />
                )}
                {isPlottable(floor) && (
                    <FactChip
                        icon={<Anchor className="w-3 h-3" />}
                        label="No-growth floor"
                        value={formatCurrency(floor * fxRate, currency)}
                    />
                )}
                {modelCount > 0 && (
                    <FactChip
                        icon={<Layers className="w-3 h-3" />}
                        label="Blended"
                        value={`${modelCount} model${modelCount === 1 ? '' : 's'}`}
                    />
                )}
            </div>
        </div>
    );
};

export default ValuationHeadlineCard;
