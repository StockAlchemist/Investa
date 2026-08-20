import React from 'react';
import {
    BarChart,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Bar,
    ResponsiveContainer
} from 'recharts';
import { FinancialRatio } from '../../../lib/api';
import { StatementPeriod, periodAxisLabel } from '../../../lib/statement_chart';
import { formatCalendarDate } from '../../../lib/market_time';

interface RatioChartProps {
    data: FinancialRatio[];
    dataKey: string;
    title: React.ReactNode;
    color: string;
    suffix?: string;
    compact?: boolean;
    periodType?: StatementPeriod;
}

export const RatioChart: React.FC<RatioChartProps> = ({
    data,
    dataKey,
    title,
    color,
    suffix = "",
    compact = false,
    periodType = 'annual'
}) => {
    const sanitizedId = `gradient-${dataKey.replace(/[^a-zA-Z0-9]/g, '')}`;
    const formatValue = (val: number) => {
        if (!isFinite(val)) return "-";
        return compact
            ? new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(val)
            : `${val.toFixed(2)}${suffix}`;
    };

    const hasData = data && data.length > 0;
    const validPoints = hasData
        ? data.filter(d => {
            const v = d[dataKey];
            return typeof v === 'number' && isFinite(v);
        })
        : [];

    return (
        <div className="bg-muted/50 p-5 rounded-2xl border border-border/40">
            <div className="flex items-center justify-between gap-2 mb-4">
                <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-wider truncate">{title}</h4>
                {validPoints.length > 0 && (
                    <span className="text-xs font-bold font-mono" style={{ color }}>
                        {formatValue(Number(validPoints[validPoints.length - 1][dataKey]))}
                    </span>
                )}
            </div>

            <div className="h-48 w-full">
                {validPoints.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
                        No historical data filed
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                            <defs>
                                <linearGradient id={sanitizedId} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor={color} stopOpacity={0.95} />
                                    <stop offset="100%" stopColor={color} stopOpacity={0.45} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" opacity={0.15} vertical={false} />
                            <XAxis
                                dataKey="Period"
                                axisLine={false}
                                tickLine={false}
                                tick={{ fontSize: 10 }}
                                className="fill-muted-foreground"
                                tickFormatter={(val) => periodAxisLabel(String(val), periodType)}
                                minTickGap={periodType === 'quarterly' ? 24 : 8}
                            />
                            <YAxis
                                axisLine={false}
                                tickLine={false}
                                tick={{ fontSize: 10 }}
                                className="fill-muted-foreground"
                                domain={['auto', 'auto']}
                                tickFormatter={(val) => formatValue(Number(val))}
                            />
                            <Tooltip
                                wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                                content={({ active, payload, label }) => {
                                    if (active && payload && payload.length) {
                                        const rawVal = payload[0].value;
                                        if (rawVal == null || !isFinite(Number(rawVal))) return null;
                                        return (
                                            <div className="bg-white/95 dark:bg-slate-950/95 backdrop-blur-md p-3 rounded-xl text-xs border border-border shadow-lg">
                                                <p className="font-medium text-foreground mb-1">{formatCalendarDate(String(label))}</p>
                                                <div className="flex items-center gap-2">
                                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                                                    <span className="text-muted-foreground">{title}:</span>
                                                    <span className="font-bold text-foreground">
                                                        {formatValue(Number(rawVal))}
                                                    </span>
                                                </div>
                                            </div>
                                        );
                                    }
                                    return null;
                                }}
                                cursor={{ fill: 'var(--border)', opacity: 0.15 }}
                            />
                            <Bar
                                dataKey={dataKey}
                                fill={`url(#${sanitizedId})`}
                                radius={[4, 4, 0, 0]}
                                maxBarSize={36}
                                animationDuration={800}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                )}
            </div>
        </div>
    );
};
