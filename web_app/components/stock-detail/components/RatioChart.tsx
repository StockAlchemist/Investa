import React from 'react';
import {
    AreaChart,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Area,
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
    const formatValue = (val: number) =>
        compact
            ? new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(val)
            : `${val}${suffix}`;

    return (
        <div className="bg-muted p-6 rounded-2xl">
            <h4 className="text-sm font-semibold text-muted-foreground mb-6 uppercase tracking-wider">{title}</h4>
            <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id={sanitizedId} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={color} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" opacity={0.1} vertical={false} />
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
                            tickFormatter={(val) => formatValue(Number(val))}
                        />
                        <Tooltip
                            wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                            content={({ active, payload, label }) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div className="bg-white/95 dark:bg-slate-950/95 backdrop-blur-md p-3 rounded-xl text-xs">
                                            <p className="font-medium text-foreground mb-1">{formatCalendarDate(String(label))}</p>
                                            <div className="flex items-center gap-2">
                                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                                                <span className="text-muted-foreground">{title}:</span>
                                                <span className="font-bold text-foreground">
                                                    {compact ? formatValue(Number(payload[0].value)) : `${Number(payload[0].value).toFixed(2)}${suffix}`}
                                                </span>
                                            </div>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                            cursor={{ stroke: 'var(--border)', strokeWidth: 1, strokeDasharray: '3 3' }}
                        />
                        <Area
                            type="monotone"
                            dataKey={dataKey}
                            stroke={color}
                            strokeWidth={3}
                            fillOpacity={1}
                            fill={`url(#${sanitizedId})`}
                            animationDuration={1500}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
