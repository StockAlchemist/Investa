import React, { useId } from 'react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';

interface TrendSparklineProps {
    data?: number[];
    color?: string; // Optional Hex color to force a specific color
    width?: number | string;
    height?: number | string;
}

export function TrendSparkline({
    data,
    color,
    width = "100%",
    height = 32,
}: TrendSparklineProps) {
    const id = useId();

    if (!data || data.length < 2) {
        return <div className="text-muted-foreground text-xs text-center w-full">N/A</div>;
    }

    const chartData = data.map((v) => ({ value: v }));

    // Auto color based on trend if no explicit color provided
    const isUp = data[data.length - 1] >= data[0];
    const autoColor = isUp ? "#10b981" : "#f43f5e"; // Emerald or Rose
    const finalColor = color || autoColor;

    const gradientId = `sparkline-gradient-${id}`;

    return (
        <div style={{ width, height }} className="filter drop-shadow-sm min-w-[60px] max-w-[120px]">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                    <defs>
                        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={finalColor} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={finalColor} stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <YAxis hide domain={['dataMin', 'dataMax']} />
                    <Area
                        type="monotone"
                        dataKey="value"
                        stroke={finalColor}
                        fill={`url(#${gradientId})`}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}
