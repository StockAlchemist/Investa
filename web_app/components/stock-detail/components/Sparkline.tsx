import React, { useId } from 'react';
import { ResponsiveContainer, AreaChart, Area, YAxis, ReferenceLine } from 'recharts';

export const Sparkline: React.FC<{ data: number[] }> = ({ data }) => {
    const id = useId();
    if (!data || data.length < 2) return null;

    // Filter out null/undefined and reverse to chronological order (oldest to newest)
    const values = [...data].filter(v => v !== null && v !== undefined).reverse();
    if (values.length < 2) return null;

    const baseline = values[0];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const off = range <= 0 ? 0 : (max - baseline) / range;

    return (
        <div className="h-10 w-28 mx-auto">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={values.map((v, i) => ({ value: v, index: i }))}>
                    <defs>
                        <linearGradient id={`splitFill-${id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset={off} stopColor="#1F9D6C" stopOpacity={0.15} />
                            <stop offset={off} stopColor="#D2491F" stopOpacity={0.15} />
                        </linearGradient>
                        <linearGradient id={`splitStroke-${id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset={off} stopColor="#1F9D6C" stopOpacity={1} />
                            <stop offset={off} stopColor="#D2491F" stopOpacity={1} />
                        </linearGradient>
                    </defs>
                    <YAxis hide domain={['dataMin', 'dataMax']} />
                    <ReferenceLine y={baseline} stroke="#6A6C74" strokeDasharray="2 2" strokeOpacity={0.3} />
                    <Area
                        type="monotone"
                        dataKey="value"
                        baseValue={baseline}
                        stroke={`url(#splitStroke-${id})`}
                        fill={`url(#splitFill-${id})`}
                        strokeWidth={1.5}
                        isAnimationActive={false}
                        dot={(props: { cx?: number; cy?: number; index?: number }) => {
                            const { cx, cy, index } = props;
                            if (index === values.length - 1) {
                                const color = values[values.length - 1] >= baseline ? "#1F9D6C" : "#D2491F";
                                return (
                                    <circle key="dot" cx={cx} cy={cy} r={2} fill={color} stroke="none" />
                                );
                            }
                            return <React.Fragment key={index} />;
                        }}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};
