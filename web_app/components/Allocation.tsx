import React, { useState } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend, Sector } from 'recharts';
import { Holding } from '../lib/api';
import { formatCurrency } from '../lib/utils';

interface AllocationProps {
    holdings: Holding[];
    currency: string;
}

const COLORS = [
    '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1', '#a4de6c', '#d0ed57'
];

interface AggregatedData {
    name: string;
    value: number;
    [key: string]: unknown;
}

interface AllocationPieChartProps {
    title: string;
    data: AggregatedData[];
    currency: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const renderActiveShape = (props: any) => {
    const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props;
    return (
        <g>
            <Sector
                cx={cx}
                cy={cy}
                innerRadius={innerRadius}
                outerRadius={outerRadius + 8}
                startAngle={startAngle}
                endAngle={endAngle}
                fill={fill}
            />
        </g>
    );
};

function AllocationPieChart({ title, data, currency }: AllocationPieChartProps) {
    const [activeIndex, setActiveIndex] = useState<number | undefined>(undefined);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const onPieEnter = (_: any, index: number) => setActiveIndex(index);
    const onPieLeave = () => setActiveIndex(undefined);

    return (
        <div className="bg-card p-4 rounded-xl shadow-sm flex flex-col h-[32rem] transition-all hover:shadow-md">
            <h3 className="text-lg font-semibold text-foreground mb-4 text-center">{title}</h3>
            <div className="flex-grow min-h-0 relative">
                <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                    <PieChart>
                        <Pie
                            data={data}
                            cx="50%"
                            cy="35%"
                            labelLine={false}
                            outerRadius={110}
                            fill="#8884d8"
                            dataKey="value"
                            stroke="var(--pie-stroke)"
                            onMouseEnter={onPieEnter}
                            onMouseLeave={onPieLeave}
                            // @ts-ignore
                            activeIndex={activeIndex}
                            activeShape={renderActiveShape}
                        >
                            {data.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={COLORS[index % COLORS.length]}
                                    fillOpacity={activeIndex === undefined || activeIndex === index ? 1 : 0.3}
                                    className="transition-all duration-300 outline-none"
                                />
                            ))}
                        </Pie>
                        <Tooltip
                            wrapperStyle={{ opacity: 1, zIndex: 1000 }}
                            contentStyle={{
                                backgroundColor: 'transparent',
                                border: 'none',
                                borderRadius: '0.75rem',
                                boxShadow: 'none'
                            }}
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div className="bg-white/95 dark:bg-slate-950/95 backdrop-blur-md border border-border p-3 rounded-xl shadow-2xl">
                                            <p className="font-medium text-foreground">{payload[0].name}</p>
                                            <p className="text-sm text-muted-foreground">
                                                {formatCurrency(payload[0].value as number, currency)}
                                            </p>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <Legend layout="horizontal" align="center" verticalAlign="bottom" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                    </PieChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

export default function Allocation({ holdings, currency }: AllocationProps) {
    if (!holdings || holdings.length === 0) {
        return <div className="p-4 text-center text-gray-500">No holdings data available.</div>;
    }

    const marketValueKey = `Market Value (${currency})`;

    const aggregateData = (key: keyof Holding | 'Sector' | 'Industry' | 'Country' | 'quoteType'): AggregatedData[] => {
        const aggregation: Record<string, number> = {};

        holdings.forEach(h => {
            const value = (h[marketValueKey] as number) || 0;
            // For Country, prioritize 'geography' (from overrides or improved data) over 'Country'
            let category = 'Unknown';
            if (key === 'Country') {
                category = (h['geography'] as string) || (h['Country'] as string) || 'Unknown';
            } else {
                category = (h[key] as string) || 'Unknown';
            }

            aggregation[category] = (aggregation[category] || 0) + value;
        });

        return Object.entries(aggregation)
            .map(([name, value]) => ({ name, value }))
            .sort((a, b) => b.value - a.value);
    };

    const assetTypeData = aggregateData('quoteType');
    const sectorData = aggregateData('Sector');
    const industryData = aggregateData('Industry');
    const countryData = aggregateData('Country');

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4">
            <AllocationPieChart title="Allocation by Asset Type" data={assetTypeData} currency={currency} />
            <AllocationPieChart title="Allocation by Sector" data={sectorData} currency={currency} />
            <AllocationPieChart title="Allocation by Industry" data={industryData} currency={currency} />
            <AllocationPieChart title="Allocation by Country" data={countryData} currency={currency} />
        </div>
    );
}
