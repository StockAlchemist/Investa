import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
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

    const renderPieChart = (title: string, data: AggregatedData[]) => (
        <div className="bg-card p-4 rounded-xl shadow-sm border border-border flex flex-col h-[32rem]">
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
                        >
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                        </Pie>
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'var(--menu-solid)',
                                border: '1px solid var(--border)',
                                borderRadius: '0.75rem',
                                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
                            }}
                            content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div className="border border-border p-3 rounded-lg shadow-xl" style={{ backgroundColor: 'var(--menu-solid)' }}>
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

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4">
            {renderPieChart("Allocation by Asset Type", assetTypeData)}
            {renderPieChart("Allocation by Sector", sectorData)}
            {renderPieChart("Allocation by Industry", industryData)}
            {renderPieChart("Allocation by Country", countryData)}
        </div>
    );
}
