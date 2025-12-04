import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Holding } from '../lib/api';

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
    [key: string]: any;
}

export default function Allocation({ holdings, currency }: AllocationProps) {
    if (!holdings || holdings.length === 0) {
        return <div className="p-4 text-center text-gray-500">No holdings data available.</div>;
    }

    const marketValueKey = `Market Value (${currency})`;

    const aggregateData = (key: keyof Holding | 'Sector' | 'Industry' | 'Country' | 'quoteType'): AggregatedData[] => {
        const aggregation: Record<string, number> = {};
        let totalValue = 0;

        holdings.forEach(h => {
            const value = h[marketValueKey] || 0;
            const category = (h[key] as string) || 'Unknown';
            aggregation[category] = (aggregation[category] || 0) + value;
            totalValue += value;
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
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700 flex flex-col h-96">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 text-center">{title}</h3>
            <div className="flex-grow">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={data}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                        >
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                        </Pie>
                        <Tooltip
                            formatter={(value: number) => new Intl.NumberFormat(undefined, { style: 'currency', currency: currency }).format(value)}
                        />
                        <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: '12px' }} />
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
            {renderPieChart("Allocation by Geography", countryData)}
        </div>
    );
}
