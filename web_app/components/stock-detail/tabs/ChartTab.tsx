import React from 'react';
import StockPriceChart from '../../StockPriceChart';

interface ChartTabProps {
    symbol: string;
    currency: string;
    avgCost?: number;
    fxRate?: number;
    accounts?: string[];
    exchange?: string;
}

export const ChartTab: React.FC<ChartTabProps> = ({
    symbol,
    currency,
    avgCost,
    fxRate,
    accounts = [],
    exchange,
}) => {
    return (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <StockPriceChart
                symbol={symbol}
                currency={currency}
                avgCost={avgCost}
                fxRate={fxRate}
                accounts={accounts}
                hidePrice={true}
                exchange={exchange}
            />
        </div>
    );
};
