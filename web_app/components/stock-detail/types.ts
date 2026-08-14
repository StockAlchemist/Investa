export interface StockDetailModalProps {
    symbol: string;
    isOpen: boolean;
    onClose: () => void;
    currency: string;
    initialTab?: string;
}

export type TabType = 'overview' | 'chart' | 'financials' | 'ratios' | 'valuation' | 'holdings' | 'analysis' | 'news';

export interface ChartSeries {
    key: string;
    label: string;
    color: string;
}

export type ChartPoint = Record<string, string | number | null>;

export interface UserPosition {
    Quantity: number;
    "Market Value": number;
    "Cost Basis": number;
    "Total Buy Cost": number;
    "Unreal. Gain": number;
    "Total Gain": number;
    "Dividends": number;
    "Weighted IRR": number;
    fx_rate: number;
}
