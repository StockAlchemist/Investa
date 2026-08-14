import { ColumnGroupDef } from './types';

export const CASH_DUST_THRESHOLD = 0.01;

// Mapping from UI Header to Data Key Prefix (or exact key)
// Based on src/utils.py
export const COLUMN_DEFINITIONS: { [header: string]: string } = {
    "Account": "Account",
    "Symbol": "Symbol",
    "Sector": "Sector",
    "Industry": "Industry",
    "Quantity": "Quantity",
    "Day Chg": "Day Change", // Suffix added dynamically
    "Day Chg %": "Day Change %",
    "Avg Cost": "Avg Cost",
    "Price": "Price",
    "Cost Basis": "Cost Basis",
    "Mkt Val": "Market Value",
    "% of Total": "pct_of_total",
    "Unreal. G/L": "Unreal. Gain",
    "Unreal. G/L %": "Unreal. Gain %",
    "Real. G/L": "Realized Gain",
    "Divs": "Dividends",
    "Fees": "Commissions",
    "Total G/L": "Total Gain",
    "Total Ret %": "Total Return %",
    "IRR (%)": "IRR (%)",
    "Total Buy Cost": "Total Buy Cost",
    "Yield (Cost) %": "Div. Yield (Cost) %",
    "Yield (Mkt) %": "Div. Yield (Current) %",
    "FX G/L %": "FX Gain/Loss %",
    "Est. Income": "Est. Ann. Income",
    "1M Trend": "sparkline_1m",
    "Tags": "Tags",
    "Contribution %": "Contribution %",
    "AI Score": "ai_score",
    "Intrinsic Value": "intrinsic_value",
};

// Columns that have been renamed, so a layout saved under the old name is
// restored rather than dropped.
export const RENAMED_COLUMNS: Record<string, string> = {
    "7d Trend": "1M Trend",
};

export const DEFAULT_VISIBLE_COLUMNS = [
    "Symbol", "1M Trend", "Quantity", "% of Total", "Price", "Mkt Val", "Day Chg", "Day Chg %", "Unreal. G/L"
];

export const GROUPING_LABEL_MAP: Record<string, string> = {
    'Market': 'Market',
    'Currency': 'Currency',
    'Sector': 'Sector',
    'Industry': 'Industry',
    'quoteType': 'Investment Type',
    'Country': 'Country',
};

export const INVESTMENT_TYPE_MAP: Record<string, string> = {
    'EQUITY': 'Stocks',
    'ETF': 'ETFs',
    'CASH': 'Cash',
    'MUTUALFUND': 'Mutual Funds',
};

export const CURRENCY_MAP: Record<string, string> = {
    'USD': 'US Dollar',
    'THB': 'Thai Baht',
    'EUR': 'Euro',
    'GBP': 'British Pound',
    'SGD': 'Singapore Dollar',
    'JPY': 'Japanese Yen',
    'HKD': 'Hong Kong Dollar',
};

export const COLUMN_GROUPS: ColumnGroupDef[] = [
    { label: 'Core', cols: ['Symbol', 'Account', 'Quantity', 'Price', 'Mkt Val', '% of Total', '1M Trend'] },
    { label: 'Daily', cols: ['Day Chg', 'Day Chg %'] },
    { label: 'Returns', cols: ['Unreal. G/L', 'Unreal. G/L %', 'Real. G/L', 'Total G/L', 'Total Ret %', 'IRR (%)'] },
    { label: 'Cost', cols: ['Avg Cost', 'Cost Basis', 'Total Buy Cost'] },
    { label: 'Income', cols: ['Divs', 'Est. Income', 'Yield (Cost) %', 'Yield (Mkt) %'] },
    { label: 'Details', cols: ['Sector', 'Industry', 'FX G/L %', 'Fees', 'Tags', 'Contribution %', 'AI Score', 'Intrinsic Value'] },
];
