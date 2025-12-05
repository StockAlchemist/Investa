export const CURRENCY_SYMBOLS: { [key: string]: string } = {
    "USD": "$",
    "THB": "฿",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CAD": "$",
    "AUD": "$",
    "CHF": "Fr",
    "CNY": "¥",
    "HKD": "$",
    "SGD": "$",
};

export function formatCurrency(value: number, currency: string): string {
    const symbol = CURRENCY_SYMBOLS[currency] || currency;

    // Handle cases where we want specific formatting
    // For now, standard locale string with symbol prefix
    return `${symbol}${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}
