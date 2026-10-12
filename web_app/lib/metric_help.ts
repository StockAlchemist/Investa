export function metricHelp(title: string): string | undefined {
    if (/TWR/i.test(title)) return 'Time-weighted return measures investment performance after removing the effect of deposits and withdrawals. Annualized figures express the equivalent yearly rate.';
    if (/IRR|MWR/i.test(title)) return 'Money-weighted return (IRR) accounts for the amounts and dates of your cash flows. Annualized figures express the equivalent yearly rate.';
    if (/Yield.*Cost|Yield on Cost/i.test(title)) return 'Estimated annual income divided by the original cost of the holdings. It differs from yield based on current market value.';
    if (/Dividend Yield|Yield.*Mkt/i.test(title)) return 'Estimated annual dividend income divided by current market value. Future payments may change.';
    return undefined;
}
