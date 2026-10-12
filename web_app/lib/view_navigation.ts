import { ALL_NAV } from './navigation';

export interface ViewSelection {
    section: string;
    currency: string;
    accounts: string[];
    symbol: string | null;
}

export function readView(search: string): Partial<ViewSelection> {
    const query = new URLSearchParams(search);
    const section = query.get('section');
    const currency = query.get('currency');
    const symbol = query.get('symbol');
    return {
        ...(ALL_NAV.some(item => item.id === section) ? { section: section! } : {}),
        ...(currency && /^[A-Z]{3,6}$/.test(currency) ? { currency } : {}),
        ...(query.has('section') ? { accounts: query.getAll('account').filter(a => a.length <= 100) } : {}),
        ...(query.has('section') ? { symbol: symbol && /^[A-Za-z0-9.^=\-$ ]{1,40}$/.test(symbol) ? symbol.toUpperCase() : null } : {}),
    };
}

export function writeView(selection: ViewSelection, search = ''): string {
    const query = new URLSearchParams(search);
    for (const key of ['section', 'currency', 'account', 'symbol']) query.delete(key);
    query.set('section', selection.section);
    query.set('currency', selection.currency);
    [...new Set(selection.accounts)].sort().forEach(account => query.append('account', account));
    if (selection.symbol) query.set('symbol', selection.symbol);
    return `?${query.toString()}`;
}
