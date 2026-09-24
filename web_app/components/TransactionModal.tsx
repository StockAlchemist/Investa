import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { Transaction } from '../lib/api';
import { marketToday } from '../lib/market_time';

// Dynamic transaction form blob: keys hold strings or auto-computed numbers,
// and fields are added/updated dynamically by name.
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- intentionally permissive: form values are strings or auto-computed numbers keyed by field name
type TxForm = Record<string, any>;

interface TransactionModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSubmit: (data: Transaction) => Promise<void>;
    initialData?: TxForm | null;
    mode: 'add' | 'edit';
    accountCurrencyMap: { [account: string]: string };
    existingAccounts?: string[];
    existingSymbols?: string[];
    accountCashModeMap?: Record<string, string>;
}

const TRANSACTION_TYPES = [
    'Buy', 'Sell', 'Dividend', 'Transfer', 'Interest', 'Fees', 'Tax', 'Deposit', 'Withdrawal', 'Spin-off', 'Split', 'Short Sell', 'Buy To Cover'
];

// Helper matching src/finutils.py
const CASH_SYMBOL_CSV = "$CASH";
const isCashSymbol = (symbol: string) => {
    if (!symbol) return false;
    const s = symbol.toLowerCase();
    return s.startsWith(CASH_SYMBOL_CSV.toLowerCase()) || s.startsWith('cash (');
};

export default function TransactionModal({ isOpen, onClose, onSubmit, initialData, mode, accountCurrencyMap, existingAccounts = [], existingSymbols = [], accountCashModeMap = {} }: TransactionModalProps) {
    const [formData, setFormData] = useState<TxForm>({
        Date: marketToday(),
        Type: 'Buy',
        Symbol: '',
        Quantity: '',
        "Price/Share": '',
        Commission: '',
        "Total Amount": '',
        Account: '',
        "Local Currency": 'USD',
        Note: '',
        "Split Ratio": '',
        "To Account": '',
        "From Account": '',
        "Auto-add Cash": true
    });

    // Track if total was manually edited to avoid auto-overwrite
    const [totalLockedByUser, setTotalLockedByUser] = useState(false);

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [activeSuggestionField, setActiveSuggestionField] = useState<'Symbol' | 'Account' | 'From Account' | 'To Account' | null>(null);

    // --- EFFECT: Reset/Init Form ---
    useEffect(() => {
        if (isOpen) {
            if (mode === 'edit' && initialData) {
                const formattedDate = initialData.Date ? initialData.Date.split('T')[0] : '';

                const fromAcc = initialData.Account || '';
                const toAcc = initialData["To Account"] || '';

                // Normalize the stored Type to the exact select-option string so
                // the dropdown shows it selected (an unmatched value silently
                // falls back to the first option, "Buy"). Ignore case AND
                // hyphen/space differences: the same corporate action can arrive
                // as 'Spin-off' (parser/option), 'Spin-Off' (DB, via .title()),
                // or 'spin off' (engine canonical form) — all must map here.
                let initType = initialData.Type || 'Buy';
                const canon = (s: string) => s.toLowerCase().replace(/[\s-]+/g, '');
                const matchedType = TRANSACTION_TYPES.find(t => canon(t) === canon(initType));
                if (matchedType) {
                    initType = matchedType;
                }

                setFormData({
                    ...initialData,
                    Date: formattedDate,
                    Type: initType,
                    Symbol: initialData.Symbol || '',
                    Quantity: initialData.Quantity || '',
                    "Price/Share": initialData["Price/Share"] || '',
                    Commission: initialData.Commission || '',
                    "Split Ratio": initialData["Split Ratio"] || '',
                    "Total Amount": initialData["Total Amount"] ? Math.abs(initialData["Total Amount"]) : '',
                    "From Account": fromAcc,
                    "To Account": toAcc,
                    "Auto-add Cash": !!initialData["Auto-add Cash"],
                    Note: initialData.Note || ''
                });
            } else {
                setFormData({
                    Date: marketToday(),
                    Type: 'Buy',
                    Symbol: '',
                    Quantity: '',
                    "Price/Share": '',
                    Commission: '',
                    "Total Amount": '',
                    Account: '',
                    "Local Currency": 'USD',
                    Note: '',
                    "Split Ratio": '',
                    "To Account": '',
                    "From Account": '',
                    "Auto-add Cash": true
                });
            }
            setError(null);
            setTotalLockedByUser(false);
            setLoading(false);
        }
    }, [isOpen, initialData, mode]);

    // --- EFFECT: Update Fields matching Desktop Logic ---
    useEffect(() => {
        if (!isOpen) return;

        const txType = (formData.Type || 'Buy').toLowerCase();
        const symbol = (formData.Symbol || '').toUpperCase();
        const isCash = isCashSymbol(symbol);
        const isTransfer = txType === 'transfer';

        setFormData((prev: TxForm) => {
            const newData = { ...prev };

            // 1. Handle Cash Symbols
            if (isCash) {
                if (['deposit', 'withdrawal', 'buy', 'sell'].includes(txType)) {
                    // Price locked to 1.0
                    if (newData['Price/Share'] !== 1.0) newData['Price/Share'] = 1.0;
                    // Total locked to Quantity
                    if (newData.Quantity !== '' && !isNaN(Number(newData.Quantity))) {
                        newData['Total Amount'] = Number(newData.Quantity);
                    }
                }
            }
            // 2. Handle Stock Trades
            else if (['buy', 'sell', 'short sell', 'buy to cover'].includes(txType)) {
                // Auto-calc Total if not locked
                if (!totalLockedByUser) {
                    const qty = parseFloat(newData.Quantity);
                    const price = parseFloat(newData['Price/Share']);
                    const comm = parseFloat(newData.Commission) || 0;
                    if (!isNaN(qty) && !isNaN(price)) {
                        if (['buy', 'buy to cover'].includes(txType)) {
                            newData['Total Amount'] = parseFloat(((qty * price) + comm).toFixed(2));
                        } else {
                            newData['Total Amount'] = parseFloat(((qty * price) - comm).toFixed(2));
                        }
                    } else if (isNaN(qty) || isNaN(price)) {
                        newData['Total Amount'] = '';
                    }
                }
            }

            // Sync Generic Account with From Account for transfers
            if (isTransfer) {
                if (newData.Account !== newData['From Account']) {
                    newData.Account = newData['From Account'];
                }
            } else {
                if (newData['From Account'] !== newData.Account) {
                    newData['From Account'] = newData.Account;
                }
            }

            // 3. Clear Auto-add Cash if not applicable or if account uses Auto cash mode
            const accountMode = (accountCashModeMap[prev.Account || ''] || 'Manual');
            if ((!canAutoAddCash || accountMode === 'Auto') && newData["Auto-add Cash"]) {
                newData["Auto-add Cash"] = false;
            }

            const hasChanged = Object.keys(newData).some(key => newData[key] !== prev[key]);
            if (hasChanged) {
                return newData;
            }
            return prev;
        });

        // eslint-disable-next-line react-hooks/exhaustive-deps -- intentionally recomputes derived fields only on these form inputs; adding the cash-mode map/flags would retrigger unnecessarily
    }, [formData.Type, formData.Symbol, formData.Quantity, formData['Price/Share'], formData.Commission, formData['From Account'], formData.Account, totalLockedByUser, isOpen]);


    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const { name, value, type } = e.target;
        let val: string | number = value;

        if (type === 'number') {
            val = value === '' ? '' : parseFloat(value);
        } else if (name === 'Symbol' || name === 'Local Currency') {
            val = value.toUpperCase();
        }

        if (name === 'Total Amount') {
            setTotalLockedByUser(!!value);
        }

        // When user changes Quantity, Price, or Commission, unlock total so it auto-recalculates
        if (['Quantity', 'Price/Share', 'Commission'].includes(name)) {
            setTotalLockedByUser(false);
        }

        setFormData((prev: TxForm) => {
            const newData = { ...prev, [name]: val };

            if (name === 'Type') {
                if (val === 'Dividend') {
                    newData['Quantity'] = '';
                    newData['Price/Share'] = '';
                } else if (val === 'Split') {
                    newData['Quantity'] = '';
                    newData['Price/Share'] = '';
                    newData['Total Amount'] = '';
                    newData['Commission'] = '';
                } else if (val === 'Transfer') {
                    newData['Price/Share'] = '';
                    newData['Total Amount'] = '';
                    newData['Commission'] = '';
                    newData['Split Ratio'] = '';
                }
            }

            if (name === 'Account' || name === 'From Account') {
                const accName = val.toString();
                const mappedCurrency = accountCurrencyMap[accName];
                if (mappedCurrency) {
                    newData['Local Currency'] = mappedCurrency;
                }
            }

            return newData;
        });
    };

    const handleSuggestionClick = (value: string, field: 'Symbol' | 'Account' | 'From Account' | 'To Account') => {
        setFormData((prev: TxForm) => {
            const newData = { ...prev, [field]: value };
            if (field === 'Symbol' && value === '$CASH') {
                newData['Price/Share'] = 1.0;
            }
            if (field === 'Account' || field === 'From Account') {
                const mappedCurrency = accountCurrencyMap[value];
                if (mappedCurrency) newData['Local Currency'] = mappedCurrency;
            }
            return newData;
        });
        setActiveSuggestionField(null);
    };

    const renderSuggestions = (field: 'Symbol' | 'Account' | 'From Account' | 'To Account', suggestions: string[]) => {
        if (activeSuggestionField !== field) return null;

        const currentValue = (formData[field] || '').toString().toUpperCase();
        const filtered = suggestions.filter(item => item.toUpperCase().includes(currentValue));

        if (filtered.length === 0) return null;

        return (
            <div className="menu-panel absolute z-10 w-full mt-1 max-h-40 overflow-y-auto">
                <ul className="py-1">
                    {filtered.map(item => (
                        <li
                            key={item}
                            onMouseDown={(e) => { e.preventDefault(); handleSuggestionClick(item, field); }}
                            className="px-3 py-2 cursor-pointer hover:bg-muted text-foreground"
                        >
                            {item}
                        </li>
                    ))}
                </ul>
            </div>
        );
    };


    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        const txType = (formData.Type || '').toLowerCase();
        const symbol = (formData.Symbol || '').toUpperCase();
        const acc = formData.Account ? formData.Account.trim() : '';
        const fromAcc = formData['From Account'] ? formData['From Account'].trim() : '';
        const toAcc = formData['To Account'] ? formData['To Account'].trim() : '';

        const qty = parseFloat(formData.Quantity);
        const price = parseFloat(formData["Price/Share"]);
        let comm = parseFloat(formData.Commission);
        if (isNaN(comm)) comm = 0;

        if (!symbol) { setError("Symbol cannot be empty."); setLoading(false); return; }

        if (txType === 'transfer') {
            if (!fromAcc || !toAcc) {
                setError("From and To accounts are required for a Transfer."); setLoading(false); return;
            }
        } else if (txType !== 'split' && txType !== 'stock split') {
            // Account is required for everything EXCEPT splits/transfers
            if (!acc) {
                setError("Account cannot be empty."); setLoading(false); return;
            }
        }

        const isCash = isCashSymbol(symbol);

        if (isCash) {
            if (['deposit', 'withdrawal', 'buy', 'sell'].includes(txType)) {
                if (isNaN(qty) || qty <= 0) {
                    setError("Amount (Quantity) must be positive for cash operations."); setLoading(false); return;
                }
            }
        } else if (txType === 'transfer') {
            if (isNaN(qty) || qty <= 0) {
                setError("Quantity must be positive for a Transfer."); setLoading(false); return;
            }
        } else if (['buy', 'sell', 'short sell', 'buy to cover'].includes(txType)) {
            if (isNaN(qty) || qty <= 0) {
                setError("Quantity must be positive."); setLoading(false); return;
            }
            // Allow a zero price for free-stock acquisitions (rewards, gifts,
            // spinoffs) — the cost basis is genuinely 0. Only reject negatives.
            if (isNaN(price) || price < 0) {
                setError("Price/Unit cannot be negative."); setLoading(false); return;
            }
        } else if (txType === 'dividend') {
            const total = parseFloat(formData['Total Amount']);
            if (!isNaN(total)) {
                if (total < 0) { setError("Dividend Total Amount cannot be negative."); setLoading(false); return; }
            } else {
                if (isNaN(qty) || qty <= 0) { setError("Dividend Quantity must be positive if Total is missing."); setLoading(false); return; }
                if (isNaN(price) || price <= 0) { setError("Dividend Price must be positive if Total is missing."); setLoading(false); return; }
            }
        }

        try {
            let finalAmount = parseFloat(formData["Total Amount"]);
            if (isNaN(finalAmount) && !isNaN(qty) && !isNaN(price)) {
                if (['buy', 'buy to cover'].includes(txType)) {
                    finalAmount = (qty * price) + comm;
                } else if (['sell', 'short sell'].includes(txType)) {
                    finalAmount = (qty * price) - comm;
                } else {
                    finalAmount = qty * price;
                }
            }
            if (['transfer', 'split'].includes(txType)) finalAmount = 0;
            if (['deposit', 'withdrawal', 'buy', 'sell'].includes(txType) && isCash) finalAmount = qty;

            let signedAmount = Math.abs(finalAmount || 0);

            if (['Buy', 'Withdrawal', 'Fees', 'Tax', 'Split', 'Buy To Cover'].includes(formData.Type)) {
                signedAmount = -Math.abs(signedAmount);
            } else {
                signedAmount = Math.abs(signedAmount);
            }

            const submissionData = {
                ...formData,
                "Quantity": isNaN(qty) ? 0 : qty,
                "Price/Share": isNaN(price) ? 0 : price,
                "Commission": isNaN(comm) ? 0 : comm,
                "Split Ratio": Number(formData["Split Ratio"] || 0),
                "Total Amount": signedAmount,
                "Account": (txType === 'split' || txType === 'stock split') ? 'All Accounts' : (txType === 'transfer' ? fromAcc : acc),
                "To Account": txType === 'transfer' ? toAcc : ''
            };

            await onSubmit(submissionData as Transaction);
            onClose();
        } catch (err: unknown) {
            console.error(err);
            setError('Failed to save transaction');
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    const txType = (formData.Type || 'Buy').toLowerCase();
    const isTransfer = txType === 'transfer';
    const isCash = isCashSymbol(formData.Symbol);
    const isSplit = txType === 'split' || txType === 'stock split';
    const canAutoAddCash = (['buy', 'sell', 'short sell', 'buy to cover'].includes(txType)) && !isCash;
    const selectedAccount = isTransfer ? formData['From Account'] : formData.Account;
    const isAccountAutoCash = (accountCashModeMap[selectedAccount || ''] || 'Manual') === 'Auto';

    const isQtyDisabled = isSplit;
    const isPriceDisabled = isTransfer || isSplit || (isCash && ['deposit', 'withdrawal', 'buy', 'sell'].includes(txType));
    const isTotalDisabled = isTransfer || isSplit;
    const isCommDisabled = isTransfer || isSplit;
    const isSplitRatioDisabled = !isSplit;

    return (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-[rgb(22_23_27/0.36)]">
            <div className="flex min-h-full items-center justify-center p-4">
                <div role="dialog" aria-modal="true" aria-labelledby="tx-modal-title" className="bg-card text-card-foreground p-6 rounded-hero w-full max-w-md shadow-[0_24px_64px_rgb(22_23_27/0.24)] ring-1 ring-black/5 dark:ring-white/10 relative">
                    <div className="flex justify-between items-center mb-4">
                        <h2 id="tx-modal-title" className="page-title text-[28px] leading-8 text-foreground">
                            {mode === 'edit' ? 'Edit Transaction' : 'Add Transaction'}
                        </h2>
                        <button type="button" onClick={onClose} aria-label="Close" className="w-9 h-9 flex items-center justify-center rounded-control bg-muted text-ink-2 hover:text-foreground transition-colors">
                            <X className="w-4 h-4" aria-hidden="true" />
                        </button>
                    </div>

                    {error && <div className="mb-4 p-2 bg-red-100 text-down rounded text-sm">{error}</div>}

                    <form onSubmit={handleSubmit} className="space-y-3">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {/* Date */}
                            <div>
                                <label className="block text-[13px] font-medium text-ink-2 mb-1.5">Date *</label>
                                <input
                                    type="date"
                                    name="Date"
                                    value={formData.Date}
                                    onChange={handleChange}
                                    className="w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow]"
                                    required
                                />
                            </div>

                            {/* Type */}
                            <div>
                                <label htmlFor="tx-type-select" className="block text-[13px] font-medium text-ink-2 mb-1.5">Type *</label>
                                <select
                                    id="tx-type-select"
                                    aria-label="Transaction Type"
                                    name="Type"
                                    value={formData.Type}
                                    onChange={handleChange}
                                    className="w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow]"
                                >
                                    {TRANSACTION_TYPES.map(type => (
                                        <option key={type} value={type}>{type}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Symbol */}
                        <div className="relative">
                            <label className="block text-[13px] font-medium text-ink-2 mb-1.5">Symbol *</label>
                            <input
                                type="text"
                                name="Symbol"
                                value={formData.Symbol}
                                onChange={handleChange}
                                onFocus={() => setActiveSuggestionField('Symbol')}
                                onBlur={() => setTimeout(() => setActiveSuggestionField(curr => curr === 'Symbol' ? null : curr), 100)}
                                placeholder="e.g. AAPL"
                                autoComplete="off"
                                className="w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow] uppercase"
                                required
                            />
                            {renderSuggestions('Symbol', existingSymbols)}
                        </div>

                        {/* Accounts */}
                        {isTransfer ? (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                <div className="relative">
                                    <label className="block text-[13px] font-medium text-ink-2 mb-1.5">From *</label>
                                    <input
                                        type="text"
                                        name="From Account"
                                        value={formData['From Account']}
                                        onChange={handleChange}
                                        onFocus={() => setActiveSuggestionField('From Account')}
                                        onBlur={() => setTimeout(() => setActiveSuggestionField(curr => curr === 'From Account' ? null : curr), 100)}
                                        className="w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow]"
                                        required
                                    />
                                    {renderSuggestions('From Account', existingAccounts)}
                                </div>
                                <div className="relative">
                                    <label className="block text-[13px] font-medium text-ink-2 mb-1.5">To *</label>
                                    <input
                                        type="text"
                                        name="To Account"
                                        value={formData['To Account']}
                                        onChange={handleChange}
                                        onFocus={() => setActiveSuggestionField('To Account')}
                                        onBlur={() => setTimeout(() => setActiveSuggestionField(curr => curr === 'To Account' ? null : curr), 100)}
                                        className="w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow]"
                                        required
                                    />
                                    {renderSuggestions('To Account', existingAccounts)}
                                </div>
                            </div>
                        ) : (
                            <div className="relative">
                                <label className={`block text-[13px] font-medium mb-1.5 ${isSplit ? 'text-muted-foreground/60' : 'text-ink-2'}`}>Account *</label>
                                <input
                                    type="text"
                                    name="Account"
                                    value={isSplit ? 'All Accounts' : formData.Account}
                                    onChange={handleChange}
                                    onFocus={() => setActiveSuggestionField('Account')}
                                    onBlur={() => setTimeout(() => setActiveSuggestionField(curr => curr === 'Account' ? null : curr), 100)}
                                    placeholder="e.g. Brokerage"
                                    autoComplete="off"
                                    disabled={isSplit}
                                    className={`w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow] ${isSplit ? 'opacity-50 cursor-not-allowed italic' : ''}`}
                                    required={!isSplit}
                                />
                                {!isSplit && renderSuggestions('Account', existingAccounts)}
                            </div>
                        )}

                        <div className="grid grid-cols-2 gap-3">
                            {/* Quantity */}
                            <div>
                                <label className="block text-[13px] font-medium text-ink-2 mb-1.5">Quantity</label>
                                <input
                                    type="number"
                                    name="Quantity"
                                    value={formData.Quantity}
                                    onChange={handleChange}
                                    disabled={isQtyDisabled}
                                    className={`w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow] ${isQtyDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                            </div>
                            {/* Price */}
                            <div>
                                <label className="block text-[13px] font-medium text-ink-2 mb-1.5">Price/Share</label>
                                <input
                                    type="number"
                                    name="Price/Share"
                                    value={formData["Price/Share"]}
                                    onChange={handleChange}
                                    disabled={isPriceDisabled}
                                    className={`w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow] ${isPriceDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            {/* Total Amount */}
                            <div>
                                <label className="block text-[13px] font-medium text-ink-2 mb-1.5">Total Amount</label>
                                <input
                                    type="number"
                                    name="Total Amount"
                                    value={formData["Total Amount"]}
                                    onChange={handleChange}
                                    disabled={isTotalDisabled}
                                    className={`w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow] ${isTotalDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                            </div>

                            {/* Commission */}
                            <div>
                                <label className="block text-[13px] font-medium text-ink-2 mb-1.5">Commission</label>
                                <input
                                    type="number"
                                    name="Commission"
                                    value={formData.Commission}
                                    onChange={handleChange}
                                    disabled={isCommDisabled}
                                    className={`w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow] ${isCommDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                            </div>
                        </div>

                        {/* Split Ratio */}
                        <div>
                            <label className={`block text-[13px] font-medium mb-1.5 ${isSplitRatioDisabled ? 'text-muted-foreground/60' : 'text-ink-2'}`}>Split Ratio (Optional)</label>
                            <input
                                type="number"
                                name="Split Ratio"
                                value={formData["Split Ratio"]}
                                onChange={handleChange}
                                disabled={isSplitRatioDisabled}
                                placeholder={isSplit ? "e.g. 2 for 2:1" : ""}
                                className={`w-full h-10 px-3 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow] ${isSplitRatioDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                            />
                        </div>

                        {/* Note */}
                        <div>
                            <label className="block text-[13px] font-medium text-ink-2 mb-1.5">Note</label>
                            <textarea
                                name="Note"
                                value={formData.Note || ''}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-input rounded-control bg-card text-foreground focus:outline-none focus:border-primary focus:shadow-[0_0_0_3px_hsl(var(--primary-tint))] transition-[border-color,box-shadow] h-20"
                            />
                        </div>

                        {canAutoAddCash && (
                            <div className="flex items-center gap-2 py-1">
                                <input
                                    type="checkbox"
                                    id="auto-add-cash"
                                    name="Auto-add Cash"
                                    checked={!!formData["Auto-add Cash"]}
                                    onChange={(e) => setFormData((prev: TxForm) => ({ ...prev, "Auto-add Cash": e.target.checked }))}
                                    disabled={isAccountAutoCash}
                                    className={`h-4 w-4 rounded accent-[hsl(var(--primary))] ${isAccountAutoCash ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                                <label
                                    htmlFor="auto-add-cash"
                                    className={`text-sm font-medium ${isAccountAutoCash ? 'text-muted-foreground/60' : 'text-ink-2'}`}
                                    title={isAccountAutoCash ? 'Not available: this account uses Auto cash mode' : undefined}
                                >
                                    Auto-add internal cash & commission withdrawal
                                    {isAccountAutoCash && <span className="ml-1 text-xs">(Auto cash mode)</span>}
                                </label>
                            </div>
                        )}

                        {/* Actions */}
                        <div className="flex justify-end gap-2 mt-6">
                            <button
                                type="button"
                                onClick={onClose}
                                className="h-11 px-5 rounded-control border border-border bg-card text-foreground font-medium hover:bg-muted transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                disabled={loading}
                                className="h-11 px-5 bg-primary text-primary-foreground font-semibold rounded-control hover:bg-primary-hover disabled:opacity-50 transition-colors"
                            >
                                {loading ? 'Saving...' : (mode === 'edit' ? 'Update Transaction' : 'Add Transaction')}
                            </button>
                        </div>

                    </form>
                </div>
            </div>
        </div>
    );
}
