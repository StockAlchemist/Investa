import React, { useState, useEffect } from 'react';
import { Transaction } from '../lib/api';

interface TransactionModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSubmit: (data: Transaction) => Promise<void>;
    initialData?: any; // Using any for flexibility with edit data
    mode: 'add' | 'edit';
    accountCurrencyMap: { [account: string]: string };
    existingAccounts?: string[];
    existingSymbols?: string[];
}

const TRANSACTION_TYPES = [
    'Buy', 'Sell', 'Dividend', 'Transfer', 'Interest', 'Fees', 'Deposit', 'Withdrawal', 'Spin-off', 'Split'
];

// Helper matching src/finutils.py
const CASH_SYMBOL_CSV = "$CASH";
const isCashSymbol = (symbol: string) => {
    if (!symbol) return false;
    const s = symbol.toLowerCase();
    return s.startsWith(CASH_SYMBOL_CSV.toLowerCase()) || s.startsWith('cash (');
};

export default function TransactionModal({ isOpen, onClose, onSubmit, initialData, mode, accountCurrencyMap, existingAccounts = [], existingSymbols = [] }: TransactionModalProps) {
    const [formData, setFormData] = useState<any>({
        Date: new Date().toISOString().split('T')[0],
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
        "From Account": ''
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

                let fromAcc = initialData.Account || '';
                let toAcc = initialData["To Account"] || '';

                setFormData({
                    ...initialData,
                    Date: formattedDate,
                    Quantity: initialData.Quantity || '',
                    "Price/Share": initialData["Price/Share"] || '',
                    Commission: initialData.Commission || '',
                    "Split Ratio": initialData["Split Ratio"] || '',
                    "Total Amount": initialData["Total Amount"] ? Math.abs(initialData["Total Amount"]) : '',
                    "From Account": fromAcc,
                    "To Account": toAcc
                });
            } else {
                setFormData({
                    Date: new Date().toISOString().split('T')[0],
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
                    "From Account": ''
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

        setFormData((prev: any) => {
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
                    if (!isNaN(qty) && !isNaN(price)) {
                        newData['Total Amount'] = parseFloat((qty * price).toFixed(2));
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

            const hasChanged = Object.keys(newData).some(key => newData[key] !== prev[key]);
            if (hasChanged) {
                return newData;
            }
            return prev;
        });

    }, [formData.Type, formData.Symbol, formData.Quantity, formData['Price/Share'], formData['From Account'], totalLockedByUser, isOpen]);


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

        setFormData((prev: any) => {
            const newData = { ...prev, [name]: val };

            const txType = (newData.Type || '').toLowerCase();

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
        setFormData((prev: any) => {
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
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg max-h-40 overflow-y-auto">
                <ul className="py-1">
                    {filtered.map(item => (
                        <li
                            key={item}
                            onMouseDown={(e) => { e.preventDefault(); handleSuggestionClick(item, field); }}
                            className="px-3 py-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100"
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

        let qty = parseFloat(formData.Quantity);
        let price = parseFloat(formData["Price/Share"]);
        let comm = parseFloat(formData.Commission);
        if (isNaN(comm)) comm = 0;

        if (!symbol) { setError("Symbol cannot be empty."); setLoading(false); return; }

        if (txType === 'transfer') {
            if (!fromAcc || !toAcc) {
                setError("From and To accounts are required for a Transfer."); setLoading(false); return;
            }
        } else {
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
            if (isNaN(price) || price <= 0) {
                setError("Price/Unit must be positive."); setLoading(false); return;
            }
        } else if (txType === 'dividend') {
            let total = parseFloat(formData['Total Amount']);
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
                finalAmount = qty * price;
            }
            if (['transfer', 'split'].includes(txType)) finalAmount = 0;
            if (['deposit', 'withdrawal', 'buy', 'sell'].includes(txType) && isCash) finalAmount = qty;

            let signedAmount = Math.abs(finalAmount || 0);

            if (['Buy', 'Withdrawal', 'Fees', 'Split'].includes(formData.Type)) {
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
                "Account": txType === 'transfer' ? fromAcc : acc,
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
    const isDividend = txType === 'dividend';

    const isQtyDisabled = isSplit;
    const isPriceDisabled = isTransfer || isSplit || (isCash && ['deposit', 'withdrawal', 'buy', 'sell'].includes(txType));
    const isTotalDisabled = isTransfer || isSplit;
    const isCommDisabled = isTransfer || isSplit;
    const isSplitRatioDisabled = !isSplit;

    return (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-black/50 backdrop-blur-sm">
            <div className="flex min-h-full items-center justify-center p-4">
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg w-full max-w-md shadow-xl relative">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                            {mode === 'edit' ? 'Edit Transaction' : 'Add Transaction'}
                        </h2>
                        <button onClick={onClose} className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                            ✕
                        </button>
                    </div>

                    {error && <div className="mb-4 p-2 bg-red-100 text-red-600 rounded text-sm">{error}</div>}

                    <form onSubmit={handleSubmit} className="space-y-3">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {/* Date */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Date *</label>
                                <input
                                    type="date"
                                    name="Date"
                                    value={formData.Date}
                                    onChange={handleChange}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                                    required
                                />
                            </div>

                            {/* Type */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Type *</label>
                                <select
                                    name="Type"
                                    value={formData.Type}
                                    onChange={handleChange}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                                >
                                    {TRANSACTION_TYPES.map(type => (
                                        <option key={type} value={type}>{type}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Symbol */}
                        <div className="relative">
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Symbol *</label>
                            <input
                                type="text"
                                name="Symbol"
                                value={formData.Symbol}
                                onChange={handleChange}
                                onFocus={() => setActiveSuggestionField('Symbol')}
                                onBlur={() => setTimeout(() => setActiveSuggestionField(null), 100)}
                                placeholder="e.g. AAPL"
                                autoComplete="off"
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 uppercase"
                                required
                            />
                            {renderSuggestions('Symbol', existingSymbols)}
                        </div>

                        {/* Accounts */}
                        {isTransfer ? (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                <div className="relative">
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">From *</label>
                                    <input
                                        type="text"
                                        name="From Account"
                                        value={formData['From Account']}
                                        onChange={handleChange}
                                        onFocus={() => setActiveSuggestionField('From Account')}
                                        onBlur={() => setTimeout(() => setActiveSuggestionField(null), 100)}
                                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                                        required
                                    />
                                    {renderSuggestions('From Account', existingAccounts)}
                                </div>
                                <div className="relative">
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">To *</label>
                                    <input
                                        type="text"
                                        name="To Account"
                                        value={formData['To Account']}
                                        onChange={handleChange}
                                        onFocus={() => setActiveSuggestionField('To Account')}
                                        onBlur={() => setTimeout(() => setActiveSuggestionField(null), 100)}
                                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                                        required
                                    />
                                    {renderSuggestions('To Account', existingAccounts)}
                                </div>
                            </div>
                        ) : (
                            <div className="relative">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Account *</label>
                                <input
                                    type="text"
                                    name="Account"
                                    value={formData.Account}
                                    onChange={handleChange}
                                    onFocus={() => setActiveSuggestionField('Account')}
                                    onBlur={() => setTimeout(() => setActiveSuggestionField(null), 100)}
                                    placeholder="e.g. Brokerage"
                                    autoComplete="off"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                                    required
                                />
                                {renderSuggestions('Account', existingAccounts)}
                            </div>
                        )}

                        <div className="grid grid-cols-2 gap-3">
                            {/* Quantity */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Quantity</label>
                                <input
                                    type="number"
                                    name="Quantity"
                                    value={formData.Quantity}
                                    onChange={handleChange}
                                    disabled={isQtyDisabled}
                                    className={`w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 ${isQtyDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                            </div>
                            {/* Price */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Price/Share</label>
                                <input
                                    type="number"
                                    name="Price/Share"
                                    value={formData["Price/Share"]}
                                    onChange={handleChange}
                                    disabled={isPriceDisabled}
                                    className={`w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 ${isPriceDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            {/* Total Amount */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Total Amount</label>
                                <input
                                    type="number"
                                    name="Total Amount"
                                    value={formData["Total Amount"]}
                                    onChange={handleChange}
                                    disabled={isTotalDisabled}
                                    className={`w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 ${isTotalDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                            </div>

                            {/* Commission */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Commission</label>
                                <input
                                    type="number"
                                    name="Commission"
                                    value={formData.Commission}
                                    onChange={handleChange}
                                    disabled={isCommDisabled}
                                    className={`w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 ${isCommDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                                />
                            </div>
                        </div>

                        {/* Split Ratio */}
                        <div>
                            <label className={`block text-sm font-medium mb-1 ${isSplitRatioDisabled ? 'text-gray-400' : 'text-gray-700 dark:text-gray-300'}`}>Split Ratio (Optional)</label>
                            <input
                                type="number"
                                name="Split Ratio"
                                value={formData["Split Ratio"]}
                                onChange={handleChange}
                                disabled={isSplitRatioDisabled}
                                placeholder={isSplit ? "e.g. 2 for 2:1" : ""}
                                className={`w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 ${isSplitRatioDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                            />
                        </div>

                        {/* Note */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Note</label>
                            <textarea
                                name="Note"
                                value={formData.Note || ''}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 h-20"
                            />
                        </div>

                        {/* Actions */}
                        <div className="flex justify-end gap-2 mt-6">
                            <button
                                type="button"
                                onClick={onClose}
                                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded dark:text-gray-300 dark:hover:bg-gray-700"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                disabled={loading}
                                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
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
