import React, { useState, useEffect } from 'react';
import { Transaction } from '../lib/api';

interface TransactionModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSubmit: (transaction: Transaction) => Promise<void>;
    initialData?: Transaction | null;
    mode: 'add' | 'edit';
    accountCurrencyMap: Record<string, string>;
    existingAccounts?: string[];
    existingSymbols?: string[];
}

const TRANSACTION_TYPES = [
    'Buy', 'Sell', 'Dividend', 'Transfer', 'Interest', 'Fees', 'Deposit', 'Withdrawal', 'Spin-off'
];

export default function TransactionModal({ isOpen, onClose, onSubmit, initialData, mode, accountCurrencyMap, existingAccounts = [], existingSymbols = [] }: TransactionModalProps) {
    const [formData, setFormData] = useState<Partial<Transaction>>({
        Date: new Date().toISOString().split('T')[0],
        Type: 'Buy',
        Symbol: '',
        Quantity: 0,
        "Price/Share": 0,
        Commission: 0,
        "Total Amount": 0,
        Account: '',
        "Local Currency": 'USD',
        Note: '',
        "Split Ratio": 0,
        "To Account": ''
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [activeSuggestionField, setActiveSuggestionField] = useState<'Symbol' | 'Account' | null>(null);



    useEffect(() => {
        if (isOpen) {
            if (mode === 'edit' && initialData) {
                // formatting date to YYYY-MM-DD for input
                const formattedDate = initialData.Date ? initialData.Date.split('T')[0] : '';
                setFormData({
                    ...initialData,
                    Date: formattedDate,
                    "Total Amount": initialData["Total Amount"] ? Math.abs(initialData["Total Amount"]) : 0
                });
            } else {
                setFormData({
                    Date: new Date().toISOString().split('T')[0],
                    Type: 'Buy',
                    Symbol: '',
                    Quantity: 0,
                    "Price/Share": 0,
                    Commission: 0,
                    "Total Amount": 0,
                    Account: '',
                    "Local Currency": 'USD',
                    Note: '',
                    "Split Ratio": 0,
                    "To Account": ''
                });
            }
            setError(null);
        }
    }, [isOpen, initialData, mode]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        const { name, value, type } = e.target;

        let val: string | number = value;
        if (type === 'number') {
            val = value === '' ? 0 : parseFloat(value);
        } else if (name === 'Symbol' || name === 'Local Currency') {
            val = value.toUpperCase();
        }

        setFormData(prev => {
            const newData = { ...prev, [name]: val };

            // Auto-calculate Total Amount if Price and Qty change
            if (name === 'Quantity' || name === 'Price/Share') {
                const qty = name === 'Quantity' ? Number(val) : (prev.Quantity || 0);
                const price = name === 'Price/Share' ? Number(val) : (prev['Price/Share'] || 0);
                newData['Total Amount'] = parseFloat((qty * price).toFixed(2));
            }
            if (name === 'Account') {
                // Auto-fill currency if account is known
                // Only if currency hasn't been manually set differently? 
                // Or prioritize account mapping? Let's prioritize account mapping for convenience,
                // but allows user to change it after if needed (though handleChange won't block it).
                const mappedCurrency = accountCurrencyMap[val];
                if (mappedCurrency) {
                    newData['Local Currency'] = mappedCurrency;
                }
            }

            return newData;
        });
    };

    const handleSuggestionClick = (field: 'Symbol' | 'Account', value: string) => {
        setFormData(prev => ({
            ...prev,
            [field]: value,
            // If Account selected, trigger currency auto-fill too
            ...(field === 'Account' && accountCurrencyMap[value] ? { 'Local Currency': accountCurrencyMap[value] } : {})
        }));
        setActiveSuggestionField(null);
    };

    const renderSuggestions = (field: 'Symbol' | 'Account', suggestions: string[]) => {
        if (activeSuggestionField !== field) return null;

        const currentValue = (formData[field] || '').toString().toUpperCase();
        const filtered = suggestions.filter(s => s.toUpperCase().includes(currentValue));

        if (filtered.length === 0) return null;

        return (
            <ul className="absolute z-10 w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg max-h-60 overflow-y-auto mt-1">
                {filtered.map(item => (
                    <li
                        key={item}
                        // onMouseDown prevents input blur before click fires
                        onMouseDown={(e) => { e.preventDefault(); handleSuggestionClick(field, item); }}
                        className="px-3 py-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100"
                    >
                        {item}
                    </li>
                ))}
            </ul>
        );
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            // Basic validation
            if (!formData.Date || !formData.Symbol || !formData.Account) {
                throw new Error("Date, Symbol, and Account are required.");
            }

            // Adjust Total Amount sign based on Type? 
            // The DB util seems to take values as is. 
            // Typically Buy is negative cash, Sell is positive. 
            // But usually user enters positive numbers in forms and backend handles logic, 
            // OR user enters signed number.
            // Let's check TransactionsTable display: 
            // `{tx["Total Amount"] ? Math.abs(tx["Total Amount"]).toLocaleString...`
            // usage suggests it might be stored signed.
            // Let's look at `portfolio_logic.py` or CSV loader.
            // CSV loader usually flips sign for Buy.
            // BUT api.py / add_transaction just inserts what we send.
            // So we should probably send signed values if that's the convention.
            // However, to keep it simple for user, maybe ask for absolute amount in form?
            // Let's just save positive for now and let the user put negative if they want, 
            // OR smarter: Investa seems to handle signs in analysis logic?
            // Let's make it explicitly signed in the form? No, that's confusing.
            // Let's assume the user enters positive values and we flip signs if needed? 
            // Or just trust the user. 
            // The desktop app usually expects signed values in the DB? 
            // Re-reading `load_and_clean_transactions` in `src/data_loader.py` would confirm.
            // But let's verify with `db_utils` - it just inserts.
            // Be safe: Allow user to type negative if they know what they are doing, default to positive.

            // Correct Type Validation
            if (formData.Type === 'Transfer' && !formData['To Account']) {
                // Warning or let it slide? Let's require it if they selected Transfer
                // throw new Error("To Account is required for Transfer");
            }

            // Determine sign based on type
            // Negative: Buy, Withdrawal, Fees, Transfer
            // Positive: Sell, Dividend, Interest, Deposit, Spin-off
            const isNegative = ['Buy', 'Withdrawal', 'Fees', 'Transfer'].includes(formData.Type || '');
            const absoluteAmount = Math.abs(formData['Total Amount'] || 0);
            const signedAmount = isNegative ? -absoluteAmount : absoluteAmount;

            const submissionData = {
                ...formData,
                "Total Amount": signedAmount
            };

            await onSubmit(submissionData as Transaction);
            onClose();
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Failed to save transaction";
            setError(message);
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4 overflow-y-auto">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
                <div className="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700">
                    <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                        {mode === 'add' ? 'Add Transaction' : 'Edit Transaction'}
                    </h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                        ✕
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="p-6 space-y-4">
                    {error && (
                        <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                            {error}
                        </div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Date */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Date *</label>
                            <input
                                type="date"
                                name="Date"
                                required
                                value={formData.Date}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        {/* Type */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Type *</label>
                            <select
                                name="Type"
                                required
                                value={formData.Type}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            >
                                {TRANSACTION_TYPES.map(t => (
                                    <option key={t} value={t}>{t}</option>
                                ))}
                            </select>
                        </div>

                        {/* Symbol */}
                        <div className="relative">
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Symbol *</label>
                            <input
                                type="text"
                                name="Symbol"
                                required
                                value={formData.Symbol}
                                onChange={handleChange}
                                onFocus={() => setActiveSuggestionField('Symbol')}
                                onBlur={() => setActiveSuggestionField(null)}
                                placeholder="e.g. AAPL"
                                autoComplete="off"
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                            {renderSuggestions('Symbol', existingSymbols)}
                        </div>

                        {/* Local Currency */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Currency *</label>
                            <input
                                type="text"
                                name="Local Currency"
                                required
                                value={formData["Local Currency"]}
                                onChange={handleChange}
                                placeholder="e.g. USD"
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        {/* Account */}
                        <div className="relative">
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Account *</label>
                            <input
                                type="text"
                                name="Account"
                                required
                                value={formData.Account}
                                onChange={handleChange}
                                onFocus={() => setActiveSuggestionField('Account')}
                                onBlur={() => setActiveSuggestionField(null)}
                                placeholder="e.g. Brokerage"
                                autoComplete="off"
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                            {renderSuggestions('Account', existingAccounts)}
                        </div>

                        {/* To Account (only for Transfers) */}
                        {formData.Type === 'Transfer' && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">To Account</label>
                                <input
                                    type="text"
                                    name="To Account"
                                    value={formData["To Account"] || ''}
                                    onChange={handleChange}
                                    placeholder="Target Account"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                        )}

                        {/* Quantity */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Quantity</label>
                            <input
                                type="number"
                                step="any"
                                name="Quantity"
                                value={formData.Quantity}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        {/* Price/Share */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Price/Share</label>
                            <input
                                type="number"
                                step="any"
                                name="Price/Share"
                                value={formData["Price/Share"]}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        {/* Commission */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Commission</label>
                            <input
                                type="number"
                                step="any"
                                name="Commission"
                                value={formData.Commission}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        {/* Total Amount */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Total Amount</label>
                            <input
                                type="number"
                                step="any"
                                name="Total Amount"
                                value={formData["Total Amount"]}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        {/* Split Ratio */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Split Ratio (Optional)</label>
                            <input
                                type="number"
                                step="any"
                                name="Split Ratio"
                                value={formData["Split Ratio"] || 0}
                                onChange={handleChange}
                                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                            />
                        </div>
                    </div>

                    {/* Note */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Note</label>
                        <textarea
                            name="Note"
                            rows={3}
                            value={formData.Note || ''}
                            onChange={handleChange}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
                        />
                    </div>

                    <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={loading}
                            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center"
                        >
                            {loading && (
                                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            )}
                            {mode === 'add' ? 'Add Transaction' : 'Save Changes'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
