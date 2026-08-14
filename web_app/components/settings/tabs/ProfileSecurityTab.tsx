import React, { useState } from 'react';
import { UserCircle, ShieldAlert, Smartphone, LogOut, Info, CheckCircle, AlertCircle, Loader2, Save } from 'lucide-react';
import { updateUserProfile, changePassword, deleteUser } from '../../../lib/api';
import { useAuth } from '../../../context/AuthContext';
import {
    cardClassName,
    sectionTitleClassName,
    labelClassName,
    inputClassName,
    primaryButtonClassName
} from '../constants';

export const ProfileSecurityTab: React.FC = () => {
    const { user, refreshUser, logout } = useAuth();

    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [passwordStatus, setPasswordStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
    const [isChangingPassword, setIsChangingPassword] = useState(false);

    const handleChangePassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setPasswordStatus(null);

        if (newPassword !== confirmPassword) {
            setPasswordStatus({ type: 'error', message: "New passwords do not match" });
            return;
        }

        if (newPassword.length < 4) {
            setPasswordStatus({ type: 'error', message: "Password must be at least 4 characters" });
            return;
        }

        setIsChangingPassword(true);
        try {
            const res = await changePassword(currentPassword, newPassword);
            setPasswordStatus({ type: 'success', message: res.message || "Password changed successfully" });
            setCurrentPassword('');
            setNewPassword('');
            setConfirmPassword('');
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            setPasswordStatus({ type: 'error', message: message });
        } finally {
            setIsChangingPassword(false);
        }
    };

    const handleDeleteAccount = async () => {
        if (!window.confirm("Are you sure you want to delete your account? This action cannot be undone and will delete all your data.")) return;
        if (!window.confirm("Please confirm again: DELETE ACCOUNT PERMANENTLY?")) return;

        try {
            await deleteUser();
            logout();
        } catch (err) {
            alert("Failed to delete account: " + String(err));
        }
    };

    return (
        <div className="space-y-8 max-w-3xl">
            {/* Profile Information */}
            <div className={cardClassName}>
                <div className="mb-2">
                    <h3 className={sectionTitleClassName}>
                        <UserCircle className="w-5 h-5 text-cyan-500" />
                        Profile Information
                    </h3>
                </div>
                <p className="text-sm text-muted-foreground mb-6">Identifiers and display name shown across the app.</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-1">
                        <label className={labelClassName}>Username</label>
                        <p className="font-mono text-lg bg-black/5 dark:bg-white/5 px-4 py-2.5 rounded-xl border border-black/5 dark:border-white/5">{user?.username}</p>
                    </div>
                    <div className="space-y-1">
                        <label className={labelClassName}>User ID</label>
                        <p className="font-mono text-lg bg-black/5 dark:bg-white/5 px-4 py-2.5 rounded-xl border border-black/5 dark:border-white/5">{user?.id}</p>
                    </div>
                    <div className="md:col-span-2 space-y-1">
                        <label className={labelClassName}>Alias (Display Name)</label>
                        <input
                            type="text"
                            defaultValue={user?.alias || ''}
                            placeholder="e.g. My Portfolio"
                            className={inputClassName}
                            onBlur={async (e) => {
                                const newAlias = e.target.value.trim();
                                if (newAlias !== (user?.alias || '')) {
                                    try {
                                        await updateUserProfile({ alias: newAlias });
                                        await refreshUser();
                                    } catch {
                                        alert("Failed to update alias");
                                    }
                                }
                            }}
                        />
                        <p className="text-[11px] text-muted-foreground mt-2 pl-1 flex items-center gap-1.5">
                            <Info className="w-3.5 h-3.5" />
                            This name will be displayed in the user menu. Leave empty to use username.
                        </p>
                    </div>
                </div>
            </div>

            {/* Security: Password Change */}
            <div className={cardClassName}>
                <div className="mb-2">
                    <h3 className={sectionTitleClassName}>
                        <ShieldAlert className="w-5 h-5 text-amber-500" />
                        Security
                    </h3>
                </div>
                <p className="text-sm text-muted-foreground mb-6">Change your login password.</p>
                <form onSubmit={handleChangePassword} className="space-y-5">
                    <div className="space-y-1">
                        <label className={labelClassName}>Current Password</label>
                        <input
                            type="password"
                            value={currentPassword}
                            onChange={(e) => setCurrentPassword(e.target.value)}
                            className={inputClassName}
                            required
                        />
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                        <div className="space-y-1">
                            <label className={labelClassName}>New Password</label>
                            <input
                                type="password"
                                value={newPassword}
                                onChange={(e) => setNewPassword(e.target.value)}
                                className={inputClassName}
                                required
                            />
                        </div>
                        <div className="space-y-1">
                            <label className={labelClassName}>Confirm Password</label>
                            <input
                                type="password"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                className={inputClassName}
                                required
                            />
                        </div>
                    </div>

                    {passwordStatus && (
                        <div className={`text-sm p-4 rounded-xl flex items-center gap-3 animate-in fade-in ${passwordStatus.type === 'success' ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20'}`}>
                            {passwordStatus.type === 'success' ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                            {passwordStatus.message}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={isChangingPassword}
                        className={primaryButtonClassName}
                    >
                        {isChangingPassword ? <Loader2 className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
                        Change Password
                    </button>
                </form>
            </div>

            {/* Session & Account Deletion */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white/60 dark:bg-zinc-900/60 backdrop-blur-xl p-6 rounded-2xl border border-white/40 dark:border-white/10 shadow-sm flex flex-col justify-between">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <Smartphone className="w-5 h-5 text-foreground" />
                            <h4 className="font-bold text-foreground">Sign Out Device</h4>
                        </div>
                        <p className="text-sm text-muted-foreground mb-6">
                            End your current session on this device.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={() => logout()}
                        className="w-full py-2.5 rounded-xl border-2 border-border bg-transparent hover:bg-secondary font-medium transition-colors text-foreground flex items-center justify-center gap-2 cursor-pointer"
                    >
                        <LogOut className="w-4 h-4" />
                        Sign Out
                    </button>
                </div>

                <div className="bg-red-50/50 dark:bg-red-950/20 backdrop-blur-xl p-6 rounded-2xl border border-red-200 dark:border-red-900/50 shadow-sm flex flex-col justify-between relative overflow-hidden group">
                    <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-red-500 to-rose-500 transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-500" />
                    <div>
                        <h4 className="font-bold text-red-700 dark:text-red-400 mb-2">Delete Account</h4>
                        <p className="text-sm text-red-600/80 dark:text-red-400/80 mb-6 leading-relaxed">
                            Permanently delete your profile, portfolio data, and settings. This action is irreversible.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={handleDeleteAccount}
                        className="w-full py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-xl font-medium shadow-sm transition-colors focus:ring-2 focus:ring-offset-2 focus:ring-red-500 cursor-pointer"
                    >
                        Delete Account Permanently
                    </button>
                </div>
            </div>
        </div>
    );
};
