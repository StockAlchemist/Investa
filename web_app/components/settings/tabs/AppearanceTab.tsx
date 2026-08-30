import React, { useEffect, useState } from 'react';
import { useTheme } from 'next-themes';
import { Monitor, Sun, Moon } from 'lucide-react';
import { cn } from '../../../lib/utils';
import {
    cardClassName,
    cardHeadClassName,
    sectionTitleClassName,
    chipClassName,
    chipActiveClassName,
} from '../constants';

const THEMES = [
    { id: 'system', label: 'System', icon: Monitor, hint: 'Follow the device appearance' },
    { id: 'light', label: 'Light', icon: Sun, hint: 'Always use the light theme' },
    { id: 'dark', label: 'Dark', icon: Moon, hint: 'Always use the dark theme' },
] as const;

/**
 * Appearance lives in Settings, not in the chrome: the theme is a preference
 * the user sets once, so it belongs beside the other preferences rather than
 * occupying a permanent slot in the sidebar, the control bar and the mobile
 * bottom nav. Mirrors `AppearanceSettingsView` in the native clients.
 */
export const AppearanceTab: React.FC = () => {
    const { theme, setTheme, resolvedTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    // The stored theme is only known on the client; rendering it during
    // hydration would mismatch the server-rendered markup.
    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setMounted(true);
    }, []);

    const current = mounted ? (theme ?? 'system') : undefined;

    return (
        <div className={cardClassName}>
            <div className={cardHeadClassName}>
                <h3 className={sectionTitleClassName}>Theme</h3>
            </div>

            <p className="text-sm text-muted-foreground mb-4">
                Choose the light or dark theme, or follow whatever the device is set to.
            </p>

            <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Theme">
                {THEMES.map(({ id, label, icon: Icon, hint }) => {
                    const isActive = current === id;
                    return (
                        <button
                            key={id}
                            type="button"
                            role="radio"
                            aria-checked={isActive}
                            title={hint}
                            onClick={() => setTheme(id)}
                            className={cn(
                                isActive ? chipActiveClassName : chipClassName,
                                'cursor-pointer',
                                !isActive && 'hover:bg-muted hover:text-foreground',
                            )}
                        >
                            <Icon className="w-3.5 h-3.5 shrink-0" />
                            <span>{label}</span>
                        </button>
                    );
                })}
            </div>

            {mounted && theme === 'system' && (
                <p className="text-xs text-muted-foreground mt-3">
                    Currently showing the {resolvedTheme === 'dark' ? 'dark' : 'light'} theme.
                </p>
            )}
        </div>
    );
};
