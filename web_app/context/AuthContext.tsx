"use client";

import React, { createContext, useContext, useEffect, useRef, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { fetchCurrentUser, logoutRequest, User } from "../lib/api";

interface AuthContextType {
    user: User | null;
    isLoading: boolean;
    login: () => Promise<void>;
    logout: () => void;
    refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(() => {
        if (typeof window !== 'undefined') {
            try {
                const cachedUser = localStorage.getItem("investa_user");
                if (cachedUser) return JSON.parse(cachedUser);
            } catch {}
        }
        return null;
    });
    const [isLoading, setIsLoading] = useState(false);
    const router = useRouter();

    const clearLocalSession = useCallback(() => {
        setUser(prev => {
            if (prev === null) return prev;
            return null;
        });
        try { localStorage.removeItem("investa_user"); } catch {}
    }, []);

    const fetchUser = useCallback(async () => {
        try {
            const userData = await fetchCurrentUser();
            if (userData) {
                setUser(prev => {
                    if (prev && prev.id === userData.id && prev.username === userData.username) return prev;
                    return userData;
                });
                try { localStorage.setItem("investa_user", JSON.stringify(userData)); } catch {}
            } else {
                clearLocalSession();
            }
        } catch {
            clearLocalSession();
        } finally {
            setIsLoading(false);
        }
    }, [clearLocalSession]);

    // Validate session in background, deferring when unauthenticated to avoid competing with initial paint
    useEffect(() => {
        const cached = typeof window !== 'undefined' ? localStorage.getItem("investa_user") : null;
        if (cached) {
            fetchUser();
        } else {
            if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
                const id = window.requestIdleCallback(() => fetchUser(), { timeout: 3000 });
                return () => {
                    window.cancelIdleCallback(id);
                };
            } else {
                const timer = setTimeout(() => fetchUser(), 1500);
                return () => clearTimeout(timer);
            }
        }
    }, [fetchUser]);

    const userRef = useRef<User | null>(null);
    useEffect(() => {
        userRef.current = user;
    }, [user]);

    const logout = useCallback(() => {
        logoutRequest();
        clearLocalSession();
        router.push("/login");
    }, [clearLocalSession, router]);

    const login = useCallback(async () => {
        setIsLoading(true);
        await fetchUser();
        router.push("/");
    }, [fetchUser, router]);

    const refreshUser = useCallback(async () => {
        await fetchUser();
    }, [fetchUser]);

    useEffect(() => {
        const handleExpired = () => {
            if (userRef.current) logout();
        };
        window.addEventListener('auth:expired', handleExpired);
        return () => window.removeEventListener('auth:expired', handleExpired);
    }, [logout]);

    const contextValue = useMemo(() => ({
        user,
        isLoading,
        login,
        logout,
        refreshUser,
    }), [user, isLoading, login, logout, refreshUser]);

    return (
        <AuthContext.Provider value={contextValue}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        return {
            user: null,
            isLoading: false,
            login: async () => {},
            logout: () => {},
            refreshUser: async () => {},
        };
    }
    return context;
}
