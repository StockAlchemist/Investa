"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchCurrentUser, logoutRequest, SessionExpiredError, User } from "../lib/api";

interface AuthContextType {
    user: User | null;
    isLoading: boolean;
    login: () => Promise<void>;
    logout: () => void;
    refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();

    // Auth now lives in an httpOnly cookie the browser sends automatically, so
    // there's no token for JS to read. Optimistically restore the cached user
    // (profile only, not the token) to render immediately, then validate the
    // cookie via /auth/me in the background.
    useEffect(() => {
        const cachedUser = localStorage.getItem("investa_user");
        if (cachedUser) {
            try {
                setUser(JSON.parse(cachedUser));
                setIsLoading(false);
            } catch {
                // Corrupt cache — fall back to the blocking fetch below.
            }
        }
        fetchUser();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Listen for 401 events dispatched by the API layers so an expired/invalid
    // cookie triggers logout.
    useEffect(() => {
        const handleExpired = () => logout();
        window.addEventListener('auth:expired', handleExpired);
        return () => window.removeEventListener('auth:expired', handleExpired);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Clear local session state without a redirect or a server round-trip — used
    // when the cookie turns out to be absent/invalid during bootstrap.
    const clearLocalSession = () => {
        setUser(null);
        try { localStorage.removeItem("investa_user"); } catch {}
    };

    const fetchUser = async () => {
        try {
            const userData = await fetchCurrentUser();
            setUser(userData);
            // Cache the profile (no token) for optimistic restore next load.
            try { localStorage.setItem("investa_user", JSON.stringify(userData)); } catch {}
        } catch (error) {
            if (!(error instanceof SessionExpiredError)) {
                console.error("Failed to fetch user:", error);
            }
            // Not logged in / cookie invalid — clear state and let route guards
            // redirect (don't force-navigate here, matching the prior no-token path).
            clearLocalSession();
        } finally {
            setIsLoading(false);
        }
    };

    const login = async () => {
        // The login POST already set the httpOnly cookie server-side; just load
        // the user (sent via cookie) and navigate.
        setIsLoading(true);
        await fetchUser();
        router.push("/");
    };

    const logout = () => {
        logoutRequest(); // best-effort: clears the cookie server-side
        clearLocalSession();
        router.push("/login");
    };

    const refreshUser = async () => {
        await fetchUser();
    };

    return (
        <AuthContext.Provider value={{ user, isLoading, login, logout, refreshUser }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
}
