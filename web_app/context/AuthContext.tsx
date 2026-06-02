"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchCurrentUser, User } from "../lib/api";

// Define shapes
// interface User remove definition as imported

interface AuthContextType {
    user: User | null;
    token: string | null;
    isLoading: boolean;
    login: (token: string) => Promise<void>;
    logout: () => void;
    refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const router = useRouter();

    // Load token from storage on mount. If we also have a cached user, restore
    // it optimistically and drop the loading gate immediately so the dashboard
    // (and its persisted React Query cache) render without waiting on /auth/me —
    // the token is still validated in the background below.
    useEffect(() => {
        const storedToken = localStorage.getItem("access_token");
        if (storedToken) {
            setToken(storedToken);
            const cachedUser = localStorage.getItem("investa_user");
            if (cachedUser) {
                try {
                    setUser(JSON.parse(cachedUser));
                    setIsLoading(false);
                } catch {
                    // Corrupt cache — fall back to the blocking fetch below.
                }
            }
            fetchUser(storedToken);
        } else {
            setIsLoading(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Listen for 401 events dispatched by authFetch so expired tokens trigger logout
    useEffect(() => {
        const handleExpired = () => logout();
        window.addEventListener('auth:expired', handleExpired);
        return () => window.removeEventListener('auth:expired', handleExpired);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const fetchUser = async (authToken: string) => {
        try {
            const userData = await fetchCurrentUser(authToken);
            setUser(userData);
            // Cache for optimistic restore on the next load.
            try { localStorage.setItem("investa_user", JSON.stringify(userData)); } catch {}
        } catch (error) {
            console.error("Failed to fetch user:", error);
            logout();
        } finally {
            setIsLoading(false);
        }
    };

    const login = async (newToken: string) => {
        setIsLoading(true); // Ensure loading state is active during transition
        localStorage.setItem("access_token", newToken);
        setToken(newToken);
        await fetchUser(newToken); // Wait for user data to be fetched

        // Handle navigation in Electron (file:// protocol)
        if (typeof window !== 'undefined' && window.location.protocol === 'file:') {
            window.location.href = 'index.html';
        } else {
            router.push("/");
        }
    };

    const logout = () => {
        localStorage.removeItem("access_token");
        localStorage.removeItem("investa_user");
        setToken(null);
        setUser(null);

        // Handle navigation in Electron (file:// protocol)
        if (typeof window !== 'undefined' && window.location.protocol === 'file:') {
            // Force navigation to the HTML file directly
            window.location.href = 'login.html';
        } else {
            router.push("/login");
        }
    };

    const refreshUser = async () => {
        if (token) {
            await fetchUser(token);
        }
    }

    return (
        <AuthContext.Provider value={{ user, token, isLoading, login, logout, refreshUser }}>
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
