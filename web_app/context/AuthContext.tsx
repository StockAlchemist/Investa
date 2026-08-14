"use client";

import React, { createContext, useContext, useEffect, useRef, useState } from "react";
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

    // Validate session in background, deferring when unauthenticated to avoid competing with initial paint
    useEffect(() => {
        const cached = typeof window !== 'undefined' ? localStorage.getItem("investa_user") : null;
        if (cached) {
            fetchUser();
        } else {
            if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
                const id = (window as any).requestIdleCallback(() => fetchUser(), { timeout: 3000 });
                return () => (window as any).cancelIdleCallback(id);
            } else {
                const timer = setTimeout(() => fetchUser(), 1500);
                return () => clearTimeout(timer);
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Read inside the listener below, which is registered once and would
    // otherwise close over the first render's `user`.
    const userRef = useRef<User | null>(null);
    useEffect(() => {
        userRef.current = user;
    }, [user]);

    // Listen for 401 events dispatched by the API layers so an expired/invalid
    // cookie triggers logout.
    //
    // Only when somebody is actually signed in. A 401 while logged out is the
    // ordinary answer — every provider mounted above the router keeps fetching
    // on the login and register pages — and treating it as an expiry logged the
    // visitor out of a session they never had and redirected them to /login.
    // That made /register unreachable: arriving there logged out is the point of
    // it, and the deferred watchlist fetch bounced you before you could type.
    // Guarding the listener rather than the dispatchers is what keeps this
    // fixed, since any globally mounted fetch would otherwise reintroduce it.
    useEffect(() => {
        const handleExpired = () => {
            if (userRef.current) logout();
        };
        window.addEventListener('auth:expired', handleExpired);
        return () => window.removeEventListener('auth:expired', handleExpired);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const clearLocalSession = () => {
        setUser(prev => {
            if (prev === null) return prev;
            return null;
        });
        try { localStorage.removeItem("investa_user"); } catch {}
    };

    const fetchUser = async () => {
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
    };

    const login = async () => {
        setIsLoading(true);
        await fetchUser();
        router.push("/");
    };

    const logout = () => {
        logoutRequest();
        clearLocalSession();
        router.push("/login");
    };

    const refreshUser = async () => {
        await fetchUser();
    };

    const contextValue = React.useMemo(() => ({
        user,
        isLoading,
        login,
        logout,
        refreshUser,
    }), [user, isLoading]);

    return (
        <AuthContext.Provider value={contextValue}>
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
