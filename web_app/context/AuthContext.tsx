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

    // Load token from storage on mount
    useEffect(() => {
        const storedToken = localStorage.getItem("access_token");
        if (storedToken) {
            setToken(storedToken);
            fetchUser(storedToken);
        } else {
            setIsLoading(false);
        }
    }, []);

    const fetchUser = async (authToken: string) => {
        try {
            const userData = await fetchCurrentUser(authToken);
            setUser(userData);
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
