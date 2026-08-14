import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Providers from "@/components/Providers";
import { WebVitals } from "@/components/WebVitals";
import { AuthProvider } from "@/context/AuthContext";
import LazyAIChat from "@/components/LazyAIChat";

const inter = Inter({ subsets: ["latin"] });

export const viewport: Viewport = {
  themeColor: "#06b6d4",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  title: {
    default: "Investa - Portfolio Tracker & Financial Analytics",
    template: "%s | Investa",
  },
  description: "Personal Investment Portfolio Tracker, Dividend Analytics, and Financial Health Screener",
  manifest: "/manifest.json",
  icons: {
    icon: "/icon.png",
    apple: "/apple-icon.png",
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Investa",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} antialiased selection:bg-indigo-500/20 selection:text-indigo-500 min-h-screen bg-background text-foreground`} suppressHydrationWarning>
        <Providers>
          <AuthProvider>
            <WebVitals />
            {children}
            <LazyAIChat />
          </AuthProvider>
        </Providers>
      </body>
    </html>
  );
}
