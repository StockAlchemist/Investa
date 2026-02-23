import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Providers from "@/components/Providers";
import { WebVitals } from "@/components/WebVitals";
import { AuthProvider } from "@/context/AuthContext";

const inter = Inter({ subsets: ["latin"] });

export const viewport: Viewport = {
  themeColor: "#06b6d4",
};

export const metadata: Metadata = {
  title: "Investa",
  description: "Personal Investment Portfolio Tracker",
  manifest: "/manifest.json",
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
      <body className={`${inter.className} antialiased selection:bg-cyan-500/20 selection:text-cyan-500 min-h-screen bg-background text-foreground`} suppressHydrationWarning>
        <Providers>
          <AuthProvider>
            <WebVitals />
            {children}
          </AuthProvider>
        </Providers>
      </body>
    </html>
  );
}
