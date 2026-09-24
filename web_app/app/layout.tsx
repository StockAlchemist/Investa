import type { Metadata, Viewport } from "next";
import { Geist, Instrument_Serif } from "next/font/google";
import "./globals.css";
import Providers from "@/components/Providers";
import { WebVitals } from "@/components/WebVitals";
import { AuthProvider } from "@/context/AuthContext";
import LazyAIChat from "@/components/LazyAIChat";

// Ledger's type pair. Geist carries the interface and every figure;
// Instrument Serif is for page titles and the hero figure only. Both expose a
// CSS variable that globals.css maps onto `font-sans` and `font-display`.
const geist = Geist({ subsets: ["latin"], variable: "--font-geist" });
const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-instrument-serif",
});

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#F5F4EF" },
    { media: "(prefers-color-scheme: dark)", color: "#0F1013" },
  ],
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  // The phone shell's bars pad themselves with `env(safe-area-inset-*)`, which
  // resolves to 0 unless the viewport covers the notch and home indicator.
  viewportFit: "cover",
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
    <html lang="en" className={`${geist.variable} ${instrumentSerif.variable}`} suppressHydrationWarning>
      <body className="font-sans antialiased selection:bg-primary/20 min-h-screen bg-background text-foreground" suppressHydrationWarning>
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
