import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Providers from "@/components/Providers";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Investa",
  description: "Personal Investment Portfolio Tracker",
  appleWebApp: {
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
      <body className={`${inter.className} antialiased selection:bg-cyan-500/20 selection:text-cyan-500 min-h-screen bg-background text-foreground`}>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
