import type { NextConfig } from "next";
import withPWAInit from "@ducanh2912/next-pwa";

const withPWA = withPWAInit({
  dest: "public",
  disable: process.env.NODE_ENV === "development",
});

const API_URL = process.env.API_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  compress: true,
  allowedDevOrigins: ["100.66.59.98", "localhost:3000", "muon.tail33e9df.ts.net", "muon", "*.ts.net"],
  devIndicators: false,
  turbopack: {}, // Required to prevent error when using next-pwa in Next 16
  async headers() {
    return [
      {
        source: '/:all*(svg|jpg|png|webp|ico|woff|woff2)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${API_URL}/api/:path*`,
      },
      {
        source: '/docs',
        destination: `${API_URL}/docs`,
      },
      {
        source: '/openapi.json',
        destination: `${API_URL}/openapi.json`,
      },
    ];
  },
};

export default withPWA(nextConfig);
