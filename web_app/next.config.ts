import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["100.66.59.98", "localhost:3000", "muon.tail33e9df.ts.net"],
  devIndicators: false,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
      {
        source: '/docs',
        destination: 'http://127.0.0.1:8000/docs',
      },
      {
        source: '/openapi.json',
        destination: 'http://127.0.0.1:8000/openapi.json',
      },
    ];
  },
};

export default nextConfig;
