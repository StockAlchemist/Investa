import type { NextConfig } from "next";

const isDesktop = process.env.APP_ENV === 'desktop';

const nextConfig: NextConfig = {
  allowedDevOrigins: ["100.66.59.98", "localhost:3000", "muon.tail33e9df.ts.net"],
  devIndicators: false,
  output: isDesktop ? 'export' : undefined,
  assetPrefix: isDesktop ? './' : undefined, // Fix loading assets in Electron (file:// protocol)
  images: {
    unoptimized: isDesktop,
  },
  async rewrites() {
    // Rewrites are not supported in static exports
    if (isDesktop) return [];

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
