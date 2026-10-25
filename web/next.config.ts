import type { NextConfig } from "next";

// GitHub Pages serves the site from /<repo>/; CI sets NEXT_PUBLIC_BASE_PATH=/<repo>. Empty locally.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

const nextConfig: NextConfig = {
  output: "export",
  basePath: basePath || undefined,
  assetPrefix: basePath || undefined,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
