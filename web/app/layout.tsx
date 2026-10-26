import type { Metadata, Viewport } from "next";
import { Fraunces, Inter } from "next/font/google";
import MotionPrefs from "@/components/MotionPrefs";
import Nav from "@/components/Nav";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-fraunces", weight: ["600", "700"] });

export const metadata: Metadata = {
  title: "QRummy — Beat the AI at Indian Rummy",
  description:
    "Play a 3-deck, 13-card Indian Rummy variant against a reinforcement-learning AI, with a live coach that explains every move.",
};

export const viewport: Viewport = { themeColor: "#0e1218", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable}`}>
      <body className="min-h-dvh flex flex-col">
        <Nav />
        <main className="flex-1">
          <MotionPrefs>{children}</MotionPrefs>
        </main>
        <footer className="border-t border-line text-muted text-xs px-4 py-4 text-center">
          QRummy · RL agent trained with{" "}
          <a href="https://github.com/datamllab/rlcard" className="underline hover:text-ink" target="_blank" rel="noreferrer">
            RLCard
          </a>{" "}
          (Zha et al., IJCAI 2020) · $0 build
        </footer>
      </body>
    </html>
  );
}
