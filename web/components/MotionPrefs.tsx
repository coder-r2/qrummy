"use client";

import { MotionConfig } from "motion/react";

// Honour the OS "reduce motion" setting: transforms are skipped, opacity fades stay.
export default function MotionPrefs({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
