"use client";

import { useRef } from "react";
import { motion, useScroll, useSpring, useTransform } from "framer-motion";

export function ScrollPipeline() {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const { scrollYProgress } = useScroll();
  
  // Kowalski spring physics on the scroll progress
  const smoothProgress = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    mass: 0.5,
  });

  // Map progress to the stroke dashoffset (1 = fully hidden, 0 = fully drawn)
  const pathLength = useTransform(smoothProgress, [0, 1], [0, 1]);
  
  // Map progress to gradient stop position for color transition
  const gradientStop = useTransform(smoothProgress, [0, 0.5, 1], ["0%", "50%", "100%"]);

  return (
    <div
      ref={containerRef}
      className="fixed left-4 md:left-8 top-0 h-screen w-8 z-30 pointer-events-none hidden lg:block"
    >
      <svg
        className="h-full w-full"
        viewBox="0 0 32 800"
        preserveAspectRatio="none"
        fill="none"
      >
        <defs>
          <linearGradient id="pipeline-gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00F0FF" stopOpacity="0.8" />
            <stop offset="60%" stopColor="#FF4500" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#FF4500" stopOpacity="0.4" />
          </linearGradient>
          <linearGradient id="pipeline-track" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.03" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0.01" />
          </linearGradient>
        </defs>
        
        {/* Track (always visible, very dim) */}
        <motion.path
          d="M 16 0 L 16 800"
          stroke="url(#pipeline-track)"
          strokeWidth="2"
          strokeLinecap="round"
        />
        
        {/* Active fill (follows scroll) */}
        <motion.path
          d="M 16 0 L 16 800"
          stroke="url(#pipeline-gradient)"
          strokeWidth="2"
          strokeLinecap="round"
          style={{
            pathLength: pathLength,
          }}
        />
        
        {/* Glow dot at the leading edge */}
        <motion.circle
          r="4"
          fill="#00F0FF"
          style={{
            cy: useTransform(smoothProgress, [0, 1], [0, 800]),
            cx: 16,
            filter: "drop-shadow(0 0 6px #00F0FF) drop-shadow(0 0 12px #00F0FF)",
          }}
        />
      </svg>
    </div>
  );
}
