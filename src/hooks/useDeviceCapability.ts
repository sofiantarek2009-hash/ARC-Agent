"use client";
import { useState, useEffect } from "react";

export function useDeviceCapability() {
  const [isMobile, setIsMobile] = useState(false);
  const [isLowTier, setIsLowTier] = useState(false);

  useEffect(() => {
    // Basic mobile detection via screen width
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    // Hardware concurrency check (CPU cores)
    // Low tier devices (e.g. older phones) typically have < 4 logical cores
    const checkPerformance = () => {
      if (typeof navigator !== "undefined" && navigator.hardwareConcurrency) {
        setIsLowTier(navigator.hardwareConcurrency < 4);
      }
    };

    checkMobile();
    checkPerformance();

    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  return { isMobile, isLowTier, disableWebGL: isMobile && isLowTier };
}
