"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { name: "Simulation", target: "/services/simulation" },
  { name: "Inspection", target: "/services/inspection" },
  { name: "Training", target: "/services/training" },
  { name: "Chemicals", target: "/services/chemicals" },
  { name: "About", target: "/about" },
];

export function EliteNavbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [visible, setVisible] = useState(true);
  const progressRef = useRef<HTMLDivElement>(null);
  
  let lastScrollY = typeof window !== "undefined" ? window.scrollY : 0;

  useEffect(() => {
    if (typeof window === "undefined") return;
    
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      setScrolled(currentScrollY > 50);

      // Hide navbar on scroll down, show on scroll up
      if (currentScrollY > lastScrollY && currentScrollY > 200) {
        setVisible(false);
      } else {
        setVisible(true);
      }
      lastScrollY = currentScrollY;

      // Update progress bar
      if (progressRef.current) {
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolledPercent = (winScroll / height) * 100;
        progressRef.current.style.width = scrolledPercent + "%";
      }
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, target: string) => {
    e.preventDefault();
    setMobileOpen(false);
    
    if (target === "#contact") {
      if (window.location.pathname !== "/") {
        window.location.href = "/#contact";
      } else {
        const el = document.querySelector(target);
        if (el) {
          const y = el.getBoundingClientRect().top + window.scrollY - 80;
          window.scrollTo({ top: y, behavior: "smooth" });
        }
      }
    } else if (target.startsWith("#")) {
      const el = document.querySelector(target);
      if (el) {
        const y = el.getBoundingClientRect().top + window.scrollY - 80;
        window.scrollTo({ top: y, behavior: "smooth" });
      }
    } else {
      window.location.href = target;
    }
  };

  const isNotHome = typeof window !== "undefined" && window.location.pathname !== "/";

  return (
    <>
      <motion.header
        initial={{ y: 0, opacity: 1 }}
        animate={{
          y: visible ? 0 : -100,
          opacity: visible ? 1 : 0,
        }}
        transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
        className={cn(
          "fixed z-50 top-0 left-0 w-full transition-colors duration-500",
          scrolled
            ? "bg-[#0B0C0E]/90 backdrop-blur-xl border-b border-white/[0.05]"
            : "bg-transparent border-transparent"
        )}
      >
        <nav className="max-w-[1440px] mx-auto flex items-center justify-between px-6 lg:px-12 py-5 lg:py-6">
          <a
            href="/"
            onClick={(e) => handleNavClick(e, "/")}
            className="font-bold tracking-tighter font-display shrink-0 flex items-center gap-2 text-xl group"
          >
            <span className="text-white">NORTHSTAR</span>
          </a>

          <div className="hidden lg:flex items-center gap-8">
            {isNotHome && (
              <a
                href="/"
                onClick={(e) => handleNavClick(e, "/")}
                className="text-xs font-mono font-bold uppercase tracking-[0.1em] text-white transition-colors duration-300 relative after:content-[''] after:absolute after:-bottom-2 after:left-0 after:w-full after:h-px after:bg-white"
              >
                HOME
              </a>
            )}
            {NAV_LINKS.map((link) => (
              <a
                key={link.name}
                href={link.target}
                onClick={(e) => handleNavClick(e, link.target)}
                className="text-xs font-mono uppercase tracking-[0.1em] text-white/60 hover:text-white transition-colors duration-300 relative after:content-[''] after:absolute after:-bottom-2 after:left-0 after:w-0 after:h-px after:bg-white after:transition-all hover:after:w-full"
              >
                {link.name}
              </a>
            ))}
          </div>

          <motion.a
            href="#contact"
            onClick={(e) => handleNavClick(e, "#contact")}
            whileTap={{ scale: 0.97 }}
            className="hidden lg:flex items-center px-6 py-2.5 bg-white text-[#0B0C0E] text-xs font-bold font-mono uppercase tracking-[0.1em] hover:bg-white/90 transition-colors duration-200"
          >
            Get in Touch
          </motion.a>

          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="lg:hidden flex flex-col justify-center items-center w-10 h-10 gap-1.5 shrink-0 relative z-50"
            aria-label="Toggle Menu"
          >
            <motion.span
              animate={mobileOpen ? { rotate: 45, y: 5 } : { rotate: 0, y: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 25 }}
              className="block w-6 h-[2px] bg-white"
            />
            <motion.span
              animate={mobileOpen ? { opacity: 0, scaleX: 0 } : { opacity: 1, scaleX: 1 }}
              transition={{ type: "spring", stiffness: 400, damping: 25 }}
              className="block w-6 h-[2px] bg-white"
            />
            <motion.span
              animate={mobileOpen ? { rotate: -45, y: -7 } : { rotate: 0, y: 0 }}
              transition={{ type: "spring", stiffness: 400, damping: 25 }}
              className="block w-6 h-[2px] bg-white"
            />
          </button>
        </nav>

        <div className="absolute bottom-0 left-0 right-0 h-[1px] bg-white/[0.05]">
          <div
            ref={progressRef}
            className="h-full bg-hazard-orange transition-[width] duration-100"
            style={{ width: "0%" }}
          />
        </div>
      </motion.header>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
            className="fixed inset-0 z-40 bg-[#0B0C0E]/95 backdrop-blur-3xl flex flex-col justify-center lg:hidden"
          >
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />
            
            <nav className="flex flex-col gap-6 w-full px-8 relative z-10">
              {isNotHome && (
                <motion.a
                  href="/"
                  onClick={(e) => handleNavClick(e, "/")}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="text-4xl font-display font-bold text-white hover:text-white transition-colors border-b border-white/10 pb-6 mb-2"
                >
                  Home
                </motion.a>
              )}
              {NAV_LINKS.map((link, i) => (
                <motion.a
                  key={link.name}
                  href={link.target}
                  onClick={(e) => handleNavClick(e, link.target)}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ delay: i * 0.05 }}
                  className="text-4xl font-display font-medium text-white/80 hover:text-white transition-colors"
                >
                  {link.name}
                </motion.a>
              ))}
              <motion.a
                href="#contact"
                onClick={(e) => handleNavClick(e, "#contact")}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ delay: NAV_LINKS.length * 0.05 + 0.1 }}
                className="mt-8 px-8 py-4 bg-white text-[#0B0C0E] text-sm font-bold font-mono text-center uppercase tracking-widest active:scale-[0.97] transition-transform"
              >
                Get in Touch
              </motion.a>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
