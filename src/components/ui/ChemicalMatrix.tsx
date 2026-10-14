"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { cn } from "@/lib/utils";

interface ChemicalCategory {
  name: string;
  products: string[];
}

interface ChemicalMatrixProps {
  categories: ChemicalCategory[];
  className?: string;
}

export function ChemicalMatrix({ categories, className }: ChemicalMatrixProps) {
  const [activeCategory, setActiveCategory] = useState(0);

  return (
    <div className={cn("grid grid-cols-1 md:grid-cols-[280px_1fr] gap-0 rounded-2xl overflow-hidden glass-panel-heavy", className)}>
      {/* Left Panel: Category Selector */}
      <div className="flex flex-col bg-white/[0.02] border-r border-white/5 z-10">
        <div className="p-4 border-b border-white/5">
          <span className="text-xs font-mono uppercase tracking-widest text-white/30">
            Chemical Categories
          </span>
        </div>
        {categories.map((cat, i) => (
          <button
            key={cat.name}
            onClick={() => setActiveCategory(i)}
            className={cn(
              "relative text-left px-5 py-4 text-sm transition border-b border-white/5",
              "hover:bg-white/5",
              activeCategory === i
                ? "text-neon-cyan font-semibold bg-white/[0.05]"
                : "text-white/50"
            )}
          >
            {/* Active indicator bar */}
            {activeCategory === i && (
              <motion.div
                layoutId="chemical-indicator"
                className="absolute left-0 top-0 bottom-0 w-[3px] bg-neon-cyan"
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
              />
            )}
            <span className="relative z-10">{cat.name}</span>
            <span className="block text-xs text-white/20 font-mono mt-1">
              {cat.products.length} compounds
            </span>
          </button>
        ))}
      </div>

      {/* Right Panel: Product Display & WebGL Background */}
      <div className="relative min-h-[400px] p-6 lg:p-10 bg-[#030712] overflow-hidden">
        {/* Abstract Wave Background */}
        <div className="absolute inset-0 z-0">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeCategory}
              initial={{ opacity: 0, scale: 1.05 }}
              animate={{ opacity: 0.6, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.6 }}
              className="absolute inset-0"
            >
              <Image
                src={`/assets/chemical-${categories[activeCategory].name.toLowerCase().replace(' ', '-').replace(' ', '-')}.jpg`}
                alt={categories[activeCategory].name}
                fill
                className="object-cover mix-blend-screen"
                onError={(e) => {
                  e.currentTarget.src = "/assets/aes-chemical-hud.jpg";
                }}
              />
            </motion.div>
          </AnimatePresence>
          {/* Subtle gradient overlay to ensure text readability */}
          <div className="absolute inset-0 bg-gradient-to-r from-[#030712] via-[#030712]/70 to-transparent pointer-events-none" />
          <div className="absolute inset-0 bg-gradient-to-t from-[#030712] via-transparent to-transparent pointer-events-none" />
        </div>

        {/* Foreground Content */}
        <div className="relative z-10">
          <AnimatePresence mode="wait">
            <motion.div
              key={categories[activeCategory].name}
              initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="flex flex-col gap-2"
          >
            <h4 className="text-lg font-bold mb-4 font-display">
              {categories[activeCategory].name}
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {categories[activeCategory].products.map((product, j) => (
                <motion.div
                  key={product}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{
                    type: "spring",
                    stiffness: 350,
                    damping: 25,
                    delay: j * 0.04,
                  }}
                  className="flex items-center gap-3 px-4 py-3 rounded-lg bg-white/[0.03] border border-white/5 hover:border-hazard-orange/30 hover:bg-white/[0.05] transition-colors group glass-panel"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-hazard-orange/60 group-hover:bg-hazard-orange transition-colors shrink-0 shadow-glow-orange-sm" />
                  <span className="text-sm text-white/70 group-hover:text-white/90 transition-colors font-mono">
                    {product}
                  </span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
