"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { cn } from "@/lib/utils";
import { Target, Activity, Cpu, ShieldAlert, Navigation, Compass, Beaker } from "lucide-react";

// The four operational pillars with their corresponding assets
const SHOWCASE_DATA = [
  {
    id: "drilling",
    title: "High-Fidelity Simulation",
    subtitle: "RDS-100® Digital Core",
    image: "/assets/we-rds-console.jpg",
    color: "#00F0FF", // simulation cyan
    icon: Target,
    metrics: { scope: "Full-Scale", phase: "Active Execution", region: "MENA" },
    description: "Full-scale physical replication of complex well control and drilling engineering scenarios.",
  },
  {
    id: "simulation",
    title: "WeXplore VR Immersion",
    subtitle: "Virtual Environment Training",
    image: "/assets/we-vr-training.jpg",
    color: "#B026FF", // vr purple
    icon: Compass,
    metrics: { scope: "IWCF / IADC", phase: "Assurance", region: "MENA" },
    description: "Immersive scenario-based environments exposing personnel to critical field risks before deployment.",
  },
  {
    id: "inspection",
    title: "Structural Integrity",
    subtitle: "API 4G Cat III Audits",
    image: "/assets/aes-rig-inspection.jpg",
    color: "#FF6B00", // hazard orange
    icon: Activity,
    metrics: { scope: "API 4G", phase: "Certification", region: "Global" },
    description: "Certified heavy derrick inspection, DROPS surveys, and specialized NDT testing methodologies.",
  },
  {
    id: "chemicals",
    title: "Fluid Treatment Systems",
    subtitle: "Industrial Chemical Synthesis",
    image: "/assets/chemical-plant.jpg",
    color: "#00FF66", // chemical green
    icon: Beaker,
    metrics: { scope: "Treatment", phase: "Deployment", region: "Egypt" },
    description: "Optimized synthesis of demulsifiers, corrosion inhibitors, and specialty biocides for high-yield separation.",
  },
];

const springConfig = { type: "spring" as const, stiffness: 350, damping: 30 };

/* ─────────────────────────────────────────────
   TELEMETRY STREAM COMPONENT
───────────────────────────────────────────── */
export function CinematicShowcase() {
  const [activeIndex, setActiveIndex] = useState(0);
  const activeItem = SHOWCASE_DATA[activeIndex];

  return (
    <div className="relative w-full h-[650px] md:h-[800px] rounded-2xl md:rounded-[2rem] overflow-hidden border border-white/10 bg-void shadow-2xl flex flex-col md:flex-row">
      
      {/* ──────────────────────────────────────
          LEFT NAVIGATION / CONTROL PANEL
      ────────────────────────────────────── */}
      <div className="relative z-30 w-full md:w-80 lg:w-96 bg-void/80 md:bg-void/40 backdrop-blur-2xl border-b md:border-r border-white/5 flex flex-col shrink-0">
        <div className="p-6 md:p-8 border-b border-white/5 shrink-0">
          <span className="text-xs font-mono uppercase tracking-widest text-white/50 block mb-2">Operations</span>
          <h3 className="text-xl font-bold tracking-tight">Visual Engineering</h3>
        </div>

        <div className="flex-1 overflow-y-auto no-scrollbar p-4 md:p-6 flex flex-col gap-2">
          {SHOWCASE_DATA.map((item, index) => {
            const isActive = index === activeIndex;
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                onClick={() => setActiveIndex(index)}
                className="group relative flex items-start gap-4 p-4 text-left rounded-xl transition-colors hover:bg-white/5 active:scale-[0.98]"
              >
                {isActive && (
                  <motion.div
                    layoutId="active-indicator"
                    className="absolute inset-0 bg-white/5 rounded-xl border border-white/10 z-0"
                    transition={springConfig}
                  />
                )}
                
                <div className="relative z-10 flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <Icon
                      size={14}
                      className={isActive ? "" : "text-white/40"}
                      style={{ color: isActive ? item.color : undefined }}
                    />
                    <span
                      className={cn(
                        "font-bold text-sm tracking-tight transition-colors",
                        isActive ? "text-white" : "text-white/60 group-hover:text-white/80"
                      )}
                    >
                      {item.title}
                    </span>
                  </div>
                  <span className="text-xs font-mono text-white/40">{item.subtitle}</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Dynamic Metric Readouts (Bottom of Nav) */}
        <div className="p-6 md:p-8 border-t border-white/5 bg-black/20 shrink-0 hidden md:block">
          <AnimatePresence mode="popLayout">
            <motion.div
              key={activeItem.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={springConfig}
              className="grid grid-cols-2 gap-4"
            >
              {Object.entries(activeItem.metrics).map(([key, value]) => (
                <div key={key} className="flex flex-col gap-1">
                  <span className="text-[10px] font-mono text-white/40 uppercase tracking-widest">{key}</span>
                  <span
                    className="text-xs font-mono font-bold"
                    style={{ color: key === "status" ? activeItem.color : "#fff" }}
                  >
                    {value}
                  </span>
                </div>
              ))}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* ──────────────────────────────────────
          RIGHT VIEWPORT / CINEMATIC IMAGERY
      ────────────────────────────────────── */}
      <div className="relative flex-1 h-full overflow-hidden bg-black">
        <AnimatePresence initial={false}>
          <motion.div
            key={activeItem.id}
            initial={{ opacity: 0, scale: 1.05 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="absolute inset-0"
          >
            <Image
              src={activeItem.image}
              alt={activeItem.title}
              fill
              className="object-cover opacity-80"
              priority
            />
            {/* Cinematic Gradient Overlays */}
            <div className="absolute inset-0 bg-gradient-to-t from-void via-void/10 to-transparent" />
            <div className="absolute inset-0 bg-gradient-to-r from-void via-void/5 to-transparent" />
            
            {/* ──────────────────────────────────────
                DIGITAL HUD OVERLAYS (Visual Engineering)
            ────────────────────────────────────── */}
            
            {/* Central Target Crosshair */}
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.3, ...springConfig }}
              className="absolute inset-0 flex items-center justify-center pointer-events-none"
            >
              <div className="w-64 h-64 border border-white/10 rounded-full relative">
                <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-px h-4 bg-white/30" />
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-px h-4 bg-white/30" />
                <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-px bg-white/30" />
                <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-4 h-px bg-white/30" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Navigation 
                    size={24} 
                    className="opacity-30" 
                    style={{ color: activeItem.color }} 
                  />
                </div>
              </div>
            </motion.div>

            {/* Scanning Laser Line */}
            <motion.div
              animate={{ top: ["0%", "100%", "0%"] }}
              transition={{ duration: 8, ease: "linear", repeat: Infinity }}
              className="absolute left-0 right-0 h-[1px] z-10 opacity-40"
              style={{
                background: `linear-gradient(90deg, transparent, ${activeItem.color}, transparent)`,
                boxShadow: `0 0 20px ${activeItem.color}`,
              }}
            />
          </motion.div>
        </AnimatePresence>

        {/* Dynamic Overlay Description */}
        <div className="absolute bottom-6 left-6 md:bottom-10 md:left-10 max-w-md z-20 pointer-events-none">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeItem.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={springConfig}
            >
              <h2 className="text-3xl font-bold tracking-tight mb-3 drop-shadow-lg text-white">
                {activeItem.title}
              </h2>
              <p className="text-white/70 text-sm leading-relaxed backdrop-blur-md bg-black/20 p-4 rounded-xl border border-white/5">
                {activeItem.description}
              </p>
            </motion.div>
          </AnimatePresence>
        </div>

        
        
        {/* Frame Brackets */}
        <div className="absolute top-6 left-6 w-8 h-8 border-t-2 border-l-2 border-white/20 z-10" />
        <div className="absolute top-6 right-6 w-8 h-8 border-t-2 border-r-2 border-white/20 z-10" />
        <div className="absolute bottom-6 left-6 w-8 h-8 border-b-2 border-l-2 border-white/20 z-10 hidden md:block" />
        
      </div>
    </div>
  );
}
