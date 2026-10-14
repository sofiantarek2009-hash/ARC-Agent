"use client";

import React, { useState, Suspense } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Canvas } from "@react-three/fiber";
import { Environment, OrbitControls, ContactShadows } from "@react-three/drei";
import { SmartModelViewer } from "./SmartModelViewer";
import { X, Box, Loader2 } from "lucide-react";

export interface AssetDef {
  id: string;
  name: string;
  file: string;
  desc?: string;
}

interface AssetBentoGridProps {
  title: string;
  subtitle?: string;
  assets: AssetDef[];
}

export function AssetBentoGrid({ title, subtitle, assets }: AssetBentoGridProps) {
  const [activeAsset, setActiveAsset] = useState<AssetDef | null>(null);

  return (
    <section className="py-24 bg-[#0B0C0E] border-t border-white/5 relative z-10">
      <div className="max-w-[1440px] mx-auto px-6 lg:px-12">
        <div className="mb-12">
          <h2 className="text-sm font-mono uppercase tracking-widest text-hazard-orange mb-4">{subtitle || "Digital Twin Inventory"}</h2>
          <h3 className="text-3xl md:text-5xl font-display font-light text-white">{title}</h3>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {assets.map((asset) => (
            <button
              key={asset.id}
              onClick={() => setActiveAsset(asset)}
              className="group relative h-48 rounded-2xl border border-white/10 bg-[#12141A] overflow-hidden flex flex-col items-start justify-end p-6 hover:border-hazard-orange/50 transition-colors text-left"
            >
              <div className="absolute inset-0 bg-gradient-to-t from-[#0B0C0E] to-transparent z-10" />
              <div className="absolute top-6 right-6 opacity-30 group-hover:opacity-100 group-hover:text-hazard-orange transition-all z-20">
                <Box className="w-6 h-6" />
              </div>
              <h4 className="relative z-20 font-display text-xl text-white group-hover:text-hazard-orange transition-colors">{asset.name}</h4>
              <p className="relative z-20 font-mono text-[10px] uppercase tracking-widest text-white/40 mt-2">
                Press to inspect asset
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Fullscreen Interactive Modal */}
      <AnimatePresence>
        {activeAsset && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-[#0B0C0E]/95 backdrop-blur-xl flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/10 relative z-20 bg-[#0B0C0E]">
              <div>
                <div className="font-mono text-[10px] text-hazard-orange uppercase tracking-[0.2em] mb-1">Asset Viewer</div>
                <h2 className="font-display text-2xl font-semibold text-white">{activeAsset.name}</h2>
              </div>
              <button
                onClick={() => setActiveAsset(null)}
                className="w-12 h-12 flex items-center justify-center rounded-full bg-white/5 hover:bg-hazard-orange hover:text-[#0B0C0E] transition-colors text-white"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            {/* 3D Viewport */}
            <div className="flex-1 relative">
              <div className="absolute bottom-8 right-8 z-20 pointer-events-none text-right">
                <p className="font-mono text-xs uppercase tracking-widest text-white/40">Drag to Rotate</p>
                <p className="font-mono text-xs uppercase tracking-widest text-white/40">Scroll to Zoom</p>
              </div>

              <Suspense fallback={
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <Loader2 className="w-8 h-8 text-hazard-orange animate-spin mb-4" />
                  <span className="font-mono text-xs uppercase tracking-widest text-white/50">Processing Geometry...</span>
                </div>
              }>
                <Canvas camera={{ position: [5, 5, 5], fov: 45 }} gl={{ antialias: true, alpha: true }}>
                  <ambientLight intensity={0.5} />
                  <directionalLight position={[10, 10, 10]} intensity={1.5} castShadow />
                  <pointLight position={[-10, -10, -10]} intensity={0.5} />
                  
                  <SmartModelViewer url={activeAsset.file} />
                  
                  <OrbitControls makeDefault enableDamping dampingFactor={0.05} autoRotate autoRotateSpeed={0.5} />
                  <Environment preset="city" />
                  <ContactShadows position={[0, -2, 0]} opacity={0.6} scale={20} blur={2} far={4} />
                </Canvas>
              </Suspense>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
