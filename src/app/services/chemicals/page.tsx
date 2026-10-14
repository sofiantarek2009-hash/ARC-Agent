"use client";

import { useRef, Suspense } from "react";
import Image from "next/image";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { EliteNavbar } from "@/components/ui/EliteNavbar";
import { IndustrialFooter } from "@/components/ui/IndustrialFooter";
import { ServiceDiorama } from "@/components/3d/ServiceDiorama";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, Environment, OrbitControls, Center, ContactShadows } from "@react-three/drei";
import * as THREE from "three";
import { ChemicalMatrix } from "@/components/ui/ChemicalMatrix";
import FluidParticleSim from "@/components/3d/FluidParticleSim";
import { NORTH_STAR_DATA } from "@/data/company";

gsap.registerPlugin(ScrollTrigger);

export default function ChemicalsPage() {
  const containerRef = useRef<HTMLElement>(null);
  
  useGSAP(() => {
    // Intro text animation
    gsap.fromTo(
      ".hero-text",
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 1, stagger: 0.1, ease: "power4.out", delay: 0.2 }
    );
    
    // Scroll triggers for fade-up elements
    const elements = gsap.utils.toArray(".fade-up-element");
    elements.forEach((el: any) => {
      gsap.fromTo(
        el,
        { opacity: 0, y: 40 },
        {
          opacity: 1, 
          y: 0,
          duration: 0.8,
          ease: "power3.out",
          scrollTrigger: {
            trigger: el,
            start: "top 85%",
            toggleActions: "play none none reverse"
          }
        }
      );
    });
  }, { scope: containerRef });

  return (
    <main ref={containerRef} className="min-h-screen bg-[#030712] selection:bg-neon-cyan/30">
      <EliteNavbar />
      
      {/* Hero Section */}
      <section className="relative w-full h-[60svh] min-h-[500px] flex flex-col justify-end overflow-hidden pt-24 border-b border-white/[0.05]">
        <div className="absolute inset-0 z-0">
          <Image
            src="/assets/data-network-abstract.jpg"
            alt="Industrial Chemicals"
            fill
            priority
            className="object-cover opacity-40 mix-blend-luminosity"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#030712] via-[#030712]/60 to-transparent pointer-events-none" />
          <div className="absolute inset-0 bg-gradient-to-r from-[#030712] via-[#030712]/40 to-transparent pointer-events-none" />
        </div>
        
        <div className="relative z-10 w-full px-6 lg:px-12 pb-16 max-w-7xl mx-auto">
          <div className="hero-text inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 bg-white/5 backdrop-blur-md mb-6">
            <span className="flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-1.5 w-1.5 rounded-full bg-neon-cyan opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-neon-cyan" />
            </span>
            <span className="text-[10px] font-mono uppercase tracking-widest text-white/70">
              AES & WE Solutions
            </span>
          </div>
          <h1 className="hero-text text-5xl md:text-7xl font-display font-bold text-white leading-tight tracking-tight mb-6">
            Industrial Chemicals
          </h1>
          <p className="hero-text text-lg text-white/60 max-w-2xl font-sans leading-relaxed">
            {NORTH_STAR_DATA.services.chemicals.desc}
          </p>
        </div>
      </section>

      {/* Content Section */}
      <section className="relative py-24 w-full px-6 lg:px-12 overflow-hidden">
        {/* Background 3D Sim */}
        <div className="absolute inset-0 z-0">
          <FluidParticleSim />
          {/* Gradients to fade sim at top/bottom edges */}
          <div className="absolute inset-0 bg-gradient-to-b from-[#030712] via-transparent to-[#030712] pointer-events-none" />
        </div>
        
        <div className="relative z-10 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 mb-24">
          <div className="fade-up-element flex flex-col justify-center">
            <div className="inline-flex items-center gap-2 mb-6">
              <div className="w-8 h-[1px] bg-neon-cyan" />
              <span className="text-xs font-mono uppercase tracking-widest text-neon-cyan">Laboratory & Manufacturing</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-display font-bold text-white mb-6">
              Advanced Formulations & High-Yield Manufacturing
            </h2>
            <p className="text-white/70 leading-relaxed mb-8">
              Our state-of-the-art laboratory facilities engineer precision chemical formulations designed for high-stress industrial environments. By integrating rigorous testing with scaled manufacturing capabilities, we ensure every compound delivers unmatched reliability and optimal performance.
            </p>
            <div className="flex flex-wrap gap-4">
               <div className="glass-panel p-4 rounded-lg flex-1 min-w-[140px] border border-white/10">
                 <div className="text-3xl font-display font-bold text-white mb-1">ISO</div>
                 <div className="text-[10px] text-white/50 uppercase font-mono tracking-wider">9001 Certified</div>
               </div>
               <div className="glass-panel p-4 rounded-lg flex-1 min-w-[140px] border border-white/10">
                 <div className="text-3xl font-display font-bold text-white mb-1">24/7</div>
                 <div className="text-[10px] text-white/50 uppercase font-mono tracking-wider">Quality Control</div>
               </div>
            </div>
          </div>
          
          <div className="fade-up-element flex flex-col justify-center">
            <div className="glass-panel-heavy p-8 md:p-10 rounded-2xl border border-white/10 relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity duration-500">
                <svg width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
              </div>
              
              <div className="relative z-10">
                <h3 className="text-xl md:text-2xl font-display font-bold text-white mb-3">Download MSDS</h3>
                <p className="text-sm text-white/60 mb-8 max-w-sm leading-relaxed">
                  Access comprehensive Material Safety Data Sheets for all our proprietary blends, active agents, and industrial solutions.
                </p>
                <button className="w-full flex items-center justify-between px-6 py-4 rounded-xl bg-white/[0.03] border border-white/10 hover:border-hazard-orange/50 hover:bg-hazard-orange/10 transition-all duration-300 active:scale-[0.97]">
                  <span className="font-mono text-xs font-bold uppercase tracking-widest text-white/80 group-hover:text-hazard-orange transition-colors">Access Document Portal</span>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white/50 group-hover:text-hazard-orange transition-colors">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="relative z-10 max-w-7xl mx-auto fade-up-element">
          <div className="mb-10 text-center md:text-left">
            <h2 className="text-3xl md:text-4xl font-display font-bold text-white mb-3">Chemical Product Matrix</h2>
            <p className="text-white/50 text-sm font-mono uppercase tracking-widest">Select a category to view specialized formulations</p>
          </div>
          <ChemicalMatrix 
            categories={NORTH_STAR_DATA.services.chemicals.categories} 
            className="shadow-2xl shadow-black/50"
          />
        </div>
      </section>

      <ServiceDiorama type="chemicals" />
<IndustrialFooter />
    </main>
  );
}
