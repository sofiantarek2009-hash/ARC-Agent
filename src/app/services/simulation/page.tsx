"use client";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, Environment, OrbitControls, Center } from "@react-three/drei";
import React, { Suspense, useRef } from "react";
import * as THREE from "three";


import Image from "next/image";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { EliteNavbar } from "@/components/ui/EliteNavbar";
import { IndustrialFooter } from "@/components/ui/IndustrialFooter";
import { ServiceDiorama } from "@/components/3d/ServiceDiorama";
import { SimulatorShowcase } from "@/components/ui/SimulatorShowcase";
import { Monitor, Cpu, ShieldCheck, Zap } from "lucide-react";

gsap.registerPlugin(ScrollTrigger);

export default function SimulationPage() {
  const container = useRef<HTMLDivElement>(null);
  const heroImgRef = useRef<HTMLImageElement>(null);
  const specsRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    // Parallax hero image
    if (heroImgRef.current) {
      gsap.to(heroImgRef.current, {
        yPercent: 30,
        ease: "none",
        scrollTrigger: {
          trigger: container.current,
          start: "top top",
          end: "bottom top",
          scrub: true,
        },
      });
    }

    // Specs staggered reveal
    const specCards = gsap.utils.toArray(".spec-card");
    if (specCards.length > 0 && specsRef.current) {
      gsap.fromTo(
        specCards,
        { y: 40, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.8,
          stagger: 0.15,
          ease: "power3.out",
          scrollTrigger: {
            trigger: specsRef.current,
            start: "top 85%",
            toggleActions: "play none none reverse",
          },
        }
      );
    }
  }, { scope: container });

  const specs = [
    {
      icon: Monitor,
      title: "Full-Scale Cab",
      desc: "Immersive replica environment featuring actual rig floor controls and authentic operational feel."
    },
    {
      icon: Cpu,
      title: "Cyber Chair",
      desc: "Ergonomic command station integrated with the latest Driller's cabin systems and UI."
    },
    {
      icon: ShieldCheck,
      title: "IWCF & IADC Approved",
      desc: "Fully certified simulation models meeting the highest international training standards."
    },
    {
      icon: Zap,
      title: "Virtual Reality WEXPLORE",
      desc: "Next-generation VR module providing 360-degree spatial awareness for critical maneuvers."
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-white selection:bg-neon-cyan/30 selection:text-cyan-50 font-sans" ref={container}>
      <EliteNavbar />
      
      <main className="pb-16 overflow-hidden">
        {/* Cinematic Parallax Hero */}
        <section className="relative h-[70vh] min-h-[500px] w-full flex items-center justify-center overflow-hidden">
          <div className="absolute inset-0 z-0 bg-slate-950">
            <Image
              ref={heroImgRef}
              src="/assets/media_1789894416489.jpg"
              alt="RDS-100 Simulator Console"
              fill
              className="object-cover opacity-60 mix-blend-luminosity scale-110"
              sizes="100vw"
              priority
            />
          </div>
          {/* Dark gradient overlays */}
          <div className="absolute inset-0 z-10 bg-gradient-to-b from-slate-950/80 via-transparent to-slate-950" />
          <div className="absolute inset-0 z-10 bg-gradient-to-t from-slate-950 via-slate-950/20 to-transparent" />
          
          <div className="relative z-20 text-center px-6 mt-16">
            <h1 className="text-5xl md:text-7xl font-display font-bold tracking-tight mb-4 drop-shadow-2xl">
              Technology & <span className="text-neon-cyan">Simulation</span>
            </h1>
            <p className="text-lg md:text-xl text-slate-300 max-w-2xl mx-auto font-mono">
              Empowering crews with high-fidelity operational scenarios, mitigating risks before they materialize on the rig floor.
            </p>
          </div>
        </section>

        {/* 3D Simulator Showcase */}
        <section className="relative z-20 -mt-20 md:-mt-32 max-w-[1400px] mx-auto px-6 md:px-12">
          <div className="glass-panel-heavy rounded-2xl border border-white/10 overflow-hidden backdrop-blur-xl bg-slate-900/60 shadow-2xl">
            <div className="h-[600px] w-full">
              <SimulatorShowcase />
            </div>
          </div>
        </section>

        {/* Technical Specs Grid */}
        <section ref={specsRef} className="py-24 max-w-7xl mx-auto px-6 md:px-12">
          <div className="mb-16 text-center">
            <h2 className="text-sm font-mono text-hazard-orange tracking-widest uppercase mb-4">
              RDS-100 Capabilities
            </h2>
            <h3 className="text-3xl md:text-4xl font-display font-medium text-white">
              Engineered for Realism
            </h3>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {specs.map((spec, idx) => (
              <div key={idx} className="spec-card glass-panel p-8 rounded-xl border border-white/5 hover:border-neon-cyan/40 transition-all duration-300 group">
                <div className="w-12 h-12 rounded-lg bg-slate-800 flex items-center justify-center mb-6 border border-white/10 group-hover:bg-neon-cyan/10 transition-colors">
                  <spec.icon className="w-6 h-6 text-neon-cyan" />
                </div>
                <h4 className="text-xl font-display font-medium text-white mb-3">
                  {spec.title}
                </h4>
                <p className="text-slate-400 text-sm leading-relaxed font-sans">
                  {spec.desc}
                </p>
              </div>
            ))}
          </div>
        </section>
      </main>

      
      

<ServiceDiorama type="simulation" />
<IndustrialFooter />
    </div>
  );
}
