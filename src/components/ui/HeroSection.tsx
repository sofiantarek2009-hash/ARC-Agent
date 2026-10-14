"use client";
import { Suspense } from "react";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import { motion } from "framer-motion";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { NORTH_STAR_DATA } from "@/data/company";

gsap.registerPlugin(ScrollTrigger);

export function HeroSection() {
  const heroRef = useRef<HTMLElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // Fallback to avoid hydration mismatch
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  useGSAP(() => {
    if (!contentRef.current) return;

    // Entrance Animation
    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });

    tl.fromTo(
      ".hero-badge",
      { y: 20, opacity: 0 },
      { y: 0, opacity: 1, duration: 1, delay: 0.2 }
    )
    .fromTo(
      ".hero-title",
      { y: 30, opacity: 0 },
      { y: 0, opacity: 1, duration: 1.2 },
      "-=0.8"
    )
    .fromTo(
      ".hero-desc",
      { y: 20, opacity: 0 },
      { y: 0, opacity: 1, duration: 1 },
      "-=0.9"
    )
    .fromTo(
      ".hero-cta",
      { y: 20, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.8, stagger: 0.1 },
      "-=0.8"
    )
    .fromTo(
      ".hero-stats",
      { opacity: 0 },
      { opacity: 1, duration: 1 },
      "-=0.4"
    );

    // Scroll Exit Animation (Parallax)
    gsap.to(".hero-bg-img", {
      scrollTrigger: {
        trigger: heroRef.current,
        start: "top top",
        end: "bottom top",
        scrub: true,
      },
      y: 150,
      opacity: 0.4,
      ease: "none",
    });

  }, { scope: heroRef, dependencies: [mounted] });

  return (
    <section 
      ref={heroRef} 
      id="hero"
      className="relative w-full h-[100svh] min-h-[800px] flex flex-col bg-[#0B0C0E] overflow-hidden pt-28" // pt-28 clears the new top navbar
    >
      {/* Background Media */}
      <div className="absolute inset-0 z-0 hero-bg-img">
        <Image
          src="/assets/hero-rig-split.jpg"
          alt="Offshore Rig"
          fill
          priority
          quality={100}
          className="object-cover object-center opacity-40 mix-blend-luminosity"
        />
        
      {/* Professional, subtle gradient overlays (Darkened heavily at the bottom for text contrast) */}
        <div className="absolute inset-0 bg-gradient-to-b from-[#0B0C0E]/20 via-transparent to-[#0B0C0E] pointer-events-none" />
        <div className="absolute inset-0 bg-black/30 pointer-events-none" />
      </div>

      {/* Content Container - Aligned to bottom left, extremely clean */}
      <div 
        ref={contentRef}
        className="relative z-10 w-full max-w-[1440px] mx-auto px-6 lg:px-12 pb-16 flex flex-col flex-1 justify-end"
      >
        <div className="max-w-4xl">
          <div className="hero-badge inline-flex items-center gap-3 mb-6">
            <span className="w-8 h-px bg-white/50" />
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/70">
              {NORTH_STAR_DATA.company.tagline}
            </span>
          </div>

          <h1 className="hero-title text-4xl md:text-5xl lg:text-7xl font-display font-light text-white leading-[1.1] tracking-tight mb-8">
            Where Physical Engineering<br className="hidden lg:block" />
            <span className="font-bold">Meets Digital Simulation.</span>
          </h1>

          <p className="hero-desc text-base md:text-lg text-white/60 max-w-2xl font-sans leading-relaxed mb-10">
            {NORTH_STAR_DATA.company.profile}
          </p>

          {/* Clean Corporate CTAs */}
          <div className="flex flex-wrap items-center gap-4">
            <a
              href="#capabilities"
              className="hero-cta inline-flex px-8 py-4 bg-white text-[#0B0C0E] text-xs font-bold font-mono uppercase tracking-[0.1em] hover:bg-white/90 transition-colors"
            >
              Explore Capabilities
            </a>
            
            <a
              href="#contact"
              className="hero-cta inline-flex px-8 py-4 border border-white/30 text-white text-xs font-bold font-mono uppercase tracking-[0.1em] hover:bg-white hover:text-[#0B0C0E] transition-colors"
            >
              Schedule Inspection
            </a>
          </div>
        </div>
      </div>

      {/* Stats Bar - Extremely minimal */}
      <div className="hero-stats relative z-10 w-full border-t border-white/[0.05] bg-[#0B0C0E]/80 backdrop-blur-md">
        <div className="max-w-[1440px] mx-auto flex flex-wrap">
          {NORTH_STAR_DATA.company.stats.map((stat, i) => (
            <div 
              key={i} 
              className="flex-1 min-w-[200px] border-r border-white/[0.05] last:border-r-0 px-6 lg:px-12 py-6"
            >
              <div className="font-mono text-[10px] uppercase tracking-[0.15em] text-white/40 mb-1">
                {stat.label}
              </div>
              <div className="font-display text-2xl lg:text-3xl font-light text-white">
                {stat.value}<span className="text-white/50">{stat.suffix}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
