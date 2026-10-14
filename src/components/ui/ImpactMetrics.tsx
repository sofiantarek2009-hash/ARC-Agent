"use client";

import React, { useRef, useEffect } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { NORTH_STAR_DATA } from "@/data/company";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const metricsData = NORTH_STAR_DATA.company.stats;

export function ImpactMetrics() {
  const sectionRef = useRef<HTMLElement>(null);
  const numbersRef = useRef<(HTMLSpanElement | null)[]>([]);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (prefersReducedMotion) {
      numbersRef.current.forEach((el, index) => {
        if (el) {
          el.innerText = metricsData[index].value.toString();
        }
      });
      return;
    }

    const ctx = gsap.context(() => {
      numbersRef.current.forEach((el, index) => {
        if (!el) return;
        const target = metricsData[index].value;
        const obj = { val: 0 };

        gsap.to(obj, {
          val: target,
          duration: 2,
          ease: "power3.out",
          scrollTrigger: {
            trigger: sectionRef.current,
            start: "top 80%",
            toggleActions: "play none none none",
          },
          onUpdate: () => {
            el.innerText = Math.round(obj.val).toString();
          },
        });
      });
      
      // Quote reveal animation
      gsap.fromTo(".quote-element",
        { y: 40, opacity: 0 },
        { 
          y: 0, 
          opacity: 1, 
          stagger: 0.2, 
          duration: 1, 
          ease: "power3.out",
          scrollTrigger: {
            trigger: ".quote-container",
            start: "top 85%"
          }
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="w-full bg-[#0B0C0E] py-24 md:py-32 relative overflow-hidden"
    >
      <div className="max-w-[1440px] mx-auto px-6 lg:px-12">
        {/* Crisp Corporate Metrics Grid */}
        <div className="flex overflow-x-auto snap-x snap-mandatory md:grid md:grid-cols-3 lg:grid-cols-5 border-t border-l border-white/10 mb-24 lg:mb-32 hide-scrollbar">
          {metricsData.map((metric, i) => (
            <div
              key={i}
              className="flex-none w-[75vw] sm:w-[50vw] md:w-auto snap-center flex flex-col justify-end p-8 border-r border-b border-white/10 h-48 md:h-56 bg-[#0B0C0E] hover:bg-white/[0.02] transition-colors duration-300"
            >
              <div className="text-5xl lg:text-6xl font-display font-medium tabular-nums tracking-tight mb-2 text-white flex items-end">
                <span ref={(el) => { numbersRef.current[i] = el; }}>{metric.value}</span>
                <span className="text-3xl text-hazard-orange ml-1">{metric.suffix}</span>
              </div>
              <div className="font-mono text-[10px] uppercase tracking-widest text-white/40">
                {metric.label}
              </div>
            </div>
          ))}
        </div>

        {/* High-End Typographic Statement */}
        <div className="quote-container grid grid-cols-1 md:grid-cols-12 gap-8 items-center border-l-2 border-hazard-orange pl-8 py-4">
          <div className="md:col-span-8">
            <h2 className="quote-element font-display text-3xl md:text-5xl lg:text-6xl text-white font-medium tracking-tight leading-[1.1]">
              Ensuring execution certainty through <span className="text-white/40">integrated engineering and capability assurance.</span>
            </h2>
          </div>
          <div className="md:col-span-4 flex md:justify-end">
            <div className="quote-element text-left md:text-right">
              <span className="block font-mono text-[10px] text-hazard-orange uppercase tracking-widest mb-2">Core Principle</span>
              <p className="font-sans text-sm text-white/50 max-w-[200px] ml-auto">
                Setting the global benchmark for technical execution, safety, and operational excellence across the energy sector.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
