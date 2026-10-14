"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(ScrollTrigger);

const FOOTER_LINKS = {
  services: [
    { name: "Technology & Simulation", href: "/services/simulation" },
    { name: "Engineering & Inspection", href: "/services/inspection" },
    { name: "Training & Competency", href: "/services/training" },
    { name: "Industrial Chemicals", href: "/services/chemicals" },
  ],
  compliance: [
    { name: "IWCF Primary Accredited", href: "#" },
    { name: "IADC WellSharp® Provider", href: "#" },
    { name: "API 7K / 4G / RP 2D", href: "#" },
    { name: "ISO 9001:2015 Certified", href: "#" },
    { name: "ABS / DNV / LEEA", href: "#" },
  ],
  company: [
    { name: "About", href: "/about" },
    { name: "Careers", href: "/about#careers" },
    { name: "Contact", href: "/#contact" },
    { name: "Privacy Policy", href: "/privacy" },
  ],
};

export function IndustrialFooter() {
  const footerRef = useRef<HTMLElement>(null);
  const wordmarkRef = useRef<HTMLDivElement>(null);

  // Scroll-triggered wordmark reveal
  useGSAP(() => {
    if (!wordmarkRef.current) return;
    gsap.fromTo(
      wordmarkRef.current,
      { opacity: 0, scale: 0.95 },
      {
        opacity: 1,
        scale: 1,
        duration: 1.2,
        ease: "power3.out",
        scrollTrigger: {
          trigger: wordmarkRef.current,
          start: "top 90%",
          toggleActions: "play none none reverse",
        },
      }
    );
  }, { scope: footerRef });

  return (
    <footer ref={footerRef} className="relative bg-[#08090B] border-t border-white/[0.06]">
      {/* ─── MAIN GRID ─── */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 lg:gap-8 px-6 lg:px-12 py-20">
        {/* Col 1: Brand + Email CTA */}
        <div className="lg:col-span-1 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-6">
              <span className="font-display font-bold text-white text-2xl tracking-tight">NORTHSTAR</span>
            </div>
            <p className="text-white/70 text-base leading-relaxed mb-8 max-w-[280px]">
              Integrated Technical Consultation, Simulation Manufacturing, & Asset Integrity Solutions for the Global Energy Sector.
            </p>
          </div>
          <form
            onSubmit={(e) => e.preventDefault()}
            className="flex flex-col sm:flex-row gap-3 w-full max-w-sm"
          >
            <input
              type="email"
              placeholder="Work Email"
              className="flex-1 bg-transparent border-b border-white/20 text-white px-2 py-2 font-mono text-sm tracking-wider placeholder:text-white/40 focus:border-white focus:outline-none transition-colors"
            />
            <button
              type="submit"
              className="px-6 py-2 border border-white text-white font-mono font-bold text-xs tracking-widest uppercase hover:bg-white hover:text-black transition-colors active:scale-[0.97]"
            >
              Contact
            </button>
          </form>
        </div>

        {/* Col 2: Services */}
        <div>
          <h4 className="font-display text-lg font-semibold text-white mb-6 uppercase tracking-wider">
            Services
          </h4>
          <ul className="space-y-4">
            {FOOTER_LINKS.services.map((link) => (
              <li key={link.name}>
                <a
                  href={link.href}
                  className="text-white/70 text-base hover:text-white transition-colors duration-200"
                >
                  {link.name}
                </a>
              </li>
            ))}
          </ul>
        </div>

        {/* Col 3: Compliance */}
        <div>
          <h4 className="font-display text-lg font-semibold text-white mb-6 uppercase tracking-wider">
            Compliance
          </h4>
          <ul className="space-y-4">
            {FOOTER_LINKS.compliance.map((link) => (
              <li key={link.name}>
                <span className="text-white/70 text-base">
                  {link.name}
                </span>
              </li>
            ))}
          </ul>
        </div>

        {/* Col 4: Company */}
        <div>
          <h4 className="font-display text-lg font-semibold text-white mb-6 uppercase tracking-wider">
            Company
          </h4>
          <ul className="space-y-4">
            {FOOTER_LINKS.company.map((link) => (
              <li key={link.name}>
                <a
                  href={link.href}
                  className="text-white/70 text-base hover:text-white transition-colors duration-200"
                >
                  {link.name}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ─── GIANT WORDMARK ─── */}
      <div
        ref={wordmarkRef}
        className="overflow-hidden border-t border-white/[0.04] pt-8 pb-4 px-6 flex justify-center"
      >
        <div
          className="font-display font-black text-center uppercase leading-[0.85] select-none pointer-events-none"
          style={{
            fontSize: "clamp(3rem, 15vw, 15rem)",
            letterSpacing: "-0.04em",
            color: "rgba(255,255,255,0.03)",
          }}
        >
          NORTHSTAR
        </div>
      </div>

      {/* ─── BOTTOM BAR ─── */}
      <div className="flex flex-col sm:flex-row items-center justify-between px-6 lg:px-12 py-8 border-t border-white/[0.04] gap-4">
        <span className="font-sans text-sm text-white/40 tracking-wide">
          © {new Date().getFullYear()} NorthStar Energy Solutions. All Rights Reserved.
        </span>
        <a
          href="https://www.linkedin.com/company/northstar-energy-solutions"
          target="_blank"
          rel="noopener noreferrer"
          className="font-sans text-sm text-white/40 tracking-wide hover:text-white transition-colors"
        >
          LinkedIn
        </a>
      </div>
    </footer>
  );
}
