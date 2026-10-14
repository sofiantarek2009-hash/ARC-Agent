"use client";

import { useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const panelsData = [
  {
    id: "01",
    title: "Technology & Simulation",
    image: "/assets/we-rds-hero-1.jpg",
    list: ["RDS-100®", "WEXPLORE® VR", "Digital Solutions", "CAP"],
    link: "/services/simulation",
  },
  {
    id: "02",
    title: "Engineering & Inspection",
    image: "/assets/aes-rope-access-hud.jpg",
    list: ["API 4G Cat III", "IRATA", "UWILD", "NDT"],
    link: "/services/inspection",
  },
  {
    id: "03",
    title: "Training & Competency",
    image: "/assets/training-center-iso.jpg",
    list: ["IWCF Level 3/4", "IADC WellSharp"],
    link: "/services/training",
  },
  {
    id: "04",
    title: "Industrial Chemicals",
    image: "/assets/aes-chemicals.jpg",
    list: ["Demulsifiers", "Corrosion Inhibitors", "Biocides"],
    link: "/services/chemicals",
  },
];

export function HorizontalCapabilities() {
  const containerRef = useRef<HTMLDivElement>(null);

  // Smooth entry animation for the cards
  useGSAP(() => {
    const cards = gsap.utils.toArray(".capability-card");
    
    cards.forEach((card: any) => {
      gsap.fromTo(
        card,
        { scale: 0.95, opacity: 0, y: 50 },
        {
          scale: 1,
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: "power3.out",
          scrollTrigger: {
            trigger: card,
            start: "top 85%",
            toggleActions: "play none none reverse",
          },
        }
      );
    });
  }, { scope: containerRef });

  return (
    <section
      ref={containerRef}
      className="relative w-full bg-void px-6 py-24 md:px-12"
    >
      <div className="mb-16 md:mb-24 max-w-4xl">
        <h2 className="font-display text-4xl md:text-5xl lg:text-7xl font-bold text-white mb-6">
          Core Capabilities
        </h2>
        <p className="text-white/60 text-lg md:text-xl font-sans max-w-2xl">
          Integrated solutions covering the full spectrum of energy operations, from digital twin simulation to heavy physical inspection.
        </p>
      </div>

      <div className="relative flex flex-col gap-8 pb-32">
        {panelsData.map((panel, index) => {
          // Calculate the sticky top offset so they stack like a deck of cards
          // Dynamic offset: smaller offset on mobile
          const isMobile = typeof window !== 'undefined' ? window.innerWidth < 768 : false;
          const baseOffset = isMobile ? 80 : 100;
          const stepOffset = isMobile ? 15 : 30;
          const stickyTopOffset = `calc(${baseOffset}px + ${index * stepOffset}px)`;

          return (
            <div
              key={panel.id}
              className="capability-card sticky flex flex-col md:flex-row min-h-[500px] w-full rounded-3xl overflow-hidden border border-white/[0.08] bg-[#0B0C0E] shadow-2xl"
              style={{ top: stickyTopOffset, zIndex: index + 10 }}
            >
              {/* Content Half */}
              <div className="relative z-10 flex w-full md:w-1/2 flex-col justify-center p-8 md:p-16 lg:p-20 bg-gradient-to-br from-[#12141A] to-[#08090B]">
                <span className="mb-6 block font-mono text-sm tracking-widest uppercase text-hazard-orange">
                  {panel.id} / 04
                </span>
                
                <h3 className="mb-8 font-display text-3xl font-bold text-white md:text-4xl lg:text-5xl leading-tight">
                  {panel.title}
                </h3>
                
                <ul className="mb-12 flex flex-wrap gap-3">
                  {panel.list.map((item, i) => (
                    <li
                      key={i}
                      className="rounded-full border border-white/10 bg-white/5 px-5 py-2.5 font-mono text-xs uppercase tracking-wider text-white/80"
                    >
                      {item}
                    </li>
                  ))}
                </ul>

                <div className="mt-auto pt-8">
                  <Link
                    href={panel.link}
                    className="group relative inline-flex items-center justify-center overflow-hidden rounded-full bg-white px-8 py-4 font-mono text-xs font-bold uppercase tracking-wider text-[#0B0C0E] transition-transform duration-300 ease-[0.23,1,0.32,1] active:scale-[0.97]"
                  >
                    <span className="relative z-10 transition-colors duration-300 group-hover:text-white">
                      Explore Segment
                    </span>
                    <div className="absolute inset-0 z-0 h-full w-0 bg-hazard-orange transition-all duration-300 ease-[0.23,1,0.32,1] group-hover:w-full"></div>
                  </Link>
                </div>
              </div>

              {/* Image Half */}
              <div className="relative h-[300px] w-full md:h-auto md:w-1/2 overflow-hidden border-l border-white/[0.05]">
                <Image
                  src={panel.image}
                  alt={panel.title}
                  fill
                  sizes="(max-width: 768px) 100vw, 50vw"
                  className="object-cover transition-transform duration-1000 ease-out hover:scale-105"
                  priority={index === 0}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#0B0C0E]/80 to-transparent md:bg-gradient-to-l md:from-transparent md:to-[#0B0C0E]/80"></div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
