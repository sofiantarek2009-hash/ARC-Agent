"use client";

import React, { useEffect, useRef, Suspense } from "react";
import Image from "next/image";
import gsap from "gsap";
import ScrollTrigger from "gsap/ScrollTrigger";
import { EliteNavbar } from "@/components/ui/EliteNavbar";
import { IndustrialFooter } from "@/components/ui/IndustrialFooter";
import { ServiceDiorama } from "@/components/3d/ServiceDiorama";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, Environment, OrbitControls, Center, ContactShadows } from "@react-three/drei";
import * as THREE from "three";

gsap.registerPlugin(ScrollTrigger);

export default function InspectionPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const heroRef = useRef<HTMLDivElement>(null);
  const heroImageRef = useRef<HTMLImageElement>(null);
  const bentoRef = useRef<HTMLDivElement>(null);
  const galleryRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    let ctx = gsap.context(() => {
      // Parallax Hero
      gsap.to(heroImageRef.current, {
        y: "20%",
        ease: "none",
        scrollTrigger: {
          trigger: heroRef.current,
          start: "top top",
          end: "bottom top",
          scrub: true,
        },
      });

      // Bento Grid Entrance
      gsap.from(".bento-item", {
        y: 60,
        opacity: 0,
        duration: 1.2,
        stagger: 0.1,
        ease: "cubic-bezier(0.23, 1, 0.32, 1)",
        scrollTrigger: {
          trigger: bentoRef.current,
          start: "top 85%",
        },
      });

      // Gallery Stagger
      gsap.from(".gallery-item", {
        y: 80,
        opacity: 0,
        scale: 0.95,
        duration: 1.4,
        stagger: 0.15,
        ease: "cubic-bezier(0.23, 1, 0.32, 1)",
        scrollTrigger: {
          trigger: galleryRef.current,
          start: "top 80%",
        },
      });
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <main className="min-h-screen bg-[#0A0E17] text-white overflow-hidden" ref={containerRef}>
      <EliteNavbar />

      {/* Hero Section */}
      <section 
        ref={heroRef}
        className="relative h-[80vh] w-full flex items-center justify-center overflow-hidden"
      >
        <div className="absolute inset-0 z-0 h-[120%] -top-[10%] w-full">
          <Image
            ref={heroImageRef}
            src="/assets/aes-rig-inspection.jpg"
            alt="Rig Inspection"
            fill
            className="object-cover opacity-60"
            sizes="100vw"
            priority
          />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#0A0E17]/50 to-[#0A0E17] z-10" />
        
        <div className="relative z-20 text-center px-4 max-w-5xl mx-auto mt-20">
          <h1 className="text-5xl md:text-7xl font-bold font-display tracking-tight text-white drop-shadow-2xl">
            Engineering & <span className="text-[#00F2FE]">Inspection</span>
          </h1>
          <p className="mt-6 text-xl text-gray-300 font-sans max-w-2xl mx-auto">
            Ensuring operational excellence and structural integrity in the most demanding environments.
          </p>
        </div>
      </section>

      {/* Bento Grid Content */}
      <section className="py-24 px-6 md:px-12 max-w-7xl mx-auto relative z-20" ref={bentoRef}>
        <div className="mb-16 text-center">
          <h2 className="text-3xl md:text-5xl font-display font-semibold mb-4">Core Capabilities</h2>
          <div className="w-24 h-1 bg-[#FF5722] mx-auto rounded-full" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 auto-rows-[320px]">
          {/* Item 1: API 4G Cat III */}
          <div className="bento-item glass-panel relative overflow-hidden rounded-2xl md:col-span-2 group flex flex-col justify-end p-8">
            <Image src="/assets/aes-rig-inspection.jpg" alt="API 4G Cat III" fill className="object-cover transition-transform duration-700 group-hover:scale-105" sizes="(max-width: 768px) 100vw, 66vw" />
            <div className="absolute inset-0 bg-gradient-to-t from-[#0A0E17] via-[#0A0E17]/40 to-transparent z-10" />
            <div className="absolute inset-0 bg-[#00F2FE]/5 group-hover:bg-[#00F2FE]/10 transition-colors duration-500 z-10" />
            <div className="relative z-20">
              <h3 className="text-3xl font-display font-bold text-white mb-2">API 4G Cat III</h3>
              <p className="text-gray-300 font-sans max-w-md">Comprehensive drilling structure inspections compliant with API specifications, minimizing downtime and extending asset lifecycle.</p>
            </div>
          </div>

          {/* Item 2: IRATA Rope Access */}
          <div className="bento-item relative overflow-hidden rounded-2xl group">
            <Image 
              src="/assets/aes-rope-access-hud.jpg" 
              alt="IRATA Rope Access" 
              fill 
              className="object-cover transition-transform duration-700 group-hover:scale-105"
              sizes="(max-width: 768px) 100vw, 33vw"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#0A0E17] via-[#0A0E17]/40 to-transparent z-10" />
            <div className="absolute bottom-0 left-0 p-8 z-20">
              <h3 className="text-2xl font-display font-bold text-white mb-2">IRATA Rope Access</h3>
              <p className="text-sm text-gray-300">Rapid deployment for high-altitude non-destructive testing.</p>
            </div>
          </div>

          {/* Item 3: UWILD & Marine */}
          <div className="bento-item relative overflow-hidden rounded-2xl group md:col-span-1">
            <Image 
              src="/assets/aes-underwater-ndt.jpg" 
              alt="UWILD & Marine" 
              fill 
              className="object-cover transition-transform duration-700 group-hover:scale-105"
              sizes="(max-width: 768px) 100vw, 33vw"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#0A0E17] via-[#0A0E17]/40 to-transparent z-10" />
            <div className="absolute bottom-0 left-0 p-8 z-20">
              <h3 className="text-2xl font-display font-bold text-white mb-2">UWILD & Marine</h3>
              <p className="text-sm text-gray-300">Underwater inspections in lieu of dry-docking for structural assessment.</p>
            </div>
          </div>

          {/* Item 4: NDT Services */}
          <div className="bento-item relative overflow-hidden rounded-2xl group md:col-span-2">
            <Image 
              src="/assets/aes-mpi-inspection.jpg" 
              alt="NDT Services" 
              fill 
              className="object-cover transition-transform duration-700 group-hover:scale-105"
              sizes="(max-width: 768px) 100vw, 66vw"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#0A0E17] via-[#0A0E17]/40 to-transparent z-10" />
            <div className="absolute bottom-0 left-0 p-8 z-20">
              <h3 className="text-3xl font-display font-bold text-white mb-2">NDT Services</h3>
              <p className="text-gray-300 font-sans max-w-md">Advanced non-destructive testing methodologies ensuring flawless detection and reporting.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Gallery Section */}
      <section className="py-24 px-6 md:px-12 bg-[#121826]/30 border-t border-white/5" ref={galleryRef}>
        <div className="max-w-7xl mx-auto">
          <div className="mb-16">
            <h2 className="text-3xl md:text-4xl font-display font-semibold mb-4 text-[#00F2FE]">Restored Assets</h2>
            <p className="text-gray-400 max-w-2xl">A showcase of recently revitalized structural components and major lifting units.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="gallery-item glass-panel p-2 rounded-xl group">
              <div className="relative h-64 md:h-80 w-full overflow-hidden rounded-lg">
                <Image 
                  src="/assets/aes-rope-access-hud.jpg" 
                  alt="Crane Lift" 
                  fill 
                  className="object-cover group-hover:scale-110 transition-transform duration-700 ease-in-out"
                  sizes="(max-width: 768px) 100vw, 33vw"
                />
              </div>
              <p className="p-4 text-center font-mono text-sm tracking-wide text-gray-300 uppercase">Crane Lift Operation</p>
            </div>

            <div className="gallery-item glass-panel p-2 rounded-xl group md:translate-y-12">
              <div className="relative h-64 md:h-80 w-full overflow-hidden rounded-lg">
                <Image 
                  src="/assets/aes-rig-inspection.jpg" 
                  alt="Drilling Rig" 
                  fill 
                  className="object-cover group-hover:scale-110 transition-transform duration-700 ease-in-out"
                  sizes="(max-width: 768px) 100vw, 33vw"
                />
              </div>
              <p className="p-4 text-center font-mono text-sm tracking-wide text-gray-300 uppercase">Drilling Rig Inspection</p>
            </div>

            <div className="gallery-item glass-panel p-2 rounded-xl group">
              <div className="relative h-64 md:h-80 w-full overflow-hidden rounded-lg">
                <Image 
                  src="/assets/underwater-inspection.jpg" 
                  alt="Offshore Hull" 
                  fill 
                  className="object-cover group-hover:scale-110 transition-transform duration-700 ease-in-out"
                  sizes="(max-width: 768px) 100vw, 33vw"
                />
              </div>
              <p className="p-4 text-center font-mono text-sm tracking-wide text-gray-300 uppercase">Offshore Hull Integrity</p>
            </div>
          </div>
        </div>
      </section>

      <ServiceDiorama type="inspection" />
<IndustrialFooter />
    
      

</main>
  );
}
