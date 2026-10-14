"use client";

import React, { useRef } from "react";
import Image from "next/image";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { EliteNavbar } from "@/components/ui/EliteNavbar";
import { IndustrialFooter } from "@/components/ui/IndustrialFooter";
import { ServiceDiorama } from "@/components/3d/ServiceDiorama";


gsap.registerPlugin(ScrollTrigger);

export default function AboutPage() {
  const container = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    // Split-screen Parallax
    const leftPanel = container.current?.querySelector(".left-panel");
    const rightPanel = container.current?.querySelector(".right-panel");

    if (leftPanel && rightPanel) {
      gsap.to(leftPanel, {
        yPercent: 15,
        ease: "none",
        scrollTrigger: {
          trigger: container.current,
          start: "top top",
          end: "bottom top",
          scrub: true,
        },
      });

      gsap.to(rightPanel, {
        yPercent: -15,
        ease: "none",
        scrollTrigger: {
          trigger: container.current,
          start: "top top",
          end: "bottom top",
          scrub: true,
        },
      });
    }

    // Milestones staggered reveal
    const milestones = gsap.utils.toArray(".milestone-card");
    if (milestones.length > 0 && timelineRef.current) {
      gsap.fromTo(
        milestones,
        { y: 50, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.8,
          stagger: 0.2,
          ease: "power3.out",
          scrollTrigger: {
            trigger: timelineRef.current,
            start: "top 80%",
            end: "bottom 20%",
            toggleActions: "play none none reverse",
          },
        }
      );
    }
  }, { scope: container });

  return (
    <div className="min-h-screen bg-slate-950 text-white selection:bg-cyan-500/30 selection:text-cyan-50 font-sans" ref={container}>
      <EliteNavbar />
      
      <main className="pt-24 pb-16 overflow-hidden">
        {/* Split Screen Hero */}
        <section className="relative h-[80vh] min-h-[600px] flex flex-col md:flex-row overflow-hidden border-b border-white/10">
          
          <div className="left-panel relative w-full md:w-1/2 h-1/2 md:h-full">
            <div className="absolute inset-0 bg-slate-900/60 z-10" />
            <Image
              src="/assets/team-photo.jpg"
              alt="Well Experts Team"
              fill
              className="object-cover opacity-80 mix-blend-luminosity grayscale hover:grayscale-0 transition-all duration-700"
              sizes="(max-width: 768px) 100vw, 50vw"
              priority
            />
            <div className="absolute inset-0 z-20 flex flex-col justify-end p-8 md:p-16">
              <h2 className="text-3xl md:text-5xl font-display font-bold tracking-tight text-white mb-4">
                Well Experts
              </h2>
              <p className="text-slate-300 max-w-md font-mono text-sm leading-relaxed">
                Technical Consultation & Advanced Simulation.
                Forging the future of drilling operations through high-fidelity training and engineering solutions.
              </p>
            </div>
          </div>
          
          <div className="right-panel relative w-full md:w-1/2 h-1/2 md:h-full mt-[-10%] md:mt-0">
            <div className="absolute inset-0 bg-orange-950/40 z-10 mix-blend-multiply" />
            <Image
              src="/assets/office-signing.jpg"
              alt="AES Office Signing"
              fill
              className="object-cover opacity-80 mix-blend-luminosity hover:mix-blend-normal transition-all duration-700"
              sizes="(max-width: 768px) 100vw, 50vw"
              priority
            />
            <div className="absolute inset-0 z-20 flex flex-col justify-end p-8 md:p-16">
              <h2 className="text-3xl md:text-5xl font-display font-bold tracking-tight text-white mb-4">
                Adriatic Engineering
              </h2>
              <p className="text-slate-300 max-w-md font-mono text-sm leading-relaxed">
                Integrated Inspection & Rope Access.
                Delivering robust structural integrity assessments and industrial safety in the most demanding environments.
              </p>
            </div>
          </div>
          
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-30 hidden md:flex items-center justify-center w-24 h-24 rounded-full glass-panel-heavy border border-white/20 backdrop-blur-md">
            <span className="text-neon-cyan font-display text-2xl font-bold">+</span>
          </div>
        </section>

        {/* Narrative Section */}
        <section className="py-24 px-6 md:px-12 max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <h3 className="text-sm font-mono text-hazard-orange tracking-widest uppercase mb-4 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-hazard-orange"></span>
                Our Legacy
              </h3>
              <h2 className="text-4xl md:text-5xl font-display font-medium text-white mb-8 leading-tight">
                Two Brands. <br/>
                <span className="text-slate-400">One Uncompromising Standard.</span>
              </h2>
            </div>
            <div>
              <p className="text-lg text-slate-300 leading-relaxed mb-6 font-sans">
                Formed through the strategic integration of Well Experts (est. 2022) and Adriatic Engineering Solutions (est. 2016), NorthStar Energy Solutions provides an uncompromising standard of industrial capability. While Well Experts drives digital transformation through advanced RDS-100 simulation and data analytics, Adriatic executes critical Rig Audits, NDT inspections, and physical asset integrity programs on the ground.
              </p>
              <p className="text-slate-400 leading-relaxed font-sans">
                Together, we provide an end-to-end ecosystem for energy and industrial sectors—bridging the gap between predictive modeling and physical asset integrity.
              </p>
            </div>
          </div>
        </section>

        {/* Timeline / Milestones */}
        <section ref={timelineRef} className="py-24 bg-slate-900/50 border-y border-white/5 relative">
          {/* subtle grid background */}
          <div className="absolute inset-0 bg-[url('/assets/grid.svg')] opacity-[0.03] z-0 pointer-events-none" />
          
          <div className="max-w-7xl mx-auto px-6 md:px-12 relative z-10">
            <h3 className="text-3xl font-display font-medium text-center mb-16">
              Milestones of Excellence
            </h3>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
              {[
                {
                  year: "2016",
                  title: "Adriatic Formation",
                  desc: "Adriatic Engineering Solutions established, rapidly becoming a market leader in Asset Integrity Management, NDT, and lifting services."
                },
                {
                  year: "2022",
                  title: "Well Experts Launch",
                  desc: "Well Experts founded to pioneer drilling simulation, VR training, and digital workflow automation in the MENA region."
                },
                {
                  year: "2024",
                  title: "Oman Expansion",
                  desc: "Successful deployment of the state-of-the-art RDS-100 Advanced Drilling Simulator at Ryboa Hima, expanding our footprint to Oman."
                },
                {
                  year: "2025",
                  title: "Global Standards",
                  desc: "Achieved IWCF Primary Accredited Training status and officially became an IADC Member and IADC WellSharp® provider."
                }
              ].map((milestone, idx) => (
                <div key={idx} className="milestone-card glass-panel p-8 rounded-xl border border-white/10 hover:border-neon-cyan/30 transition-colors duration-300">
                  <div className="text-neon-cyan font-mono text-xl mb-4">{milestone.year}</div>
                  <h4 className="text-2xl font-display font-medium mb-3 text-white">{milestone.title}</h4>
                  <p className="text-slate-400 text-sm leading-relaxed">{milestone.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      
      

<ServiceDiorama type="about" />
<IndustrialFooter />
    </div>
  );
}
