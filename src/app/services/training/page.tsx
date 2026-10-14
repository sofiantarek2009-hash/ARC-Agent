"use client";

import React, { useEffect, useRef } from "react";
import Image from "next/image";
import gsap from "gsap";
import ScrollTrigger from "gsap/ScrollTrigger";
import { EliteNavbar } from "@/components/ui/EliteNavbar";
import { IndustrialFooter } from "@/components/ui/IndustrialFooter";

gsap.registerPlugin(ScrollTrigger);

const schedules = [
  {
    course: "IWCF Level 3/4",
    description: "Advanced well control certification for drilling operators and supervisors.",
    modules: [
      { day: "Day 1", topic: "Equipment Principles", duration: "8 Hours" },
      { day: "Day 2", topic: "Well Control Procedures", duration: "8 Hours" },
      { day: "Day 3", topic: "Simulator Assessment", duration: "4 Hours" },
    ]
  },
  {
    course: "IADC WellSharp",
    description: "Comprehensive training focused on critical kick detection and well control.",
    modules: [
      { day: "Day 1", topic: "Drilling Operations & Risk", duration: "8 Hours" },
      { day: "Day 2", topic: "Kick Detection & Mitigation", duration: "8 Hours" },
      { day: "Day 3", topic: "Certification Exam", duration: "4 Hours" },
    ]
  }
];

export default function TrainingPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const heroRef = useRef<HTMLDivElement>(null);
  const heroImageRef = useRef<HTMLImageElement>(null);
  const mediaRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) return;

    let ctx = gsap.context(() => {
      // Hero Parallax
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

      // Schedule Table Stagger
      gsap.from(".schedule-row", {
        x: -40,
        opacity: 0,
        stagger: 0.1,
        duration: 0.8,
        ease: "power3.out",
        scrollTrigger: {
          trigger: ".schedule-container",
          start: "top 80%",
        },
      });

      // Clip-path reveals for media
      mediaRefs.current.forEach((el) => {
        if (!el) return;
        gsap.fromTo(el, 
          { clipPath: "inset(100% 0% 0% 0%)" },
          { 
            clipPath: "inset(0% 0% 0% 0%)",
            duration: 1.5,
            ease: "cubic-bezier(0.23, 1, 0.32, 1)",
            scrollTrigger: {
              trigger: el,
              start: "top 85%",
            }
          }
        );
        
        // Image scale effect
        const img = el.querySelector("img");
        if (img) {
          gsap.fromTo(img,
            { scale: 1.2 },
            {
              scale: 1,
              duration: 1.5,
              ease: "power2.out",
              scrollTrigger: {
                trigger: el,
                start: "top 85%",
              }
            }
          );
        }
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
        className="relative h-[70vh] w-full flex items-center justify-center overflow-hidden"
      >
        <div className="absolute inset-0 z-0 h-[120%] -top-[10%] w-full">
          <Image
            ref={heroImageRef}
            src="/assets/training-center-iso.jpg"
            alt="Training Center"
            fill
            className="object-cover opacity-50"
            sizes="100vw"
            priority
          />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#0A0E17]/60 to-[#0A0E17] z-10" />
        
        <div className="relative z-20 text-center px-4 max-w-4xl mx-auto mt-16">
          <h1 className="text-5xl md:text-7xl font-bold font-display tracking-tight text-white drop-shadow-lg">
            Training & <span className="text-[#FF5722]">Competency</span>
          </h1>
          <p className="mt-6 text-xl text-gray-300 font-sans max-w-2xl mx-auto">
            Empowering the next generation of industry professionals through hands-on learning and rigorous simulation.
          </p>
        </div>
      </section>

      {/* Schedules Section */}
      <section className="py-24 px-6 md:px-12 max-w-7xl mx-auto schedule-container relative z-20">
        <div className="mb-16 text-center">
          <h2 className="text-3xl md:text-5xl font-display font-semibold mb-4">Course Schedules</h2>
          <div className="w-24 h-1 bg-[#00F2FE] mx-auto rounded-full" />
        </div>

        <div className="flex flex-col gap-16">
          {schedules.map((program, idx) => (
            <div key={idx} className="glass-panel rounded-2xl p-8 md:p-12 border border-white/10">
              <div className="mb-8">
                <h3 className="text-3xl font-display font-bold text-white mb-2">{program.course}</h3>
                <p className="text-gray-400">{program.description}</p>
              </div>

              <div className="w-full overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-white/20">
                      <th className="py-4 px-6 text-sm font-mono tracking-widest text-[#00F2FE] uppercase w-1/4">Timeline</th>
                      <th className="py-4 px-6 text-sm font-mono tracking-widest text-[#00F2FE] uppercase w-1/2">Module Topic</th>
                      <th className="py-4 px-6 text-sm font-mono tracking-widest text-[#00F2FE] uppercase w-1/4">Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {program.modules.map((mod, i) => (
                      <tr 
                        key={i} 
                        className="schedule-row border-b border-white/5 hover:bg-white/5 transition-colors duration-300"
                      >
                        <td className="py-5 px-6 font-semibold text-white">{mod.day}</td>
                        <td className="py-5 px-6 text-gray-300">{mod.topic}</td>
                        <td className="py-5 px-6 text-gray-400 font-mono">{mod.duration}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-8">
                <button className="bg-[#FF5722] hover:bg-[#E64A19] text-white px-8 py-3 rounded-full font-bold font-sans tracking-wide transition-colors active:scale-[0.97]">
                  Enroll Now
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Media Reveals Section */}
      <section className="py-24 px-6 md:px-12 bg-black/40 border-t border-white/5 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="mb-20">
            <h2 className="text-3xl md:text-5xl font-display font-semibold mb-4">Hands-On Environment</h2>
            <p className="text-gray-400 max-w-2xl">Experience practical, instructor-led training using real-world scenarios.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center mb-24">
            <div 
              className="relative h-[400px] md:h-[500px] rounded-2xl overflow-hidden w-full"
              ref={el => { mediaRefs.current[0] = el; }}
            >
              <Image 
                src="/assets/iwcf-instructor.jpg" 
                alt="IWCF Instructor"
                fill
                className="object-cover"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </div>
            <div className="space-y-6">
              <h3 className="text-3xl font-display font-bold text-white">Expert Guidance</h3>
              <p className="text-gray-400 text-lg leading-relaxed">
                Learn directly from industry veterans who bring decades of field experience to the classroom. Our instructors provide personalized feedback and intensive simulator sessions to build true operational competency.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center mb-24">
            <div className="space-y-6 order-2 md:order-1">
              <h3 className="text-3xl font-display font-bold text-white">Comprehensive Materials</h3>
              <p className="text-gray-400 text-lg leading-relaxed">
                Our curriculum is backed by up-to-date documentation and physical study aids, ensuring you have the theoretical foundation required for complex problem-solving in the field.
              </p>
            </div>
            <div 
              className="relative h-[350px] md:h-[450px] rounded-2xl overflow-hidden w-full order-1 md:order-2"
              ref={el => { mediaRefs.current[1] = el; }}
            >
              <Image 
                src="/assets/book-photo.jpg" 
                alt="Study Materials"
                fill
                className="object-cover"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
            </div>
          </div>

          <div className="relative h-[500px] md:h-[600px] rounded-2xl overflow-hidden w-full"
            ref={el => { mediaRefs.current[2] = el; }}
          >
            <Image 
              src="/assets/conference-interaction.jpg" 
              alt="Conference Interaction"
              fill
              className="object-cover"
              sizes="100vw"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />
            <div className="absolute bottom-10 left-10 md:bottom-16 md:left-16 z-20 max-w-2xl">
              <h3 className="text-3xl font-display font-bold text-white mb-4">Interactive Learning</h3>
              <p className="text-gray-300 text-lg">
                Engage with peers and instructors in a collaborative environment designed to simulate the pressures and dynamics of actual rig operations.
              </p>
            </div>
          </div>
        </div>
      </section>

      <IndustrialFooter />
    </main>
  );
}
