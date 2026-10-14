"use client";
import { motion } from "framer-motion";
import Image from "next/image";
const IMAGES = [
  {
    src: "/assets/event-networking-1.jpg",
    alt: "Industry experts networking at a global geotechnology conference",
    className: "aspect-[4/3]",
  },
  {
    src: "/assets/haliburton-team.jpg",
    alt: "Haliburton Logging Services Indonesia Rig Inspection Team",
    className: "aspect-[4/3]",
  },
  {
    src: "/assets/event-networking-2.jpg",
    alt: "Executive handshake at corporate exhibition booth",
    className: "aspect-square",
  },
  {
    src: "/assets/event-networking-3.jpg",
    alt: "Technical discussion and consultation between specialists",
    className: "aspect-[4/3]",
  },
  {
    src: "/assets/haliburton-booth.jpg",
    alt: "Haliburton exhibition booth showcasing innovative technologies",
    className: "aspect-[3/4]",
  },
  {
    src: "/assets/conference-interaction.jpg",
    alt: "Engagement at international geothermal convention",
    className: "aspect-[4/3]",
  },
  {
    src: "/assets/conference-arab.jpg",
    alt: "Strategic MENA region partnerships and technical consulting",
    className: "aspect-[4/3]",
  },
  {
    src: "/assets/desert-rig-team.jpg",
    alt: "On-site drilling operations team at SINO-45 rig",
    className: "aspect-[3/4]",
  },
  {
    src: "/assets/iwcf-instructor.jpg",
    alt: "Well Experts instructor leading an IWCF Well Control course",
    className: "aspect-[4/3] lg:col-span-2 lg:row-span-2",
  },
  {
    src: "/assets/aes-mpi-inspection.jpg",
    alt: "AES technician performing Magnetic Particle Inspection (MPI) under UV light",
    className: "aspect-square",
  },
  {
    src: "/assets/aes-rig-inspection.jpg",
    alt: "AES engineer conducting a structural rig inspection",
    className: "aspect-[4/3]",
  },
  {
    src: "/assets/neon-icons.jpg",
    alt: "Digital transformation telemetry and instrumentation icons",
    className: "aspect-square",
  },
  {
    src: "/assets/abstract-wave.jpg",
    alt: "Computational fluid dynamics simulation of chemical matrix",
    className: "aspect-[4/3]",
  },
  {
    src: "/assets/aes-chemicals.jpg",
    alt: "AES chemical engineering fluid sampling in the field",
    className: "aspect-[3/4]",
  },
  {
    src: "/assets/we-rds-hero-2.jpg",
    alt: "RDS-100 cyber simulator digital twin training environment",
    className: "aspect-square lg:col-span-2 lg:row-span-2",
  },
];

export function MasonryGrid() {
  return (
    <section className="relative z-10 px-6 max-w-7xl mx-auto pb-32">
      <motion.div
        initial={{ opacity: 0, y: 15, scale: 0.98 }}
        whileInView={{ opacity: 1, y: 0, scale: 1 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
        className="mb-12"
      >
        <span className="text-xs font-mono uppercase tracking-widest text-hazard-orange block mb-3">
          Global Reach
        </span>
        <h2 className="text-3xl md:text-4xl font-bold">Conference & Exhibitions</h2>
        <p className="text-white/50 mt-2 max-w-lg tracking-tight">Showcasing our digital twin and competency solutions at premier industry events.</p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {IMAGES.map((img, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 15, scale: 0.98 }}
            whileInView={{ opacity: 1, y: 0, scale: 1 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1], delay: i * 0.05 }}
            className={`relative rounded-2xl overflow-hidden group cursor-pointer ${img.className}`}
          >
            <Image
              src={img.src}
              alt={img.alt}
              fill
              className="object-cover transition-transform duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] group-hover:scale-105"
              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
            />
            {/* Cinematic Gradient Hover */}
            <div className="absolute inset-0 bg-gradient-to-t from-petroleum-900 via-petroleum-900/20 to-transparent opacity-60 transition-opacity duration-500 group-hover:opacity-80" />
            
            <div className="absolute inset-x-0 bottom-0 p-8 flex flex-col justify-end translate-y-4 group-hover:translate-y-0 transition-transform duration-500">
              <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-500 delay-100 text-[10px] font-mono tracking-widest text-hazard-orange uppercase mb-2">
                Event Log
              </span>
              <p className="text-white/90 text-sm font-medium tracking-wide leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity duration-500 delay-150">
                {img.alt}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
