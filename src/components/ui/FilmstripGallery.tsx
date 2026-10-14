"use client";

import React, { useRef, useState, useEffect } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Plus } from "lucide-react";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const GALLERY_ITEMS = [
  { src: "/assets/event-networking-1.jpg", title: "ADIPEC Summit", category: "Global Events", width: "w-[400px]" },
  { src: "/assets/haliburton-team.jpg", title: "Halliburton Partnership", category: "Strategic Alliance", width: "w-[600px]" },
  { src: "/assets/iwcf-instructor.jpg", title: "IWCF Certification", category: "Training & Competency", width: "w-[400px]" },
  { src: "/assets/aes-rig-inspection.jpg", title: "API 4G Cat III Audit", category: "Field Operations", width: "w-[500px]" },
  { src: "/assets/conference-arab.jpg", title: "EGYPES Exhibition", category: "Global Events", width: "w-[600px]" },
  { src: "/assets/aes-mpi-inspection.jpg", title: "MPI NDT Inspection", category: "Field Operations", width: "w-[400px]" },
  { src: "/assets/desert-rig-team.jpg", title: "On-site Deployment", category: "Field Operations", width: "w-[500px]" },
  { src: "/assets/event-networking-2.jpg", title: "Industry Networking", category: "Global Events", width: "w-[400px]" },
  { src: "/assets/haliburton-booth.jpg", title: "Technology Showcase", category: "Strategic Alliance", width: "w-[500px]" },
  { src: "/assets/conference-interaction.jpg", title: "Client Demonstration", category: "Global Events", width: "w-[600px]" },
];

export function FilmstripGallery() {
  const containerRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [selectedImage, setSelectedImage] = useState<typeof GALLERY_ITEMS[0] | null>(null);

  useEffect(() => {
    if (!trackRef.current || !containerRef.current) return;

    // Reveal Animation on Scroll
    const items = gsap.utils.toArray(".gallery-card");
    items.forEach((item: any, i) => {
      gsap.fromTo(
        item,
        { opacity: 0, y: 50 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: "power3.out",
          scrollTrigger: {
            trigger: containerRef.current,
            start: "top 75%",
          },
          delay: i * 0.1,
        }
      );
    });

    return () => {
    };
  }, []);

  const handleItemClick = (img: typeof GALLERY_ITEMS[0]) => {
    setSelectedImage(img);
  };

  return (
    <div className="relative w-full overflow-hidden py-24 bg-[#0B0C0E]" ref={containerRef}>
      <div className="max-w-[1440px] mx-auto px-6 lg:px-12 mb-16">
        <h2 className="text-4xl md:text-5xl lg:text-6xl font-display font-light text-white mb-4 tracking-tight">
          Global <span className="font-bold">Events & Field Ops.</span>
        </h2>
        <p className="font-mono text-xs text-white/40 uppercase tracking-[0.2em]">
          Exploring technical execution and partnerships globally
        </p>
      </div>

      <div 
        ref={trackRef}
        className="flex flex-nowrap items-center w-max gap-6 pb-8 animate-marquee hover:[animation-play-state:paused]"
      >
        {[...GALLERY_ITEMS, ...GALLERY_ITEMS].map((item, i) => (
          <div 
            key={i}
            className={`gallery-card relative flex-shrink-0 h-[350px] lg:h-[450px] ${item.width} overflow-hidden rounded-xl group bg-white/5 border border-white/10 transition-colors hover:border-white/30`}
            onClick={() => handleItemClick(item)}
          >
            <Image 
              src={item.src} 
              alt={item.title} 
              fill 
              sizes="(max-width: 768px) 80vw, 40vw" 
              className="object-cover transition-transform duration-[1.5s] ease-[0.23,1,0.32,1] group-hover:scale-105" 
              draggable={false}
            />
            
            {/* Gradient Overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-60 group-hover:opacity-80 transition-opacity duration-500" />
            
            {/* Content Overlay */}
            <div className="absolute inset-0 p-6 flex flex-col justify-end transform translate-y-4 group-hover:translate-y-0 transition-transform duration-500 ease-[0.23,1,0.32,1]">
              <span className="font-mono text-[10px] text-hazard-orange uppercase tracking-widest mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-500 delay-100">
                {item.category}
              </span>
              <div className="flex items-center justify-between">
                <h3 className="font-display text-xl lg:text-2xl font-semibold text-white">
                  {item.title}
                </h3>
                <div className="w-8 h-8 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center border border-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500 delay-200">
                  <Plus className="w-4 h-4 text-white" />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox */}
      <AnimatePresence>
        {selectedImage && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-8 bg-black/90 backdrop-blur-xl"
            onClick={() => setSelectedImage(null)}
          >
            <motion.div 
              initial={{ scale: 0.95, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 20 }}
              transition={{ ease: [0.23, 1, 0.32, 1], duration: 0.5 }}
              className="relative w-full h-full max-w-6xl max-h-[85vh] rounded-xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <button 
                className="absolute top-6 right-6 z-10 font-mono text-xs tracking-widest uppercase text-white bg-black/50 px-4 py-2 rounded-full border border-white/20 hover:bg-white hover:text-black transition-colors"
                onClick={() => setSelectedImage(null)}
              >
                Close ✕
              </button>
              
              <Image 
                src={selectedImage.src} 
                alt={selectedImage.title} 
                fill 
                className="object-contain" 
                sizes="100vw"
                quality={100}
                priority
              />
              
              {/* Lightbox Caption */}
              <div className="absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-black/80 to-transparent">
                <span className="font-mono text-xs text-hazard-orange uppercase tracking-widest block mb-2">
                  {selectedImage.category}
                </span>
                <h2 className="font-display text-3xl font-bold text-white">
                  {selectedImage.title}
                </h2>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
