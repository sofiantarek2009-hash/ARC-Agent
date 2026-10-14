"use client";

import { useRef, useState } from "react";
import { HTMLMotionProps, motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { cn } from "@/lib/utils";

interface MagneticCardProps extends HTMLMotionProps<"div"> {
  children: React.ReactNode;
}

export function MagneticCard({ children, className, ...props }: MagneticCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  
  // Motion values for tracking cursor relative to center of card
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  // Emil Kowalski spring physics for that heavy, engineered feel
  const springConfig = { stiffness: 350, damping: 30, mass: 0.8 };
  const mouseXSpring = useSpring(x, springConfig);
  const mouseYSpring = useSpring(y, springConfig);

  // Maximum rotation of 10 degrees
  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["8deg", "-8deg"]);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-8deg", "8deg"]);
  
  // Glare effect positioning
  const glareX = useTransform(mouseXSpring, [-0.5, 0.5], ["100%", "0%"]);
  const glareY = useTransform(mouseYSpring, [-0.5, 0.5], ["100%", "0%"]);

  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    
    const rect = ref.current.getBoundingClientRect();
    
    // Calculate mouse position relative to card center (-0.5 to 0.5)
    const relX = (e.clientX - rect.left) / rect.width - 0.5;
    const relY = (e.clientY - rect.top) / rect.height - 0.5;
    
    x.set(relX);
    y.set(relY);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    // Return to resting position
    x.set(0);
    y.set(0);
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  return (
    <div className="perspective-1000">
      <motion.div
        ref={ref}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onMouseEnter={handleMouseEnter}
        style={{
          rotateX,
          rotateY,
          transformStyle: "preserve-3d",
        }}
        className={cn(
          "relative overflow-hidden rounded-2xl transition-shadow duration-500",
          "glass-panel border-white/5",
          isHovered ? "shadow-[0_20px_40px_rgba(0,240,255,0.1)]" : "shadow-xl",
          className
        )}
        {...props}
      >
        {/* Dynamic glare overlay */}
        <motion.div
          className="pointer-events-none absolute inset-0 z-10 transition-opacity duration-500"
          style={{
            opacity: isHovered ? 0.3 : 0,
            background: `radial-gradient(circle at center, rgba(0,240,255,0.12) 0%, rgba(0,240,255,0) 60%)`,
            left: glareX,
            top: glareY,
            transform: "translate(-50%, -50%)",
            width: "200%",
            height: "200%",
          }}
        />
        
        {/* Deep inner shadow to enhance 3D effect */}
        <div className="pointer-events-none absolute inset-0 rounded-2xl shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]" />

        {/* Content wrapper pushed into Z-space */}
        <div 
          className="relative z-0 h-full w-full"
          style={{ transform: "translateZ(30px)" }}
        >
          {children}
        </div>
      </motion.div>
    </div>
  );
}
