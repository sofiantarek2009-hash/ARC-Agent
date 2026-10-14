"use client";

import { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { usePathname } from "next/navigation";

export function CustomCursor() {
  const cursorRef = useRef<HTMLDivElement>(null);
  const followerRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const [isTouch, setIsTouch] = useState(true);
  const pathname = usePathname();

  useEffect(() => {
    // Detect touch devices
    setIsTouch("ontouchstart" in window || navigator.maxTouchPoints > 0);
  }, []);

  useEffect(() => {
    if (isTouch || !cursorRef.current || !followerRef.current) return;

    // Use GSAP quickTo for zero-latency tracking
    const cursor = {
      x: gsap.quickTo(cursorRef.current, "x", { duration: 0.1, ease: "power3" }),
      y: gsap.quickTo(cursorRef.current, "y", { duration: 0.1, ease: "power3" }),
    };

    const follower = {
      x: gsap.quickTo(followerRef.current, "x", { duration: 0.5, ease: "power3" }),
      y: gsap.quickTo(followerRef.current, "y", { duration: 0.5, ease: "power3" }),
    };

    const onMouseMove = (e: MouseEvent) => {
      cursor.x(e.clientX);
      cursor.y(e.clientY);
      follower.x(e.clientX);
      follower.y(e.clientY);
    };

    window.addEventListener("mousemove", onMouseMove);

    // Handle hover states on links and buttons
    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const isClickable = target.closest("a, button, input, .magnetic");
      const isDraggable = target.closest(".filmstrip-track");

      if (isDraggable) {
        gsap.to(followerRef.current, { scale: 3, backgroundColor: "rgba(0, 242, 254, 0.1)", border: "1px solid rgba(0, 242, 254, 0.5)", duration: 0.3 });
        if (textRef.current) textRef.current.innerText = "DRAG";
      } else if (isClickable) {
        gsap.to(followerRef.current, { scale: 1.5, backgroundColor: "rgba(255, 255, 255, 0.1)", duration: 0.3 });
        if (textRef.current) textRef.current.innerText = "";
      } else {
        gsap.to(followerRef.current, { scale: 1, backgroundColor: "transparent", border: "1px solid rgba(255, 255, 255, 0.2)", duration: 0.3 });
        if (textRef.current) textRef.current.innerText = "";
      }
    };

    window.addEventListener("mouseover", handleMouseOver);

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseover", handleMouseOver);
    };
  }, [isTouch, pathname]);

  if (isTouch) return null;

  return (
    <>
      {/* Small dot */}
      <div
        ref={cursorRef}
        className="fixed top-0 left-0 w-2 h-2 bg-neon-cyan rounded-full pointer-events-none z-[100] -translate-x-1/2 -translate-y-1/2 mix-blend-difference hidden md:block"
      />
      {/* Follower ring */}
      <div
        ref={followerRef}
        className="fixed top-0 left-0 w-8 h-8 border border-white/20 rounded-full pointer-events-none z-[99] -translate-x-1/2 -translate-y-1/2 flex items-center justify-center transition-colors hidden md:flex"
      >
        <span ref={textRef} className="text-[6px] font-mono font-bold text-neon-cyan tracking-widest" />
      </div>
    </>
  );
}
