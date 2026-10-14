"use client";

import { cn } from "@/lib/utils"; // wait, the previous import is @/lib/utils
import { useState } from "react";
import Image from "next/image";

interface MarqueeItem {
  name: string;
  domain: string;
  scope?: string;
}

interface InfiniteMarqueeProps {
  items: MarqueeItem[];
  direction?: "left" | "right";
  speed?: "fast" | "normal" | "slow";
  className?: string;
}

export function InfiniteMarquee({
  items,
  direction = "left",
  speed = "normal",
  className,
}: InfiniteMarqueeProps) {
  const speedClass = 
    speed === "fast" ? "animate-[scroll_20s_linear_infinite]" :
    speed === "normal" ? "animate-[scroll_40s_linear_infinite]" :
    "animate-[scroll_60s_linear_infinite]";

  const directionClass = direction === "right" ? "[animation-direction:reverse]" : "";

  const renderCard = (item: MarqueeItem, index: number, isDuplicate: boolean) => {
    return (
      <div
        key={`${isDuplicate ? 'dup' : 'orig'}-${item.name}-${index}`}
        aria-hidden={isDuplicate ? "true" : undefined}
        className="group relative flex shrink-0 items-center justify-center p-6 bg-void border border-white/5 rounded-xl min-w-[200px] h-[100px] transition duration-500 hover:bg-white/[0.04] hover:border-white/20"
      >
        <div className="flex flex-col items-center justify-center w-full h-full gap-3">
          {item.domain ? (
            <img 
              src={`https://icon.horse/icon/${item.domain}`}
              alt={`${item.name} logo`}
              className="w-10 h-10 object-contain grayscale opacity-50 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-300"
              onError={(e) => {
                e.currentTarget.style.display = 'none';
                const sibling = e.currentTarget.nextElementSibling as HTMLElement;
                if (sibling) {
                  sibling.style.display = 'block';
                }
              }}
            />
          ) : null}
          <span 
            className="text-sm font-display font-bold tracking-widest uppercase text-white/50 group-hover:text-white transition text-center"
            style={{ display: item.domain ? 'none' : 'block' }}
          >
            {item.name}
          </span>
        </div>

        {item.scope && (
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 w-64 p-4 bg-[#0B0C0E] border border-white/10 text-xs text-white rounded-lg shadow-2xl opacity-0 scale-95 pointer-events-none group-hover:opacity-100 group-hover:scale-100 transition duration-300 z-50">
            <div className="font-bold mb-1 text-hazard-orange uppercase tracking-widest font-mono text-[10px]">{item.name}</div>
            <div className="text-white/70 font-sans leading-relaxed">{item.scope}</div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className={cn(
        "relative flex w-full overflow-hidden group/marquee",
        className
      )}
      style={{
        maskImage: "linear-gradient(to right, transparent, black 8%, black 92%, transparent)",
        WebkitMaskImage: "linear-gradient(to right, transparent, black 8%, black 92%, transparent)",
      }}
    >
      <div
        className={cn(
          "flex min-w-full shrink-0 gap-6 py-4 items-center group-hover/marquee:[animation-play-state:paused]",
          speedClass,
          directionClass
        )}
        style={{
          width: "max-content",
        }}
      >
        {items.map((item, index) => renderCard(item, index, false))}
        {items.map((item, index) => renderCard(item, index, true))}
      </div>
    </div>
  );
}
