"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

interface BentoAccordionProps {
  items: {
    id: string;
    title: string;
    content: string | React.ReactNode;
  }[];
  className?: string;
}

export function BentoAccordion({ items, className }: BentoAccordionProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const toggle = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {items.map((item) => (
        <div key={item.id} className="group flex flex-col rounded-xl overflow-hidden bg-void/30 border border-white/5">
          <button
            onClick={() => toggle(item.id)}
            className="flex items-center justify-between p-4 text-left hover:bg-white/5 transition-colors"
          >
            <span className="font-medium text-sm text-white/90 font-sans tracking-wide">
              {item.title}
            </span>
            <span className="text-white/40 group-hover:text-simulation transition-colors">
              {expandedId === item.id ? <Minus size={16} /> : <Plus size={16} />}
            </span>
          </button>
          
          <AnimatePresence>
            {expandedId === item.id && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ type: "spring", stiffness: 350, damping: 30 }}
              >
                <div className="px-4 pb-4 pt-0 text-sm text-white/60 font-mono leading-relaxed border-t border-white/5 mt-2 pt-4">
                  {item.content}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ))}
    </div>
  );
}
