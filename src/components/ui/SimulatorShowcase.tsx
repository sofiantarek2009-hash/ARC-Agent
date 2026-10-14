"use client";
import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, Environment, PresentationControls, ContactShadows, Center } from "@react-three/drei";
import { ShieldCheck, Crosshair, Monitor, Cpu } from "lucide-react";
import * as THREE from "three";
import { useDeviceCapability } from "@/hooks/useDeviceCapability";
import Image from "next/image";

const RDSModel = () => {
  const { scene } = useGLTF("/models/rds-100.glb");
  const modelRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (modelRef.current) {
      modelRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.2;
    }
  });

  return (
    <group ref={modelRef}>
      <primitive object={scene} />
      {/* Add glow to monitors */}
      <pointLight position={[-0.3, 0.9, 0.3]} color="#00F2FE" intensity={2} distance={2} />
      <pointLight position={[0.3, 0.9, 0.3]} color="#00F2FE" intensity={2} distance={2} />
    </group>
  );
};

const VRModel = () => {
  const { scene } = useGLTF("/models/vr-headset.glb");
  const modelRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (modelRef.current) {
      modelRef.current.position.y = Math.sin(state.clock.elapsedTime * 2) * 0.05;
      modelRef.current.rotation.y = state.clock.elapsedTime * 0.5;
    }
  });

  return (
    <group ref={modelRef} scale={3} position={[0, -0.5, 0]}>
      <primitive object={scene} />
      <pointLight position={[0, 0, 0.5]} color="#FF5722" intensity={1} distance={1} />
    </group>
  );
};

const SimulatorScene = ({ activeSpec }: { activeSpec: string }) => {
  return (
    <Canvas camera={{ position: [0, 1.5, 4], fov: 45 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 10]} intensity={1} />
      
      <PresentationControls
        global
        rotation={[0.1, 0, 0]}
        polar={[-Math.PI / 3, Math.PI / 3]}
        azimuth={[-Math.PI / 1.4, Math.PI / 2]}
      >
        <Center>
          {activeSpec === 'vr' ? <VRModel /> : <RDSModel />}
        </Center>
      </PresentationControls>
      
      <Environment preset="city" />
      <ContactShadows position={[0, -1, 0]} opacity={0.5} scale={10} blur={2} far={4} color="#00F2FE" />
    </Canvas>
  );
};

const SPECS = [
  {
    id: "iwcf",
    title: "IWCF Level 3 & 4",
    icon: ShieldCheck,
    desc: "Fully accredited well control simulation for Driller and Supervisor levels.",
    color: "text-neon-cyan",
    border: "border-neon-cyan/20",
    bg: "bg-neon-cyan/10"
  },
  {
    id: "iadc",
    title: "IADC WellSharp",
    icon: Crosshair,
    desc: "Rigorous competency assurance programs and continuous driller training.",
    color: "text-hazard-orange",
    border: "border-hazard-orange/20",
    bg: "bg-hazard-orange/10"
  },
  {
    id: "vr",
    title: "WeXplore VR",
    icon: Monitor,
    desc: "Immersive digital twin environments mapping actual rig floor dimensions.",
    color: "text-white",
    border: "border-white/20",
    bg: "bg-white/10"
  }
];

export function SimulatorShowcase() {
  const [activeSpec, setActiveSpec] = useState(SPECS[0].id);

  const specImages: Record<string, string> = {
    iwcf: "/assets/we-rds-console.jpg",
    iadc: "/assets/digital-hud.jpg",
    vr: "/assets/data-network-abstract.jpg"
  };

  return (
    <section id="simulator" className="relative w-full py-24 px-6 max-w-7xl mx-auto">
      <div className="mb-12">
        <h2 className="text-sm font-mono uppercase tracking-widest text-neon-cyan mb-4 flex items-center gap-2">
          <Cpu className="w-4 h-4" />
          RDS-100 Reality Drilling Simulator
        </h2>
        <h3 className="text-4xl md:text-5xl font-bold font-display max-w-2xl">
          Digital Twin Competency Ecosystem
        </h3>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:min-h-[600px]">
        <div className="relative w-full h-[400px] lg:h-full lg:min-h-[600px] rounded-2xl overflow-hidden glass-panel-heavy border border-white/5 bg-[#0B0C0E]">
          {/* Subtle grid background */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />
          
          <AnimatePresence mode="wait">
            <motion.div
              key={activeSpec}
              initial={{ opacity: 0, scale: 1.05 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
              className="absolute inset-0"
            >
              <Image
                src={specImages[activeSpec] || "/assets/we-rds-console.jpg"}
                alt="Simulator View"
                fill
                className="object-cover opacity-90 mix-blend-screen"
              />
            </motion.div>
          </AnimatePresence>

          {/* Overlay UI */}
          <div className="absolute bottom-6 left-6 font-mono text-xs text-white/40 uppercase tracking-widest pointer-events-none flex gap-4 z-10">
            <span className="flex items-center gap-2"><Cpu className="w-3 h-3 text-neon-cyan" /> {activeSpec.toUpperCase()} ACTIVE</span>
          </div>
        </div>

        {/* Right: Interactive Specs */}
        <div className="flex flex-col justify-start pt-2 gap-4 lg:gap-6">
          {SPECS.map((spec) => {
            const isActive = activeSpec === spec.id;
            const Icon = spec.icon;
            
            return (
              <motion.div
                key={spec.id}
                onClick={() => setActiveSpec(spec.id)}
                className={`relative p-6 rounded-xl cursor-pointer transition-all duration-300 border ${isActive ? 'border-white bg-white/10 shadow-[0_0_30px_rgba(255,255,255,0.05)]' : 'border-white/10 bg-[#0B0C0E] hover:bg-white/5 hover:border-white/30'}`}
                whileHover={{ x: 4 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-start gap-5 relative z-10">
                  <div className={`p-3 rounded-lg flex-shrink-0 transition-colors duration-300 ${isActive ? 'bg-white text-black' : 'bg-white/5 text-white/50'}`}>
                    <Icon className="w-6 h-6" strokeWidth={1.5} />
                  </div>
                  <div>
                    <h4 className={`text-xl font-display font-semibold mb-2 transition-colors duration-300 ${isActive ? 'text-white' : 'text-white/70'}`}>
                      {spec.title}
                    </h4>
                    <p className={`text-sm leading-relaxed transition-colors duration-300 ${isActive ? 'text-white/80' : 'text-white/40'}`}>
                      {spec.desc}
                    </p>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

useGLTF.preload("/models/rds-100.glb");
useGLTF.preload("/models/vr-headset.glb");
