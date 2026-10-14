"use client";

import React, { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF, Environment, OrbitControls, ContactShadows, Bounds } from "@react-three/drei";
import * as THREE from "three";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";

import { Loader2 } from "lucide-react";

interface DioramaProps {
  type: "inspection" | "simulation" | "chemicals" | "about";
}

// A wrapper to load and apply corporate materials to a model
function StyledModel({ url, position, rotation, scale }: any) {
  const { scene } = useGLTF(url as string) as any;
  
  // Clone to safely mutate materials
  const clonedScene = React.useMemo(() => {
    const clone = scene.clone(true);
    const baseMetal = new THREE.MeshStandardMaterial({
      color: "#181A20", 
      metalness: 0.9,
      roughness: 0.4,
    });
    const highlightMetal = new THREE.MeshStandardMaterial({
      color: "#FF5500", // intense hazard orange for bloom
      metalness: 0.8,
      roughness: 0.2,
      emissive: "#FF3300",
      emissiveIntensity: 0.4,
    });

    clone.traverse((child: THREE.Object3D) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        const name = mesh.name.toLowerCase();
        // Deterministic highlighting based on mesh name or position to avoid flickering on re-renders
        const seed = name.length + mesh.position.x + mesh.position.y;
        if (seed % 10 > 8) {
          mesh.material = highlightMetal;
        } else {
          mesh.material = baseMetal;
        }
      }
    });
    return clone;
  }, [scene]);

  return <primitive object={clonedScene} position={position} rotation={rotation} scale={scale} />;
}

// Rotating group wrapper
function RotatingGroup({ children }: { children: React.ReactNode }) {
  const ref = useRef<THREE.Group>(null);
  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });
  return <group ref={ref}>{children}</group>;
}

export function ServiceDiorama({ type }: DioramaProps) {
  const getDioramaContent = () => {
    switch (type) {
      case "inspection":
        return (
          <RotatingGroup>
            {/* Crawler Crane inspecting Helideck */}
            <StyledModel url="/models/model (12).glb" position={[-2, 0, 0]} scale={0.5} rotation={[0, Math.PI / 4, 0]} />
            <StyledModel url="/models/model (15).glb" position={[2, 0, 0]} scale={0.3} />
          </RotatingGroup>
        );
      case "simulation":
        return (
          <RotatingGroup>
            {/* Top Drive suspended over Drill Bit */}
            <StyledModel url="/models/model (16).glb" position={[0, 4, 0]} scale={1.5} />
            <StyledModel url="/models/model (5).glb" position={[0, -2, 0]} scale={2} rotation={[0, 0, 0]} />
          </RotatingGroup>
        );
      case "chemicals":
        return (
          <RotatingGroup>
            {/* Chemical Cabinet next to Degasser and Mud Cleaner */}
            <StyledModel url="/models/model (1).glb" position={[-3, 0, -1]} scale={1.2} />
            <StyledModel url="/models/model (22).glb" position={[0, 0, 1]} scale={1.2} rotation={[0, Math.PI / 2, 0]} />
            <StyledModel url="/models/model (23).glb" position={[3, 0, -1]} scale={1.2} />
          </RotatingGroup>
        );
      case "about":
      default:
        return (
          <RotatingGroup>
            {/* Pump Jack and Power Generator */}
            <StyledModel url="/models/oil_pump_jack.glb" position={[-2, 0, 0]} scale={0.4} />
            <StyledModel url="/models/model (29).glb" position={[3, 0, 0]} scale={0.8} />
          </RotatingGroup>
        );
    }
  };

  const getDioramaText = () => {
    switch (type) {
      case "inspection": return { title: "Structural Integrity Inspection", desc: "Crawler crane and helideck integrity verification in action." };
      case "simulation": return { title: "Drilling Systems Simulation", desc: "Top drive and drill bit physics replicated in high-fidelity." };
      case "chemicals": return { title: "Processing & Fluid Treatment", desc: "Chemical storage, degasser, and mud cleaner systems." };
      case "about": return { title: "Heavy Operations Architecture", desc: "Pump jack and power generation deployment." };
    }
  };

  const { title, desc } = getDioramaText();

  return (
    <section className="relative w-full h-[600px] lg:h-[800px] bg-[#0B0C0E] border-t border-white/5 overflow-hidden">
      <div className="absolute top-12 left-6 lg:left-12 z-20 pointer-events-none">
        <div className="font-mono text-[10px] text-hazard-orange uppercase tracking-[0.2em] mb-2">Synergistic Digital Twin</div>
        <h2 className="font-display text-3xl lg:text-5xl font-semibold text-white mb-2">{title}</h2>
        <p className="font-mono text-xs text-white/50 uppercase tracking-widest">{desc}</p>
      </div>

      <div className="absolute inset-0 z-10">
        <Suspense fallback={
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0B0C0E]">
            <Loader2 className="w-8 h-8 text-hazard-orange animate-spin mb-4" />
            <span className="font-mono text-xs uppercase tracking-widest text-white/50">Assembling Synergistic Assets...</span>
          </div>
        }>
          {typeof window !== 'undefined' && (
            <Canvas camera={{ position: [10, 10, 10], fov: 45 }} gl={{ antialias: true, alpha: true }}>
              <ambientLight intensity={0.4} />
              <directionalLight position={[10, 10, 10]} intensity={1.5} castShadow />
              
              
              <Bounds fit clip observe margin={1.2}>
                {getDioramaContent()}
              </Bounds>
              
              <OrbitControls makeDefault enableDamping dampingFactor={0.05} autoRotate autoRotateSpeed={0.2} />
              <Environment preset="studio" environmentIntensity={0.8} />
              <ContactShadows position={[0, -1, 0]} opacity={0.8} scale={20} blur={2.5} far={4} color="#000000" />
              
              <EffectComposer>
                <Bloom luminanceThreshold={0.2} mipmapBlur intensity={1.5} />
                <Vignette eskil={false} offset={0.1} darkness={1.1} />
              </EffectComposer>

            </Canvas>
          )}
        </Suspense>
      </div>
    </section>
  );
}
