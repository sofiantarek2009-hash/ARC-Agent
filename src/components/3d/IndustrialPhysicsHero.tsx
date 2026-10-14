"use client";

import { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Physics, RigidBody, RapierRigidBody, InstancedRigidBodies, InstancedRigidBodyProps } from "@react-three/rapier";
import { Environment, MeshTransmissionMaterial } from "@react-three/drei";
import * as THREE from "three";

const COUNT = 40;

function FloatingSteel() {
  const api = useRef<any>(null);
  const [hovered, setHover] = useState(false);

  // Distribute instances
  const instances = useMemo(() => {
    const insts: InstancedRigidBodyProps[] = [];
    
    for (let i = 0; i < COUNT; i++) {
      insts.push({
        key: i,
        position: [
          (Math.random() - 0.5) * 15,
          (Math.random() - 0.5) * 15,
          (Math.random() - 0.5) * 10 - 5
        ],
        rotation: [Math.random() * Math.PI, Math.random() * Math.PI, 0],
        scale: [
          0.5 + Math.random() * 0.5,
          2 + Math.random() * 4,
          0.5 + Math.random() * 0.5
        ]
      });
    }

    return insts;
  }, []);

  // Attract objects to center or pointer
  useFrame((state) => {
    if (!api.current) return;
    
    const time = state.clock.getElapsedTime();
    const target = new THREE.Vector3(
      (state.pointer.x * state.viewport.width) / 2,
      (state.pointer.y * state.viewport.height) / 2,
      -5
    );

    api.current.forEach((body: any, i: number) => {
      if (!body) return;
      const p = body.translation();
      const pos = new THREE.Vector3(p.x, p.y, p.z);
      
      // Calculate vector towards pointer
      const dir = target.clone().sub(pos);
      
      // Gentle floating force + pointer attraction
      const strength = 0.5;
      const dist = dir.length();
      dir.normalize().multiplyScalar(strength / Math.max(dist, 1.0));
      
      // Add some chaotic tumble
      body.applyImpulse({ x: dir.x * 0.05, y: dir.y * 0.05, z: dir.z * 0.05 }, true);
      body.applyTorqueImpulse({
        x: Math.sin(time + i) * 0.01,
        y: Math.cos(time + i) * 0.01,
        z: 0
      }, true);
    });
  });

  return (
    <InstancedRigidBodies
      ref={api}
      instances={instances}
      colliders="cuboid"
      linearDamping={4}
      angularDamping={4}
      friction={0.1}
      restitution={0.5}
    >
      <instancedMesh
        args={[undefined, undefined, COUNT]}
        count={COUNT}
        castShadow
        receiveShadow
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
      >
        <cylinderGeometry args={[0.5, 0.5, 1, 16]} />
        <meshStandardMaterial 
          color="#0f172a" 
          metalness={0.9} 
          roughness={0.1}
          envMapIntensity={2}
        />
      </instancedMesh>
    </InstancedRigidBodies>
  );
}

export default function IndustrialPhysicsHero() {
  return (
    <div className="absolute inset-0 z-0 pointer-events-auto">
      <Canvas
        camera={{ position: [0, 0, 15], fov: 35 }}
        gl={{ antialias: false, alpha: true }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#030712"]} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={2} color="#00f0ff" />
        <directionalLight position={[-10, -10, -5]} intensity={2} color="#ff3b00" />
        <Physics gravity={[0, 0, 0]}>
          <FloatingSteel />
        </Physics>
        <Environment preset="city" />
      </Canvas>
      {/* Heavy vignette overlay */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,#030712_80%)] pointer-events-none" />
    </div>
  );
}
