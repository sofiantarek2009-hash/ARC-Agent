"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

const vertexShader = `
  uniform float uTime;
  attribute float aSize;
  attribute vec3 aOffset;
  varying vec3 vColor;
  
  void main() {
    vec3 pos = position + aOffset;
    
    // Complex fluid-like motion using sine waves
    pos.x += sin(uTime * 0.5 + aOffset.y * 2.0) * 0.5;
    pos.y += cos(uTime * 0.3 + aOffset.x * 2.0) * 0.5;
    pos.z += sin(uTime * 0.4 + aOffset.z * 2.0) * 0.5;
    
    // Slow drift upwards
    pos.y += mod(uTime * 0.5 + aOffset.y, 10.0) - 5.0;
    
    // Color depends on height and depth
    vColor = mix(vec3(0.0, 0.94, 1.0), vec3(1.0, 0.23, 0.0), (pos.y + 5.0) / 10.0);
    
    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_PointSize = aSize * (10.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  varying vec3 vColor;
  void main() {
    // Soft circle particle
    float dist = length(gl_PointCoord - vec2(0.5));
    if (dist > 0.5) discard;
    
    float alpha = smoothstep(0.5, 0.1, dist);
    gl_FragColor = vec4(vColor, alpha * 0.6);
  }
`;

const COUNT = 5000;

function Particles() {
  const pointsRef = useRef<THREE.Points>(null);
  
  const uniforms = useMemo(() => ({
    uTime: { value: 0 }
  }), []);

  const [positions, sizes] = useMemo(() => {
    const pos = new Float32Array(COUNT * 3);
    const siz = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) {
      pos[i * 3 + 0] = (Math.random() - 0.5) * 15;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 10;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 15;
      siz[i] = Math.random() * 8.0 + 2.0;
    }
    return [pos, siz];
  }, []);

  useFrame((state) => {
    if (pointsRef.current) {
      (pointsRef.current.material as THREE.ShaderMaterial).uniforms.uTime.value = state.clock.getElapsedTime();
      pointsRef.current.rotation.y = state.clock.getElapsedTime() * 0.05;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[new Float32Array(COUNT * 3), 3]}
        />
        <bufferAttribute
          attach="attributes-aOffset"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-aSize"
          args={[sizes, 1]}
        />
      </bufferGeometry>
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

export default function FluidParticleSim() {
  return (
    <div className="absolute inset-0 z-0 bg-petroleum-900 pointer-events-none overflow-hidden h-[120%] -top-[10%]">
      <Canvas camera={{ position: [0, 0, 10], fov: 45 }} gl={{ antialias: false, alpha: false }}>
        <color attach="background" args={["#030712"]} />
        <Particles />
      </Canvas>
      <div className="absolute inset-0 bg-gradient-to-b from-[#030712] via-transparent to-[#030712]" />
    </div>
  );
}
