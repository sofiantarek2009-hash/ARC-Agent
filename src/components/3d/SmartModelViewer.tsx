"use client";

import React, { useMemo, useEffect } from "react";
import { useGLTF, Bounds, Center } from "@react-three/drei";
import * as THREE from "three";

interface SmartModelViewerProps {
  url: string;
  autoRotate?: boolean;
}

export function SmartModelViewer({ url }: SmartModelViewerProps) {
  const { scene } = useGLTF(url);

  // Clone scene to avoid mutating cached GLTF, then apply corporate materials
  const clonedScene = useMemo(() => {
    const clone = scene.clone(true);
    
    // Create high-end corporate materials
    const baseMetal = new THREE.MeshStandardMaterial({
      color: "#2C303A", // Dark industrial grey
      metalness: 0.8,
      roughness: 0.3,
      envMapIntensity: 1.5,
    });
    
    const highlightMetal = new THREE.MeshStandardMaterial({
      color: "#FF6B00", // Hazard orange
      metalness: 0.6,
      roughness: 0.4,
      envMapIntensity: 1.2,
    });

    const lightMaterial = new THREE.MeshStandardMaterial({
      color: "#FFFFFF",
      emissive: "#FFFFFF",
      emissiveIntensity: 0.5,
      roughness: 0.1,
      metalness: 0.1
    });

    clone.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        // Cast shadow
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Simple heuristic: if the mesh name contains 'light', 'glass', etc, give it a different material.
        // Otherwise, mostly base metal, occasionally orange for smaller parts.
        const name = mesh.name.toLowerCase();
        
        if (name.includes("light") || name.includes("glass") || name.includes("window")) {
          mesh.material = lightMaterial;
        } else if (
          name.includes("accent") || 
          name.includes("wire") || 
          name.includes("handle") || 
          name.includes("pipe") ||
          mesh.geometry.boundingSphere?.radius && mesh.geometry.boundingSphere.radius < 0.5
        ) {
           // Apply orange to smaller details or specific names to give it that "corporate accent"
           // Wait, geometry radius check might make too many things orange. Let's be conservative.
           if (Math.random() > 0.85) {
             mesh.material = highlightMetal;
           } else {
             mesh.material = baseMetal;
           }
        } else {
          mesh.material = baseMetal;
        }
      }
    });
    
    return clone;
  }, [scene]);

  // Clean up cloned scene
  useEffect(() => {
    return () => {
      clonedScene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          if (mesh.geometry) mesh.geometry.dispose();
          if (mesh.material && !Array.isArray(mesh.material)) {
             mesh.material.dispose();
          }
        }
      });
    };
  }, [clonedScene]);

  return (
    <Bounds fit clip observe margin={1.2}>
      <primitive object={clonedScene} />
    </Bounds>
  );
}
