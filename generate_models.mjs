import { Document, Node, WebIO } from '@gltf-transform/core';
import { metalRough } from '@gltf-transform/functions';

// Creates a basic box primitive
function createBox(doc, name, size, position, colorHex) {
  // We need primitive creation which gltf-transform doesn't natively do easily.
  // Actually, building geometries from scratch in gltf-transform requires writing raw vertex buffers.
}
