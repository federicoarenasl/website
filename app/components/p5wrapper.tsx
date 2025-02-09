"use client";

import { useEffect, useRef } from "react";
import p5 from "p5";

const P5Wrapper = () => {
  const sketchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sketch = (p: p5) => {
      let points: { x: number; y: number; z: number; baseX: number; baseY: number; baseZ: number }[] = [];
      const numPoints = 500; // Number of points in the cloud
      const radius = 15; // Sphere radius
      let mouseInfluence = 500; // Stores mouse influence strength

      const setupCanvas = () => {
        const canvas = p.createCanvas(30, 30, p.WEBGL);
        canvas.elt.style.borderRadius = '50%'; // Make the canvas appear spherical
        canvas.parent(sketchRef.current!);
        p.noStroke();
        p.fill(255);
      };

      const generatePoints = () => {
        for (let i = 0; i < numPoints; i++) {
          const theta = p.random(p.TWO_PI);
          const phi = p.acos(p.random(-1, 1));

          const x = radius * p.sin(phi) * p.cos(theta);
          const y = radius * p.sin(phi) * p.sin(theta);
          const z = radius * p.cos(phi);

          points.push({ x, y, z, baseX: x, baseY: y, baseZ: z });
        }
      };

      const updateMouseInfluence = () => {
        let targetInfluence = p.map(p.mouseX, 0, p.width, 0, 5, true);
        mouseInfluence = p.lerp(mouseInfluence, targetInfluence, 0.1); // Smooth transition
      };

      const drawPoints = () => {
        points.forEach((pt) => {
          const perturbX = p.noise(pt.baseX * 0.1 + p.frameCount * 0.02) * mouseInfluence - mouseInfluence / 2;
          const perturbY = p.noise(pt.baseY * 0.1 + p.frameCount * 0.02) * mouseInfluence - mouseInfluence / 2;
          const perturbZ = p.noise(pt.baseZ * 0.1 + p.frameCount * 0.02) * mouseInfluence - mouseInfluence / 2;

          pt.x = p.lerp(pt.x, pt.baseX + perturbX, 0.1);
          pt.y = p.lerp(pt.y, pt.baseY + perturbY, 0.1);
          pt.z = p.lerp(pt.z, pt.baseZ + perturbZ, 0.1);

          p.push();
          p.translate(pt.x, pt.y, pt.z);
          p.sphere(0.5); // Small spheres as points
          p.pop();
        });
      };

      p.setup = () => {
        setupCanvas();
        generatePoints();
      };

      p.draw = () => {
        p.background(0);
        p.orbitControl(); // Allows user rotation
        updateMouseInfluence();

        p.push();
        p.rotateY(p.frameCount * 0.005); // Slow rotation
        p.rotateX(p.frameCount * 0.003);
        drawPoints();
        p.pop();
      };
    };

    const myP5 = new p5(sketch, sketchRef.current!);

    return () => {
      myP5.remove();
    };
  }, []);

  return <div ref={sketchRef} className="relative m-6 z-0" />;
};

export default P5Wrapper;
