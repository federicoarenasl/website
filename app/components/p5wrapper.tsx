"use client";

import { useEffect, useRef, useState } from "react";
import p5 from "p5";

/**
 * P5Wrapper component that renders a p5.js sketch.
 * The sketch displays a 3D sphere with points that react to mouse movement and dark mode preference.
 */
const P5Wrapper = () => {
  // Reference to the div element where the p5 canvas will be attached
  const sketchRef = useRef<HTMLDivElement>(null);
  // State to track if dark mode is enabled
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check for dark mode preference
    const matchMedia = window.matchMedia("(prefers-color-scheme: dark)");
    setIsDarkMode(matchMedia.matches);

    // Event handler for theme change
    const handleThemeChange = (event: MediaQueryListEvent) => {
      setIsDarkMode(event.matches);
    };

    // Add event listener for changes in the color scheme
    matchMedia.addEventListener("change", handleThemeChange);

    /**
     * p5.js sketch function
     * @param {p5} p - The p5 instance
     */
    const sketch = (p: p5) => {
      // Array to store points on the sphere
      let points: { x: number; y: number; z: number; baseX: number; baseY: number; baseZ: number }[] = [];
      const numPoints = 500; // Number of points on the sphere
      const radius = 15; // Radius of the sphere
      let mouseInfluence = 5; // Influence of mouse movement on points

      /**
       * Sets up the p5 canvas and its properties
       */
      const setupCanvas = () => {
        const canvas = p.createCanvas(40, 40, p.WEBGL);
        canvas.elt.style.borderRadius = "50%";
        canvas.parent(sketchRef.current!);
        p.noStroke();
      };

      /**
       * Generates random points on the surface of a sphere
       */
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

      /**
       * Updates the influence of the mouse on the points based on its position
       */
      const updateMouseInfluence = () => {
        let targetInfluenceX = p.map(p.mouseX, 0, p.width, 0, 25, true);
        let targetInfluenceY = p.map(p.mouseY, 0, p.height, 0, 25, true);
        let targetInfluence = (targetInfluenceX + targetInfluenceY) / 2;
        mouseInfluence = p.lerp(mouseInfluence, targetInfluence, 0.1);
      };

      /**
       * Draws the points on the canvas, applying perturbations based on noise
       */
      const drawPoints = () => {
        p.fill(isDarkMode ? 255 : 0); // White points in dark mode, black points in light mode

        points.forEach((pt) => {
          const perturbX = p.noise(pt.baseX * 0.1 + p.frameCount * 0.02) * mouseInfluence - mouseInfluence / 2;
          const perturbY = p.noise(pt.baseY * 0.1 + p.frameCount * 0.02) * mouseInfluence - mouseInfluence / 2;
          const perturbZ = p.noise(pt.baseZ * 0.1 + p.frameCount * 0.02) * mouseInfluence - mouseInfluence / 2;

          pt.x = p.lerp(pt.x, pt.baseX + perturbX, 0.1);
          pt.y = p.lerp(pt.y, pt.baseY + perturbY, 0.1);
          pt.z = p.lerp(pt.z, pt.baseZ + perturbZ, 0.1);

          p.push();
          p.translate(pt.x, pt.y, pt.z);
          p.sphere(0.5);
          p.pop();
        });
      };

      // p5.js setup function
      p.setup = () => {
        setupCanvas();
        generatePoints();
      };

      // p5.js draw function
      p.draw = () => {
        p.background(isDarkMode ? 0 : 255); // Dark mode → black bg, Light mode → white bg
        p.orbitControl();
        // updateMouseInfluence(); // Commenting this out to prevent the points from moving

        p.push();
        p.rotateY(p.frameCount * 0.005);
        p.rotateX(p.frameCount * 0.003);
        drawPoints();
        p.pop();
      };
    };

    // Create a new p5 instance with the sketch
    const myP5 = new p5(sketch, sketchRef.current!);

    // Cleanup function to remove p5 instance and event listener
    return () => {
      myP5.remove();
      matchMedia.removeEventListener("change", handleThemeChange);
    };
  }, [isDarkMode]);

  // Render the div that will contain the p5 canvas
  return <div ref={sketchRef} className="absolute top-0 right-0 my-6 z-0" />;
};

export default P5Wrapper;
