import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class MoESimulation {
  constructor(container, options = {}) {
    this.container = container;
    this.onStatusChange = options.onStatusChange || (() => {});
    
    this.animationId = null;
    this.isPaused = false;
    this.speed = 1.0;
    
    this.experts = [];
    this.activeRays = [];
    this.particlePulses = [];
    this.tokenQueue = [];

    this.promptText = options.prompt || "Solve math equation and write code";

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060914);
    this.scene.fog = new THREE.FogExp2(0x060914, 0.02);

    // Camera
    this.camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    this.camera.position.set(0, 10, 22);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x38bdf8, 2, 60);
    pointLight.position.set(0, 12, 10);
    this.scene.add(pointLight);

    // Grid Floor
    const grid = new THREE.GridHelper(36, 36, 0x1e293b, 0x0f172a);
    grid.position.y = -4;
    this.scene.add(grid);

    // Groups
    this.routerGroup = new THREE.Group();
    this.scene.add(this.routerGroup);

    this.expertsGroup = new THREE.Group();
    this.scene.add(this.expertsGroup);

    this.raysGroup = new THREE.Group();
    this.scene.add(this.raysGroup);

    this.tokensGroup = new THREE.Group();
    this.scene.add(this.tokensGroup);

    // Build Visual Components
    this.createRouter();
    this.createExperts();
    this.processPromptTokens(this.promptText);

    // Window Resize
    this.onResize = this.onResize.bind(this);
    window.addEventListener('resize', this.onResize);
  }

  createLabelSprite(text, colorStr = '#ffffff') {
    const canvas = document.createElement('canvas');
    canvas.width = 384;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 3;
    
    ctx.beginPath();
    ctx.roundRect(4, 4, 376, 88, 16);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 26px sans-serif';
    ctx.fillStyle = colorStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 192, 48);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(2.6, 0.65, 1);
    return sprite;
  }

  createRouter() {
    // Router Core Octahedron
    const geo = new THREE.OctahedronGeometry(1.5, 0);
    const mat = new THREE.MeshPhongMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      wireframe: false,
      transparent: true,
      opacity: 0.85
    });
    this.routerMesh = new THREE.Mesh(geo, mat);
    this.routerGroup.add(this.routerMesh);

    // Orbital Ring
    const ringGeo = new THREE.TorusGeometry(2.4, 0.08, 16, 100);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x7dd3fc, wireframe: true });
    this.routerRing = new THREE.Mesh(ringGeo, ringMat);
    this.routerRing.rotation.x = Math.PI / 3;
    this.routerGroup.add(this.routerRing);

    // Label
    const label = this.createLabelSprite("Gating Router (Top-2 Softmax)", "#38bdf8");
    label.position.set(0, 2.8, 0);
    this.routerGroup.add(label);

    this.routerGroup.position.set(0, 2, 0);
  }

  createExperts() {
    const expertDefs = [
      { name: "Math & Logic", color: 0x06b6d4, type: "math" },
      { name: "Code & Syntax", color: 0x10b981, type: "code" },
      { name: "Creative Writing", color: 0xec4899, type: "creative" },
      { name: "Multilingual", color: 0xf59e0b, type: "lang" },
      { name: "Reasoning", color: 0x8b5cf6, type: "reason" },
      { name: "Retrieval RAG", color: 0x3b82f6, type: "rag" },
      { name: "Summarization", color: 0x64748b, type: "summary" },
      { name: "Fact Check", color: 0x14b8a6, type: "fact" }
    ];

    this.experts = [];
    const radius = 11;

    expertDefs.forEach((def, idx) => {
      const angle = (idx / expertDefs.length) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      const group = new THREE.Group();

      // Pod geometry
      const podGeo = new THREE.BoxGeometry(1.8, 2.2, 1.8);
      const podMat = new THREE.MeshPhongMaterial({
        color: def.color,
        emissive: def.color,
        emissiveIntensity: 0.25,
        transparent: true,
        opacity: 0.85
      });
      const podMesh = new THREE.Mesh(podGeo, podMat);
      group.add(podMesh);

      // Wireframe border
      const edges = new THREE.EdgesGeometry(podGeo);
      const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 });
      group.add(new THREE.LineSegments(edges, lineMat));

      // Label
      const label = this.createLabelSprite(`Expert ${idx + 1}: ${def.name}`, `#${def.color.toString(16).padStart(6, '0')}`);
      label.position.set(0, 2, 0);
      group.add(label);

      // Load Gauge Bar
      const barGeo = new THREE.CylinderGeometry(0.2, 0.2, 1.5, 16);
      const barMat = new THREE.MeshBasicMaterial({ color: def.color });
      const barMesh = new THREE.Mesh(barGeo, barMat);
      barMesh.position.set(1.4, 0, 0);
      group.add(barMesh);

      group.position.set(x, 0, z);
      this.expertsGroup.add(group);

      this.experts.push({
        def,
        group,
        mesh: podMesh,
        barMesh,
        pos: new THREE.Vector3(x, 0, z),
        load: 0.1
      });
    });
  }

  processPromptTokens(text) {
    while(this.tokensGroup.children.length > 0) {
      this.tokensGroup.remove(this.tokensGroup.children[0]);
    }
    this.tokenQueue = [];

    const words = text.trim().split(/\s+/).slice(0, 5);
    words.forEach((word, idx) => {
      const group = new THREE.Group();
      const geo = new THREE.SphereGeometry(0.45, 16, 16);
      const mat = new THREE.MeshPhongMaterial({ color: 0xf59e0b, emissive: 0xd97706 });
      group.add(new THREE.Mesh(geo, mat));

      const label = this.createLabelSprite(word, "#f59e0b");
      label.position.set(0, 0.8, 0);
      label.scale.set(1.8, 0.45, 1);
      group.add(label);

      // Stack tokens along negative Z axis approaching router
      group.position.set(0, 2, -14 - idx * 3.5);
      this.tokensGroup.add(group);

      this.tokenQueue.push({ word, group, initialZ: -14 - idx * 3.5 });
    });

    this.triggerRoutingCycle();
  }

  triggerRoutingCycle() {
    // Clear old rays
    while(this.raysGroup.children.length > 0) {
      this.raysGroup.remove(this.raysGroup.children[0]);
    }
    this.particlePulses = [];

    // Select Top-2 experts based on active token domain
    const randIdx1 = Math.floor(Math.random() * this.experts.length);
    let randIdx2 = (randIdx1 + Math.floor(1 + Math.random() * 3)) % this.experts.length;

    const routerPos = this.routerGroup.position.clone();

    [randIdx1, randIdx2].forEach((expIdx, order) => {
      const expert = this.experts[expIdx];
      expert.load = Math.min(1.0, expert.load + 0.35);

      const midPos = new THREE.Vector3().addVectors(routerPos, expert.pos).multiplyScalar(0.5);
      midPos.y += 3; // Bezier height

      const curve = new THREE.QuadraticBezierCurve3(routerPos, midPos, expert.pos);
      const points = curve.getPoints(30);
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      
      const lineMat = new THREE.LineBasicMaterial({
        color: order === 0 ? 0x38bdf8 : 0x10b981,
        linewidth: 3,
        transparent: true,
        opacity: 0.95
      });
      const line = new THREE.Line(lineGeo, lineMat);
      this.raysGroup.add(line);

      // Glowing routing particle pulse
      const pGeo = new THREE.SphereGeometry(0.2, 16, 16);
      const pMat = new THREE.MeshBasicMaterial({ color: order === 0 ? 0x38bdf8 : 0x10b981 });
      const particle = new THREE.Mesh(pGeo, pMat);
      this.raysGroup.add(particle);

      this.particlePulses.push({ particle, curve, progress: 0, speed: 0.02 });
    });

    const activeNames = `${this.experts[randIdx1].def.name} (${orderWeight(0)}) & ${this.experts[randIdx2].def.name} (${orderWeight(1)})`;
    this.onStatusChange(1, `MoE Router active: Dynamically routed token to Top-2 Experts: ${activeNames}`);
  }

  start() {
    const loop = () => {
      this.animationId = requestAnimationFrame(loop);
      this.render();
    };
    loop();
  }

  render() {
    this.controls.update();

    if (!this.isPaused) {
      // Rotate Router Core
      this.routerMesh.rotation.y += 0.02 * this.speed;
      this.routerMesh.rotation.x += 0.01 * this.speed;
      this.routerRing.rotation.z -= 0.015 * this.speed;

      // Animate Token Queue forward toward Router
      this.tokenQueue.forEach((t) => {
        t.group.position.z += 0.05 * this.speed;
        if (t.group.position.z > 2) {
          t.group.position.z = t.initialZ;
          this.triggerRoutingCycle();
        }
      });

      // Animate Routing Particles along Bezier curves
      this.particlePulses.forEach(p => {
        p.progress += p.speed * this.speed;
        if (p.progress > 1) p.progress = 0;
        const pos = p.curve.getPoint(p.progress);
        p.particle.position.copy(pos);
      });

      // Decay Expert Loads
      this.experts.forEach(exp => {
        exp.load = Math.max(0.1, exp.load - 0.002 * this.speed);
        exp.barMesh.scale.set(1, exp.load * 2, 1);
        exp.mesh.material.emissiveIntensity = exp.load;
      });
    }

    this.renderer.render(this.scene, this.camera);
  }

  setPrompt(text) {
    this.promptText = text;
    this.processPromptTokens(text);
  }

  setSpeed(val) {
    this.speed = val;
  }

  pause() {
    this.isPaused = true;
  }

  resume() {
    this.isPaused = false;
  }

  onResize() {
    if (!this.container) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  destroy() {
    if (this.animationId) cancelAnimationFrame(this.animationId);
    window.removeEventListener('resize', this.onResize);
    if (this.renderer && this.renderer.domElement) {
      this.container.removeChild(this.renderer.domElement);
      this.renderer.dispose();
    }
  }
}

function orderWeight(idx) {
  return idx === 0 ? "68%" : "32%";
}
