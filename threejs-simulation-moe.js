import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class MoESimulation {
  constructor(container, options = {}) {
    this.container = container;
    this.onStatusChange = options.onStatusChange || (() => {});
    this.onStageChanged = options.onStageChanged || (() => {});
    this.onRoutingUpdated = options.onRoutingUpdated || (() => {});
    
    this.animationId = null;
    this.isPaused = false;
    this.speed = 1.0;
    
    this.experts = [];
    this.activeForwardRays = [];
    this.activeReturnRays = [];
    this.particlePulses = [];
    this.tokenQueue = [];

    // Stages: 0=Token Gating Input, 1=Router Softmax Top-2 Dispatch, 2=Expert Parallel FFN Compute, 3=Weighted-Sum Residual Merge
    this.currentStage = 0;
    this.totalStages = 4;
    this.stageTimer = 0;
    this.stageDuration = 140;

    this.promptText = options.prompt || "Solve math equation and write python code";

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060914);
    this.scene.fog = new THREE.FogExp2(0x060914, 0.018);

    // Camera
    this.camera = new THREE.PerspectiveCamera(54, width / height, 0.1, 1000);
    this.camera.position.set(0, 11, 23);

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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    this.scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x38bdf8, 2.2, 70);
    pointLight.position.set(0, 14, 10);
    this.scene.add(pointLight);

    // Grid Floor
    const grid = new THREE.GridHelper(36, 36, 0x1e293b, 0x0f172a);
    grid.position.y = -4;
    this.scene.add(grid);

    // Groups
    this.routerGroup = new THREE.Group();
    this.scene.add(this.routerGroup);

    this.accumulatorGroup = new THREE.Group();
    this.scene.add(this.accumulatorGroup);

    this.expertsGroup = new THREE.Group();
    this.scene.add(this.expertsGroup);

    this.raysGroup = new THREE.Group();
    this.scene.add(this.raysGroup);

    this.tokensGroup = new THREE.Group();
    this.scene.add(this.tokensGroup);

    // Build Visual Components
    this.createRouter();
    this.createAccumulator();
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

    ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 3;
    
    ctx.beginPath();
    ctx.roundRect(4, 4, 376, 88, 16);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 25px sans-serif';
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
      emissiveIntensity: 0.6,
      wireframe: false,
      transparent: true,
      opacity: 0.9
    });
    this.routerMesh = new THREE.Mesh(geo, mat);
    this.routerGroup.add(this.routerMesh);

    // Orbital Ring
    const ringGeo = new THREE.TorusGeometry(2.3, 0.07, 16, 100);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x7dd3fc, wireframe: true });
    this.routerRing = new THREE.Mesh(ringGeo, ringMat);
    this.routerRing.rotation.x = Math.PI / 3;
    this.routerGroup.add(this.routerRing);

    // Label
    const label = this.createLabelSprite("Gating Router W_g (Top-2)", "#38bdf8");
    label.position.set(0, 2.7, 0);
    this.routerGroup.add(label);

    this.routerGroup.position.set(0, 2, -2);
  }

  createAccumulator() {
    // Output weighted-sum accumulator node
    const geo = new THREE.IcosahedronGeometry(1.1, 1);
    const mat = new THREE.MeshPhongMaterial({
      color: 0xc084fc,
      emissive: 0x9333ea,
      emissiveIntensity: 0.7,
      wireframe: false,
      transparent: true,
      opacity: 0.9
    });
    this.accumulatorMesh = new THREE.Mesh(geo, mat);
    this.accumulatorGroup.add(this.accumulatorMesh);

    const label = this.createLabelSprite("Output Residual: y = ∑ Pᵢ Eᵢ(x)", "#c084fc");
    label.position.set(0, 2.2, 0);
    label.scale.set(3.2, 0.75, 1);
    this.accumulatorGroup.add(label);

    this.accumulatorGroup.position.set(0, 2, 7);
  }

  createExperts() {
    const expertDefs = [
      { name: "Math & Logic", color: 0x06b6d4, type: "math", keywords: ["math", "equation", "calculate", "algebra", "number", "sum", "solve"] },
      { name: "Code & Syntax", color: 0x10b981, type: "code", keywords: ["code", "python", "syntax", "function", "def", "async", "script", "write"] },
      { name: "Creative Writing", color: 0xec4899, type: "creative", keywords: ["creative", "story", "write", "poem", "fantasy", "fiction", "novel"] },
      { name: "Multilingual", color: 0xf59e0b, type: "lang", keywords: ["translate", "spanish", "french", "multilingual", "language", "german"] },
      { name: "Reasoning", color: 0x8b5cf6, type: "reason", keywords: ["reason", "logic", "deduce", "infer", "think", "deep", "solve", "models"] },
      { name: "Retrieval RAG", color: 0x3b82f6, type: "rag", keywords: ["retrieve", "rag", "vector", "database", "search", "document", "knowledge"] },
      { name: "Summarization", color: 0x64748b, type: "summary", keywords: ["summary", "summarize", "brief", "digest", "tl;dr", "short"] },
      { name: "Fact Check", color: 0x14b8a6, type: "fact", keywords: ["fact", "verify", "true", "check", "evidence", "truth", "accuracy"] }
    ];

    this.experts = [];
    const radius = 11.5;

    expertDefs.forEach((def, idx) => {
      const angle = (idx / expertDefs.length) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      const group = new THREE.Group();

      // Pod geometry (Expert FFN Network)
      const podGeo = new THREE.BoxGeometry(2.0, 2.4, 2.0);
      const podMat = new THREE.MeshPhongMaterial({
        color: def.color,
        emissive: def.color,
        emissiveIntensity: 0.25,
        transparent: true,
        opacity: 0.88
      });
      const podMesh = new THREE.Mesh(podGeo, podMat);
      group.add(podMesh);

      // Wireframe border
      const edges = new THREE.EdgesGeometry(podGeo);
      const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.45 });
      group.add(new THREE.LineSegments(edges, lineMat));

      // Label
      const label = this.createLabelSprite(`E${idx + 1}: ${def.name}`, `#${def.color.toString(16).padStart(6, '0')}`);
      label.position.set(0, 2.2, 0);
      group.add(label);

      // Gating Probability Badge
      const weightBadge = this.createLabelSprite("0.00", "#ffffff");
      weightBadge.position.set(0, -1.8, 0);
      weightBadge.scale.set(1.4, 0.4, 1);
      group.add(weightBadge);

      // Load Gauge Bar
      const barGeo = new THREE.CylinderGeometry(0.2, 0.2, 1.6, 16);
      const barMat = new THREE.MeshBasicMaterial({ color: def.color });
      const barMesh = new THREE.Mesh(barGeo, barMat);
      barMesh.position.set(1.5, 0, 0);
      group.add(barMesh);

      group.position.set(x, 0.5, z);
      this.expertsGroup.add(group);

      this.experts.push({
        def,
        group,
        mesh: podMesh,
        barMesh,
        weightBadge,
        pos: new THREE.Vector3(x, 0.5, z),
        load: 0.1,
        prob: 0
      });
    });
  }

  computeSemanticRouting(text) {
    const lower = text.toLowerCase();
    const words = lower.split(/\s+/);

    // Score experts by keyword matching
    const scores = this.experts.map(exp => {
      let score = 0.5; // baseline prior
      exp.def.keywords.forEach(kw => {
        if (lower.includes(kw)) score += 3.0;
        words.forEach(w => {
          if (w.startsWith(kw) || kw.startsWith(w)) score += 1.5;
        });
      });
      return score;
    });

    // Softmax over scores to obtain router probabilities
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = expScores.map(s => s / sumExp);

    // Get Top-2 indices
    const indexed = probs.map((p, i) => ({ prob: p, index: i }));
    indexed.sort((a, b) => b.prob - a.prob);

    const top1 = indexed[0];
    const top2 = indexed[1];

    // Renormalize Top-2 probabilities: P1 + P2 = 1.0
    const topSum = top1.prob + top2.prob;
    const p1 = top1.prob / topSum;
    const p2 = top2.prob / topSum;

    return [
      { index: top1.index, weight: p1 },
      { index: top2.index, weight: p2 }
    ];
  }

  processPromptTokens(text) {
    while(this.tokensGroup.children.length > 0) {
      this.tokensGroup.remove(this.tokensGroup.children[0]);
    }
    this.tokenQueue = [];

    const words = text.trim().split(/\s+/).slice(0, 5);
    words.forEach((word, idx) => {
      const group = new THREE.Group();
      const geo = new THREE.SphereGeometry(0.48, 18, 18);
      const mat = new THREE.MeshPhongMaterial({ color: 0xf59e0b, emissive: 0xd97706, emissiveIntensity: 0.7 });
      group.add(new THREE.Mesh(geo, mat));

      const label = this.createLabelSprite(word, "#f59e0b");
      label.position.set(0, 0.85, 0);
      label.scale.set(1.8, 0.45, 1);
      group.add(label);

      // Stack tokens along negative Z axis approaching router
      group.position.set(0, 2, -15 - idx * 3.5);
      this.tokensGroup.add(group);

      this.tokenQueue.push({ word, group, initialZ: -15 - idx * 3.5 });
    });

    this.triggerRoutingCycle();
  }

  triggerRoutingCycle() {
    // Clear old rays
    while(this.raysGroup.children.length > 0) {
      this.raysGroup.remove(this.raysGroup.children[0]);
    }
    this.particlePulses = [];

    // Compute deterministic semantic routing for the prompt
    const top2 = this.computeSemanticRouting(this.promptText);
    const routerPos = this.routerGroup.position.clone();
    const accumPos = this.accumulatorGroup.position.clone();

    // Reset expert weights
    this.experts.forEach(exp => {
      exp.prob = 0;
    });

    top2.forEach((route, order) => {
      const expert = this.experts[route.index];
      expert.prob = route.weight;
      expert.load = Math.min(1.0, expert.load + route.weight * 0.7);

      const color = order === 0 ? 0x38bdf8 : 0x10b981;
      const weightPct = `${Math.round(route.weight * 100)}%`;

      // 1. Forward Path: Router -> Expert
      const midForward = new THREE.Vector3().addVectors(routerPos, expert.pos).multiplyScalar(0.5);
      midForward.y += 3;
      const fwdCurve = new THREE.QuadraticBezierCurve3(routerPos, midForward, expert.pos);
      const fwdPoints = fwdCurve.getPoints(24);
      const fwdLineGeo = new THREE.BufferGeometry().setFromPoints(fwdPoints);
      const fwdLineMat = new THREE.LineBasicMaterial({ color, linewidth: 2, transparent: true, opacity: 0.85 });
      this.raysGroup.add(new THREE.Line(fwdLineGeo, fwdLineMat));

      // Forward Particle
      const pGeo1 = new THREE.SphereGeometry(0.18, 14, 14);
      const pMat1 = new THREE.MeshBasicMaterial({ color });
      const particle1 = new THREE.Mesh(pGeo1, pMat1);
      this.raysGroup.add(particle1);
      this.particlePulses.push({ particle: particle1, curve: fwdCurve, progress: 0, speed: 0.018 });

      // 2. Return Path: Expert -> Output Accumulator (Weighted Sum)
      const midReturn = new THREE.Vector3().addVectors(expert.pos, accumPos).multiplyScalar(0.5);
      midReturn.y += 3;
      const retCurve = new THREE.QuadraticBezierCurve3(expert.pos, midReturn, accumPos);
      const retPoints = retCurve.getPoints(24);
      const retLineGeo = new THREE.BufferGeometry().setFromPoints(retPoints);
      const retLineMat = new THREE.LineDashedMaterial({
        color: 0xc084fc,
        dashSize: 0.4,
        gapSize: 0.2,
        linewidth: 2,
        transparent: true,
        opacity: 0.85
      });
      const retLine = new THREE.Line(retLineGeo, retLineMat);
      retLine.computeLineDistances();
      this.raysGroup.add(retLine);

      // Return Particle
      const pGeo2 = new THREE.SphereGeometry(0.18, 14, 14);
      const pMat2 = new THREE.MeshBasicMaterial({ color: 0xc084fc });
      const particle2 = new THREE.Mesh(pGeo2, pMat2);
      this.raysGroup.add(particle2);
      this.particlePulses.push({ particle: particle2, curve: retCurve, progress: 0.5, speed: 0.018 });
    });

    const exp1 = this.experts[top2[0].index];
    const exp2 = this.experts[top2[1].index];
    const p1Str = `${Math.round(top2[0].weight * 100)}%`;
    const p2Str = `${Math.round(top2[1].weight * 100)}%`;

    this.onRoutingUpdated({
      expert1: `${exp1.def.name} (${p1Str})`,
      expert2: `${exp2.def.name} (${p2Str})`
    });

    this.onStatusChange(1, `Semantic MoE Gating Router: Activated Top-2 Specialists: ${exp1.def.name} (${p1Str}) & ${exp2.def.name} (${p2Str}) -> Weighted Sum Merge`);
  }

  goToStage(stageIndex) {
    this.currentStage = (stageIndex + this.totalStages) % this.totalStages;
    this.stageTimer = this.currentStage * this.stageDuration;
    this.notifyStageUpdate();
  }

  stepForward() {
    this.pause();
    this.goToStage(this.currentStage + 1);
  }

  stepBackward() {
    this.pause();
    this.goToStage(this.currentStage - 1);
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
      this.stageTimer += 1 * this.speed;

      // Rotate Router & Accumulator Core
      this.routerMesh.rotation.y += 0.02 * this.speed;
      this.routerMesh.rotation.x += 0.01 * this.speed;
      this.routerRing.rotation.z -= 0.015 * this.speed;
      this.accumulatorMesh.rotation.y -= 0.02 * this.speed;

      // Animate Token Queue forward toward Router
      this.tokenQueue.forEach((t) => {
        t.group.position.z += 0.05 * this.speed;
        if (t.group.position.z > 2) {
          t.group.position.z = t.initialZ;
          this.triggerRoutingCycle();
        }
      });

      // Animate Routing Particles along Curves
      this.particlePulses.forEach(p => {
        p.progress += p.speed * this.speed;
        if (p.progress > 1) p.progress = 0;
        const pos = p.curve.getPoint(p.progress);
        p.particle.position.copy(pos);
      });

      // Decay Expert Loads
      this.experts.forEach(exp => {
        exp.load = Math.max(0.1, exp.load - 0.002 * this.speed);
        exp.barMesh.scale.set(1, Math.max(0.2, exp.load * 2), 1);
        exp.mesh.material.emissiveIntensity = Math.max(0.2, exp.load);
      });

      const totalLoopFrames = this.stageDuration * this.totalStages;
      const stageIndex = Math.floor((this.stageTimer % totalLoopFrames) / this.stageDuration);
      if (stageIndex !== this.currentStage) {
        this.currentStage = stageIndex;
        this.notifyStageUpdate();
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  notifyStageUpdate() {
    const descriptions = [
      "Stage 1: Token Stream Input -> Router Gating Network W_g",
      "Stage 2: Top-2 Softmax Gating: Dynamic Specialist Expert Activation",
      "Stage 3: Parallel Expert FFN Computation (Isolated Expert Sub-Networks)",
      "Stage 4: Weighted Sum Convergence into Residual Stream: y = ∑ Pᵢ Eᵢ(x)"
    ];
    const desc = descriptions[this.currentStage] || "";
    this.onStatusChange(this.currentStage, desc);
    this.onStageChanged(this.currentStage);
  }

  setPrompt(text) {
    this.promptText = text;
    this.stageTimer = 0;
    this.currentStage = 0;
    this.processPromptTokens(text);
    this.notifyStageUpdate();
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

