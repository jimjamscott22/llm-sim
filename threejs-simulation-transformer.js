import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class TransformerSimulation {
  constructor(container, options = {}) {
    this.container = container;
    this.onStatusChange = options.onStatusChange || (() => {});
    this.onTokenGenerated = options.onTokenGenerated || (() => {});
    
    this.promptText = options.prompt || "Deep learning models transform intelligence";
    this.tokens = [];
    this.generatedTokens = [];
    
    this.animationId = null;
    this.isPaused = false;
    this.speed = 1.0;
    
    // Stages: 0=Tokenize, 1=Embedding, 2=Attention, 3=Layers, 4=Softmax/Sampling
    this.currentStage = 0;
    this.stageTimer = 0;
    this.stageDuration = 120; // frame count per stage at 1.0x speed
    
    this.attentionCurves = [];
    this.particlePulses = [];
    this.softmaxBars = [];

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060913);
    this.scene.fog = new THREE.FogExp2(0x060913, 0.025);

    // Camera
    this.camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    this.camera.position.set(0, 8, 16);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.1; // Don't go below floor

    // Lights
    const ambientLight = new THREE.AmbientLight(0x38bdf8, 0.6);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.5);
    dirLight1.position.set(10, 20, 10);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xc084fc, 1.2);
    dirLight2.position.set(-10, 15, -10);
    this.scene.add(dirLight2);

    // Grid Floor
    const grid = new THREE.GridHelper(30, 30, 0x1e293b, 0x0f172a);
    grid.position.y = -3;
    this.scene.add(grid);

    // Dynamic Groups
    this.tokensGroup = new THREE.Group();
    this.scene.add(this.tokensGroup);

    this.attentionGroup = new THREE.Group();
    this.scene.add(this.attentionGroup);

    this.layersGroup = new THREE.Group();
    this.scene.add(this.layersGroup);

    this.softmaxGroup = new THREE.Group();
    this.scene.add(this.softmaxGroup);

    // Build static architecture elements
    this.buildTransformerArchitecture();
    this.processPrompt(this.promptText);

    // Window resize
    this.onResize = this.onResize.bind(this);
    window.addEventListener('resize', this.onResize);
  }

  buildTransformerArchitecture() {
    // 3D Transformer Stack Planes (Layer 1, Layer 2, Layer 3)
    const layerColors = [0x0284c7, 0x7c3aed, 0xc084fc];
    for (let i = 0; i < 3; i++) {
      const planeGeo = new THREE.PlaneGeometry(16, 8);
      const planeMat = new THREE.MeshPhongMaterial({
        color: layerColors[i],
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
        wireframe: false
      });
      const plane = new THREE.Mesh(planeGeo, planeMat);
      plane.rotation.x = Math.PI / 2;
      plane.position.y = i * 2.2 - 1;
      this.layersGroup.add(plane);

      // Wireframe border
      const edges = new THREE.EdgesGeometry(planeGeo);
      const lineMat = new THREE.LineBasicMaterial({ color: layerColors[i], transparent: true, opacity: 0.4 });
      const border = new THREE.LineSegments(edges, lineMat);
      border.rotation.x = Math.PI / 2;
      border.position.y = i * 2.2 - 1;
      this.layersGroup.add(border);

      // Layer label sprite
      const sprite = this.createPillSprite(`Transformer Layer ${i + 1}`, "#a855f7");
      sprite.position.set(-9, i * 2.2 - 1, 0);
      sprite.scale.set(3, 0.6, 1);
      this.layersGroup.add(sprite);
    }
  }

  createTokenSprite(text, colorStr = '#38bdf8') {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 4;
    
    ctx.beginPath();
    ctx.roundRect(4, 4, 248, 120, 16);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 32px sans-serif';
    ctx.fillStyle = colorStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 64);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(1.8, 0.9, 1);
    return sprite;
  }

  createPillSprite(text, colorStr = '#ffffff') {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 3;
    
    ctx.beginPath();
    ctx.roundRect(4, 4, 504, 88, 20);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 28px sans-serif';
    ctx.fillStyle = colorStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 48);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    return new THREE.Sprite(spriteMat);
  }

  processPrompt(text) {
    // Clear old tokens
    while(this.tokensGroup.children.length > 0) {
      const child = this.tokensGroup.children[0];
      this.tokensGroup.remove(child);
    }

    const words = text.trim().split(/\s+/).slice(0, 6);
    this.tokens = [];
    const spacing = 2.2;
    const startX = -((words.length - 1) * spacing) / 2;

    words.forEach((word, idx) => {
      const group = new THREE.Group();
      
      // 3D Cube Node representing Token Embedding
      const geo = new THREE.BoxGeometry(1.2, 0.8, 1.2);
      const mat = new THREE.MeshPhongMaterial({
        color: 0x0284c7,
        emissive: 0x0369a1,
        transparent: true,
        opacity: 0.85
      });
      const mesh = new THREE.Mesh(geo, mat);
      group.add(mesh);

      // Label
      const sprite = this.createTokenSprite(word, idx === words.length - 1 ? "#f59e0b" : "#38bdf8");
      sprite.position.set(0, 0.9, 0);
      group.add(sprite);

      const pos = new THREE.Vector3(startX + idx * spacing, -2, 0);
      group.position.copy(pos);

      this.tokensGroup.add(group);
      this.tokens.push({ word, group, initialPos: pos.clone(), mesh, sprite, idx });
    });

    this.rebuildAttentionConnections();
    this.rebuildSoftmaxBars();
  }

  rebuildAttentionConnections() {
    // Clear existing attention curves & particles
    while(this.attentionGroup.children.length > 0) {
      this.attentionGroup.remove(this.attentionGroup.children[0]);
    }
    this.attentionCurves = [];
    this.particlePulses = [];

    const num = this.tokens.length;
    if (num < 2) return;

    // Create curved attention arcs between token pairs
    for (let i = 0; i < num; i++) {
      for (let j = 0; j < num; j++) {
        if (i === j) continue;
        
        const weight = Math.min(1.0, 0.2 + (Math.sin(i * 3 + j * 7) + 1) * 0.4);
        if (weight < 0.35) continue; // Only strong attention heads

        const p1 = this.tokens[i].group.position.clone().add(new THREE.Vector3(0, 0.4, 0));
        const p2 = this.tokens[j].group.position.clone().add(new THREE.Vector3(0, 0.4, 0));

        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        mid.y += Math.abs(i - j) * 1.1 + 0.5; // Arc height

        const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
        const points = curve.getPoints(30);
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
        
        const color = i === num - 1 ? 0xf59e0b : 0x818cf8;
        const lineMat = new THREE.LineBasicMaterial({
          color: color,
          transparent: true,
          opacity: weight * 0.75
        });
        
        const line = new THREE.Line(lineGeo, lineMat);
        this.attentionGroup.add(line);

        // Particle packet traveling on curve
        const pGeo = new THREE.SphereGeometry(0.12, 12, 12);
        const pMat = new THREE.MeshBasicMaterial({ color: color });
        const particle = new THREE.Mesh(pGeo, pMat);
        this.attentionGroup.add(particle);

        this.particlePulses.push({
          particle,
          curve,
          progress: Math.random(),
          speed: (0.008 + weight * 0.015)
        });
      }
    }
  }

  rebuildSoftmaxBars() {
    while(this.softmaxGroup.children.length > 0) {
      this.softmaxGroup.remove(this.softmaxGroup.children[0]);
    }
    this.softmaxBars = [];

    // Top k candidate predictions for next token
    const candidates = [
      { word: "intelligence", prob: 0.48, color: 0x10b981 },
      { word: "systems", prob: 0.26, color: 0x3b82f6 },
      { word: "future", prob: 0.14, color: 0xa855f7 },
      { word: "data", prob: 0.12, color: 0x64748b }
    ];

    const barStartX = 4;
    const barZ = -2;

    candidates.forEach((cand, idx) => {
      const group = new THREE.Group();
      const height = cand.prob * 5 + 0.2;
      
      const geo = new THREE.CylinderGeometry(0.35, 0.35, height, 16);
      const mat = new THREE.MeshPhongMaterial({
        color: cand.color,
        emissive: cand.color,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.85
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.y = height / 2;
      group.add(mesh);

      // Label sprite above bar
      const label = this.createTokenSprite(`${cand.word} (${Math.round(cand.prob * 100)}%)`, 
        idx === 0 ? "#10b981" : "#94a3b8");
      label.position.set(0, height + 0.6, 0);
      label.scale.set(1.6, 0.8, 1);
      group.add(label);

      group.position.set(barStartX + idx * 1.8, 4, barZ);
      this.softmaxGroup.add(group);
      this.softmaxBars.push({ cand, group, mesh, label });
    });

    // Header label for Softmax
    const headerSprite = this.createPillSprite("Softmax Logit Probabilities (Top-k)", "#10b981");
    headerSprite.position.set(barStartX + 2.7, 7.5, barZ);
    headerSprite.scale.set(4, 0.7, 1);
    this.softmaxGroup.add(headerSprite);
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

      // Animate attention pulse particles along curves
      this.particlePulses.forEach(item => {
        item.progress += item.speed * this.speed;
        if (item.progress > 1) item.progress = 0;
        const pos = item.curve.getPoint(item.progress);
        item.particle.position.copy(pos);
      });

      // Animate token levitation and layer propagation depending on stage
      const stageIndex = Math.floor((this.stageTimer % (this.stageDuration * 5)) / this.stageDuration);
      
      if (stageIndex !== this.currentStage) {
        this.currentStage = stageIndex;
        this.notifyStageUpdate();
      }

      // Stage specific animations
      if (this.currentStage === 0) {
        // Tokenize & Vector Embedding
        this.tokens.forEach((t, i) => {
          t.group.position.y = t.initialPos.y + Math.sin(this.stageTimer * 0.05 + i) * 0.15;
          t.mesh.material.emissiveIntensity = 0.3;
        });
      } else if (this.currentStage === 1) {
        // Attention calculation (tokens elevate to Transformer Layer 1)
        this.tokens.forEach((t, i) => {
          t.group.position.y = THREE.MathUtils.lerp(t.group.position.y, -1 + Math.sin(this.stageTimer * 0.08 + i) * 0.2, 0.1);
        });
      } else if (this.currentStage === 2) {
        // FeedForward & Norm passing up to Layer 3
        this.tokens.forEach((t, i) => {
          t.group.position.y = THREE.MathUtils.lerp(t.group.position.y, 3.4 + Math.sin(this.stageTimer * 0.08 + i) * 0.2, 0.1);
          t.mesh.material.emissiveIntensity = 0.8;
        });
      } else if (this.currentStage === 3) {
        // Softmax & Next Token Sampling
        this.softmaxBars.forEach((bar, i) => {
          const pulse = 1 + (i === 0 ? Math.sin(this.stageTimer * 0.15) * 0.1 : 0);
          bar.mesh.scale.set(pulse, 1, pulse);
        });
      } else if (this.currentStage === 4) {
        // Autoregressive Token Generation Trigger
        if (Math.floor(this.stageTimer) % this.stageDuration === 0) {
          this.appendGeneratedToken();
        }
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  appendGeneratedToken() {
    const candidateWords = ["intelligence", "networks", "systems", "capabilities", "models"];
    const nextWord = candidateWords[Math.floor(Math.random() * candidateWords.length)];
    this.promptText += " " + nextWord;
    this.generatedTokens.push(nextWord);
    this.onTokenGenerated(nextWord, this.promptText);

    // Reprocess full token stream dynamically
    this.processPrompt(this.promptText);
  }

  notifyStageUpdate() {
    const descriptions = [
      "Stage 1: Tokenization & Embedding Space Mapping",
      "Stage 2: Multi-Head Self-Attention Calculation",
      "Stage 3: Stacked Transformer Layer Propagation",
      "Stage 4: Logits & Softmax Probability Distribution",
      "Stage 5: Autoregressive Next-Token Sampling & Feedback Loop"
    ];
    this.onStatusChange(this.currentStage, descriptions[this.currentStage] || "");
  }

  setPrompt(newText) {
    this.promptText = newText;
    this.stageTimer = 0;
    this.currentStage = 0;
    this.processPrompt(newText);
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
