import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class TransformerSimulation {
  constructor(container, options = {}) {
    this.container = container;
    this.onStatusChange = options.onStatusChange || (() => {});
    this.onTokenGenerated = options.onTokenGenerated || (() => {});
    this.onStageChanged = options.onStageChanged || (() => {});
    
    this.promptText = options.prompt || "Deep learning models transform intelligence";
    this.tokens = [];
    this.generatedTokens = [];
    
    this.animationId = null;
    this.isPaused = false;
    this.speed = 1.0;
    
    // Stages: 
    // 0: Tokenize & Embedding
    // 1: Q, K, V Projections
    // 2: Multi-Head Causal Self-Attention (QK^T / sqrt(d_k))
    // 3: Layer Norm & Feed-Forward Network (FFN / MLP)
    // 4: Logits, Softmax & Temperature/Top-P Sampling
    // 5: Autoregressive Token Generation
    this.currentStage = 0;
    this.totalStages = 6;
    this.stageTimer = 0;
    this.stageDuration = 140; // frames per stage at 1.0x speed
    
    // Sampling Hyperparameters
    this.temperature = 0.7;
    this.topP = 0.9;
    
    this.attentionCurves = [];
    this.particlePulses = [];
    this.softmaxBars = [];
    this.qkvMeshes = [];

    // Pre-defined vocabulary candidate logits for simulation
    this.baseCandidates = [
      { word: "intelligence", logit: 4.2, color: 0x10b981 },
      { word: "systems", logit: 3.5, color: 0x38bdf8 },
      { word: "capabilities", logit: 2.7, color: 0xc084fc },
      { word: "representations", logit: 2.1, color: 0xf59e0b },
      { word: "networks", logit: 1.4, color: 0x06b6d4 },
      { word: "parameters", logit: 0.6, color: 0x64748b }
    ];

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060913);
    this.scene.fog = new THREE.FogExp2(0x060913, 0.022);

    // Camera
    this.camera = new THREE.PerspectiveCamera(52, width / height, 0.1, 1000);
    this.camera.position.set(0, 9, 18);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.1;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x38bdf8, 0.65);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.6);
    dirLight1.position.set(10, 20, 10);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xc084fc, 1.3);
    dirLight2.position.set(-10, 15, -10);
    this.scene.add(dirLight2);

    // Grid Floor
    const grid = new THREE.GridHelper(32, 32, 0x1e293b, 0x0f172a);
    grid.position.y = -3.5;
    this.scene.add(grid);

    // Dynamic Groups
    this.tokensGroup = new THREE.Group();
    this.scene.add(this.tokensGroup);

    this.qkvGroup = new THREE.Group();
    this.scene.add(this.qkvGroup);

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
    // 3D Transformer Stack Planes (Layer 1: Self-Attn, Layer 2: FFN/MLP, Layer 3: Output Logits)
    const layerDefs = [
      { name: "Layer 1: Multi-Head Self-Attention", color: 0x0284c7, y: -0.5 },
      { name: "Layer 2: Feed-Forward Network (MLP & Norm)", color: 0x7c3aed, y: 1.8 },
      { name: "Layer 3: Unembedding Projection & Logits", color: 0xc084fc, y: 4.1 }
    ];

    layerDefs.forEach(layer => {
      const planeGeo = new THREE.PlaneGeometry(16, 7.5);
      const planeMat = new THREE.MeshPhongMaterial({
        color: layer.color,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide
      });
      const plane = new THREE.Mesh(planeGeo, planeMat);
      plane.rotation.x = Math.PI / 2;
      plane.position.y = layer.y;
      this.layersGroup.add(plane);

      // Wireframe border
      const edges = new THREE.EdgesGeometry(planeGeo);
      const lineMat = new THREE.LineBasicMaterial({ color: layer.color, transparent: true, opacity: 0.45 });
      const border = new THREE.LineSegments(edges, lineMat);
      border.rotation.x = Math.PI / 2;
      border.position.y = layer.y;
      this.layersGroup.add(border);

      // Layer label sprite
      const sprite = this.createPillSprite(layer.name, "#94a3b8");
      sprite.position.set(-9.2, layer.y, 0);
      sprite.scale.set(3.4, 0.6, 1);
      this.layersGroup.add(sprite);
    });
  }

  createTokenSprite(text, colorStr = '#38bdf8') {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 4;
    
    ctx.beginPath();
    ctx.roundRect(4, 4, 248, 120, 16);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 30px sans-serif';
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

    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 3;
    
    ctx.beginPath();
    ctx.roundRect(4, 4, 504, 88, 20);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 24px sans-serif';
    ctx.fillStyle = colorStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 48);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    return new THREE.Sprite(spriteMat);
  }

  createMiniBadgeSprite(text, colorStr = '#ffffff') {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 3;
    
    ctx.beginPath();
    ctx.roundRect(3, 3, 122, 58, 12);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 28px monospace';
    ctx.fillStyle = colorStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(0.6, 0.3, 1);
    return sprite;
  }

  processPrompt(text) {
    // Clear old tokens and QKV elements
    while(this.tokensGroup.children.length > 0) {
      this.tokensGroup.remove(this.tokensGroup.children[0]);
    }
    while(this.qkvGroup.children.length > 0) {
      this.qkvGroup.remove(this.qkvGroup.children[0]);
    }

    const words = text.trim().split(/\s+/).slice(0, 6);
    this.tokens = [];
    this.qkvMeshes = [];
    const spacing = 2.3;
    const startX = -((words.length - 1) * spacing) / 2;

    words.forEach((word, idx) => {
      const group = new THREE.Group();
      
      // 3D Cube Node representing Token Embedding (Residual Stream Vector)
      const geo = new THREE.BoxGeometry(1.3, 0.7, 1.3);
      const mat = new THREE.MeshPhongMaterial({
        color: 0x0284c7,
        emissive: 0x0369a1,
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0.9
      });
      const mesh = new THREE.Mesh(geo, mat);
      group.add(mesh);

      // Label sprite
      const isLast = idx === words.length - 1;
      const sprite = this.createTokenSprite(word, isLast ? "#f59e0b" : "#38bdf8");
      sprite.position.set(0, 0.85, 0);
      group.add(sprite);

      const pos = new THREE.Vector3(startX + idx * spacing, -2.5, 0);
      group.position.copy(pos);

      this.tokensGroup.add(group);

      // Explicit Q, K, V Projection Blocks attached underneath each token
      const qkvSubGroup = new THREE.Group();
      
      // Q (Query) block - Amber
      const qGeo = new THREE.BoxGeometry(0.35, 0.35, 0.35);
      const qMat = new THREE.MeshPhongMaterial({ color: 0xf59e0b, emissive: 0xd97706, emissiveIntensity: 0.5 });
      const qMesh = new THREE.Mesh(qGeo, qMat);
      qMesh.position.set(-0.4, -0.65, 0);
      qkvSubGroup.add(qMesh);
      const qBadge = this.createMiniBadgeSprite("Q", "#f59e0b");
      qBadge.position.set(-0.4, -1.0, 0);
      qkvSubGroup.add(qBadge);

      // K (Key) block - Cyan
      const kGeo = new THREE.BoxGeometry(0.35, 0.35, 0.35);
      const kMat = new THREE.MeshPhongMaterial({ color: 0x38bdf8, emissive: 0x0284c7, emissiveIntensity: 0.5 });
      const kMesh = new THREE.Mesh(kGeo, kMat);
      kMesh.position.set(0, -0.65, 0);
      qkvSubGroup.add(kMesh);
      const kBadge = this.createMiniBadgeSprite("K", "#38bdf8");
      kBadge.position.set(0, -1.0, 0);
      qkvSubGroup.add(kBadge);

      // V (Value) block - Emerald
      const vGeo = new THREE.BoxGeometry(0.35, 0.35, 0.35);
      const vMat = new THREE.MeshPhongMaterial({ color: 0x10b981, emissive: 0x059669, emissiveIntensity: 0.5 });
      const vMesh = new THREE.Mesh(vGeo, vMat);
      vMesh.position.set(0.4, -0.65, 0);
      qkvSubGroup.add(vMesh);
      const vBadge = this.createMiniBadgeSprite("V", "#10b981");
      vBadge.position.set(0.4, -1.0, 0);
      qkvSubGroup.add(vBadge);

      qkvSubGroup.position.copy(pos);
      this.qkvGroup.add(qkvSubGroup);

      this.tokens.push({ 
        word, 
        group, 
        initialPos: pos.clone(), 
        mesh, 
        sprite, 
        idx,
        qkvSubGroup,
        qMesh, kMesh, vMesh
      });
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

    // Build Causal Self-Attention Arcs (j <= i: token can only attend to prior or current tokens)
    for (let i = 0; i < num; i++) {
      for (let j = 0; j <= i; j++) {
        if (i === j) continue;

        // Compute simulated dot-product attention affinity score (QK^T / sqrt(d_k))
        // Tokens with closer contextual distance get higher attention
        const tokenDistance = Math.abs(i - j);
        const dotProductAffinity = Math.max(0.15, 1.0 - (tokenDistance * 0.22) + (Math.sin(i * 2 + j * 5) * 0.15));
        
        if (dotProductAffinity < 0.25) continue;

        const p1 = this.tokens[i].group.position.clone().add(new THREE.Vector3(0, 0.4, 0));
        const p2 = this.tokens[j].group.position.clone().add(new THREE.Vector3(0, 0.4, 0));

        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        mid.y += (i - j) * 0.9 + 0.6; // Arc height

        const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
        const points = curve.getPoints(24);
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
        
        const isTargetToken = i === num - 1;
        const color = isTargetToken ? 0xf59e0b : 0x38bdf8;
        const lineMat = new THREE.LineBasicMaterial({
          color: color,
          transparent: true,
          opacity: dotProductAffinity * 0.8
        });
        
        const line = new THREE.Line(lineGeo, lineMat);
        this.attentionGroup.add(line);

        // Particle packet traveling along Q->K attention path
        const pGeo = new THREE.SphereGeometry(0.1, 12, 12);
        const pMat = new THREE.MeshBasicMaterial({ color: color });
        const particle = new THREE.Mesh(pGeo, pMat);
        this.attentionGroup.add(particle);

        this.particlePulses.push({
          particle,
          curve,
          progress: Math.random(),
          speed: (0.01 + dotProductAffinity * 0.015)
        });
      }
    }
  }

  rebuildSoftmaxBars() {
    while(this.softmaxGroup.children.length > 0) {
      this.softmaxGroup.remove(this.softmaxGroup.children[0]);
    }
    this.softmaxBars = [];

    // Calculate Softmax with Temperature (T) and Top-P (Nucleus) filter
    // 1. Scale logits by Temperature: z_i / T
    const scaledLogits = this.baseCandidates.map(c => ({
      ...c,
      scaledLogit: c.logit / Math.max(0.05, this.temperature)
    }));

    // Max logit subtraction for numerical stability
    const maxLogit = Math.max(...scaledLogits.map(c => c.scaledLogit));
    const expVals = scaledLogits.map(c => Math.exp(c.scaledLogit - maxLogit));
    const sumExp = expVals.reduce((a, b) => a + b, 0);
    
    // Raw softmax probabilities
    let probs = scaledLogits.map((c, i) => ({
      ...c,
      prob: expVals[i] / sumExp
    }));

    // Sort descending for Top-P Nucleus thresholding
    probs.sort((a, b) => b.prob - a.prob);

    // Apply Top-P cumulative mask
    let cumProb = 0;
    probs = probs.map(cand => {
      cumProb += cand.prob;
      const inTopP = cumProb - cand.prob < this.topP;
      return { ...cand, inTopP };
    });

    const barStartX = 4.2;
    const barZ = -2;

    probs.slice(0, 5).forEach((cand, idx) => {
      const group = new THREE.Group();
      const height = Math.max(0.2, cand.prob * 5.5);
      
      const geo = new THREE.CylinderGeometry(0.3, 0.3, height, 16);
      const barColor = cand.inTopP ? cand.color : 0x475569;
      const mat = new THREE.MeshPhongMaterial({
        color: barColor,
        emissive: barColor,
        emissiveIntensity: cand.inTopP ? 0.35 : 0.05,
        transparent: true,
        opacity: cand.inTopP ? 0.9 : 0.35
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.y = height / 2;
      group.add(mesh);

      // Label sprite above bar
      const pctStr = `${Math.round(cand.prob * 100)}%`;
      const labelText = cand.inTopP ? `${cand.word} (${pctStr})` : `[P-filtered] ${cand.word}`;
      const labelColor = cand.inTopP ? (idx === 0 ? "#10b981" : "#38bdf8") : "#64748b";
      const label = this.createTokenSprite(labelText, labelColor);
      label.position.set(0, height + 0.6, 0);
      label.scale.set(1.6, 0.75, 1);
      group.add(label);

      group.position.set(barStartX + idx * 1.6, 4.4, barZ);
      this.softmaxGroup.add(group);
      this.softmaxBars.push({ cand, group, mesh, label, inTopP: cand.inTopP });
    });

    // Softmax Title Pill Sprite
    const headerSprite = this.createPillSprite(
      `Softmax (T=${this.temperature.toFixed(2)}, Top-P=${this.topP.toFixed(2)})`, 
      "#10b981"
    );
    headerSprite.position.set(barStartX + 3.2, 7.8, barZ);
    headerSprite.scale.set(4.4, 0.7, 1);
    this.softmaxGroup.add(headerSprite);
  }

  setTemperature(val) {
    this.temperature = parseFloat(val);
    this.rebuildSoftmaxBars();
  }

  setTopP(val) {
    this.topP = parseFloat(val);
    this.rebuildSoftmaxBars();
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

      // Animate attention pulse particles along curves
      this.particlePulses.forEach(item => {
        item.progress += item.speed * this.speed;
        if (item.progress > 1) item.progress = 0;
        const pos = item.curve.getPoint(item.progress);
        item.particle.position.copy(pos);
      });

      const totalLoopFrames = this.stageDuration * this.totalStages;
      const stageIndex = Math.floor((this.stageTimer % totalLoopFrames) / this.stageDuration);
      
      if (stageIndex !== this.currentStage) {
        this.currentStage = stageIndex;
        this.notifyStageUpdate();
      }

      // Stage-specific visual behaviors
      this.applyStageVisuals(this.currentStage);
    } else {
      // In paused/stepped state, hold visual configuration
      this.applyStageVisuals(this.currentStage);
    }

    this.renderer.render(this.scene, this.camera);
  }

  applyStageVisuals(stage) {
    if (stage === 0) {
      // 0: Tokenization & Word Vector Embedding
      this.tokens.forEach((t, i) => {
        t.group.position.y = THREE.MathUtils.lerp(t.group.position.y, t.initialPos.y + Math.sin(this.stageTimer * 0.05 + i) * 0.1, 0.1);
        t.qkvSubGroup.position.y = t.group.position.y;
        t.qkvSubGroup.visible = false;
        t.mesh.material.emissiveIntensity = 0.3;
      });
      this.attentionGroup.visible = false;
    } else if (stage === 1) {
      // 1: Q, K, V Projections
      this.tokens.forEach((t, i) => {
        t.group.position.y = THREE.MathUtils.lerp(t.group.position.y, -1.8, 0.1);
        t.qkvSubGroup.position.y = t.group.position.y;
        t.qkvSubGroup.visible = true;
        // Pulse QKV Blocks
        const pulse = 1 + Math.sin(this.stageTimer * 0.1 + i) * 0.15;
        t.qMesh.scale.set(pulse, pulse, pulse);
        t.kMesh.scale.set(pulse, pulse, pulse);
        t.vMesh.scale.set(pulse, pulse, pulse);
      });
      this.attentionGroup.visible = false;
    } else if (stage === 2) {
      // 2: Multi-Head Causal Self-Attention (Layer 1)
      this.tokens.forEach((t, i) => {
        t.group.position.y = THREE.MathUtils.lerp(t.group.position.y, -0.5 + Math.sin(this.stageTimer * 0.06 + i) * 0.15, 0.1);
        t.qkvSubGroup.position.y = t.group.position.y;
        t.qkvSubGroup.visible = true;
      });
      this.attentionGroup.visible = true;
    } else if (stage === 3) {
      // 3: Feed-Forward MLP & Add/Norm (Layer 2)
      this.tokens.forEach((t, i) => {
        t.group.position.y = THREE.MathUtils.lerp(t.group.position.y, 1.8 + Math.sin(this.stageTimer * 0.06 + i) * 0.15, 0.1);
        t.qkvSubGroup.position.y = t.group.position.y;
        t.qkvSubGroup.visible = false;
        t.mesh.material.emissiveIntensity = 0.8;
      });
      this.attentionGroup.visible = false;
    } else if (stage === 4) {
      // 4: Logits, Softmax & Temperature/Top-P Sampling (Layer 3)
      this.tokens.forEach((t, i) => {
        t.group.position.y = THREE.MathUtils.lerp(t.group.position.y, 4.1, 0.1);
        t.qkvSubGroup.position.y = t.group.position.y;
        t.qkvSubGroup.visible = false;
      });
      this.softmaxBars.forEach((bar, i) => {
        if (bar.inTopP && i === 0) {
          const pulse = 1 + Math.sin(this.stageTimer * 0.15) * 0.08;
          bar.mesh.scale.set(pulse, 1, pulse);
        }
      });
    } else if (stage === 5) {
      // 5: Autoregressive Next-Token Sampling & Feedback Loop
      if (!this.isPaused && Math.floor(this.stageTimer) % this.stageDuration === 0) {
        this.appendGeneratedToken();
      }
    }
  }

  appendGeneratedToken() {
    const candidateWords = ["intelligence", "systems", "capabilities", "representations", "networks"];
    const nextWord = candidateWords[Math.floor(Math.random() * candidateWords.length)];
    this.promptText += " " + nextWord;
    this.generatedTokens.push(nextWord);
    this.onTokenGenerated(nextWord, this.promptText);

    this.processPrompt(this.promptText);
  }

  notifyStageUpdate() {
    const descriptions = [
      "Stage 1: Tokenization & Subword Vector Embeddings (Residual Stream)",
      "Stage 2: Q (Query), K (Key), and V (Value) Linear Projections",
      "Stage 3: Multi-Head Causal Self-Attention: Softmax(QK^T / √d_k) · V",
      "Stage 4: Residual Add & LayerNorm -> Feed-Forward Network (MLP)",
      "Stage 5: Unembedding Matrix -> Logits -> Softmax Temperature & Top-P Sampling",
      "Stage 6: Autoregressive Next-Token Generation & Sequence Loopback"
    ];
    const desc = descriptions[this.currentStage] || "";
    this.onStatusChange(this.currentStage, desc);
    this.onStageChanged(this.currentStage);
  }

  setPrompt(newText) {
    this.promptText = newText;
    this.stageTimer = 0;
    this.currentStage = 0;
    this.processPrompt(newText);
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

