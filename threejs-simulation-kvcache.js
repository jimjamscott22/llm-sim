import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class KVCacheSimulation {
  constructor(container, options = {}) {
    this.container = container;
    this.onStatusChange = options.onStatusChange || (() => {});
    this.onStageChanged = options.onStageChanged || (() => {});
    this.onMetricsUpdated = options.onMetricsUpdated || (() => {});

    this.animationId = null;
    this.isPaused = false;
    this.speed = 1.0;
    this.promptText = options.prompt || "Key Value Cache accelerates transformer inference";

    this.kvCells = [];
    this.tokenNodes = [];
    this.tokenCount = 0;
    this.mode = 'with_cache'; // 'with_cache' vs 'without_cache'

    // Stages: 0=Prefill Phase (Prompt K/V Cache Ingestion), 1=Autoregressive Step 1, 2=Autoregressive Step 2, 3=Autoregressive Step 3
    this.currentStage = 0;
    this.totalStages = 4;
    this.stageTimer = 0;
    this.stageDuration = 140;

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x050814);
    this.scene.fog = new THREE.FogExp2(0x050814, 0.018);

    // Camera
    this.camera = new THREE.PerspectiveCamera(54, width / height, 0.1, 1000);
    this.camera.position.set(0, 10, 21);

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

    const pointLight = new THREE.PointLight(0x10b981, 2.2, 55);
    pointLight.position.set(0, 12, 10);
    this.scene.add(pointLight);

    // Grid Floor
    const grid = new THREE.GridHelper(32, 32, 0x1e293b, 0x0f172a);
    grid.position.y = -4;
    this.scene.add(grid);

    // Groups
    this.memoryGroup = new THREE.Group();
    this.scene.add(this.memoryGroup);

    this.queryGroup = new THREE.Group();
    this.scene.add(this.queryGroup);

    this.laserGroup = new THREE.Group();
    this.scene.add(this.laserGroup);

    // Build KV Cache Memory Grid Architecture
    this.buildKVCacheGrid();
    this.processPrompt(this.promptText);

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
    sprite.scale.set(2.4, 0.6, 1);
    return sprite;
  }

  buildKVCacheGrid() {
    while(this.memoryGroup.children.length > 0) {
      this.memoryGroup.remove(this.memoryGroup.children[0]);
    }

    const bankLabels = [
      { name: "Key Cache Tensor Bank (K)", color: 0x38bdf8, z: -3.5 },
      { name: "Value Cache Tensor Bank (V)", color: 0x10b981, z: 3.5 }
    ];

    this.kvCells = [];

    bankLabels.forEach(bank => {
      const planeGeo = new THREE.PlaneGeometry(17, 5);
      const planeMat = new THREE.MeshPhongMaterial({
        color: bank.color,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide
      });
      const plane = new THREE.Mesh(planeGeo, planeMat);
      plane.rotation.x = Math.PI / 2;
      plane.position.set(0, 0, bank.z);
      this.memoryGroup.add(plane);

      // Header Sprite
      const sprite = this.createLabelSprite(bank.name, `#${bank.color.toString(16).padStart(6, '0')}`);
      sprite.position.set(-9.6, 0, bank.z);
      sprite.scale.set(3.2, 0.7, 1);
      this.memoryGroup.add(sprite);

      // Grid of 3D Memory Slots (8 slots per bank)
      for (let slot = 0; slot < 8; slot++) {
        const x = -7 + slot * 2.0;
        const cellGeo = new THREE.BoxGeometry(1.5, 0.8, 1.5);
        const cellMat = new THREE.MeshPhongMaterial({
          color: bank.color,
          emissive: bank.color,
          emissiveIntensity: 0.2,
          transparent: true,
          opacity: 0.75
        });
        const cellMesh = new THREE.Mesh(cellGeo, cellMat);
        cellMesh.position.set(x, 0.4, bank.z);
        this.memoryGroup.add(cellMesh);

        // Cell border
        const edges = new THREE.EdgesGeometry(cellGeo);
        const lineMat = new THREE.LineBasicMaterial({ color: bank.color, transparent: true, opacity: 0.5 });
        cellMesh.add(new THREE.LineSegments(edges, lineMat));

        this.kvCells.push({ bank: bank.name, slot, x, z: bank.z, mesh: cellMesh, active: false, label: null });
      }
    });
  }

  processPrompt(text) {
    this.tokenCount = 0;
    this.currentStage = 0;

    // Reset memory cells
    this.kvCells.forEach(cell => {
      cell.active = false;
      cell.tokenText = "";
      cell.mesh.material.emissiveIntensity = 0.2;
      if (cell.label) {
        this.memoryGroup.remove(cell.label);
        cell.label = null;
      }
    });

    const words = text.trim().split(/\s+/).slice(0, 5);
    this.tokenNodes = words;

    // Fill initial KV cache slots for prompt tokens (Prefill Phase)
    words.forEach((word, idx) => {
      this.activateKVSlot(idx, word);
    });

    this.updateMetrics();
    this.notifyStageUpdate();
  }

  activateKVSlot(slotIdx, tokenText) {
    this.kvCells.forEach(cell => {
      if (cell.slot === slotIdx) {
        cell.active = true;
        cell.tokenText = tokenText;
        cell.mesh.material.emissiveIntensity = 0.85;

        if (!cell.label) {
          const sprite = this.createLabelSprite(`[${tokenText}]`, cell.bank.includes("(K)") ? "#38bdf8" : "#10b981");
          sprite.position.set(cell.x, 1.25, cell.z);
          sprite.scale.set(1.4, 0.35, 1);
          this.memoryGroup.add(sprite);
          cell.label = sprite;
        }
      }
    });
  }

  setMode(newMode) {
    this.mode = newMode;
    this.triggerDecodingStep(this.tokenCount);
  }

  toggleMode() {
    this.setMode(this.mode === 'with_cache' ? 'without_cache' : 'with_cache');
  }

  triggerDecodingStep(stepIndex) {
    this.tokenCount = stepIndex;

    const newTokens = ["accelerates", "transformer", "latency", "throughput", "efficient"];
    const newToken = newTokens[this.tokenCount % newTokens.length];

    // Clear old queries and lasers
    while(this.queryGroup.children.length > 0) {
      this.queryGroup.remove(this.queryGroup.children[0]);
    }
    while(this.laserGroup.children.length > 0) {
      this.laserGroup.remove(this.laserGroup.children[0]);
    }

    const activeSlots = this.kvCells.filter(c => c.active && c.bank.includes("(K)"));

    if (this.mode === 'with_cache') {
      // WITH KV CACHE: Single new Query token shoots O(1) direct lookup rays
      const queryGeo = new THREE.SphereGeometry(0.7, 24, 24);
      const queryMat = new THREE.MeshPhongMaterial({ color: 0x10b981, emissive: 0x059669, emissiveIntensity: 0.9 });
      const queryMesh = new THREE.Mesh(queryGeo, queryMat);
      const queryPos = new THREE.Vector3(0, 5.2, 0);
      queryMesh.position.copy(queryPos);
      this.queryGroup.add(queryMesh);

      const label = this.createLabelSprite(`Single Query Q_new: "${newToken}" [O(1) Step]`, "#10b981");
      label.position.set(0, 6.6, 0);
      label.scale.set(3.4, 0.75, 1);
      this.queryGroup.add(label);

      // Clean Laser lines directly to cached KV slots
      this.kvCells.filter(c => c.active).forEach(cell => {
        const targetPos = new THREE.Vector3(cell.x, 0.4, cell.z);
        const points = [queryPos, targetPos];
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.8, linewidth: 2 });
        this.laserGroup.add(new THREE.Line(lineGeo, lineMat));
      });
    } else {
      // WITHOUT KV CACHE: Full O(N^2) quadratic recomputation storm!
      // Must re-query all past tokens Q_1, Q_2, ... Q_new against each other!
      const totalTokens = Math.min(8, this.tokenNodes.length + this.tokenCount);
      
      for (let qIdx = 0; qIdx < totalTokens; qIdx++) {
        const qX = -7 + qIdx * 2.0;
        const qPos = new THREE.Vector3(qX, 5.2, (Math.sin(qIdx) * 1.5));

        const queryGeo = new THREE.SphereGeometry(0.35, 16, 16);
        const queryMat = new THREE.MeshPhongMaterial({ color: 0xf43f5e, emissive: 0xe11d48, emissiveIntensity: 0.9 });
        const qMesh = new THREE.Mesh(queryGeo, queryMat);
        qMesh.position.copy(qPos);
        this.queryGroup.add(qMesh);

        // Criss-cross laser storm to all prior tokens
        for (let kIdx = 0; kIdx <= qIdx; kIdx++) {
          this.kvCells.filter(c => c.slot === kIdx).forEach(cell => {
            const targetPos = new THREE.Vector3(cell.x, 0.4, cell.z);
            const points = [qPos, targetPos];
            const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
            const lineMat = new THREE.LineBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0.65 });
            this.laserGroup.add(new THREE.Line(lineGeo, lineMat));
          });
        }
      }

      // Warning Header
      const warningLabel = this.createLabelSprite(`WITHOUT CACHE: O(N²) Quadratic Recomputation Storm!`, "#f43f5e");
      warningLabel.position.set(0, 7.2, 0);
      warningLabel.scale.set(4.4, 0.8, 1);
      this.queryGroup.add(warningLabel);
    }

    // Append new KV slot for decoded token
    const nextSlot = Math.min(7, this.tokenNodes.length + this.tokenCount);
    this.activateKVSlot(nextSlot, newToken);

    this.updateMetrics();
  }

  updateMetrics() {
    const totalTokens = Math.min(8, this.tokenNodes.length + this.tokenCount);
    const vramBytes = totalTokens * 2 * 32 * 128 * 2; // (tokens * 2(K,V) * layers * d_model * fp16)
    const vramKB = Math.round(vramBytes / 1024);

    const isCached = this.mode === 'with_cache';
    const complexity = isCached ? "O(1) Step Lookup" : "O(N²) Full Recompute";
    const flops = isCached ? "~87% Saved" : "0% (Wasted FLOPS)";
    const vramStr = isCached ? `${vramKB} KB active` : "0 KB (No Cache)";

    this.onMetricsUpdated({
      complexity,
      flops,
      vram: vramStr,
      mode: this.mode
    });

    const statusMsg = isCached
      ? `⚡ With KV Cache: Decoded Token ${this.tokenCount + 1} using single Query lookup against cached K/V tensors. ${flops}`
      : `💥 WITHOUT KV Cache: Decoded Token ${this.tokenCount + 1} by recomputing all ${totalTokens} tokens across the full quadratic sequence!`;

    this.onStatusChange(this.currentStage, statusMsg);
  }

  goToStage(stageIndex) {
    this.currentStage = (stageIndex + this.totalStages) % this.totalStages;
    this.stageTimer = this.currentStage * this.stageDuration;
    this.triggerDecodingStep(this.currentStage);
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
    let frame = 0;
    const loop = () => {
      this.animationId = requestAnimationFrame(loop);
      this.render(frame++);
    };
    loop();
  }

  render(frame) {
    this.controls.update();

    if (!this.isPaused) {
      this.stageTimer += 1 * this.speed;

      // Pulse active KV Cache Memory Cells
      this.kvCells.forEach((cell, i) => {
        if (cell.active) {
          const pulse = 0.6 + Math.sin(this.stageTimer * 0.05 + i) * 0.25;
          cell.mesh.material.emissiveIntensity = pulse;
        }
      });

      const totalLoopFrames = this.stageDuration * this.totalStages;
      const stageIndex = Math.floor((this.stageTimer % totalLoopFrames) / this.stageDuration);
      
      if (stageIndex !== this.currentStage) {
        this.currentStage = stageIndex;
        this.triggerDecodingStep(this.currentStage);
        this.notifyStageUpdate();
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  notifyStageUpdate() {
    const descriptions = [
      "Stage 1: Prefill Phase — Compute and ingest initial Prompt K/V Tensors into Cache Banks",
      "Stage 2: Autoregressive Step 1 — Evaluate Query token against cached tensors",
      "Stage 3: Autoregressive Step 2 — Append newly generated token K/V to memory bank",
      "Stage 4: Autoregressive Step 3 — Scale sequence length without quadratic FLOPS explosion"
    ];
    const desc = descriptions[this.currentStage] || "";
    this.onStageChanged(this.currentStage);
  }

  setPrompt(text) {
    this.promptText = text;
    this.stageTimer = 0;
    this.processPrompt(text);
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

