import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class KVCacheSimulation {
  constructor(container, options = {}) {
    this.container = container;
    this.onStatusChange = options.onStatusChange || (() => {});

    this.animationId = null;
    this.isPaused = false;
    this.speed = 1.0;
    this.promptText = options.prompt || "Key Value Cache accelerates transformer inference";

    this.kvCells = [];
    this.tokenNodes = [];
    this.stage = 'prefill'; // 'prefill' -> 'decode'
    this.tokenCount = 0;

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x050814);
    this.scene.fog = new THREE.FogExp2(0x050814, 0.02);

    // Camera
    this.camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    this.camera.position.set(0, 9, 20);

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

    const pointLight = new THREE.PointLight(0x10b981, 2, 50);
    pointLight.position.set(0, 10, 10);
    this.scene.add(pointLight);

    // Grid Floor
    const grid = new THREE.GridHelper(30, 30, 0x1e293b, 0x0f172a);
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
    sprite.scale.set(2.4, 0.6, 1);
    return sprite;
  }

  buildKVCacheGrid() {
    // 3D Key Matrix Bank & Value Matrix Bank Planes
    const bankLabels = [
      { name: "Key Cache Tensor Bank (K)", color: 0x38bdf8, z: -3 },
      { name: "Value Cache Tensor Bank (V)", color: 0x10b981, z: 3 }
    ];

    this.kvCells = [];

    bankLabels.forEach(bank => {
      const planeGeo = new THREE.PlaneGeometry(16, 5);
      const planeMat = new THREE.MeshPhongMaterial({
        color: bank.color,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide
      });
      const plane = new THREE.Mesh(planeGeo, planeMat);
      plane.rotation.x = Math.PI / 2;
      plane.position.set(0, 0, bank.z);
      this.memoryGroup.add(plane);

      // Header Sprite
      const sprite = this.createLabelSprite(bank.name, `#${bank.color.toString(16).padStart(6, '0')}`);
      sprite.position.set(-9, 0, bank.z);
      this.memoryGroup.add(sprite);

      // Grid of 3D Memory Slots (8 slots per bank)
      for (let slot = 0; slot < 8; slot++) {
        const x = -7 + slot * 2;
        const cellGeo = new THREE.BoxGeometry(1.4, 0.8, 1.4);
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

        this.kvCells.push({ bank: bank.name, slot, x, z: bank.z, mesh: cellMesh, active: false });
      }
    });
  }

  processPrompt(text) {
    this.stage = 'prefill';
    this.tokenCount = 0;

    const words = text.trim().split(/\s+/).slice(0, 6);
    this.tokenNodes = words;

    // Fill initial KV cache slots for prompt tokens (Prefill Phase)
    words.forEach((word, idx) => {
      this.activateKVSlot(idx, word);
    });

    this.onStatusChange(1, `Prefill Phase Complete: Computed and stored K & V tensors for ${words.length} prompt tokens into 3D KV Cache memory bank.`);
  }

  activateKVSlot(slotIdx, tokenText) {
    this.kvCells.forEach(cell => {
      if (cell.slot === slotIdx) {
        cell.active = true;
        cell.tokenText = tokenText;
        cell.mesh.material.emissiveIntensity = 0.85;

        // Label token on cell
        if (!cell.label) {
          const sprite = this.createLabelSprite(`[${tokenText}]`, "#10b981");
          sprite.position.set(cell.x, 1.2, cell.z);
          sprite.scale.set(1.4, 0.35, 1);
          this.memoryGroup.add(sprite);
          cell.label = sprite;
        }
      }
    });
  }

  triggerDecodingStep() {
    this.stage = 'decode';
    this.tokenCount++;

    const newTokens = ["is", "fast", "efficient", "scalable", "ready"];
    const newToken = newTokens[this.tokenCount % newTokens.length];

    // Clear old query
    while(this.queryGroup.children.length > 0) {
      this.queryGroup.remove(this.queryGroup.children[0]);
    }
    while(this.laserGroup.children.length > 0) {
      this.laserGroup.remove(this.laserGroup.children[0]);
    }

    // Query node for new single token
    const queryGeo = new THREE.SphereGeometry(0.7, 24, 24);
    const queryMat = new THREE.MeshPhongMaterial({ color: 0xf59e0b, emissive: 0xd97706, emissiveIntensity: 0.9 });
    const queryMesh = new THREE.Mesh(queryGeo, queryMat);
    const queryPos = new THREE.Vector3(0, 5, 0);
    queryMesh.position.copy(queryPos);
    this.queryGroup.add(queryMesh);

    const label = this.createLabelSprite(`New Token Query: "${newToken}"`, "#f59e0b");
    label.position.set(0, 6.3, 0);
    this.queryGroup.add(label);

    // Laser rays querying active KV Cache slots without recomputing past prompt
    this.kvCells.filter(c => c.active).forEach(cell => {
      const targetPos = new THREE.Vector3(cell.x, 0.4, cell.z);
      const points = [queryPos, targetPos];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.7 });
      const line = new THREE.Line(lineGeo, lineMat);
      this.laserGroup.add(line);
    });

    // Append new KV slot for decoded token
    const nextSlot = Math.min(7, this.tokenNodes.length + (this.tokenCount % 3));
    this.activateKVSlot(nextSlot, newToken);

    this.onStatusChange(2, `Decoding Phase (Token ${this.tokenCount}): O(1) single Query vector lookup against cached KV Tensors! FLOPS saved: ~85%`);
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
      // Pulse active KV Cache Memory Cells
      this.kvCells.forEach((cell, i) => {
        if (cell.active) {
          const pulse = 0.6 + Math.sin(frame * 0.05 + i) * 0.25;
          cell.mesh.material.emissiveIntensity = pulse;
        }
      });

      // Periodically trigger autoregressive decoding step
      if (frame % Math.floor(140 / this.speed) === 0) {
        this.triggerDecodingStep();
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  setPrompt(text) {
    this.promptText = text;
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
