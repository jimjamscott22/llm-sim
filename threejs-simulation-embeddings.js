import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class EmbeddingSimulation {
  constructor(container, options = {}) {
    this.container = container;
    this.onStatusChange = options.onStatusChange || (() => {});
    
    this.animationId = null;
    this.isPaused = false;
    this.speed = 1.0;
    this.queryText = options.query || "Retrieval Augmented Generation with Vector Database";

    this.nodes = [];
    this.topKLines = [];
    this.radarWaveRadius = 0;
    this.radarActive = false;

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x050714);
    this.scene.fog = new THREE.FogExp2(0x050714, 0.02);

    // Camera
    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    this.camera.position.set(0, 10, 20);

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

    const pointLight = new THREE.PointLight(0x38bdf8, 2, 50);
    pointLight.position.set(0, 15, 10);
    this.scene.add(pointLight);

    // 3D Axis Helper & Vector Grid
    const grid = new THREE.GridHelper(30, 30, 0x1e293b, 0x0f172a);
    grid.position.y = -5;
    this.scene.add(grid);

    // Axis visualizers
    this.createAxisLabels();

    // Groups
    this.nodesGroup = new THREE.Group();
    this.scene.add(this.nodesGroup);

    this.linesGroup = new THREE.Group();
    this.scene.add(this.linesGroup);

    // Build Vector Dataset & Query Node
    this.populateVectorSpace();
    this.createRadarPulse();
    this.performRAGSearch(this.queryText);

    // Window Resize
    this.onResize = this.onResize.bind(this);
    window.addEventListener('resize', this.onResize);
  }

  createLabelSprite(text, colorStr = '#ffffff') {
    const canvas = document.createElement('canvas');
    canvas.width = 384;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
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

  createAxisLabels() {
    const xLabel = this.createLabelSprite("Dimension 1: Semantic Topic", "#38bdf8");
    xLabel.position.set(12, -4.5, 0);
    this.scene.add(xLabel);

    const yLabel = this.createLabelSprite("Dimension 2: Abstraction Level", "#a855f7");
    yLabel.position.set(0, 10, 0);
    this.scene.add(yLabel);

    const zLabel = this.createLabelSprite("Dimension 3: Domain Cluster", "#10b981");
    zLabel.position.set(0, -4.5, 12);
    this.scene.add(zLabel);
  }

  populateVectorSpace() {
    // Clusters: 0=AI/LLM, 1=Database/RAG, 2=Code/System, 3=Physics
    const clusters = [
      { name: "AI & LLM Docs", color: 0x38bdf8, center: new THREE.Vector3(-6, 2, -4), items: ["Transformer Architecture", "Attention Mechanism", "Prompt Engineering", "Fine-Tuning LoRA"] },
      { name: "RAG & Vector DB Docs", color: 0xf59e0b, center: new THREE.Vector3(2, 3, 2), items: ["HNSW Vector Index", "Cosine Similarity", "Embedding Chunking", "Hybrid Dense Search"] },
      { name: "System & Code Docs", color: 0x10b981, center: new THREE.Vector3(6, -1, -5), items: ["FastAPI Backend", "Async Event Loop", "Docker Container", "CUDA GPU Memory"] },
      { name: "General Knowledge", color: 0xa855f7, center: new THREE.Vector3(-4, -2, 6), items: ["Quantum Physics", "Graph Theory", "Thermodynamics", "Information Theory"] }
    ];

    this.nodes = [];

    clusters.forEach(cluster => {
      cluster.items.forEach(item => {
        const offset = new THREE.Vector3(
          (Math.random() - 0.5) * 4,
          (Math.random() - 0.5) * 3,
          (Math.random() - 0.5) * 4
        );
        const pos = cluster.center.clone().add(offset);

        // Node Mesh
        const geo = new THREE.SphereGeometry(0.5, 24, 24);
        const mat = new THREE.MeshPhongMaterial({
          color: cluster.color,
          emissive: cluster.color,
          emissiveIntensity: 0.4,
          transparent: true,
          opacity: 0.9
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(pos);
        this.nodesGroup.add(mesh);

        // Label
        const label = this.createLabelSprite(item, "#e2e8f0");
        label.position.copy(pos).add(new THREE.Vector3(0, 0.8, 0));
        label.scale.set(1.8, 0.45, 1);
        this.nodesGroup.add(label);

        this.nodes.push({ item, cluster: cluster.name, pos, mesh, label, color: cluster.color });
      });
    });
  }

  createRadarPulse() {
    const geo = new THREE.SphereGeometry(1, 32, 32);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      wireframe: true,
      transparent: true,
      opacity: 0.5
    });
    this.radarMesh = new THREE.Mesh(geo, mat);
    this.scene.add(this.radarMesh);
  }

  performRAGSearch(queryText) {
    this.queryText = queryText;

    // Remove old Query Node if exists
    if (this.queryGroup) this.scene.remove(this.queryGroup);

    this.queryGroup = new THREE.Group();
    
    // Glowing diamond geometry for Query Vector
    const geo = new THREE.OctahedronGeometry(0.9);
    const mat = new THREE.MeshPhongMaterial({
      color: 0xf59e0b,
      emissive: 0xd97706,
      emissiveIntensity: 0.8,
      wireframe: false
    });
    const mesh = new THREE.Mesh(geo, mat);
    this.queryGroup.add(mesh);

    // Query Label
    const sprite = this.createLabelSprite(`RAG Query: "${queryText.slice(0, 24)}..."`, "#f59e0b");
    sprite.position.set(0, 1.3, 0);
    sprite.scale.set(3.2, 0.8, 1);
    this.queryGroup.add(sprite);

    // Position Query Node near RAG Vector DB cluster
    const queryPos = new THREE.Vector3(1, 3.5, 1.5);
    this.queryGroup.position.copy(queryPos);
    this.scene.add(this.queryGroup);

    // Reset Radar Pulse
    this.radarMesh.position.copy(queryPos);
    this.radarWaveRadius = 0.5;
    this.radarActive = true;

    // Clear top-k lines
    while(this.linesGroup.children.length > 0) {
      this.linesGroup.remove(this.linesGroup.children[0]);
    }

    // Compute Cosine Distances to all nodes
    this.nodes.forEach(node => {
      const dist = queryPos.distanceTo(node.pos);
      node.dist = dist;
    });

    // Sort by distance (Top K nearest neighbors)
    this.nodes.sort((a, b) => a.dist - b.dist);
    const topK = this.nodes.slice(0, 4);

    topK.forEach((neighbor, idx) => {
      // Connect line
      const points = [queryPos, neighbor.pos];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineDashedMaterial({
        color: 0xf59e0b,
        dashSize: 0.3,
        gapSize: 0.15,
        linewidth: 2,
        transparent: true,
        opacity: 0.9
      });
      const line = new THREE.Line(lineGeo, lineMat);
      line.computeLineDistances();
      this.linesGroup.add(line);

      // Distance score badge
      const similarityScore = (1 / (1 + neighbor.dist * 0.15)).toFixed(3);
      const badge = this.createLabelSprite(`Top #${idx + 1} Similarity: ${similarityScore}`, "#10b981");
      const midPos = new THREE.Vector3().addVectors(queryPos, neighbor.pos).multiplyScalar(0.5);
      badge.position.copy(midPos).add(new THREE.Vector3(0, 0.4, 0));
      badge.scale.set(2, 0.5, 1);
      this.linesGroup.add(badge);
    });

    this.onStatusChange(1, `RAG Vector Search Complete: Retrieved ${topK.length} nearest document chunks from Vector Database.`);
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
      // Rotate Query diamond node
      if (this.queryGroup) {
        this.queryGroup.children[0].rotation.y += 0.02 * this.speed;
        this.queryGroup.children[0].rotation.x += 0.01 * this.speed;
      }

      // Expand RAG Sonar Radar Pulse
      if (this.radarActive) {
        this.radarWaveRadius += 0.08 * this.speed;
        if (this.radarWaveRadius > 14) {
          this.radarWaveRadius = 0.5;
        }
        this.radarMesh.scale.set(this.radarWaveRadius, this.radarWaveRadius, this.radarWaveRadius);
        this.radarMesh.material.opacity = Math.max(0, 0.6 - (this.radarWaveRadius / 14));
      }

      // Gentle floating animation for vector space nodes
      const time = Date.now() * 0.001;
      this.nodes.forEach((node, i) => {
        node.mesh.position.y = node.pos.y + Math.sin(time * 2 + i) * 0.1;
        node.label.position.y = node.mesh.position.y + 0.8;
      });
    }

    this.renderer.render(this.scene, this.camera);
  }

  setQuery(queryStr) {
    this.performRAGSearch(queryStr);
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
