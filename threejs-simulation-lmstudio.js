import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class LMStudioSimulation {
  constructor(container) {
    this.container = container;
    this.animationId = null;
    this.isPaused = false;
    this.speed = 1.0;
    this.stepCount = 0;
    this.totalSteps = 100;
    
    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0c14);

    // Camera
    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    this.camera.position.set(0, 2, 8);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x38bdf8, 2, 50);
    pointLight.position.set(0, 5, 5);
    this.scene.add(pointLight);

    // Grid Floor
    const grid = new THREE.GridHelper(20, 20, 0x1e293b, 0x0f172a);
    grid.position.y = -2;
    this.scene.add(grid);

    // Build Visual Components
    this.createComponents();

    // Event listener
    this.onResize = this.onResize.bind(this);
    window.addEventListener('resize', this.onResize);
  }

  createTextSprite(text, colorStr = '#ffffff') {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 4;
    
    // Rounded rect
    const r = 16;
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(512 - r, 0);
    ctx.quadraticCurveTo(512, 0, 512, r);
    ctx.lineTo(512, 128 - r);
    ctx.quadraticCurveTo(512, 128, 512 - r, 128);
    ctx.lineTo(r, 128);
    ctx.quadraticCurveTo(0, 128, 0, 128 - r);
    ctx.lineTo(0, r);
    ctx.quadraticCurveTo(0, 0, r, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 36px sans-serif';
    ctx.fillStyle = colorStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 64);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(3, 0.75, 1);
    return sprite;
  }

  createComponents() {
    // 1. User Input Box
    this.inputSprite = this.createTextSprite("Input: Hello LLM!", "#f59e0b");
    this.inputSprite.position.set(-4, 0.5, 0);
    this.scene.add(this.inputSprite);

    // 2. LLM Core (Pulsing Sphere with Outer Shell)
    const coreGroup = new THREE.Group();
    
    const coreGeo = new THREE.SphereGeometry(1, 32, 32);
    const coreMat = new THREE.MeshPhongMaterial({
      color: 0x0284c7,
      emissive: 0x0369a1,
      wireframe: true,
      transparent: true,
      opacity: 0.8
    });
    this.coreMesh = new THREE.Mesh(coreGeo, coreMat);
    coreGroup.add(this.coreMesh);

    const innerGeo = new THREE.IcosahedronGeometry(0.6, 2);
    const innerMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: false
    });
    this.innerMesh = new THREE.Mesh(innerGeo, innerMat);
    coreGroup.add(this.innerMesh);

    this.scene.add(coreGroup);
    this.coreGroup = coreGroup;

    // 3. Attention Lines
    this.numHeads = 8;
    this.attentionLines = [];
    this.attentionLineGroup = new THREE.Group();
    
    for (let i = 0; i < this.numHeads; i++) {
      const angle = (i / this.numHeads) * Math.PI * 2;
      const targetPos = new THREE.Vector3(
        Math.cos(angle) * 2.5,
        Math.sin(angle) * 2.5,
        (Math.random() - 0.5) * 1.5
      );

      const points = [new THREE.Vector3(0, 0, 0), targetPos];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x818cf8,
        transparent: true,
        opacity: 0.6
      });
      const line = new THREE.Line(lineGeo, lineMat);
      this.attentionLineGroup.add(line);
      this.attentionLines.push({ line, targetPos });
    }
    this.scene.add(this.attentionLineGroup);

    // 4. Output Response Sprite
    this.outputSprite = this.createTextSprite("Output: Thinking...", "#10b981");
    this.outputSprite.position.set(4, 0.5, 0);
    this.scene.add(this.outputSprite);
  }

  updateLabel(sprite, text, colorStr) {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = colorStr;
    ctx.lineWidth = 4;
    
    const r = 16;
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(512 - r, 0);
    ctx.quadraticCurveTo(512, 0, 512, r);
    ctx.lineTo(512, 128 - r);
    ctx.quadraticCurveTo(512, 128, 512 - r, 128);
    ctx.lineTo(r, 128);
    ctx.quadraticCurveTo(0, 128, 0, 128 - r);
    ctx.lineTo(0, r);
    ctx.quadraticCurveTo(0, 0, r, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.font = 'Bold 34px sans-serif';
    ctx.fillStyle = colorStr;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 256, 64);

    sprite.material.map.dispose();
    sprite.material.map = new THREE.CanvasTexture(canvas);
    sprite.material.map.needsUpdate = true;
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
      this.stepCount += 0.5 * this.speed;
      
      // Core rotation
      this.coreMesh.rotation.y += 0.01 * this.speed;
      this.coreMesh.rotation.x += 0.005 * this.speed;
      this.innerMesh.rotation.y -= 0.02 * this.speed;

      // Pulsing scale
      const scale = 1 + Math.sin(this.stepCount * 0.1) * 0.15;
      this.coreGroup.scale.set(scale, scale, scale);

      // Attention line animation
      this.attentionLineGroup.rotation.z += 0.01 * this.speed;

      // Stage progression simulation
      const cycle = Math.floor((this.stepCount % 120) / 40);
      if (cycle === 0) {
        this.updateLabel(this.outputSprite, "Output: Processing...", "#3b82f6");
      } else if (cycle === 1) {
        this.updateLabel(this.outputSprite, "Output: Generating...", "#a855f7");
      } else {
        this.updateLabel(this.outputSprite, "Output: Hello World!", "#10b981");
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  pause() {
    this.isPaused = true;
  }

  resume() {
    this.isPaused = false;
  }

  setSpeed(val) {
    this.speed = val;
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
