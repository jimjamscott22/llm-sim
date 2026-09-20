import { LMStudioSimulation } from './threejs-simulation-lmstudio.js';
import { TransformerSimulation } from './threejs-simulation-transformer.js';
import { EmbeddingSimulation } from './threejs-simulation-embeddings.js';
import { MoESimulation } from './threejs-simulation-moe.js';
import { KVCacheSimulation } from './threejs-simulation-kvcache.js';

class App {
  constructor() {
    this.container = document.getElementById('canvas-container');
    this.currentSimulation = null;
    this.currentVersion = 'transformer'; // default mode
    this.isPaused = false;

    this.initUI();
    this.loadSimulation('transformer');
  }

  initUI() {
    // Version Buttons
    const versionBtns = document.querySelectorAll('.version-btn');
    versionBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const ver = e.currentTarget.dataset.version;
        if (ver !== this.currentVersion) {
          versionBtns.forEach(b => b.classList.remove('active'));
          e.currentTarget.classList.add('active');
          this.loadSimulation(ver);
        }
      });
    });

    // Stepper & Playback Controls
    const playPauseBtn = document.getElementById('play-pause-btn');
    playPauseBtn.addEventListener('click', () => {
      if (this.isPaused) {
        this.currentSimulation?.resume();
        this.isPaused = false;
        playPauseBtn.innerHTML = '<span>⏸</span> Pause';
        playPauseBtn.classList.remove('paused');
      } else {
        this.currentSimulation?.pause();
        this.isPaused = true;
        playPauseBtn.innerHTML = '<span>▶</span> Play';
        playPauseBtn.classList.add('paused');
      }
    });

    const stepPrevBtn = document.getElementById('step-prev-btn');
    stepPrevBtn.addEventListener('click', () => {
      this.currentSimulation?.stepBackward();
      this.isPaused = true;
      playPauseBtn.innerHTML = '<span>▶</span> Play';
      playPauseBtn.classList.add('paused');
    });

    const stepNextBtn = document.getElementById('step-next-btn');
    stepNextBtn.addEventListener('click', () => {
      this.currentSimulation?.stepForward();
      this.isPaused = true;
      playPauseBtn.innerHTML = '<span>▶</span> Play';
      playPauseBtn.classList.add('paused');
    });

    // Speed Slider
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-val');
    speedSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      speedValue.textContent = val.toFixed(1) + 'x';
      this.currentSimulation?.setSpeed(val);
    });

    // Transformer Sampling Sliders
    const tempSlider = document.getElementById('temp-slider');
    const tempVal = document.getElementById('temp-val');
    tempSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      tempVal.textContent = val.toFixed(2);
      if (typeof this.currentSimulation?.setTemperature === 'function') {
        this.currentSimulation.setTemperature(val);
      }
    });

    const toppSlider = document.getElementById('topp-slider');
    const toppVal = document.getElementById('topp-val');
    toppSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      toppVal.textContent = val.toFixed(2);
      if (typeof this.currentSimulation?.setTopP === 'function') {
        this.currentSimulation.setTopP(val);
      }
    });

    // KV Cache Mode Toggle Button
    const toggleKvBtn = document.getElementById('toggle-kv-mode-btn');
    toggleKvBtn.addEventListener('click', () => {
      if (typeof this.currentSimulation?.toggleMode === 'function') {
        this.currentSimulation.toggleMode();
      }
    });

    // Prompt Form Submit
    const promptForm = document.getElementById('prompt-form');
    const promptInput = document.getElementById('prompt-input');
    promptForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const val = promptInput.value.trim();
      if (!val) return;

      if (typeof this.currentSimulation?.setPrompt === 'function') {
        this.currentSimulation.setPrompt(val);
      } else if (typeof this.currentSimulation?.setQuery === 'function') {
        this.currentSimulation.setQuery(val);
      }
    });

    // Preset Prompts Dropdown
    const presetSelect = document.getElementById('preset-select');
    presetSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (!val) return;
      promptInput.value = val;
      if (typeof this.currentSimulation?.setPrompt === 'function') {
        this.currentSimulation.setPrompt(val);
      } else if (typeof this.currentSimulation?.setQuery === 'function') {
        this.currentSimulation.setQuery(val);
      }
    });
  }

  updateStagePills(stageNames) {
    const container = document.getElementById('stage-pills-container');
    const row = document.getElementById('stage-pills-row');
    if (!stageNames || stageNames.length === 0) {
      row.style.display = 'none';
      return;
    }
    row.style.display = 'flex';
    container.innerHTML = '';

    stageNames.forEach((name, idx) => {
      const btn = document.createElement('button');
      btn.className = `stage-pill ${idx === 0 ? 'active' : ''}`;
      btn.dataset.stage = idx;
      btn.textContent = name;
      btn.addEventListener('click', () => {
        this.currentSimulation?.goToStage(idx);
        this.isPaused = true;
        const playPauseBtn = document.getElementById('play-pause-btn');
        playPauseBtn.innerHTML = '<span>▶</span> Play';
        playPauseBtn.classList.add('paused');
      });
      container.appendChild(btn);
    });
  }

  highlightActiveStage(stageIdx) {
    const pills = document.querySelectorAll('.stage-pill');
    pills.forEach((p, idx) => {
      if (idx === stageIdx) {
        p.classList.add('active');
      } else {
        p.classList.remove('active');
      }
    });
  }

  loadSimulation(version) {
    if (this.currentSimulation) {
      this.currentSimulation.destroy();
      this.currentSimulation = null;
    }

    this.currentVersion = version;
    this.isPaused = false;
    
    const playPauseBtn = document.getElementById('play-pause-btn');
    playPauseBtn.innerHTML = '<span>⏸</span> Pause';
    playPauseBtn.classList.remove('paused');

    const promptInput = document.getElementById('prompt-input');
    const promptLabel = document.getElementById('prompt-label');
    const statusText = document.getElementById('status-text');

    // Toggle Dynamic Panels
    const transformerPanel = document.getElementById('transformer-panel');
    const kvcachePanel = document.getElementById('kvcache-panel');
    const moePanel = document.getElementById('moe-panel');

    transformerPanel.style.display = version === 'transformer' ? 'flex' : 'none';
    kvcachePanel.style.display = version === 'kvcache' ? 'flex' : 'none';
    moePanel.style.display = version === 'moe' ? 'flex' : 'none';

    if (version === 'transformer') {
      promptLabel.textContent = "LLM Prompt Input:";
      promptInput.value = "Deep learning models transform intelligence";
      
      this.updateStagePills([
        "1. Tokenize", 
        "2. QKV Proj", 
        "3. Self-Attn", 
        "4. FFN & Norm", 
        "5. Softmax", 
        "6. Next Token"
      ]);

      this.currentSimulation = new TransformerSimulation(this.container, {
        prompt: promptInput.value,
        onStatusChange: (stage, desc) => {
          statusText.textContent = desc;
        },
        onStageChanged: (stage) => {
          this.highlightActiveStage(stage);
        },
        onTokenGenerated: (token, fullText) => {
          const genContainer = document.getElementById('generated-stream');
          if (genContainer) {
            genContainer.textContent = fullText;
          }
        }
      });
      statusText.textContent = "Interactive 3D Transformer Pipeline: Tokenization -> Q,K,V Projections -> Self-Attention -> Softmax -> Generation";
    } else if (version === 'embeddings') {
      promptLabel.textContent = "RAG Query Input:";
      promptInput.value = "Retrieval Augmented Generation with Vector Database";
      this.updateStagePills([]);

      this.currentSimulation = new EmbeddingSimulation(this.container, {
        query: promptInput.value,
        onStatusChange: (stage, desc) => {
          statusText.textContent = desc;
        }
      });
      statusText.textContent = "3D Vector Space & RAG Retriever: High-dimensional semantic embeddings & Top-K Cosine search";
    } else if (version === 'moe') {
      promptLabel.textContent = "Task / Prompt Input:";
      promptInput.value = "Solve math equation and write python code";

      this.updateStagePills([
        "1. Token Input", 
        "2. Gating Top-2", 
        "3. Expert FFN", 
        "4. Weighted Merge"
      ]);

      this.currentSimulation = new MoESimulation(this.container, {
        prompt: promptInput.value,
        onStatusChange: (stage, desc) => {
          statusText.textContent = desc;
        },
        onStageChanged: (stage) => {
          this.highlightActiveStage(stage);
        },
        onRoutingUpdated: (routing) => {
          document.getElementById('moe-expert-1').textContent = routing.expert1;
          document.getElementById('moe-expert-2').textContent = routing.expert2;
        }
      });
      statusText.textContent = "Mixture of Experts (MoE): Semantic Gating Router dynamically activating Top-2 Specialist Expert Networks";
    } else if (version === 'kvcache') {
      promptLabel.textContent = "Prompt Input:";
      promptInput.value = "Key Value Cache accelerates transformer inference";

      this.updateStagePills([
        "1. Prefill K/V", 
        "2. Decode Step 1", 
        "3. Decode Step 2", 
        "4. Decode Step 3"
      ]);

      this.currentSimulation = new KVCacheSimulation(this.container, {
        prompt: promptInput.value,
        onStatusChange: (stage, desc) => {
          statusText.textContent = desc;
        },
        onStageChanged: (stage) => {
          this.highlightActiveStage(stage);
        },
        onMetricsUpdated: (metrics) => {
          const btn = document.getElementById('toggle-kv-mode-btn');
          const compElem = document.getElementById('kv-complexity');
          const flopsElem = document.getElementById('kv-flops');
          const vramElem = document.getElementById('kv-vram');

          compElem.textContent = metrics.complexity;
          flopsElem.textContent = metrics.flops;
          vramElem.textContent = metrics.vram;

          if (metrics.mode === 'with_cache') {
            btn.className = 'toggle-cache-btn with-cache';
            btn.innerHTML = '<span>⚡ With KV Cache: O(1) Query</span>';
            compElem.style.color = 'var(--accent-emerald)';
            flopsElem.style.color = 'var(--accent-emerald)';
          } else {
            btn.className = 'toggle-cache-btn without-cache';
            btn.innerHTML = '<span>💥 Without Cache: O(N²) Storm</span>';
            compElem.style.color = 'var(--accent-rose)';
            flopsElem.style.color = 'var(--accent-rose)';
          }
        }
      });
      statusText.textContent = "3D KV Cache Memory: Prefill Phase vs O(1) Autoregressive Decoding Single-Query Lookups";
    } else if (version === 'lmstudio') {
      promptLabel.textContent = "Simple Prompt:";
      promptInput.value = "Hello LLM!";
      this.updateStagePills([]);

      this.currentSimulation = new LMStudioSimulation(this.container);
      statusText.textContent = "LM Studio Conceptual Processing Core";
    }

    this.currentSimulation.start();
  }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
  new App();
});

