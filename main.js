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

    // Control Buttons
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

    // Speed Slider
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-val');
    speedSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      speedValue.textContent = val.toFixed(1) + 'x';
      this.currentSimulation?.setSpeed(val);
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

    if (version === 'transformer') {
      promptLabel.textContent = "LLM Prompt Input:";
      promptInput.value = "Deep learning models transform intelligence";
      
      this.currentSimulation = new TransformerSimulation(this.container, {
        prompt: promptInput.value,
        onStatusChange: (stage, desc) => {
          statusText.textContent = desc;
        },
        onTokenGenerated: (token, fullText) => {
          const genContainer = document.getElementById('generated-stream');
          if (genContainer) {
            genContainer.textContent = fullText;
          }
        }
      });
      statusText.textContent = "Interactive 3D Transformer Pipeline: Tokenization -> Self-Attention -> Softmax -> Autoregressive Generation";
    } else if (version === 'embeddings') {
      promptLabel.textContent = "RAG Query Input:";
      promptInput.value = "Retrieval Augmented Generation with Vector Database";

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

      this.currentSimulation = new MoESimulation(this.container, {
        prompt: promptInput.value,
        onStatusChange: (stage, desc) => {
          statusText.textContent = desc;
        }
      });
      statusText.textContent = "Mixture of Experts (MoE): Gating Router dynamically activating Top-2 Specialist Expert Networks";
    } else if (version === 'kvcache') {
      promptLabel.textContent = "Prompt Input:";
      promptInput.value = "Key Value Cache accelerates transformer inference";

      this.currentSimulation = new KVCacheSimulation(this.container, {
        prompt: promptInput.value,
        onStatusChange: (stage, desc) => {
          statusText.textContent = desc;
        }
      });
      statusText.textContent = "3D KV Cache Memory: Prefill Phase vs O(1) Autoregressive Decoding Single-Query Lookups";
    } else if (version === 'lmstudio') {
      promptLabel.textContent = "Simple Prompt:";
      promptInput.value = "Hello LLM!";

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
