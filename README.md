# 🧠 Interactive 3D LLM & Transformer Visualizer Suite

[![Three.js](https://img.shields.io/badge/Three.js-r161-black?style=for-the-badge&logo=three.js)](https://threejs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.x-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![JavaScript](https://img.shields.io/badge/ES6+-Modern_JS-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An interactive, real-time 3D visualization suite built with **Three.js** and modern WebGL to explore the inner mechanics, data flows, and memory architectures of Large Language Models (LLMs) and Transformer neural networks.

---

## 🌟 Visualizer Modules

The suite features 5 modular 3D visualization modes accessible via the top navigation bar:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                LLM 3D Visualizer                                       │
│  [ v2: 3D Transformer ] [ v3: RAG & Vector Space ] [ v4: MoE ] [ v5: KV Cache ] [ v1 ] │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. 🔬 `v2: 3D Transformer Pipeline` (`threejs-simulation-transformer.js`)
Visualizes the complete end-to-end forward pass of a generative Transformer:
- **Tokenization Stage**: Splits input strings into discrete tokens positioned in 3D space.
- **Embedding & Positional Encoding**: Projects tokens into high-dimensional representation blocks.
- **Multi-Head Self-Attention**: Renders dynamic 3D bezier attention curves connecting interdependent tokens with glowing query-key affinity weights.
- **Feed-Forward Layers**: Animates data flow through multi-layer perceptron (MLP) blocks.
- **Softmax & Sampling**: Displays 3D vertical probability bars and samples next tokens autoregressively.
- **Live Stream Overlay**: Real-time token streaming console.

---

### 2. 🌌 `v3: RAG & Vector Space` (`threejs-simulation-embeddings.js`)
Explores high-dimensional semantic spaces and Retrieval-Augmented Generation (RAG):
- **3D Semantic Clusters**: Visualizes clustered vector embeddings categorized across knowledge domains (e.g., Code, Science, Math, General).
- **Query Projection**: Projects input prompts into semantic coordinate space.
- **Cosine Distance Radar**: Animates a spherical radar proximity wave detecting nearest neighbor vector points.
- **Top-K Retrieval**: Draws glowing vector distance vectors between the query and retrieved context nodes.

---

### 3. 🔀 `v4: Mixture of Experts (MoE)` (`threejs-simulation-moe.js`)
Demystifies Sparse Mixture of Experts architectures (as seen in Mixtral, DeepSeek, and GPT-4):
- **Central Gating Router**: Analyzes incoming prompt tokens and calculates sparse top-$k$ routing weights.
- **Specialist Expert Nodes**: 3D geometric expert clusters representing specialized domain models (*Math*, *Code*, *Logic*, *Creative*, *General*).
- **Dynamic Sparse Routing**: Emits directed particle beams and activation lines toward the top-2 selected expert sub-networks per token.
- **Expert Load Monitoring**: Visual feedback of expert activation states and throughput.

---

### 4. ⚡ `v5: KV Cache Memory Architecture` (`threejs-simulation-kvcache.js`)
Illustrates inference optimization via Key-Value (KV) caching:
- **Prefill vs. Decoding Phase**: Compares parallel prompt ingestion with step-by-step $O(1)$ constant-time autoregressive decoding.
- **3D KV Matrix Grid**: Interactive memory grid dynamically filling Key ($K$) and Value ($V$) activation states in GPU VRAM.
- **Lookup Line Dynamics**: Shows how new query tokens attend to previously cached keys without recomputing past states.
- **Memory Footprint Telemetry**: Visual representation of context window growth and memory allocation.

---

### 5. 🔮 `v1: LM Studio Core` (`threejs-simulation-lmstudio.js`)
Foundational conceptual visualization showing orbital tensor rings, token ingestion pipelines, and core neural node activations.

---

## 🎮 Interactive Controls & UI

| Control | Description |
| :--- | :--- |
| **Orbit Controls** | **Left Click + Drag** to rotate camera; **Right Click + Drag** to pan; **Scroll Wheel** to zoom in/out. |
| **Prompt Input** | Type custom prompts or queries to see real-time updates across tokenization, routing, or vector search. |
| **Preset Templates** | Quick-select curated prompts tailored for each simulation mode. |
| **Speed Slider** | Dynamically adjust animation speed from `0.2x` (slow-motion inspection) up to `3.0x` (high-throughput). |
| **Play / Pause** | Freeze simulation at any time to inspect 3D geometries, attention lines, and vector coordinates. |
| **Status Banner** | Real-time phase indicator explaining the exact computational step currently rendering. |

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** (v18+ recommended) & **npm**  
  *or* any static web server (Python, Caddy, NGINX, Live Server, etc.)

---

### Method 1: Modern Dev Server (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llm-sim
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the local Vite development server:**
   ```bash
   npm run dev
   ```
   Open your browser at `http://localhost:3000`.

4. **Build production bundle:**
   ```bash
   npm run build
   npm run preview
   ```

---

### Method 2: Zero-Install Quick Start

Because the project utilizes native ES Modules and browser import maps, you can run it immediately with zero local build tools using any static HTTP server:

**Using NPX (Node):**
```bash
npx serve . -p 3000
```

**Using Python:**
```bash
python -m http.server 3000
```

**Using `uv` (Fast Python Package Manager):**
```bash
uv run python -m http.server 3000
```

---

## 📂 Project Structure

```
llm-sim/
├── index.html                           # Main entry HTML, Glassmorphism UI, and Import Map
├── main.js                              # Orchestrator app class & simulation lifecycle manager
├── threejs-simulation-transformer.js    # v2: Full Transformer pipeline simulation
├── threejs-simulation-embeddings.js     # v3: Vector space & RAG retrieval simulation
├── threejs-simulation-moe.js            # v4: Mixture of Experts (MoE) gating simulation
├── threejs-simulation-kvcache.js        # v5: KV Cache memory architecture simulation
├── threejs-simulation-lmstudio.js       # v1: Conceptual LM core visualizer
├── package.json                         # Project scripts, Three.js, and Vite configuration
├── vite.config.js                       # Vite development server configuration
├── .gitignore                           # Git ignore rules for Node, Python, and OS artifacts
└── README.md                            # Project documentation
```

---

## 🛠️ Technology Stack

- **[Three.js (r161)](https://threejs.org/)**: 3D rendering engine, shaders, procedural geometries, particle systems, and materials.
- **[OrbitControls](https://threejs.org/docs/#examples/en/controls/OrbitControls)**: Smooth inertial camera controls with damping.
- **[Vite](https://vitejs.dev/)**: Next-generation frontend tooling and development server.
- **ES Modules & Import Maps**: Modular architecture that runs natively both bundled and unbundled.
- **Glassmorphism CSS3**: Sleek dark-mode user interface with backdrop filters, custom inputs, and dynamic animations.

---

## 🤝 Contributing

Contributions, feature suggestions, and visualizer expansions are welcome! Feel free to:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewVisualizer`)
3. Commit your Changes (`git commit -m 'Add new Transformer Attention Head visualizer'`)
4. Push to the Branch (`git push origin feature/NewVisualizer`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
