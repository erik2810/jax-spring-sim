# jax-spring-sim · viewer

Interactive WebGPU viewer for the JAX particle-spring simulator. The Python
backend computes trajectories with JAX and streams them here over a WebSocket
(binary protocol); this app renders them with three.js (WebGPU renderer + TSL).

## Stack

- **Vite + React 19**
- **three.js** (`three/webgpu` renderer, `three/tsl` node materials)
- **Tailwind CSS v4** (`@tailwindcss/vite`)
- **KaTeX** (`react-katex`) for the energy equation

## Run

The viewer needs the backend running first:

```bash
# from the repo root
uv run python -m jax_spring_sim.server      # FastAPI on :8000
```

Then, in this directory:

```bash
npm install
npm run dev                                  # Vite on :5173
```

Vite proxies `/ws` and `/health` to the backend (see `vite.config.js`), so no
CORS or env setup is needed in dev. To point at a different backend, set
`VITE_WS_URL` (e.g. `VITE_WS_URL=ws://host:8000/ws npm run dev`).

A WebGPU-capable browser is recommended; the `three/webgpu` renderer falls back
to WebGL2 where WebGPU is unavailable (shown in the header badge).

## Scenes

- **Catenary** — a cable pinned at both ends settling under gravity.
- **Cloth drape** — a pinned cloth grid draping in 3D.
- **Inverse design** — `value_and_grad` recovering hidden rest lengths; the
  chain morphs toward the target shape (green) as Adam runs.

## Structure

```
src/
├── App.jsx               layout, controls, playback state
├── lib/
│   ├── protocol.js       parse the binary frame protocol (DataView)
│   ├── useSimSocket.js   WebSocket hook (buffers frames in a ref)
│   └── colormap.js       viridis colour ramp for the value channel
├── viewer/
│   ├── SpringScene.js    three.js WebGPU scene (instanced beads + springs, TSL)
│   └── SpringViewer.jsx  React <-> scene bridge + playback clock
└── components/           scene metadata + icons
```
