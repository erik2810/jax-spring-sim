// three.js WebGPU scene for the particle-spring system. Particles are an
// InstancedMesh coloured per-instance by the value channel; springs are
// LineSegments. TSL (three/tsl) provides a fresnel rim on the particles.

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Fn, cameraPosition, color, float, normalWorld, positionWorld } from 'three/tsl';
import { viridis } from '../lib/colormap.js';

const PIN_COLOR = new THREE.Color(0xffae3b);
const BG = 0x0a0e14;

export class SpringScene {
  constructor() {
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
    this.group = new THREE.Group();
    this.spheres = null;
    this.lines = null;
    this.targetLines = null;
    this.meta = null;
    this.baseRadius = 0.2;
    this.userMoved = false;
    this.backendName = 'unknown';
    this._dummy = new THREE.Object3D();
    this._color = new THREE.Color();
    this._fitBox = new THREE.Box3();
    this._fitPoint = new THREE.Vector3();
    this._framedRadius = 0;
    this._getFrame = null;
  }

  async init(canvas) {
    const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
    await renderer.init();
    this.renderer = renderer;
    this.backendName = renderer.backend?.isWebGPUBackend ? 'WebGPU' : 'WebGL2';

    const w = canvas.clientWidth || 1;
    const h = canvas.clientHeight || 1;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BG);
    scene.fog = new THREE.Fog(BG, 60, 220);
    this.scene = scene;

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 4000);
    camera.position.set(0, 4, 40);
    this.camera = camera;

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.addEventListener('start', () => {
      this.userMoved = true; // stop auto-framing once the user takes over
    });
    this.controls = controls;

    scene.add(new THREE.HemisphereLight(0xbcd4ff, 0x10141c, 1.2));
    const dir = new THREE.DirectionalLight(0xffffff, 2.2);
    dir.position.set(8, 16, 10);
    scene.add(dir);

    const grid = new THREE.GridHelper(120, 60, 0x1b2433, 0x10161f);
    grid.position.y = -0.001;
    scene.add(grid);

    scene.add(this.group);
    renderer.setAnimationLoop(() => this._tick());
  }

  setFrameProvider(fn) {
    this._getFrame = fn;
  }

  _sphereMaterial() {
    const mat = new THREE.MeshStandardNodeMaterial({ roughness: 0.45, metalness: 0.15 });
    // Fresnel rim via TSL, applied as emissive so the per-instance colour
    // (instanceColor) still drives the albedo.
    const fresnel = Fn(() => {
      const viewDir = cameraPosition.sub(positionWorld).normalize();
      const nDotV = normalWorld.dot(viewDir).saturate();
      return float(1.0).sub(nDotV).pow(2.5);
    });
    mat.emissiveNode = color(0x2b6cff).mul(fresnel()).mul(0.7);
    return mat;
  }

  setTopology(meta) {
    this.meta = meta;
    this.userMoved = false;
    this._fitBox.makeEmpty();
    this._framedRadius = 0;
    this._disposeContent();

    const n = meta.n;
    // Beads of a roughly constant world size (spacing is ~1 in every scene),
    // gently shrunk for very large counts so they do not fully merge.
    this.baseRadius = 0.2 * Math.min(1, Math.pow(50 / Math.max(8, n), 0.15));

    const sphereGeo = new THREE.SphereGeometry(1, 16, 12);
    const spheres = new THREE.InstancedMesh(sphereGeo, this._sphereMaterial(), n);
    spheres.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    for (let i = 0; i < n; i++) spheres.setColorAt(i, this._color.setRGB(0.2, 0.4, 0.6));
    spheres.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.spheres = spheres;
    this.group.add(spheres);

    const edges = meta.edges;
    const segPos = new Float32Array(edges.length * 2 * 3);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(segPos, 3));
    const lineMat = new THREE.LineBasicNodeMaterial({ transparent: true, opacity: 0.4 });
    lineMat.colorNode = color(0x6f8bb8);
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    lines.frustumCulled = false;
    this.lines = lines;
    this.group.add(lines);

    if (meta.target) {
      const t = meta.target;
      const tPos = new Float32Array(edges.length * 2 * 3);
      for (let e = 0; e < edges.length; e++) {
        const [a, b] = edges[e];
        tPos.set([t[a][0], t[a][1], t[a][2]], e * 6);
        tPos.set([t[b][0], t[b][1], t[b][2]], e * 6 + 3);
      }
      const tg = new THREE.BufferGeometry();
      tg.setAttribute('position', new THREE.BufferAttribute(tPos, 3));
      const tm = new THREE.LineBasicNodeMaterial({ transparent: true, opacity: 0.55 });
      tm.colorNode = color(0x37d39a);
      const targetLines = new THREE.LineSegments(tg, tm);
      targetLines.frustumCulled = false;
      this.targetLines = targetLines;
      this.group.add(targetLines);
    }
  }

  updateFrame(positions, value) {
    if (!this.spheres || !this.meta) return;
    const n = this.meta.n;
    const fixed = this.meta.fixed;
    const [vmin, vmax] = this.meta.valueRange || [0, 1];
    const span = vmax - vmin || 1;

    for (let i = 0; i < n; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      const pinned = fixed[i] > 0.5;
      this._dummy.position.set(x, y, z);
      this._dummy.scale.setScalar(pinned ? this.baseRadius * 1.7 : this.baseRadius);
      this._dummy.updateMatrix();
      this.spheres.setMatrixAt(i, this._dummy.matrix);
      if (pinned) {
        this.spheres.setColorAt(i, PIN_COLOR);
      } else {
        const t = value ? (value[i] - vmin) / span : 0.5;
        const [r, g, b] = viridis(t);
        this.spheres.setColorAt(i, this._color.setRGB(r, g, b));
      }
    }
    this.spheres.instanceMatrix.needsUpdate = true;
    if (this.spheres.instanceColor) this.spheres.instanceColor.needsUpdate = true;

    const edges = this.meta.edges;
    const arr = this.lines.geometry.attributes.position.array;
    for (let e = 0; e < edges.length; e++) {
      const [a, b] = edges[e];
      arr[e * 6] = positions[a * 3];
      arr[e * 6 + 1] = positions[a * 3 + 1];
      arr[e * 6 + 2] = positions[a * 3 + 2];
      arr[e * 6 + 3] = positions[b * 3];
      arr[e * 6 + 4] = positions[b * 3 + 1];
      arr[e * 6 + 5] = positions[b * 3 + 2];
    }
    this.lines.geometry.attributes.position.needsUpdate = true;

    this._autoFrame(positions, n);
  }

  // Fit the camera to the trajectory's extent so far. The bounding box grows as
  // the system moves (e.g. a chain sagging), so we re-fit while it expands and
  // stop once the user takes manual control of the orbit camera.
  _autoFrame(positions, n) {
    if (this.userMoved) return;
    for (let i = 0; i < n; i++) {
      this._fitPoint.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      this._fitBox.expandByPoint(this._fitPoint);
    }
    if (this._fitBox.isEmpty()) return;

    const size = this._fitBox.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 1) * 0.5;
    if (this._framedRadius && radius <= this._framedRadius * 1.08) return; // stable enough
    this._framedRadius = radius;

    const center = this._fitBox.getCenter(new THREE.Vector3());
    const dist = (radius / Math.tan((this.camera.fov * Math.PI) / 360)) * 1.7;
    this.controls.target.copy(center);
    this.camera.position.set(center.x, center.y + radius * 0.2, center.z + dist);
    this.camera.near = Math.max(0.01, dist / 200);
    this.camera.far = dist * 40;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _tick() {
    if (this._getFrame) {
      const f = this._getFrame();
      if (f) this.updateFrame(f.positions, f.value);
    }
    if (this.controls) this.controls.update();
    if (this.renderer && this.scene && this.camera) this.renderer.render(this.scene, this.camera);
  }

  resize(w, h) {
    if (!this.renderer || !this.camera) return;
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  _disposeContent() {
    for (const obj of [this.spheres, this.lines, this.targetLines]) {
      if (!obj) continue;
      this.group.remove(obj);
      obj.geometry.dispose();
      obj.material.dispose();
    }
    this.spheres = null;
    this.lines = null;
    this.targetLines = null;
  }

  dispose() {
    if (this.renderer) {
      this.renderer.setAnimationLoop(null);
      this.renderer.dispose();
    }
    if (this.controls) this.controls.dispose();
  }
}
