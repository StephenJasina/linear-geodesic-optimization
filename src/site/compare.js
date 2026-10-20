import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DURATION = 8;
const DISPLAY_HEIGHT = 6;
const COLORS = { before: '#38b9df', after: '#4fc98b', third: '#b88bea', fourth: '#f0c05c', route: '#e88728' };
const stops = [
  [0, '#173544'], [.15, '#275f71'], [.4, '#369797'],
  [.7, '#afd46b'], [1, '#ffe27b'],
].map(([position, color]) => [position, new THREE.Color(color)]);
const panelIds = ['before', 'after', 'third', 'fourth'];
const views = [];
const timeline = document.getElementById('timeline');
const playButton = document.getElementById('play-button');
const timeReadout = document.getElementById('time-readout');
const loadError = document.getElementById('load-error');
let comparison = null;
let progress = 0;
let playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let lastFrame = performance.now();
let syncingCamera = false;

function colorAt(value) {
  const t = THREE.MathUtils.clamp(value, 0, 1);
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const blend = (t - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
      return stops[i - 1][1].clone().lerp(stops[i][1], blend);
    }
  }
  return stops[stops.length - 1][1].clone();
}

function validate(data) {
  if (!Number.isInteger(data.side) || data.side < 2 || !Array.isArray(data.network?.points)) {
    throw new Error('The comparison JSON needs a mesh side length and network.');
  }
  if (!Array.isArray(data.surfaces) || data.surfaces.length < 2 || data.surfaces.length > 4) {
    throw new Error('The comparison JSON needs two to four surfaces.');
  }
  for (const id of panelIds.slice(0, data.surfaces.length)) {
    const surface = data.surfaces.find((item) => item.id === id);
    if (!surface || surface.heights?.length !== data.side * data.side || !Array.isArray(surface.geodesic) || surface.geodesic.length < 2 || !Array.isArray(surface.route) || surface.route.length < 2) {
      throw new Error(`The comparison JSON needs ${id} heights and a geodesic.`);
    }
  }
}

function heightAt(surface, x, y, side) {
  const u = THREE.MathUtils.clamp(x, 0, 1) * (side - 1);
  const v = THREE.MathUtils.clamp(y, 0, 1) * (side - 1);
  const i = Math.floor(u), j = Math.floor(v);
  const i1 = Math.min(i + 1, side - 1), j1 = Math.min(j + 1, side - 1);
  const a = u - i, b = v - j;
  const h = surface.heights;
  return DISPLAY_HEIGHT * ((1 - a) * (1 - b) * h[i * side + j]
    + a * (1 - b) * h[i1 * side + j]
    + (1 - a) * b * h[i * side + j1]
    + a * b * h[i1 * side + j1]);
}

function makeMesh(surface, side, heightMin, heightMax) {
  const positions = new Float32Array(side * side * 3);
  const colors = new Float32Array(side * side * 3);
  const indices = [];
  const span = Math.max(heightMax - heightMin, 1e-9);
  for (let i = 0; i < side; i++) {
    for (let j = 0; j < side; j++) {
      const index = i * side + j;
      const height = surface.heights[index];
      positions.set([i / (side - 1) - .5, DISPLAY_HEIGHT * height, j / (side - 1) - .5], index * 3);
      const color = colorAt((height - heightMin) / span);
      colors.set([color.r, color.g, color.b], index * 3);
      if (i < side - 1 && j < side - 1) {
        const right = index + side;
        indices.push(index, right, index + 1, right, right + 1, index + 1);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({
    vertexColors: true, side: THREE.DoubleSide, shininess: 55,
  }));
}

function makeGrid(surface, side) {
  const points = [];
  for (let fixed = 0; fixed < side; fixed += 5) {
    for (let moving = 0; moving < side - 1; moving++) {
      for (const [i, j] of [[[fixed, moving], [fixed, moving + 1]], [[moving, fixed], [moving + 1, fixed]]]) {
        points.push(i[0] / (side - 1) - .5, DISPLAY_HEIGHT * surface.heights[i[0] * side + i[1]] + .002, i[1] / (side - 1) - .5);
        points.push(j[0] / (side - 1) - .5, DISPLAY_HEIGHT * surface.heights[j[0] * side + j[1]] + .002, j[1] / (side - 1) - .5);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
    color: '#b5d7d9', transparent: true, opacity: .32, depthWrite: false,
  }));
}

function routePoints(data, surface) {
  const points = [];
  for (let i = 0; i < surface.route.length - 1; i++) {
    const a = surface.route[i], b = surface.route[i + 1];
    for (let k = 0; k < 36; k++) {
      const t = k / 36;
      const x = a[0] + (b[0] - a[0]) * t;
      const y = a[1] + (b[1] - a[1]) * t;
      points.push(new THREE.Vector3(x - .5, heightAt(surface, x, y, data.side) + .014, y - .5));
    }
  }
  const last = surface.route[surface.route.length - 1];
  points.push(new THREE.Vector3(last[0] - .5, heightAt(surface, ...last, data.side) + .014, last[1] - .5));
  return points;
}

function lineOnSurface(path, color) {
  const points = path.map(([x, y, z]) => new THREE.Vector3(x - .5, DISPLAY_HEIGHT * z + .018, y - .5));
  const curve = new THREE.CurvePath();
  for (let i = 1; i < points.length; i++) {
    curve.add(new THREE.LineCurve3(points[i - 1], points[i]));
  }
  const tube = new THREE.Mesh(
    new THREE.TubeGeometry(curve, Math.max(80, points.length * 2), .0055, 6, false),
    new THREE.MeshBasicMaterial({ color, depthTest: true }),
  );
  return { tube, points };
}

function makeNetwork(data, surface) {
  const points = [];
  const seen = new Set();
  for (const [from, to] of data.network.edges) {
    const key = [Math.min(from, to), Math.max(from, to)].join(':');
    if (seen.has(key)) continue;
    seen.add(key);
    const a = data.network.points[from], b = data.network.points[to];
    for (let step = 0; step < 24; step++) {
      for (const t of [step / 24, (step + 1) / 24]) {
        const x = a[0] + (b[0] - a[0]) * t;
        const y = a[1] + (b[1] - a[1]) * t;
        points.push(x - .5, heightAt(surface, x, y, data.side) + .006, y - .5);
      }
    }
  }
  const group = new THREE.Group();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  group.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
    color: '#274751', transparent: true, opacity: .6, depthTest: false,
  })));
  const marker = new THREE.SphereGeometry(.008, 8, 7);
  const material = new THREE.MeshBasicMaterial({ color: '#223d45', depthTest: false });
  for (const [x, y] of data.network.points) {
    const node = new THREE.Mesh(marker, material);
    node.position.set(x - .5, heightAt(surface, x, y, data.side) + .01, y - .5);
    group.add(node);
  }
  return group;
}

function formatLoss(value) {
  return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—';
}

function makeTopMap(data, surface, heightMin, heightMax, color) {
  const canvas = document.getElementById(`${surface.id}-map`);
  const size = 240;
  canvas.width = canvas.height = size;
  const context = canvas.getContext('2d');
  const base = document.createElement('canvas');
  base.width = base.height = size;
  const painter = base.getContext('2d');
  const pixels = painter.createImageData(data.side, data.side);
  const span = Math.max(heightMax - heightMin, 1e-9);
  for (let i = 0; i < data.side; i++) {
    for (let j = 0; j < data.side; j++) {
      const value = surface.heights[i * data.side + j];
      const colorValue = colorAt((value - heightMin) / span);
      const index = ((data.side - 1 - j) * data.side + i) * 4;
      pixels.data[index] = Math.round(colorValue.r * 255);
      pixels.data[index + 1] = Math.round(colorValue.g * 255);
      pixels.data[index + 2] = Math.round(colorValue.b * 255);
      pixels.data[index + 3] = 255;
    }
  }
  const lowResolution = document.createElement('canvas');
  lowResolution.width = lowResolution.height = data.side;
  lowResolution.getContext('2d').putImageData(pixels, 0, 0);
  painter.imageSmoothingEnabled = true;
  painter.drawImage(lowResolution, 0, 0, size, size);
  const toCanvas = ([x, y]) => [x * size, (1 - y) * size];
  painter.strokeStyle = '#45616b';
  painter.lineWidth = 1.2;
  painter.setLineDash([]);
  for (const [from, to] of data.network.edges) {
    const a = toCanvas(data.network.points[from]);
    const b = toCanvas(data.network.points[to]);
    painter.beginPath();
    painter.moveTo(...a);
    painter.lineTo(...b);
    painter.stroke();
  }
  painter.beginPath();
  surface.route.forEach((point, index) => {
    const [x, y] = toCanvas(point);
    if (index === 0) painter.moveTo(x, y); else painter.lineTo(x, y);
  });
  painter.strokeStyle = COLORS.route;
  painter.lineWidth = 4;
  painter.setLineDash([11, 7]);
  painter.stroke();
  painter.setLineDash([]);
  painter.beginPath();
  surface.geodesic.forEach((point, index) => {
    const [x, y] = toCanvas(point);
    if (index === 0) painter.moveTo(x, y); else painter.lineTo(x, y);
  });
  painter.strokeStyle = color;
  painter.lineWidth = 4;
  painter.stroke();
  surface.route.forEach((point, index) => {
    const [x, y] = toCanvas(point);
    painter.beginPath();
    painter.arc(x, y, 4.5, 0, 2 * Math.PI);
    painter.fillStyle = '#ffd2a1';
    painter.fill();
    painter.strokeStyle = '#213743';
    painter.lineWidth = 1;
    painter.stroke();
    painter.fillStyle = '#172e39';
    painter.font = '600 13px IBM Plex Mono, monospace';
    painter.fillText(surface.routeLabels?.[index] ?? String(index + 1), x + 6, y - 7);
  });
  return { canvas, context, base, toCanvas };
}

function createView(data, surface, heightMin, heightMax) {
  const host = document.getElementById(`${surface.id}-render`);
  host.replaceChildren();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, .01, 20);
  camera.position.set(1.2, .92, 1.43);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xffffff, 0);
  host.appendChild(renderer.domElement);
  scene.add(new THREE.AmbientLight(0xffffff, 1.3));
  const light = new THREE.DirectionalLight(0xffffff, 2.0);
  light.position.set(-.5, 1.5, .7);
  scene.add(light);
  scene.add(makeMesh(surface, data.side, heightMin, heightMax));
  scene.add(makeGrid(surface, data.side));
  scene.add(makeNetwork(data, surface));
  const desired = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(routePoints(data, surface)),
    new THREE.LineDashedMaterial({ color: COLORS.route, dashSize: .024, gapSize: .014, depthTest: false }),
  );
  desired.computeLineDistances();
  desired.renderOrder = 4;
  scene.add(desired);
  const color = COLORS[surface.id];
  const { tube } = lineOnSurface(surface.geodesic, color);
  tube.renderOrder = 5;
  scene.add(tube);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(.014, 14, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }),
  );
  marker.renderOrder = 6;
  scene.add(marker);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, .07, 0);
  controls.enablePan = false;
  controls.minDistance = 1.3;
  controls.maxDistance = 4;
  controls.maxPolarAngle = Math.PI * .49;
  controls.update();
  controls.addEventListener('change', () => synchronizeCamera(surface.id));
  controls.addEventListener('start', () => setPlaying(false));
  const map = makeTopMap(data, surface, heightMin, heightMax, color);
  const path = surface.geodesic.map(([x, y, z]) => new THREE.Vector3(x - .5, DISPLAY_HEIGHT * z + .018, y - .5));
  const distances = [0];
  for (let i = 1; i < path.length; i++) distances.push(distances[i - 1] + path[i].distanceTo(path[i - 1]));
  const view = { host, scene, camera, renderer, controls, marker, map, path, distances, surface };
  resizeView(view);
  return view;
}

function resizeView(view) {
  const width = Math.max(1, view.host.clientWidth);
  const height = Math.max(1, view.host.clientHeight);
  view.camera.aspect = width / height;
  view.camera.updateProjectionMatrix();
  view.renderer.setSize(width, height, false);
}

function synchronizeCamera(sourceId) {
  if (syncingCamera) return;
  syncingCamera = true;
  const source = views.find((view) => view.surface.id === sourceId);
  if (source) {
    for (const view of views) {
      if (view === source) continue;
      view.camera.position.copy(source.camera.position);
      view.camera.quaternion.copy(source.camera.quaternion);
      view.controls.target.copy(source.controls.target);
      view.controls.update();
    }
  }
  syncingCamera = false;
}

function pathPosition(view, fraction) {
  const goal = fraction * view.distances[view.distances.length - 1];
  let index = 1;
  while (index < view.distances.length - 1 && view.distances[index] < goal) index++;
  const start = view.distances[index - 1];
  const end = view.distances[index];
  const blend = end === start ? 0 : (goal - start) / (end - start);
  return view.path[index - 1].clone().lerp(view.path[index], blend);
}

function updateProgress() {
  timeline.value = Math.round(progress * 1000);
  timeReadout.textContent = `00:${String(Math.floor(progress * DURATION)).padStart(2, '0')} / 00:08`;
  for (const view of views) {
    const position = pathPosition(view, progress);
    view.marker.position.copy(position);
    view.marker.position.y += .018;
    const { context, canvas, base, toCanvas } = view.map;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(base, 0, 0);
    const [x, y] = toCanvas([position.x + .5, position.z + .5]);
    context.beginPath();
    context.arc(x, y, 6.8, 0, 2 * Math.PI);
    context.fillStyle = '#ffffff';
    context.fill();
    context.strokeStyle = '#17313d';
    context.lineWidth = 2;
    context.stroke();
  }
}

function setPlaying(value) {
  playing = value;
  playButton.innerHTML = value ? 'Pause <span aria-hidden="true">Ⅱ</span>' : 'Play <span aria-hidden="true">▶</span>';
  playButton.setAttribute('aria-label', value ? 'Pause animation' : 'Play animation');
}

function setComparison(data) {
  validate(data);
  comparison = data;
  for (const view of views) {
    view.controls.dispose();
    view.renderer.dispose();
  }
  views.length = 0;
  let minimum = Infinity, maximum = -Infinity;
  for (const surface of data.surfaces) {
    for (const height of surface.heights) {
      minimum = Math.min(minimum, height);
      maximum = Math.max(maximum, height);
    }
  }
  for (const id of panelIds) {
    const surface = data.surfaces.find((item) => item.id === id);
    document.querySelector(`.manifold-panel.${id}`).hidden = !surface;
    if (!surface) continue;
    views.push(createView(data, surface, minimum, maximum));
    document.getElementById(`${id}-title`).textContent = surface.label;
    document.getElementById(`${id}-distance`).textContent = `Geodesic ${surface.geodesicLength.toFixed(3)}`;
    document.getElementById(`${id}-baseline-fraction`).textContent = `${(100 * surface.baselineCorridorFraction).toFixed(1)}%`;
    document.getElementById(`${id}-fraction`).textContent = `${(100 * surface.corridorFraction).toFixed(1)}%`;
    document.getElementById(`${id}-baseline-deviation`).textContent = surface.baselineRouteDeviation.toFixed(3);
    document.getElementById(`${id}-deviation`).textContent = surface.routeDeviation.toFixed(3);
    document.getElementById(`${id}-route`).textContent = surface.routeLabels.join(' → ');
    document.getElementById(`${id}-curvature`).textContent = formatLoss(surface.optimization?.curvature_loss);
    document.getElementById(`${id}-smooth`).textContent = formatLoss(surface.optimization?.smooth_loss);
  }
  document.querySelector('.eyebrow').textContent = data.title;
  document.getElementById('surface-count').textContent = String(data.surfaces.length);
  document.getElementById('height-change').textContent = data.heightChange?.maximumAbsolute.toFixed(4) ?? '—';
  document.querySelector('.experiment-note').textContent = data.note ?? '';
  progress = 0;
  updateProgress();
  loadError.hidden = true;
}

playButton.addEventListener('click', () => setPlaying(!playing));
timeline.addEventListener('input', () => {
  setPlaying(false);
  progress = Number(timeline.value) / 1000;
  updateProgress();
});
document.getElementById('comparison-file').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    setComparison(JSON.parse(await file.text()));
  } catch (error) {
    loadError.textContent = error.message;
    loadError.hidden = false;
  }
});
window.addEventListener('resize', () => views.forEach(resizeView));

function animate(now) {
  const elapsed = Math.min((now - lastFrame) / 1000, .1);
  lastFrame = now;
  if (playing && views.length) {
    progress = (progress + elapsed / DURATION) % 1;
    updateProgress();
    const first = views[0];
    const offset = first.camera.position.clone().sub(first.controls.target);
    const angle = Math.atan2(offset.z, offset.x) + elapsed * 2 * Math.PI / DURATION;
    const radius = Math.hypot(offset.x, offset.z);
    first.camera.position.set(
      first.controls.target.x + radius * Math.cos(angle),
      first.controls.target.y + offset.y,
      first.controls.target.z + radius * Math.sin(angle),
    );
    first.controls.update();
    synchronizeCamera(first.surface.id);
  }
  for (const view of views) view.renderer.render(view.scene, view.camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

fetch(`${import.meta.env.BASE_URL}route-comparison.json`)
  .then((response) => {
    if (!response.ok) throw new Error('Could not load the example comparison.');
    return response.json();
  })
  .then(setComparison)
  .catch((error) => {
    loadError.textContent = error.message;
    loadError.hidden = false;
  });
