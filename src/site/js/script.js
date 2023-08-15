import { EffectComposer } from './scripts/EffectComposer.js';
import { RenderPass } from './scripts/RenderPass.js';
import { ShaderPass } from './scripts/ShaderPass.js';
import { CopyShader } from './scripts/CopyShader.js';
import { FXAAShader } from './scripts/FXAAShader.js';
import { LineGeometry } from './scripts/LineGeometry.js';
import { Line2 } from './scripts/Line2.js';
import { LineMaterial } from './scripts/LineMaterial.js';
import { SelectionBox } from './scripts/SelectionBox.js';
import { SelectionHelper } from './scripts/SelectionHelper.js';
import RBush from './scripts/rbush/index.js'
import quickselect from './scripts/quickselect/index.js'
import Draw from './scripts/ol/interaction/Draw.js';
import { transformExtent } from './scripts/ol/proj.js';

let bgcolor = 0xf3f3f3;
let graphcolor = 0xffffff;
let vertexcolor = 0x4CAF50;
let edgecolor = 0x21bf73;
let edgecolor_sec = 0x4cd995;
let canvascolor = "#ffffff";
let contcolor = 0xff0000;

let T = THREE;

// TODO: Remove this when removing the extended plane
let useSplitEdges = false;

let showHoverGraph = true;

let vertexCount = 0;
let edgeCount = 0;

let zoomWidths = [];
let zoomHeights = [];
let zoomLevels = [];

let planeXMin = -10;
let planeXMax = 10;
let planeYMin = -10;
let planeYMax = 10;
let planeW = planeXMax - planeXMin;
let planeH = planeYMax - planeYMin;
let divisions = 50;
let heightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
let opacityMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
let curvMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
let refine_data = {};

let calcHeightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));

let vertexHeight = 3;

// Cycle variables for threshold cycle
let cycle = false;
let last_cycle = Date.now();

// Flag to indicate threshold change -> update heights when thresh changed
let thresh_change = false;

// ThreeJS Scene Setup
var scene = new T.Scene();
let div = 64;
var camera = new THREE.OrthographicCamera(
  window.innerWidth / -div, window.innerWidth / div,
  window.innerHeight / div, window.innerHeight / -div,
  0.1, 1000
);
camera.position.x = -15;
camera.position.z = 20;
camera.position.y = 15;

var renderer = new THREE.WebGLRenderer({
  logarithmicDepthBuffer: true,
  antialias: false
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.autoClear = false;
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

scene.background = new THREE.Color(bgcolor);
var controls = new T.OrbitControls(camera, renderer.domElement);

const olMap = createMap();
const canv = document.createElement('canvas');
canv.id = "canvas-texture";
let ctx = canv.getContext('2d');
ctx.canvas.width = 4000;
ctx.canvas.height = 4000;
ctx.fillStyle = canvascolor;
ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
var texture = new T.CanvasTexture(ctx.canvas);
texture.minFilter = THREE.LinearFilter;
texture.center = new T.Vector2(0.5, 0.5)
texture.rotation = -Math.PI / 2

let ptGeom = new T.SphereGeometry(0.02, 32, 32);
let ptMat = new T.MeshBasicMaterial({
  color: vertexcolor
});

var clipPlaneUp = new T.Plane(new T.Vector3(0, 2, 0), 2);
var clipPlaneLeft = new T.Plane(new T.Vector3(-1, 0, 0), -planeXMin);
var clipPlaneRight = new T.Plane(new T.Vector3(1, 0, 0), planeXMax);
var clipPlaneFront = new T.Plane(new T.Vector3(0, 0, -1), -planeYMin);
var clipPlaneBack = new T.Plane(new T.Vector3(0, 0, 1), planeYMax);

var loader = new T.ImageLoader();
var aMap = loader.load("images/grayscale.png");

var geometry = new T.PlaneGeometry(
  planeW, planeH,
  divisions - 1, divisions - 1
);
var material = new T.MeshBasicMaterial({
  color: graphcolor,
  side: T.DoubleSide
});
// TODO: change back
var planeMat = new THREE.MeshPhongMaterial({
  color: graphcolor,
  clippingPlanes: [clipPlaneLeft, clipPlaneRight, clipPlaneFront, clipPlaneBack],
  vertexColors: T.VertexColors,
  side: THREE.DoubleSide,
  flatShading: false,
  shininess: 0,
  wireframe: false,
  map: texture,
  transparent: true,
  opacity: 1.0
}); //, alphaMap: T.ImageUtils.loadTexture("images/grayscale.png")} )
let transparentMat = new T.MeshLambertMaterial({
  visible: false
});
var transparentPlaneMat = new THREE.MeshPhongMaterial({
  color: graphcolor,
  clippingPlanes: [clipPlaneLeft, clipPlaneRight, clipPlaneFront, clipPlaneBack],
  vertexColors: T.VertexColors,
  side: THREE.DoubleSide,
  flatShading: false,
  shininess: 0,
  wireframe: false,
  map: texture,
  transparent: true,
  opacity: 0.5
});
let mMat = [planeMat, transparentMat, transparentPlaneMat];
var plane = new T.Mesh(geometry, mMat);

plane.rotation.set(-Math.PI / 2, 0, 0);
scene.add(plane);

controls.update();

let light = new T.PointLight(0xffffff, 0.5);
light.position.set(-7, 8, 0);
scene.add(light);

const directionalLight = new T.DirectionalLight(0xffffff, 0.6);
directionalLight.position.set(1.5, 0.1, 10);
scene.add(directionalLight);

var alight = new THREE.AmbientLight(0x404040); // soft white light
scene.add(alight);

let vertices = {};
let edges = {};
let graphs = [];
let names = {};
let linesDrawn = [];
let linesCleared = true;
let subPlanes = [];
let current_edges = {};

edgecolor_sec = 0x2cc57c;
edgecolor = 0x178e51;
var lineMat = new T.LineBasicMaterial({
  color: edgecolor,
  linewidth: 6,
  clippingPlanes: [clipPlaneUp]
});
var lineMatSec = new T.LineBasicMaterial({
  color: edgecolor_sec,
  linewidth: 1.5,
  depthFunc: T.LessDepth
});

var contourMeshLines = [];

plane.geometry.dynamic = true;

// Extended plane for wraparound distance calculation
// TODO: Remove this, as this tool should only support planar
// geometries and not cylindrical ones
var extendGeom = new T.PlaneGeometry(
  planeW, planeH * 2,
  divisions - 1, 2 * divisions * 2 - 1
);
var ePlane = new T.Mesh(extendGeom, new THREE.MeshPhongMaterial({
  color: 0x00ff00
}));
ePlane.rotation.set(-Math.PI / 2, 0, 0);
ePlane.position.z += 10;

for (let face of plane.geometry.faces) {
  face.vertexColors[0] = new T.Color(0xffffff);
  face.vertexColors[1] = new T.Color(0xffffff);
  face.vertexColors[2] = new T.Color(0xffffff);
}

var contX = [];
var contY = [];
for (let i = 0; i < heightMap.length; i++) {
  contX.push(i);
  contY.push(i);
}

// Composers and FXAA shader
let renderPass = new RenderPass(scene, camera);

let fxaaPass = new ShaderPass(FXAAShader);
fxaaPass.renderToScreen = false;
let pixelRatio = renderer.getPixelRatio();
fxaaPass.material.uniforms['resolution'].value.x = 1 / (window.innerWidth * pixelRatio);
fxaaPass.material.uniforms['resolution'].value.y = 1 / (window.innerHeight * pixelRatio);

let composer = new EffectComposer(renderer);
composer.addPass(renderPass);
composer.addPass(fxaaPass);

window.onload = function() {
  var mapCanvas = document.getElementById('map').getElementsByTagName('canvas')[0];

  let ctx = mapCanvas.getContext('2d');
  const customTexture = texture;
  const mapTexture = new T.CanvasTexture(ctx.canvas);
  mapTexture.minFilter = THREE.LinearFilter;
  mapTexture.magFilter = THREE.NearestFilter;
  mapTexture.center = new T.Vector2(0.5, 0.5);
  mapTexture.rotation = -Math.PI / 2;
  plane.material[2].map = mapTexture;

  var mapdiv = document.getElementById("map");

  mapdiv.style.display = "none";

  // TODO: Stack for -ve zoom levels for consistency
  // TODO: Enable
  // window.addEventListener('wheel', wheelEvent, true);

  var dropNodes = document.getElementById('drop-nodes');
  dropNodes.addEventListener('dragover', dragOver, false);
  dropNodes.addEventListener('drop', fileSelectNodes, false);

  var dropEdges = document.getElementById('drop-edges');
  dropEdges.addEventListener('dragover', dragOver, false);
  dropEdges.addEventListener('drop', fileSelectEdges, false);

  let btnAddVertex = document.getElementById("btn-add-vertex");
  btnAddVertex.onclick = addVertex;

  let btnAddEdge = document.getElementById("btn-add-edge");
  btnAddEdge.onclick = addEdge;

  document.getElementById("btn-calc-dist").onclick = function() {
    if (subPlanes.length != 0) {
      calcDistanceOnSurface(subPlanes[0].plane, vertices, current_edges);
    } else {
      calcDistanceOnSurface(plane, vertices, current_edges);
    }
  }

  let hideSurface = document.getElementById("hide-surface");
  let chkCalcSurface = document.getElementById("use-calc-surface");
  let useTransp = document.getElementById("use-transparency");
  let showMap = document.getElementById("show-map");
  let showGraph = document.getElementById("show-graph");

  showGraph.onchange = function() {
    if (vertices[0] != undefined) {
      for (let id in vertices) {
        if (!showGraph.checked) {
          scene.remove(vertices[id].mesh);
          scene.remove(vertices[id].label);
        } else {
          scene.add(vertices[id].mesh);
          scene.add(vertices[id].label);
        }
      }
    }
  }

  document.getElementById("threshold-slider").onchange = function() {
    thresh_change = true;
  }

  let vertexControlDiv = document.getElementById("div-vertex");
  let edgeControlDiv = document.getElementById("div-edge");

  let btnGenGraph = document.getElementById("btn-gen-graph");
  btnGenGraph.onclick = generateGraph;

  let btnGenGraphEmpty = document.getElementById("btn-gen-graph-empty");
  btnGenGraphEmpty.onclick = generateGraphNoWeights;

  let btnCalcCurv = document.getElementById("btn-calc-curv");
  btnCalcCurv.onclick = calculateCurvature;

  let btnHelp = document.getElementById("btn-help");
  btnHelp.onclick = helpClick;

  document.getElementById("btn-cycle-thresholds").onclick = cycleThresholds;

  document.getElementById("btn-calc-surface").onclick = calcSurface;

  controls.enablePan = true;
  controls.panSpeed = 1;
  controls.enableRotate = true;
  controls.enableZoom = true;
  controls.minZoom = 1;
  controls.update();

  var animate = function() {
    updateSliderVals();

    if (showMap.checked) {
      plane.material[0].map = mapTexture;
      texture = mapTexture;
      ctx = mapCanvas.getContext("2d");
    } else {
      plane.material[0].map = customTexture;
      texture = customTexture;
      ctx = canv.getContext("2d");
    }
    plane.material[0].needsUpdate = true;
    plane.material[2].needsUpdate = true;

    curvMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));

    requestAnimationFrame(animate);

    controls.update();

    // Update thresholds if cycling
    if (cycle && (Date.now() - last_cycle) / 1000 > 2) {
      let slider = document.getElementById("threshold-slider");
      let value = parseInt(slider.value);
      value += 1;
      value %= (parseInt(slider.max) + 1);
      slider.value = value;
      last_cycle = Date.now();
      thresh_change = true;
    }

    ctx.fillStyle = canvascolor

    if (graphs.length == 0) {
      current_edges = {
        ...edges
      }
    } else {
      current_edges = {
        ...graphs[document.getElementById("threshold-slider").value].edges
      };
    }

    if (!showMap.checked) {
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }

    // Draws grid lines
    if (document.getElementById("show-grid").checked) {
      drawGrid(ctx.canvas);
    }

    ctx.setLineDash([]);

    // Draw physical graph edge, texture edge
    for (let id in current_edges) {
      let lineWidth = 12;
      let borders = true;
      let edge = current_edges[id];

      if (showHoverGraph && showGraph.checked && (edge.mesh == null || linesCleared)) {
        edge.mesh = drawEdge(edge, lineMat);
      }

      let startPt = [parseFloat(edge.start.mesh.position.x), parseFloat(edge.start.mesh.position.z)];
      let endPt = [parseFloat(edge.end.mesh.position.x), parseFloat(edge.end.mesh.position.z)];

      // Draw texture edge // TODO: undo
      startPt = [
        (1 - (startPt[0] - planeXMin) / planeW) * ctx.canvas.width,
        (startPt[1] - planeYMin) * ctx.canvas.height / planeH
      ];
      endPt = [
        (1 - (endPt[0] - planeXMin) / planeW) * ctx.canvas.width,
        (endPt[1] - planeYMin) * ctx.canvas.height / planeH
      ];
      ctx.save();
      ctx.globalAlpha = 1.0;
      ctx.beginPath();

      if (borders) {
        ctx.moveTo(startPt[1], startPt[0]);
        ctx.lineTo(endPt[1], endPt[0]);
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = lineWidth + 1;
        ctx.stroke();
      }

      ctx.moveTo(startPt[1], startPt[0]);
      ctx.lineTo(endPt[1], endPt[0]);
      let color = new T.Color();
      let endColor;
      if (edge.weight >= 0) {
        endColor = new T.Color("hsl(222, 100%, 61%)");
      } else {
        endColor = new T.Color("hsl(356, 74%, 52%)");
      }
      color.lerp(endColor, Math.min(Math.abs(edge.weight), 1));
      ctx.strokeStyle = "#" + color.getHexString();
      ctx.lineWidth = lineWidth;
      ctx.stroke();
      ctx.restore();
    }

    linesCleared = false;

    // Set plane vertices' height
    heightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.));

    // Draw point on surface texture
    for (let id in vertices) {
      let radius = 2;
      let vertex = vertices[id];
      let point = [parseFloat(vertex.mesh.position.x), parseFloat(vertex.mesh.position.z)];
      point = [
        (1 - (point[0] - planeXMin) / planeW) * ctx.canvas.width,
        (point[1] - planeYMin) * ctx.canvas.height / planeH
      ];
      ctx.fillStyle = "#FF5C5C";

      ctx.beginPath();
      ctx.arc(point[1], point[0], radius, 0, 2 * Math.PI);
      ctx.fill();
    }

    let map = heightMap;
    let useTransp = document.getElementById("use-transparency");

    if (document.getElementById("use-calc-surface").checked) {
      if (graphs.length > 0) {
        map = graphs[document.getElementById("threshold-slider").value].heightmap;
      } else {
        map = calcHeightMap;
      }
    }
    if (thresh_change) {
      thresh_change = false;
      updatePlaneHeights(map);
    }

    if (map == heightMap) {
      updatePlaneHeights(map);
    }

    if (showMap.checked) {
      // For now, disable opacity changes
      opacityMap = Array(divisions).fill().map(() => Array(divisions).fill(1.0));

      calcOpacityMap(opacityMap, vertices, current_edges, map);
      aMap = createAndUpdateAlphaMapD3(opacityMap);

      plane.material[0].alphaMap = aMap;
    }

    for (let p of subPlanes) {
      for (let i = p.start[0]; i < p.end[0]; i++) {
        for (let j = p.start[1]; j < p.end[1]; j++) {
          plane.geometry.vertices[j * divisions + i].z = 0;
        }
      }

      p.plane.material.alphaMap = aMap;
      p.plane.material.map.needsUpdate = true;
      p.plane.geometry.colorsNeedUpdate = true;
    }

    // colorCurvature(plane)

    plane.material[0].needsUpdate = true;
    texture.needsUpdate = true;

    plane.geometry.groupsNeedUpdate = true;
    plane.geometry.verticesNeedUpdate = true;
    plane.geometry.colorsNeedUpdate = true;
    plane.geometry.computeVertexNormals();

    // Render
    renderer.localClippingEnabled = hideSurface.checked;
    olMap.render();
    composer.render();
  }

  animate();
}

let selectionBox = new SelectionBox(camera, scene);
let helper = new SelectionHelper(selectionBox, renderer, 'selectBox');

document.addEventListener("keydown", function(event) {
  // Shift click for select
  if (event.key == "Shift") {
    controls.enablePan = false;
    controls.update();
    document.body.style.cursor = "crosshair";
    document.addEventListener("pointerdown", pointerDown);
    document.addEventListener("pointermove", pointerMove);
    document.addEventListener("pointerup", pointerUp);
  } else if (event.key == "Escape") {
    for (let i in subPlanes) {
      let sPlane = subPlanes[i].plane;
      gsap.to(sPlane.scale, {
        duration: 1,
        x: 0.1,
        y: 0.1,
        z: 0.1,
        onComplete: function() {
          scene.remove(sPlane);
        }
      });
    }
    // TODO: zoom width, heights, levels reset

    gsap.to(camera, {
      duration: 1,
      zoom: 1,
      onUpdate: function() {
        camera.updateProjectionMatrix();
      }
    });
    gsap.to(controls.target, {
      duration: 1,
      x: 0,
      y: 0,
      z: 0,
      onUpdate: function() {
        controls.update();
      }
    });
    gsap.to(plane.position, {
      duration: 1,
      y: 0,
      onStart: function() {
        plane.visible = true;
      },
      onUpdate: function() {}
    });

    olMap.getView().setZoom(0);
    subPlanes = [];
  }
});

document.addEventListener("keyup", function(event) {
  if (event.key == "Shift") {
    document.body.style.cursor = "auto";
    controls.enablePan = true;
    controls.update();
    document.removeEventListener('pointerdown', pointerDown);
    document.removeEventListener('pointermove', pointerMove);
    document.removeEventListener('pointerup', pointerUp);
  }
});

function updateSliderVals() {
  document.getElementById("posheight-slider-val").innerHTML = parseFloat(document.getElementById("posheight-slider").value).toFixed(2);
}

function cycleThresholds() {
  let btnCycle = document.getElementById("btn-cycle-thresholds");
  cycle = !cycle;
  if (!cycle) {
    btnCycle.innerHTML = "Cycle Thresholds";
  } else {
    btnCycle.innerHTML = "Stop Cycle";
  }
  last_cycle = Date.now();
}

// TODO: This function is currently not in use. The curvature
// computation should probably be moved into the backend, as code for
// that already exists there. The colors are also likely slightly
// imprecise. Finally, this function currently relies on the size of the
// mesh to be 50x50, so it needs to be rewritten to take `divisions`
// into account
function colorCurvature(plane) {
  // CURVATURE calc
  let max_curv = 0;
  let min_curv = 1;
  for (let i = 9; i < 39; i++) {
    for (let j = 9; j < 39; j++) {
      // console.log(plane.geometry.vertices[i].x + " " + plane.geometry.vertices[i].y)
      let o = [plane.geometry.vertices[i * 50 + j].x, plane.geometry.vertices[i * 50 + j].y, plane.geometry.vertices[i * 50 + j].z];
      let a = [plane.geometry.vertices[(i - 1) * 50 + j - 1].x, plane.geometry.vertices[(i - 1) * 50 + j - 1].y, plane.geometry.vertices[(i - 1) * 50 + j - 1].z];
      let b = [plane.geometry.vertices[(i - 1) * 50 + j].x, plane.geometry.vertices[(i - 1) * 50 + j].y, plane.geometry.vertices[(i - 1) * 50 + j].z];
      let c = [plane.geometry.vertices[i * 50 + j - 1].x, plane.geometry.vertices[i * 50 + j - 1].y, plane.geometry.vertices[i * 50 + j - 1].z];
      let d = [plane.geometry.vertices[i * 50 + j + 1].x, plane.geometry.vertices[i * 50 + j + 1].y, plane.geometry.vertices[i * 50 + j + 1].z];
      let e = [plane.geometry.vertices[(i + 1) * 50 + j].x, plane.geometry.vertices[(i + 1) * 50 + j].y, plane.geometry.vertices[(i + 1) * 50 + j].z];
      let f = [plane.geometry.vertices[(i + 1) * 50 + j + 1].x, plane.geometry.vertices[(i + 1) * 50 + j + 1].y, plane.geometry.vertices[(i + 1) * 50 + j + 1].z];
      let vxs = [a, b, d, f, e, c];
      let angles = [];

      for (let k = 0; k < vxs.length; k++) {
        let posB = vxs[k];
        let posC = vxs[(k + 1) % vxs.length];
        let vec1 = [posB[0] - o[0], posB[1] - o[1], posB[2] - o[2]];
        let vec2 = [posC[0] - o[0], posC[1] - o[1], posC[2] - o[2]];
        let dot = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
        let norm_v1 = Math.sqrt(Math.pow(vec1[0], 2) + Math.pow(vec1[1], 2) + Math.pow(vec1[2], 2));
        let norm_v2 = Math.sqrt(Math.pow(vec2[0], 2) + Math.pow(vec2[1], 2) + Math.pow(vec2[2], 2));
        let angle1 = Math.acos(dot / (norm_v1 * norm_v2));
        angles.push(angle1);
      }
      let curvature = 2 * Math.PI - angles.reduce((a, b) => a + b);
      plane.geometry.vertices[i * 50 + j].curvature = curvature;
      max_curv = Math.max(max_curv, curvature);
      min_curv = Math.min(min_curv, curvature);
    }
  }

  for (let face of plane.geometry.faces) {
    const red = new THREE.Color(0xdf2935);
    const blue = new THREE.Color(0x3772ff);
    let curv = plane.geometry.vertices[face.a].curvature;
    let scale = 2;
    if (face.vertexColors[0] == undefined) {
      face.vertexColors[0] = new THREE.Color(1, 1, 1);
      face.vertexColors[1] = new THREE.Color(1, 1, 1);
      face.vertexColors[2] = new THREE.Color(1, 1, 1);
    }
    face.vertexColors[0].setRGB(1, 1, 1);
    if (curv > 0.01) {
      face.vertexColors[0].lerp(blue, Math.ceil(scale * curv / max_curv) / scale);
    } else if (curv < -0.01)
      face.vertexColors[0].lerp(red, Math.ceil(scale * curv / min_curv) / scale);

    curv = plane.geometry.vertices[face.b].curvature;
    face.vertexColors[1].setRGB(1, 1, 1);
    if (curv > 0.01)
      face.vertexColors[1].lerp(blue, Math.ceil(scale * curv / max_curv) / scale);
    else if (curv < -0.01)
      face.vertexColors[1].lerp(red, Math.ceil(scale * curv / min_curv) / scale);

    curv = plane.geometry.vertices[face.c].curvature;
    face.vertexColors[2].setRGB(1, 1, 1);
    if (curv > 0.01)
      face.vertexColors[2].lerp(blue, Math.ceil(scale * curv / max_curv) / scale);
    else if (curv < -0.01)
      face.vertexColors[2].lerp(red, Math.ceil(scale * curv / min_curv) / scale);
  }
}

function drawGrid(canvas) {
  let ctx = canvas.getContext("2d");
  let cols = divisions - 1;
  let rows = divisions - 1;

  let width = canvas.width;
  let height = canvas.height;
  let start_i = 0;
  let start_j = 0;
  if (subPlanes.length > 0) {
    let sp = subPlanes[0];
    start_i = (sp.ystart3JS - planeXMin) / planeW * canvas.width;
    width = (sp.ystart3JS + sp.height - planeXMin) / planeW * canvas.width;
    start_j = (-sp.xstart3JS - planeYMin) / planeH * canvas.height;
    height = (-sp.xstart3JS - sp.width - planeYMin) / planeH * canvas.height;
    if (height < start_j) {
      let temp = height;
      height = start_j;
      start_j = temp;
    }
  }

  ctx.beginPath();
  for (let i = start_i; i <= width; i += (width - start_i) / cols) {
    ctx.moveTo(i, 0);
    ctx.lineTo(i, canvas.height);
  }
  for (let j = start_j; j <= height; j += (height - start_j) / rows) {
    ctx.moveTo(0, j);
    ctx.lineTo(canvas.width, j);
  }
  ctx.lineWidth = 4;
  ctx.strokyStyle = "black";
  ctx.stroke();
}

function updatePlaneHeights(map) {
  let useTransp = document.getElementById("use-transparency");

  for (let i = 0; i < divisions; i++) {
    for (let j = 0; j < divisions; j++) {
      gsap.to(plane.geometry.vertices[i * divisions + j], {
        duration: 0.25,
        z: map[i][j],
      });
    }
  }

  // Set extended plane
  // TODO: This should be removed when the rest of the extended plane
  // code is
  for (let i = 0; i < divisions; i++) {
    for (let j = 0; j < divisions; j++) {
      ePlane.geometry.vertices[(i + divisions) * divisions + j].z = map[i][j];
      ePlane.geometry.vertices[i * divisions + j].z = map[i][j];
    }
  }

  ePlane.geometry.groupsNeedUpdate = true;
  ePlane.geometry.verticesNeedUpdate = true;
  ePlane.geometry.colorsNeedUpdate = true;
  ePlane.geometry.computeVertexNormals();

  for (let face of plane.geometry.faces) {
    let z1 = plane.geometry.vertices[face.a].z;
    let z2 = plane.geometry.vertices[face.b].z;
    let z3 = plane.geometry.vertices[face.c].z;
    let hide = false;
    let v = face.a;
    let i = Math.floor(v / divisions);
    let j = v % divisions;

    let transparent = true;
    let points = [];
    points.push([Math.floor(i), Math.floor(j)]);
    points.push([Math.ceil(i), Math.ceil(j)]);
    points.push([Math.floor(i), Math.ceil(j)]);
    points.push([Math.ceil(i), Math.floor(j)]);
    // if (opacityMap[Math.floor(i)][Math.floor(j)] == 1)
    //   transparent = false
    v = face.b;
    i = v / divisions;
    j = v % divisions;
    points.push([Math.floor(i), Math.floor(j)]);
    points.push([Math.ceil(i), Math.ceil(j)]);
    points.push([Math.floor(i), Math.ceil(j)]);
    points.push([Math.ceil(i), Math.floor(j)]);
    // if (opacityMap[Math.floor(i)][Math.floor(j)] == 1)
    //   transparent = false
    // if ((i < xlimit || i > heightMap.length - xlimit) || (j < ylimit || j > heightMap[0].length - ylimit))
    //   hide = true
    v = face.c;
    i = v / divisions;
    j = v % divisions;
    points.push([Math.floor(i), Math.floor(j)]);
    points.push([Math.ceil(i), Math.ceil(j)]);
    points.push([Math.floor(i), Math.ceil(j)]);
    points.push([Math.ceil(i), Math.floor(j)]);
    // if (opacityMap[Math.floor(i)][Math.floor(j)] == 1)
    //   transparent = false
    for (let p of points) {
      if (0 <= p[0] && p[0] < divisions && 0 <= p[1] && p[1] < divisions) {
        if (opacityMap[p[0]][p[1]] == 1 || Math.abs(map[p[0]][p[1]]) > 0.5) {
          transparent = false;
        }
      }
    }
    // if ((i < xlimit || i > heightMap.length - xlimit) || (j < ylimit || j > heightMap[0].length - ylimit))
    //   hide = true
    if (false && hideSurface.checked && Math.abs(z1) == 0 && Math.abs(z2) == 0 && Math.abs(z3) == 0) {
      face.materialIndex = 1; // Transparent
    } else if (false && hideSurface.checked && (Math.abs(z2 - z1) > 0.5 || Math.abs(z3 - z1) > 0.5)) { // Extra condition for tests
      face.materialIndex = 1;
    } else if (false && hideSurface.checked && (Math.abs(z2 - z1) + Math.abs(z3 - z1) + Math.abs(z3 - z2) > 0.8)) { // Extra condition for tests
      face.materialIndex = 1;
    } else if (false && hideSurface.checked && (z1 == 0 || z2 == 0 || z3 == 0)) { // Extra condition for tests // Was true
      face.materialIndex = 1;
    } else if (false && hide && hideSurface.checked) {
      face.materialIndex = 1;
    } else if (false && hideSurface.checked && (z1 < -2.4 || z2 < -2.4 || z3 < -2.4)) { // Inward edge
      face.materialIndex = 1;
    } else if (transparent && useTransp.checked) {
      face.materialIndex = 2;
    } else {
      face.materialIndex = 0;
    }
  }
}

function helpClick(event) {
  let helpDiv = document.getElementById("div-help");
  if (helpDiv.style.display === "none") {
    helpDiv.style.display = "block";
  } else {
    helpDiv.style.display = "none";
  }
}

function pointerDown(event) {
  if (event.which != 1) {
    return;
  }
  for (var item of selectionBox.collection) {
    if (Array.isArray(item.material)) {
      continue;
    }
    item.material = item.material.clone();
  }

  selectionBox.startPoint.set(
    (event.clientX / window.innerWidth) * 2 - 1,
    -(event.clientY / window.innerHeight) * 2 + 1,
    0.5
  );
}

function pointerMove(event) {
  if (helper.isDown) {
    for (var i = 0; i < selectionBox.collection.length; i++) {
      if (Array.isArray(selectionBox.collection[i].material)) {
        continue
      }
      selectionBox.collection[i].material = selectionBox.collection[i].material.clone();
    }

    selectionBox.endPoint.set(
      (event.clientX / window.innerWidth) * 2 - 1,
      -(event.clientY / window.innerHeight) * 2 + 1,
      0.5
    );

    var allSelected = selectionBox.select();
    for (var i = 0; i < allSelected.length; i++) {
      if (Array.isArray(allSelected[i].material)) {
        continue;
      }
      allSelected[i].material = allSelected[i].material.clone();
    }
  }
}

function pointerUp(event) {
  if (event.which != 1) {
    return;
  }
  selectionBox.endPoint.set(
    (event.clientX / window.innerWidth) * 2 - 1,
    -(event.clientY / window.innerHeight) * 2 + 1,
    0.5
  );

  var allSelected = selectionBox.select();
  for (var i = 0; i < allSelected.length; i++) {
    if (Array.isArray(allSelected[i].material)) {
      continue;
    }
  }
  subgraphSelect(allSelected);
}

// TODO: This function needs to be rewritten. Currently, it increases
// the resolution of the map, but it also changes the viewing window
// itself.
function wheelEvent(event) {
  olMap.updateSize();

  if (document.elementFromPoint(event.clientX, event.clientY).tagName != 'CANVAS') {
    return;
  }
  var mapdiv = document.getElementById("map")
  mapdiv.style.display = "block"
  if (event.deltaY > 0) { // zoom out
    if (zoomLevels.length != 0) {
      mapdiv.style.width = zoomWidths.pop() + 'px';
      mapdiv.style.height = zoomHeights.pop() + 'px';
      olMap.updateSize();
      olMap.getView().setZoom(zoomLevels.pop());
    }
    // olMap.getView().setCenter(ol.proj.fromLonLat([87.6, 41.8]))
  } else { // zoom in
    zoomWidths.push(mapdiv.offsetWidth);
    zoomHeights.push(mapdiv.offsetHeight);
    zoomLevels.push(olMap.getView().getZoom());
    olMap.getView().setZoom(olMap.getView().getZoom() + 0.05);
    mapdiv.style.width = mapdiv.offsetWidth * 1.05 + 'px';
    mapdiv.style.height = mapdiv.offsetHeight * 1.05 + 'px';
    olMap.updateSize();
    let p1 = ol.proj.fromLonLat([0, 0]);
    let p2 = ol.proj.fromLonLat([100 / (2 ** zoomLevels.length), 100 / (2 ** zoomLevels.length)]);
    let extents = [p1[0], p1[1], p2[0], p2[1]];
    // olMap.getLayers().array_[0].setExtent(extents);
    // olMap.getView().setCenter(ol.proj.fromLonLat([87.6, 41.8]));
  }
  mapdiv.style.display = "none";
}

function calcOpacityMap(opacityMap, vertices, edges, heightMap) {
  for (let id in edges) {
    if (edges[id].split) {
      continue;
    }
    let startPt = [edges[id].start.mesh.position.x, edges[id].start.mesh.position.z];
    let endPt = [edges[id].end.mesh.position.x, edges[id].end.mesh.position.z];

    startPt = convert3JStoOM(startPt, divisions);
    endPt = convert3JStoOM(endPt, divisions);
    for (let i = 0; i < divisions; i++) {
      for (let j = 0; j < divisions; j++) {
        if (Math.abs(dist(startPt, [i, j]) + dist([i, j], endPt) - dist(startPt, endPt)) < 0.1) {
          opacityMap[j][i] = 1;
        }
        if (heightMap[Math.floor(j / 2)][Math.floor(i / 2)] > 0.3) {
          opacityMap[j][i] = 1;
        }
      }
    }
  }
}

function calcDistanceOnSurface(plane, vertices, edges) {
  let verts = [];
  let faces = [];
  let nodes = []; // Array of mesh positions of nodes/vertices of graphs
  let send_edges = [];
  for (let vert of plane.geometry.vertices) {
    verts.push([vert.x, vert.y, vert.z]);
  }
  for (let face of plane.geometry.faces) {
    faces.push([face.a, face.b, face.c]);
  }
  for (let vert in vertices) {
    let cur_node = vertices[vert];
    nodes.push(convert3JStoVerts(cur_node.mesh.position.x, cur_node.mesh.position.z));
  }
  plane.geometry.verticesNeedUpdate = true;
  for (let id in edges) {
    if (edges[id].split) {
      continue;
    }
    let start = edges[id].start;
    let end = edges[id].end;
    let startNode = convert3JStoVerts(start.mesh.position.x, start.mesh.position.z);
    let endNode = convert3JStoVerts(end.mesh.position.x, end.mesh.position.z);
    send_edges.push([startNode, endNode]);
  }

  let send_data = {
    verts: verts,
    faces: faces,
    nodes: nodes,
    edges: send_edges
  };
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.responseType = "text"

  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
      let data = xmlHttp.responseText;
      data = JSON.parse(data);
      let distances = data.distances;
      let grads = data.grads;
      let paths = data.paths;
      drawSurfacePathsFlip(paths, edges);
    }
  }
  xmlHttp.open("post", "calc-distance");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xmlHttp.send(JSON.stringify(send_data));
}

function drawSurfacePathsFlip(paths, edges) {
  for (let i = 0, j = 0; i < paths.length; i++, j++) {
    let points = [];
    let line_color = new T.Color("hsl(0, 0%, 100%)");
    if (edges[j].weight >= 0) {
      var endColor = new T.Color("hsl(222, 100%, 61%)");
    } else {
      var endColor = new T.Color("hsl(356, 74%, 52%)");
    }
    line_color.lerp(endColor, Math.min(Math.abs(edges[j].weight), 1));
    let material = new T.LineBasicMaterial({
      color: line_color,
      linewidth: 4,
      linecap: 'round'
    });
    for (let pt of paths[i]) {
      if (subPlanes.length != 0) {
        pt[2] += subPlanes[0].plane.position.y + 0.01;
        pt[0] += subPlanes[0].plane.position.x;
        pt[1] -= subPlanes[0].plane.position.z;
      }
      points.push(new T.Vector3(pt[0], pt[2], -pt[1]));
    }
    const geometry = new T.BufferGeometry().setFromPoints(points);
    const line = new T.Line(geometry, material);
    scene.add(line);
  }
}

function subgraphSelect(selected) {
  if (selected.length <= 1) {
    return;
  }
  var data = {
    nodes: [],
    links: []
  };
  var nodes = {};
  var edges = [];
  let ids = {};
  let xRange = [planeXMax, planeXMin];
  let yRange = [planeYMax, planeYMin];

  // Get nodes and edges from selected shapes
  for (let id in selected) {
    if (selected[id].geometry.type == "SphereGeometry") {
      xRange[0] = Math.min(xRange[0], selected[id].position.x);
      xRange[1] = Math.max(xRange[1], selected[id].position.x);
      yRange[0] = Math.min(yRange[0], selected[id].position.z);
      yRange[1] = Math.max(yRange[1], selected[id].position.z);

      data.nodes.push({
        id: parseInt(id),
        city: selected[id].name,
        lat: parseFloat(selected[id].position.x),
        long: parseFloat(selected[id].position.z)
      });
      ids[selected[id].name] = parseInt(id);
      nodes[selected[id].name] = {
        lat: parseFloat(selected[id].position.x),
        long: parseFloat(selected[id].position.z),
        name: selected[id].name
      };
    } else if (selected[id].geometry.type == "BufferGeometry") {
      let start = selected[id].name.split('/')[0];
      let end = selected[id].name.split('/')[1];
      let weight = selected[id].userData.weight;
      let neg_mod = selected[id].userData.neg_mod;
      let nrw_mod = selected[id].userData.nrw_mod;
      let nheight_mod = selected[id].userData.nheight_mod;
      if (ids[start] == undefined || ids[end] == undefined) {
        continue;
      }
      let startname = Math.min(nodes[start].name, nodes[end].name);
      let endname = Math.max(nodes[start].name, nodes[end].name);

      if (refine_data[startname] && refine_data[startname][endname]) {
        let ref = refine_data[startname][endname];

        if (ref > 0) {
          neg_mod = 1.2 * neg_mod;
          nrw_mod = 1.2 * nrw_mod;
          nheight_mod = 1.2 * nheight_mod;
        } else if (ref < -1) {
          neg_mod = 0.8 * neg_mod;
          nrw_mod = 0.8 * nrw_mod;
        }
        selected[id].userData.neg_mod = neg_mod;
        selected[id].userData.nrw_mod = nrw_mod;
        selected[id].userData.nheight_mod = nheight_mod;
      }
      let edge = {
        start: nodes[start],
        end: nodes[end],
        weight: weight,
        nrw_mod: nrw_mod,
        neg_mod: neg_mod,
        nheight_mod: nheight_mod
      };
      data.links.push({
        source: ids[start],
        target: ids[end],
        ricciCurvature: weight
      });
      edges.push(edge);
    }
  }

  // TODO: Frome here on, the code in this function seems to need a lot
  // of reworking. It has been left more or less as it was found. Some
  // lines have been commented out if the corresponding features have
  // been removed.

  refine_data = [];

  let padding = 0.25;
  xRange[0] = Math.max(-7, xRange[0] - padding);
  xRange[1] = Math.min(7, xRange[1] + padding);
  yRange[0] = Math.max(-10, yRange[0] - padding);
  yRange[1] = Math.min(10, yRange[1] + padding);

  data.nodes.push({
    id: data.length,
    city: "edge1",
    lat: parseFloat(xRange[0]),
    long: parseFloat(yRange[0])
  });
  data.nodes.push({
    id: data.length,
    city: "edge2",
    lat: parseFloat(xRange[1]),
    long: parseFloat(yRange[1])
  });


  let mid = [(xRange[0] + xRange[1]) / 2, (yRange[0] + yRange[1]) / 2]
  let width = xRange[1] - xRange[0]
  let height = yRange[1] - yRange[0]
  // --- DRAW PLANE ---
  let ctx = canv.getContext('2d')
  if (document.getElementById("show-map").checked) {
    let mapCanvas = document.getElementById('map').getElementsByTagName('canvas')[0]
    ctx = mapCanvas.getContext('2d')
  }

  var subTexture = new T.CanvasTexture(ctx.canvas)
  subTexture.minFilter = THREE.LinearFilter
  subTexture.center = new T.Vector2(0.5, 0.5)
  subTexture.rotation = -Math.PI / 2
  subTexture.repeat.y = (xRange[1] - xRange[0]) / 20
  subTexture.repeat.x = (yRange[1] - yRange[0]) / 20
  subTexture.offset.x = mid[1] / 20
  subTexture.offset.y = mid[0] / 20

  var subGeom = new T.PlaneGeometry(width, height, divisions - 1, divisions - 1)
  // var material = new T.MeshBasicMaterial( { color: graphcolor, side: T.DoubleSide} )
  var subMat = new THREE.MeshPhongMaterial({
    color: graphcolor,
    clippingPlanes: [clipPlaneRight, clipPlaneLeft, clipPlaneBack, clipPlaneFront],
    vertexColors: T.VertexColors,
    side: THREE.DoubleSide,
    flatShading: false,
    shininess: 0,
    wireframe: false,
    map: subTexture
  })
  var subPlane = new T.Mesh(subGeom, subMat)

  // TODO: change
  subPlane.position.set(mid[0], 2, mid[1])
  subPlane.rotation.set(-Math.PI / 2, 0., 0.)
  let spObj = {}
  spObj.xstart3JS = xRange[0]
  spObj.width = width
  spObj.ystart3JS = yRange[0]
  spObj.height = height
  spObj.start = [Math.floor((xRange[0] + 10) * divisions / 20), Math.floor((yRange[0] + 10) * divisions / 20)]
  spObj.end = [Math.ceil((xRange[1] + 10) * divisions / 20), Math.ceil((yRange[1] + 10) * divisions / 20)]
  spObj.plane = subPlane
  subPlane.scale.set(0.1, 0.1, 0.1)
  scene.add(subPlane)
  subPlanes.push(spObj)
  let scale = Math.min(width, height) / divisions
  scale *= 10 // 10
  console.log(scale)

  // plane.position.set(0, -10, 0)

  //TODO: change back

  gsap.to(camera, {
    duration: 1,
    zoom: 2,
    onUpdate: function() {
      camera.updateProjectionMatrix();
    }
  })

  gsap.to(controls.target, {
    duration: 1,
    x: subPlane.position.x,
    y: subPlane.position.y,
    z: subPlane.position.z,
    onUpdate: function() {
      controls.update();
    }
  })

  gsap.to(plane.position, {
    duration: 1,
    y: -20,
    onUpdate: function() {},
    onComplete: function() {
      plane.visible = false
    }
  })

  gsap.to(subPlane.scale, {
    duration: 1,
    x: 1,
    y: 1,
    z: 1,
  })

  // let btn = document.createElement("button")
  // btn.innerHTML = "Zoom Out"
  // btn.id = "zoom-out"
  // document.body.appendChild(btn)
  // plane.material.transparent = true
  // plane.material.opacity = 0.5f


  // controls.target = subPlane.position
  // camera.zoom = 2
  // controls.update()
  // camera.updateProjectionMatrix()
  // ---GENERATE SURFACE
  let chkCalcSurface = document.getElementById("use-calc-surface")
  let newHeightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));

  let modifier = Math.max(width, height) / 20
  console.log(modifier, width, height)
  // modifier = 0.4

  if (!chkCalcSurface.checked) {
    let planeXMin = xRange[0]
    let planeWidth = xRange[1] - xRange[0]
    let planeYMin = yRange[0]
    let planeHeight = yRange[1] - yRange[0]

    for (let edge of edges) {
      if (edge.weight < 0)
        continue
      // let startname = Math.min(edge.start.name, edge.end.name)
      // let endname = Math.max(edge.start.name, edge.end.name)
      // console.log(startname, endname)
      // if (refine_data[startname] && refine_data[startname][endname]) {
      //   console.log(startname, endname, refine_data[startname][endname])
      // }
      let startPt = convert3JStoHMgeneric([edge.start.lat, edge.start.long], planeXMin, planeYMin, planeWidth, planeHeight)
      let endPt = convert3JStoHMgeneric([edge.end.lat, edge.end.long], planeXMin, planeYMin, planeWidth, planeHeight)
      let start = {
        x: startPt[0],
        y: startPt[1]
      }
      let end = {
        x: endPt[0],
        y: endPt[1]
      }
      let mid = {
        x: Math.floor((startPt[0] + endPt[0]) / 2),
        y: Math.floor((startPt[1] + endPt[1]) / 2)
      }
      // setHeights(start, mid, end, edge.weight, newHeightMap, modifier)
    }



    for (let edge of edges) {
      if (edge.weight > 0)
        continue
      console.log(edge)
      let startname = Math.min(edge.start.name, edge.end.name)
      let endname = Math.max(edge.start.name, edge.end.name)
      let neg_mod = edge.neg_mod
      console.log(neg_mod)
      let nrw_mod = edge.nrw_mod
      let nheight_mod = edge.nheight_mod

      let startPt = convert3JStoHMgeneric([edge.start.lat, edge.start.long], planeXMin, planeYMin, planeWidth, planeHeight)
      let endPt = convert3JStoHMgeneric([edge.end.lat, edge.end.long], planeXMin, planeYMin, planeWidth, planeHeight)
      //TODO: remove
      startPt[1] += 1
      endPt[1] += 1
      let start = {
        x: startPt[0],
        y: startPt[1]
      }
      let end = {
        x: endPt[0],
        y: endPt[1]
      }
      let mid = {
        x: Math.floor((startPt[0] + endPt[0]) / 2),
        y: Math.floor((startPt[1] + endPt[1]) / 2)
      }
      // setHeights(start, mid, end, edge.weight, newHeightMap, modifier, neg_mod, nrw_mod, nheight_mod)
    }

    setPlaneHeights(subPlane, newHeightMap)
    spObj.heightmap = newHeightMap


    return
  }

  var xmlHttp = new XMLHttpRequest();
  xmlHttp.responseType = "text"
  var smooth_pen = document.getElementById("input-smooth").value
  var niter = document.getElementById("input-niter").value
  var send_data = {
    graph: data,
    smooth_pen: smooth_pen,
    niter: niter
  }

  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
      data = xmlHttp.responseText
      data = data.substring(data.indexOf('['))
      data = JSON.parse(data)
      console.log(data)
      console.log("data recv")
      let newHeightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
      for (let i = 0; i < divisions; i++) {
        for (let j = 0; j < divisions; j++) {
          newHeightMap[j][49 - i] = data[i * divisions + j] * scale
        }
      }
      console.log(newHeightMap)
      setPlaneHeights(subPlane, newHeightMap)
    }
  }
  xmlHttp.open("post", "calc-surface");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

  xmlHttp.send(JSON.stringify(send_data));
  console.log("data sent")
}

function setPlaneHeights(plane, map) {
  for (let i = 0; i < divisions; i++) {
    for (let j = 0; j < divisions; j++) {
      plane.geometry.vertices[i * divisions + j].z = map[i][j];
    }
  }
  plane.geometry.groupsNeedUpdate = true;
  plane.geometry.verticesNeedUpdate = true;
  plane.geometry.colorsNeedUpdate = true;
  plane.geometry.computeVertexNormals();
}

function vertexNameChange() {
  // TODO: Deal with duplicate names
  if (this.value == '') {
    return;
  }
  let parentDiv = this.parentElement;
  let id = parentDiv.childNodes[0].textContent;
  let pt = vertices[id];
  let oldName = pt.name;
  pt.name = this.value;
  scene.remove(pt.label);
  pt.label = getNameSprite(this.value);
  pt.label.position.set(pt.mesh.position.x, vertexHeight + 0.5, pt.mesh.position.z);
  scene.add(pt.label);
  delete names[oldName];
  names[pt.name] = parseInt(id);
}

function vertexPositionChange() {
  if (this.value == '' || isNaN(this.value)) {
    return;
  }
  let parentDiv = this.parentElement;
  let id = parentDiv.childNodes[0].textContent;
  let pt = vertices[id];
  if (this.className == "xPos") {
    gsap.to(pt.mesh.position, {
      duration: 0.25,
      x: this.value,
      onUpdate: function() {
        olMap.render();
      }
    });
    pt.lat = this.value * 155 / (planeW / 2)
  } else {
    gsap.to(pt.mesh.position, {
      duration: 0.25,
      z: this.value,
      onUpdate: function() {
        olMap.render();
      }
    });
    pt.long = this.value * 180 / (planeH / 2)
  }
}

function addVertex(obj, x, y, drawPoint, name, lat = null, long = null) {
  if (typeof drawPoint == 'undefined') {
    drawPoint = true;
  }

  if (x == undefined) {
    if (lat != null) {
      x = lat * (planeW / 2) / 155;
    } else {
      x = 0;
    }
  }

  if (y == undefined) {
    if (long != null) {
      y = long * (planeH / 2) / 180;
    } else {
      if (vertexCount == 0) {
        y = 5;
      } else {
        y = -5;
      }
    }
  }
  if (lat == null) {
    lat = x * 155 / (planeW / 2);
  }
  if (long == null) {
    long = y * 180 / (planeH / 2);
  }


  let vDiv = document.createElement("div");
  vDiv.id = "vertex" + vertexCount;
  vDiv.className = "form-box";

  let idLbl = document.createElement("label");
  idLbl.setAttribute("for", "id");
  idLbl.textContent = vertexCount;

  if (typeof name == 'undefined') {
    name = vertexCount;
  }

  let nameLbl = document.createElement("label");
  nameLbl.setAttribute("for", "name");
  nameLbl.textContent = "Name:";

  let nameInput = document.createElement("input");
  nameInput.className = "name";
  nameInput.setAttribute("type", "text");
  nameInput.defaultValue = name;
  nameInput.onchange = vertexNameChange;

  let xPosLbl = document.createElement("label");
  xPosLbl.setAttribute("for", "xPos");
  xPosLbl.textContent = "x:";

  let xPos = document.createElement("input");
  xPos.className = "xPos";
  xPos.setAttribute("type", "text");
  xPos.defaultValue = x;
  xPos.oninput = vertexPositionChange;

  let yPosLbl = document.createElement("label");
  yPosLbl.setAttribute("for", "yPos");
  yPosLbl.textContent = "y:";

  let yPos = document.createElement("input");
  yPos.className = "yPos";
  yPos.setAttribute("type", "text");
  yPos.defaultValue = y;
  yPos.oninput = vertexPositionChange;

  let del = document.createElement("button");
  del.className = "btn-delete";
  del.innerHTML = "X";;
  del.onclick = removeVertex;

  vDiv.appendChild(idLbl);
  vDiv.appendChild(nameLbl);
  vDiv.appendChild(nameInput);
  vDiv.appendChild(xPosLbl);
  vDiv.appendChild(xPos);
  vDiv.appendChild(yPosLbl);
  vDiv.appendChild(yPos);
  vDiv.appendChild(del);
  document.getElementById("div-vertex").appendChild(vDiv);

  xPos.select();

  let newPt = new T.Mesh(ptGeom, ptMat);
  newPt.position.y = vertexHeight;
  newPt.position.x = xPos.value;
  newPt.position.z = yPos.value;
  newPt.name = name;

  let sprite = getNameSprite(name)
  sprite.position.set(xPos.value, vertexHeight + 0.1, yPos.value)

  if (showHoverGraph && drawPoint) {
    let sprite = getNameSprite(name);
    sprite.position.set(xPos.value, vertexHeight + 0.1, yPos.value);
    scene.add(sprite);
    newPt.scale.set(0.1, 0.1, 0.1);
    scene.add(newPt);
    gsap.to(newPt.scale, {
      duration: 1.5,
      x: 1,
      y: 1,
      z: 1,
      ease: 'elastic'
    });
  }
  vertices[String(vertexCount)] = new VertexObj(vertexCount, name, newPt, sprite, lat, long);
  names[name] = vertexCount;
  vertexCount++;
}

// TODO: It is unclear whether this function is needed. If it is, it
// would be for the drag-and-drop file system
function addVertexSec(obj, x, y, vertices, drawPoint = false) {
  let newPt = new T.Mesh(ptGeom, ptMat);
  newPt.position.y = vertexHeight;
  newPt.position.x = x;
  newPt.position.z = y;

  length = Object.keys(vertices).length;

  let sprite = getNameSprite(length);
  sprite.position.set(x, vertexHeight + 0.5, y);

  if (drawPoint) {
    scene.add(sprite);
    newPt.scale.set(0.1, 0.1, 0.1);
    scene.add(newPt);
    gsap.to(newPt.scale, {
      duration: 1,
      x: 1,
      y: 1,
      z: 1,
      ease: 'elastic'
    });
  }

  vertices[String(length)] = new VertexObj(length, length, newPt, sprite);
}

function removeVertex() {
  let parentDiv = this.parentElement;
  let name = parentDiv.childNodes[0].textContent;
  scene.remove(vertices[name].mesh);
  scene.remove(vertices[name].label);
  delete vertices[name];
  parentDiv.remove();
}

function generateGraph() {
  // Replace this with the default graph we want
  addVertex(null, 1.4134275618374557, -8.0, true, name = "0");
  addVertex(null, -1.4536103630714172e-15, -5.6537102473498235, true, name = "1");
  addVertex(null, -1.4134275618374557, -8.0, true, name = "2");
  addVertex(null, -1.4134275618374557, 8.0, true, name = "3");
  addVertex(null, -1.4536103630714172e-15, 5.6537102473498235, true, name = "4");
  addVertex(null, 1.4134275618374557, 8.0, true, name = "5");

  addEdge(null, 0, 1, 0.3333);
  addEdge(null, 0, 2, 0.5);
  addEdge(null, 1, 2, 0.3333);
  addEdge(null, 1, 4, -0.6667);
  addEdge(null, 3, 4, 0.3333);
  addEdge(null, 3, 5, 0.5);
  addEdge(null, 4, 5, 0.3333);
}

function generateGraphNoWeights() {
  addVertex(null, 1.4134275618374557, -8.0, true, name = "0");
  addVertex(null, -1.4536103630714172e-15, -5.6537102473498235, true, name = "1");
  addVertex(null, -1.4134275618374557, -8.0, true, name = "2");
  addVertex(null, -1.4134275618374557, 8.0, true, name = "3");
  addVertex(null, -1.4536103630714172e-15, 5.6537102473498235, true, name = "4");
  addVertex(null, 1.4134275618374557, 8.0, true, name = "5");

  addEdge(null, 0, 1, 0.);
  addEdge(null, 0, 2, 0.);
  addEdge(null, 1, 2, 0.);
  addEdge(null, 1, 4, 0.);
  addEdge(null, 3, 4, 0.);
  addEdge(null, 3, 5, 0.);
  addEdge(null, 4, 5, 0.);
}

function drawEdge(edge, lineMat) {
  let points = [];
  if (lineMat != lineMatSec) {
    points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight + 0.0001, edge.start.mesh.position.z));
    points.push(new T.Vector3(edge.end.mesh.position.x, vertexHeight + 0.0001, edge.end.mesh.position.z));
  } else {
    points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight, edge.start.mesh.position.z));
    points.push(new T.Vector3(edge.end.mesh.position.x, vertexHeight, edge.end.mesh.position.z));
  }
  points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight + 0.0001, edge.start.mesh.position.z));

  let geom = new T.BufferGeometry().setFromPoints(points);

  let mat = new T.LineBasicMaterial({
    color: edgecolor,
    linewidth: 5,
    clippingPlanes: [clipPlaneUp, clipPlaneRight, clipPlaneLeft, clipPlaneBack, clipPlaneFront, ]
  })
  let line = new T.Line(geom, mat);
  let color = new T.Color("hsl(0, 0%, 100%)");
  let endColor;
  if (edge.weight >= 0) {
    endColor = new T.Color("hsl(222, 100%, 61%)");
  } else {
    endColor = new T.Color("hsl(356, 74%, 52%)");
  }
  color.lerp(endColor, Math.min(Math.abs(edge.weight), 1));
  line.material.color.set(color);
  line.name = edge.start.name + "/" + edge.end.name;
  line.userData = {
    weight: edge.weight,
    neg_mod: edge.neg_mod,
    nrw_mod: edge.nrw_mod,
    nheight_mod: edge.nheight_mod
  };

  scene.add(line);
  linesDrawn.push(line);
  return line;
}

function addEdge(obj, start, end, weight) {
  if (typeof start == 'undefined') {
    start = 0;
  }

  if (typeof end == 'undefined') {
    end = 0;
  }

  if (typeof weight == 'undefined') {
    weight = 0;
  }

  let vDiv = document.createElement("div");
  vDiv.id = "edge" + edgeCount;
  vDiv.className = "form-box";

  let nameLbl = document.createElement("label");
  nameLbl.setAttribute("for", "name");
  nameLbl.textContent = edgeCount;

  let startLbl = document.createElement("label");
  startLbl.setAttribute("for", "start");
  startLbl.textContent = "start:";

  let startText = document.createElement("input");
  startText.className = "start";
  startText.setAttribute("type", "text");
  startText.defaultValue = start;
  startText.oninput = edgeChange;

  let endLbl = document.createElement("label");
  endLbl.setAttribute("for", "start");
  endLbl.textContent = "end:";

  let endText = document.createElement("input");
  endText.className = "end";
  endText.setAttribute("type", "text");
  endText.defaultValue = end;
  endText.oninput = edgeChange;

  let weightLbl = document.createElement("label");
  weightLbl.setAttribute("for", "weight");
  weightLbl.textContent = "weight:";

  let weightText = document.createElement("input");
  weightText.className = "weight";
  weightText.setAttribute("type", "text");
  weightText.defaultValue = weight;
  weightText.oninput = edgeChange;

  let del = document.createElement("button");
  del.className = "btn-delete";
  del.innerHTML = "X";;
  del.onclick = removeEdge;

  vDiv.appendChild(nameLbl);
  vDiv.appendChild(startLbl);
  vDiv.appendChild(startText);
  vDiv.appendChild(endLbl);
  vDiv.appendChild(endText);
  vDiv.appendChild(weightLbl);
  vDiv.appendChild(weightText);
  vDiv.appendChild(del);
  document.getElementById("div-edge").appendChild(vDiv);

  let size = Object.keys(vertices).length;

  let s = parseInt(startText.value);
  let e = parseInt(endText.value);

  weight = parseFloat(weightText.value);

  let startPt = vertices[s];
  let endPt = vertices[e];

  let edge = new EdgeObj(edgeCount, startPt, endPt, weight);
  edges[edgeCount] = edge;
  edgeCount++;
}

function addEdgeSec(obj, start, end, weight, vertices, edges) {
  let vSize = Object.keys(vertices).length;
  let eSize = Object.keys(edges).length;

  let s = parseInt(start);
  let e = parseInt(end);

  weight = parseFloat(weight);

  let startPt = vertices[s];
  let endPt = vertices[e];

  let edge = new EdgeObj(eSize, startPt, endPt, weight);
  edges[eSize] = edge;
}

function edgeChange() {
  // TODO: Deal with non existent vertices
  if (this.value == '' || isNaN(this.value)) {
    return;
  }
  let parentDiv = this.parentElement;
  let startId = parentDiv.childNodes[2].value;
  let endId = parentDiv.childNodes[4].value;
  let weight = parseFloat(parentDiv.childNodes[6].value);
  let id = parentDiv.childNodes[0].textContent;
  let edge = edges[id];
  edge.start = vertices[startId];
  edge.end = vertices[endId];
  edge.weight = weight;
  edge.checkSplit();
}

function removeEdge() {
  let parentDiv = this.parentElement;
  let id = parentDiv.childNodes[0].textContent;
  delete edges[id];
  parentDiv.remove();
}

function getNameSprite(name) {
  let canvas = document.createElement('canvas');
  let ctx = canvas.getContext('2d');

  let metrics = ctx.measureText(name);
  let textWidth = metrics.width;
  let textHeight = metrics.height;

  ctx.canvas.width = textWidth * 30 + 30;
  ctx.canvas.height = textWidth * 30 + 10;

  ctx.font = "20px Roboto Mono";
  ctx.fillStyle = "#000000";

  ctx.fillText(name, ctx.canvas.width / 2 - textWidth / 2, ctx.canvas.height / 2);

  let texture = new T.CanvasTexture(ctx.canvas);
  texture.needsUpdate = true;

  let spriteMat = new T.SpriteMaterial({
    map: texture,
    alphaTest: 0.1
  });
  let sprite = new T.Sprite(spriteMat);
  sprite.scale.set(0.05 * textWidth, 0.05 * textWidth, 0.05 * textWidth);
  return sprite;
}

let VertexObj = class {
  start = []; // Edges starting at this vertex
  end = []; // Edges ending at this vertex

  constructor(id, name, mesh, label, lat = 0, long = 0, start = [], end = []) {
    this.id = id;
    this.name = name;
    this.mesh = mesh;
    this.label = label;
    this.start = start;
    this.end = end;
    this.lat = lat;
    this.long = long;
  }
}

let EdgeObj = class {
  // TODO: Remove split edge logic
  constructor(id, start, end, weight) {
    this.id = id;
    this.start = start;
    this.end = end;
    this.weight = weight;
    this.bearing = GreatCircle.bearing(start.lat, start.long, end.lat, end.long);
    this.split = false;
    this.neg_mod = 1;
    this.nrw_mod = 1;
    this.nheight_mod = 1;
    this.mesh = null;
    if (start.long > end.long && this.bearing <= 180) {
      this.split = useSplitEdges;
      let p1 = [start.mesh.position.x, start.mesh.position.z];
      this.startSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z)], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z) + 20], [10, 10], [-10, 10]);
      this.endSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z) - 20], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)], [10, -10], [-10, -10]);
      this.startSplit = [this.startSplit[0] * 155 / 10, this.startSplit[1] * 180 / 10];
      this.endSplit = [this.endSplit[0] * 155 / 10, this.endSplit[1] * 180 / 10];
    } else if (start.long < end.long && this.bearing >= 180) {
      this.split = useSplitEdges;
      let p1 = [start.mesh.position.x, start.mesh.position.z];
      this.startSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z)], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z) - 20], [10, -10], [-10, -10]);
      this.endSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z) + 20], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)], [10, 10], [-10, 10]);
      this.startSplit = [this.startSplit[0] * 155 / 10, this.startSplit[1] * 180 / 10];
      this.endSplit = [this.endSplit[0] * 155 / 10, this.endSplit[1] * 180 / 10];
    }
  }

  checkSplit() {
    let start = this.start;
    let end = this.end;
    if (start.long > end.long && this.bearing <= 180) {
      this.split = useSplitEdges;
      this.startSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z)], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z + 20)], [7, 10], [-7, 10]);
      this.endSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z - 20)], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)], [7, -10], [-7, -10]);
    }
  }
}

let GraphObj = class {
  constructor(vertices, edges, heightmap) {
    this.vertices = vertices;
    this.heightmap = heightmap;
    this.edges = edges;
  }
}

function createMap() {
  let zoomFactor = 1.6;
  let center = [0, 0];
  var mapdiv = document.createElement('div');
  mapdiv.id = 'map';
  mapdiv.class = 'map-div';
  mapdiv.style.width = '2000px';
  mapdiv.style.height = '2000px';
  document.body.appendChild(mapdiv);
  var map = new ol.Map({
    target: 'map',
    renderer: 'canvas',
    layers: [
      new ol.layer.Tile({
        source: new ol.source.Stamen({
          layer: 'terrain'
        }),
      })
    ],
    view: new ol.View({
      center: ol.proj.fromLonLat(center),
      resolution: 40075016.68557849 / 2000 / zoomFactor
    })
  });
  return map;
}

function calculateCurvature() {
  var data = {
    nodes: [],
    links: []
  };
  for (let id in vertices) {
    data.nodes.push({
      id: id
    });
  }
  let current_edges = {
    ...edges
  };
  if (graphs.length > 0) {
    current_edges = {
      ...graphs[document.getElementById("threshold-slider").value].edges
    };
  }
  for (let id in current_edges) {
    let edge = current_edges[id];
    data.links.push({
      source: edge.start.id,
      target: edge.end.id
    });
  }
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
      data = JSON.parse(xmlHttp.responseText)
      let current_edges = {
        ...edges
      };
      if (graphs.length > 0) {
        current_edges = {
          ...graphs[document.getElementById("threshold-slider").value].edges
        };
      }
      for (let id in data.links) {
        let link = data.links[id];
        for (let id2 in current_edges) {
          let edge = current_edges[id2];
          if ((edge.start.id == link.source && edge.end.id == link.target) || (edge.start.id == link.target && edge.end.id == link.source)) {
            edge.weight = parseFloat(link.ricciCurvature);
            let edgeDiv = document.getElementById("edge" + id2);
            if (edgeDiv != null) {
              edgeDiv.querySelector(".weight").value = parseFloat(link.ricciCurvature);
            }
            break;
          }
        }
      }

      for (let line of linesDrawn) {
        scene.remove(line);
      }
      linesDrawn = [];
      linesCleared = true;
    }
  }
  xmlHttp.open("post", "calc-curvature");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

  xmlHttp.send(JSON.stringify(data));
}

function calcSurface() {
  var data = {
    nodes: [],
    links: []
  };
  let current_edges = {
    ...edges
  };
  if (graphs.length > 0) {
    current_edges = {
      ...graphs[document.getElementById("threshold-slider").value].edges
    };
  }
  length = Object.keys(vertices).length;
  for (let id in vertices) {
    let node = vertices[id];
    data.nodes.push({
      id: parseInt(id),
      city: String(node.name),
      lat: node.lat + 1E-10,
      long: node.long + 1E-10
    });
  }

  let splitCount = 0;
  for (let id in current_edges) {
    let edge = current_edges[id];
    if (edge.split) {
      data.nodes.push({
        id: data.nodes.length,
        city: "splitstart" + splitCount,
        lat: edge.startSplit[0],
        long: edge.startSplit[1]
      });
      data.nodes.push({
        id: data.nodes.length,
        city: "splitend" + splitCount,
        lat: edge.endSplit[0],
        long: edge.endSplit[1]
      });
      data.links.push({
        source: edge.start.id,
        target: data.nodes.length - 2,
        ricciCurvature: edge.weight
      });
      data.links.push({
        source: edge.end.id,
        target: data.nodes.length - 1,
        ricciCurvature: edge.weight
      });
      splitCount++;
    } else {
      data.links.push({
        source: edge.start.id,
        target: edge.end.id,
        ricciCurvature: edge.weight
      });
    }
  }
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.responseType = "text";
  var smooth_pen = document.getElementById("input-smooth").value;
  var niter = document.getElementById("input-niter").value;
  var send_data = {
    graph: data,
    smooth_pen: smooth_pen,
    niter: niter,
    map: heightMap
  };

  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
      let scale = 20.;
      document.body.style.cursor = "auto";
      data = xmlHttp.responseText;
      data = data.substring(data.indexOf('['));
      data = JSON.parse(data);
      let hm = [];
      if (graphs.length > 0) {
        hm = graphs[document.getElementById("threshold-slider").value].heightmap;
      } else {
        hm = calcHeightMap;
      }
      for (let i = 0; i < divisions; i++) {
        for (let j = 0; j < divisions; j++) {
          hm[j][divisions - 1 - i] = data[i * divisions + j] * scale;
        }
      }

      updatePlaneHeights(hm);
    }
  }
  xmlHttp.open("post", "calc-surface");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xmlHttp.send(JSON.stringify(send_data));
  document.body.style.cursor = "progress";
}

function createAndUpdateAlphaMapD3(map) {
  var svg = d3.select('#alpha-svg');
  var width = 512;
  var height = 512;
  if (d3.select('#alpha-svg').empty()) {
    var svg = d3.select('body')
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('id', 'alpha-svg')
      .attr('display', 'none');
  }
  svg.selectAll("*").remove();
  svg.append('rect')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('fill', '#333333');
  svg.append('g');

  // Add X axis
  var x = d3.scaleLinear()
    .domain([0, map.length])
    .range([0, width]);
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // Add Y axis
  var y = d3.scaleLinear()
    .domain([0, map[0].length])
    .range([height, 0]);
  svg.append("g")
    .call(d3.axisLeft(y));

  var color = d3.scaleLinear()
    .domain([0.0, 0.002]) // Points per square pixel. .002
    .range(["#333333", "white"]);
  var data = [];

  for (let i = 0; i < map.length; i++) {
    for (let j = 0; j < map[i].length; j++) {
      if (map[i][j] == 1) {
        data.push({
          x: i,
          y: j
        });
      }
    }
  }

  // Modify bandwidth
  var densityData = d3.contourDensity()
    .x(function(d) {
      return x(d.x);
    })
    .y(function(d) {
      return y(d.y);
    })
    .size([width, height])
    .bandwidth(4)
    (data);


  svg.insert("g", "g")
    .selectAll("path")
    .data(densityData)
    .enter().append("path")
    .attr("d", d3.geoPath())
    .attr("fill", function(d) {
      return color(d.value);
    });

  var alphaCanv = document.getElementById('alpha-canvas');
  if (alphaCanv == null) {
    alphaCanv = document.createElement('canvas');
    alphaCanv.id = 'alpha-canvas';
    alphaCanv.height = height;
    alphaCanv.width = width;
    alphaCanv.style.display = 'none';

    document.body.appendChild(alphaCanv);
  }

  var ctx = alphaCanv.getContext('2d');

  var svgEle = document.getElementById('alpha-svg');
  var img = document.createElement('img');
  img.setAttribute('src', "data:image/svg+xml;base64," + window.btoa(unescape(encodeURIComponent((new XMLSerializer()).serializeToString(svgEle)))));
  img.onload = function() {
    ctx.drawImage(img, 0, 0);
  }

  return new T.CanvasTexture(alphaCanv);

}

function fileSelectEdges(evt) {
  evt.stopPropagation();
  evt.preventDefault();

  var files = evt.dataTransfer.files; // FileList object.
  for (let file of files) {
    readEdgeFile(file);
  }
}

function readEdgeFile(file) {
  var reader = new FileReader();
  reader.onload = function() {
    let current_edges = {};
    let text = reader.result;
    let lines = text.split('\n');
    let i = -1;
    let inputNames = [];
    let negative_edges = [];
    if (file.name.substr(-3) != 'csv') {
      let inputNameData = lines[0].split('\"');
      let k = -1;
      for (let nameData of inputNameData) {
        k++;
        nameData = nameData.trim();
        if (nameData == '') {
          continue;
        }
        if (k % 2 == 1) {
          inputNames.push(nameData);
        } else {
          inputNames = inputNames.concat(nameData.split(' '));
        }
      }
    } else {
      inputNames = lines[0].split(',').slice(1);
    }

    if (file.name.substr(-3) != 'csv') {
      for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        let data = line.split("\"");
        let currentNode = '';
        if (data.length > 1) { // Two word name - deal with double quotes
          currentNode = data[1];
          data = data[2].split(" ");
        } else {
          data = data[0].split(" ");
          currentNode = data[0];
          data = data.splice(1);
        }
        if (data[0] == '' || isNaN(data[0])) {
          continue;
        }
        let currentId = names[currentNode];
        for (let j = 0; j < i; j++) {
          let weight = parseFloat(data[j]);
          if (weight == 0) {
            continue;;
          }
          let endNode = inputNames[j];
          let endId = names[endNode];
          if (weight < 0) {
            let n = {
              start: currentId,
              end: endId,
              weight: weight
            };
            negative_edges.push(n);
            continue;
          }
          addEdgeSec(null, currentId, endId, weight, vertices, current_edges);
        }
      }
    } else {
      for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        let data = line.split(",");
        let currentNode = data[0];
        data = data.splice(1);

        let currentId = names[currentNode];
        for (let j = 0; j < i; j++) {
          if (data[j] == '' || isNaN(data[0])) {
            continue;
          }
          let weight = parseFloat(data[j]);
          let endNode = inputNames[j];
          let endId = names[endNode];
          if (weight < 0) {
            let n = {
              start: currentId,
              end: endId,
              weight: weight
            };
            negative_edges.push(n);
            continue;
          }
          addEdgeSec(null, currentId, endId, weight, vertices, current_edges);
        }
      }
    }
    negative_edges.sort((a, b) => -(a.weight - b.weight));
    for (let e of negative_edges) {
      addEdgeSec(null, e.start, e.end, e.weight, vertices, current_edges);
    }

    let newHeightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
    let newGraph = new GraphObj(vertices, current_edges, newHeightMap);
    graphs.push(newGraph);

    document.getElementById("threshold-slider").max = graphs.length - 1;
    document.getElementById("threshold-slider").value = graphs.length - 1;
  }
  reader.readAsText(file);
}

function fileSelectNodes(evt) {
  evt.stopPropagation();
  evt.preventDefault();
  var files = evt.dataTransfer.files; // FileList object.
  var reader = new FileReader();

  reader.onload = function() {
    let text = reader.result;
    let lines = text.split('\n');
    for (let line of lines) {
      let data = line.split(',');
      if (data[1] == '' || isNaN(data[1])) {
        continue;
      }
      let projected = ol.proj.fromLonLat([data[2], data[1]]);
      // TODO: This next line seems strange. It's presumably based off
      // of https://epsg.io/3857, but it might need to be edited
      addVertex(null, projected[1] / 20048966.10 * 10, projected[0] / 20026376.39 * 10, true, data[0], parseFloat(data[1]), parseFloat(data[2]));
    }
  }
  reader.readAsText(files[0]);
}

function dragOver(evt) {
  evt.stopPropagation();
  evt.preventDefault();
  evt.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
}

function convert3JStoVerts(x, y) {
  // Depends on plane geometry
  // Change from -10,10 to 0,49
  if (subPlanes.length != 0) {
    return convert3JStoVertsSubgraph(x, y);
  }
  x = parseFloat(x);
  y = parseFloat(y);
  x -= planeXMin;
  x /= planeW;
  x *= divisions - 1;
  y -= planeYMin;
  y /= planeH;
  y *= divisions - 1;
  let retval = Math.round(y) * divisions + Math.round(x);
  return retval;
}

function convert3JStoVertsSubgraph(x, y) {
  let spObj = subPlanes[0];
  x = parseFloat(x);
  y = parseFloat(y);
  x -= spObj.xstart3JS;
  x /= spObj.width;
  x *= divisions - 1;
  y -= spObj.ystart3JS;
  y /= spObj.height;
  y *= divisions - 1;
  let retval = Math.round(y) * 50 + Math.round(x);
  return retval;
}

function convert3JStoHMgeneric(point, planeXMin, planeYMin, planeWidth, planeHeight) {
  point[0] = (point[0] - planeXMin); // Change from (min,max) to (0, newmax)
  point[1] = (point[1] - planeYMin); // Change from (min,max) to (0, newmax)
  point[0] = Math.round((point[0] / planeWidth) * (divisions - 1)); // Change from (0, planeWidth) to (0, divisions)
  point[1] = Math.round((point[1] / planeHeight) * (divisions - 1)); // Change from (0, planeHeight) to (0, divisions)
  return point;
}

function convert3JStoOM(point, divisions) {
  point[0] = (point[0] - planeXMin); // Change from (min,max) to (0, newmax)
  point[1] = (point[1] - planeYMin); // Change from (min,max) to (0, newmax)

  point[0] = Math.round((point[0] / planeW) * (divisions - 1)); // Change from (0, planeWidth) to (0, divisions)
  point[1] = Math.round((point[1] / planeH) * (divisions - 1)); // Change from (0, planeHeight) to (0, divisions)

  return point;
}

function dist(startPt, endPt) {
  return Math.sqrt((startPt[0] - endPt[0]) ** 2 + (startPt[1] - endPt[1]) ** 2)
}
