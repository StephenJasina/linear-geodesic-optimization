import { EffectComposer } from "./scripts/EffectComposer.js";
import { RenderPass } from "./scripts/RenderPass.js";
import { ShaderPass } from "./scripts/ShaderPass.js";
import { FXAAShader } from "./scripts/FXAAShader.js";

let bgcolor = 0xf3f3f3;
let graphcolor = 0xffffff;
let vertexcolor = 0x4CAF50;
let edgecolor = 0x21bf73;
let canvascolor = "#ffffff";

let T = THREE;

let vertexCount = 0;
let edgeCount = 0;

let circumferenceEarth = 40075016.68557849;
let mapCenter = [0.0, 0.0];
let mapZoomFactor = 1.6;
let mapResolution = 1000
let mapZoomLevel = 0

let planeXMin = -10;
let planeXMax = 10;
let planeYMin = -10;
let planeYMax = 10;
let planeW = planeXMax - planeXMin;
let planeH = planeYMax - planeYMin;
let divisions = 50;
let heightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
let opacityMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
let refine_data = {};

let vertexHeight = 3;

// ThreeJS Scene Setup
let scene = new T.Scene();
let div = 64;
let camera = new THREE.OrthographicCamera(
  window.innerWidth / -div, window.innerWidth / div,
  window.innerHeight / div, window.innerHeight / -div,
  0.1, 1000
);
camera.position.x = -15;
camera.position.z = 20;
camera.position.y = 15;

let renderer = new THREE.WebGLRenderer({
  logarithmicDepthBuffer: true,
  antialias: false
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.autoClear = false;
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

scene.background = new THREE.Color(bgcolor);
let controls = new T.OrbitControls(camera, renderer.domElement);

const olMap = createMap();
const blankCanvas = document.createElement("canvas");
blankCanvas.id = "canvas-texture";
let ctx = blankCanvas.getContext("2d");
ctx.canvas.width = 4000;
ctx.canvas.height = 4000;
ctx.fillStyle = canvascolor;
ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
let texture = new T.CanvasTexture(ctx.canvas);
texture.minFilter = THREE.LinearFilter;
texture.center = new T.Vector2(0.5, 0.5)
texture.rotation = -Math.PI / 2

let ptGeom = new T.SphereGeometry(0.02, 32, 32);
let ptMat = new T.MeshBasicMaterial({
  color: vertexcolor
});

let clipPlaneUp = new T.Plane(new T.Vector3(0, 2, 0), 2);
let clipPlaneLeft = new T.Plane(new T.Vector3(-1, 0, 0), -planeXMin);
let clipPlaneRight = new T.Plane(new T.Vector3(1, 0, 0), planeXMax);
let clipPlaneFront = new T.Plane(new T.Vector3(0, 0, -1), -planeYMin);
let clipPlaneBack = new T.Plane(new T.Vector3(0, 0, 1), planeYMax);

let loader = new T.ImageLoader();
let aMap = loader.load("images/grayscale.png");

let geometry = new T.PlaneGeometry(
  planeW, planeH,
  divisions - 1, divisions - 1
);
let material = new T.MeshBasicMaterial({
  color: graphcolor,
  side: T.DoubleSide
});
// TODO: change back
let planeMat = new THREE.MeshPhongMaterial({
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
let plane = new T.Mesh(geometry, planeMat);

plane.rotation.set(-Math.PI / 2, 0, 0);
scene.add(plane);

controls.update();

let light = new T.PointLight(0xffffff, 0.5);
light.position.set(-7, 8, 0);
scene.add(light);

const directionalLight = new T.DirectionalLight(0xffffff, 0.6);
directionalLight.position.set(1.5, 0.1, 10);
scene.add(directionalLight);

let alight = new THREE.AmbientLight(0x404040); // soft white light
scene.add(alight);

let vertices = {};
let edges = {};
let graphs = [];
let names = {};
let linesDrawn = [];
let linesCleared = true;
let current_edges = {};

edgecolor = 0x178e51;
let linewidth = 20;
let lineMat = new T.LineBasicMaterial({
  color: edgecolor,
  linewidth: linewidth,
  clippingPlanes: [clipPlaneUp]
});

plane.geometry.dynamic = true;

for (let face of plane.geometry.faces) {
  face.vertexColors[0] = new T.Color(0xffffff);
  face.vertexColors[1] = new T.Color(0xffffff);
  face.vertexColors[2] = new T.Color(0xffffff);
}

// Composers and FXAA shader
let renderPass = new RenderPass(scene, camera);

let fxaaPass = new ShaderPass(FXAAShader);
fxaaPass.renderToScreen = false;
let pixelRatio = renderer.getPixelRatio();
fxaaPass.material.uniforms["resolution"].value.x = 1 / (window.innerWidth * pixelRatio);
fxaaPass.material.uniforms["resolution"].value.y = 1 / (window.innerHeight * pixelRatio);

let composer = new EffectComposer(renderer);
composer.addPass(renderPass);
composer.addPass(fxaaPass);

window.onload = function() {
  let mapCanvas = document.getElementById("map").getElementsByTagName("canvas")[0];

  let ctx = mapCanvas.getContext("2d");
  const customTexture = texture;
  const mapTexture = new T.CanvasTexture(ctx.canvas);
  mapTexture.minFilter = THREE.LinearFilter;
  mapTexture.magFilter = THREE.NearestFilter;
  mapTexture.center = new T.Vector2(0.5, 0.5);
  mapTexture.rotation = -Math.PI / 2;

  let mapdiv = document.getElementById("map");

  mapdiv.style.display = "none";

  window.addEventListener("wheel", wheelEvent, true);

  let dropOutputDiv = document.getElementById("drop-output");
  dropOutputDiv.addEventListener("dragover", dragOver, false);
  dropOutputDiv.addEventListener("drop", dropOutput, false);

  let btnAddVertex = document.getElementById("btn-add-vertex");
  btnAddVertex.onclick = addVertex;

  let btnAddEdge = document.getElementById("btn-add-edge");
  btnAddEdge.onclick = addEdge;

  document.getElementById("btn-calculate-geodesics").onclick = function() {
    calcDistanceOnSurface(plane, vertices, edges);
  }

  let showMap = document.getElementById("show-map");
  let showGraph = document.getElementById("show-graph");
  let showHoverGraph = document.getElementById("show-hover-graph");

  function refresh_graph() {
    if (showGraph.checked && showHoverGraph.checked) {
      for (let id in vertices) {
        scene.add(vertices[id].mesh);
        scene.add(vertices[id].label);
      }
      for (let line of linesDrawn) {
        scene.add(line);
      }
    } else {
      for (let id in vertices) {
        scene.remove(vertices[id].mesh);
        scene.remove(vertices[id].label);
      }
      for (let line of linesDrawn) {
        scene.remove(line);
      }
    }

    if (!showGraph) {
      let canvases = [mapCanvas, blankCanvas]
      for (let canvas of canvases) {
        let context = canvas.getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }
  showGraph.onchange = refresh_graph;
  showHoverGraph.onchange = refresh_graph;

  let vertexControlDiv = document.getElementById("div-vertex");
  let edgeControlDiv = document.getElementById("div-edge");

  let btnCalcCurv = document.getElementById("btn-calc-curv");
  btnCalcCurv.onclick = calculateCurvature;

  let btnHelp = document.getElementById("btn-help");
  btnHelp.onclick = helpClick;

  controls.enablePan = true;
  controls.panSpeed = 1;
  controls.enableRotate = true;
  controls.enableZoom = true;
  controls.minZoom = 1;
  controls.update();

  let animate = function() {
    if (showMap.checked) {
      plane.material.map = mapTexture;
      texture = mapTexture;
      ctx = mapCanvas.getContext("2d");
    } else {
      plane.material.map = customTexture;
      texture = customTexture;
      ctx = blankCanvas.getContext("2d");
    }

    controls.update();

    ctx.fillStyle = canvascolor

    if (!showMap.checked) {
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }

    // Draws grid lines
    if (document.getElementById("show-grid").checked) {
      drawGrid(ctx.canvas);
    }

    ctx.setLineDash([]);

    // Draw physical graph edge, texture edge
    if (showGraph.checked) {
      for (let id in edges) {
        let borders = true;
        let edge = edges[id];

        if (showHoverGraph.checked && (edge.mesh == null || linesCleared)) {
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
          ctx.lineWidth = linewidth + 5;
          ctx.stroke();
        }

        ctx.moveTo(startPt[1], startPt[0]);
        ctx.lineTo(endPt[1], endPt[0]);
        let color = getCurvatureColor(edge.weight);
        ctx.strokeStyle = "#" + color.getHexString();
        ctx.lineWidth = linewidth;
        ctx.stroke();
        ctx.restore();
      }

      linesCleared = false;

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
    }

    plane.material.needsUpdate = true;
    texture.needsUpdate = true;

    plane.geometry.groupsNeedUpdate = true;
    plane.geometry.verticesNeedUpdate = true;
    plane.geometry.colorsNeedUpdate = true;
    plane.geometry.computeVertexNormals();

    // Render
    if (showMap.checked) {
      olMap.render();
    }
    composer.render();

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}

function resetView() {
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
  let mapdiv = document.getElementById("map");
  mapdiv.style.display = "block";
  mapZoomLevel = 0;
  mapdiv.style.width = mapResolution + "px";
  mapdiv.style.height = mapResolution + "px";
  olMap.updateSize();
  olMap.getView().setCenter(ol.proj.fromLonLat(mapCenter));
  olMap.getView().setResolution(circumferenceEarth / mapResolution / mapZoomFactor)
  mapdiv.style.display = "none";
}

document.addEventListener("keydown", function(ev) {
  if (ev.key == "Escape") {
    resetView();
  }
});

function drawGrid(canvas) {
  let ctx = canvas.getContext("2d");
  let cols = divisions - 1;
  let rows = divisions - 1;

  let width = canvas.width;
  let height = canvas.height;
  let start_i = 0;
  let start_j = 0;

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
  for (let i = 0; i < divisions; i++) {
    for (let j = 0; j < divisions; j++) {
      gsap.to(plane.geometry.vertices[i * divisions + j], {
        duration: 0.25,
        z: map[i][j],
      });
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

function wheelEvent(event) {
  if (document.elementFromPoint(event.clientX, event.clientY).tagName != "CANVAS") {
    return;
  }
  let mapdiv = document.getElementById("map");
  mapdiv.style.display = "block";
  if (event.deltaY < 0) {
    mapZoomLevel += 1;
  } else if (mapZoomLevel > 0) {
    mapZoomLevel -= 1;
  }
  let effectiveMapZoomLevel = Math.min(mapZoomLevel, 54)
  let newMapResolution = mapResolution * Math.pow(0.95, -effectiveMapZoomLevel)
  mapdiv.style.width = newMapResolution + "px";
  mapdiv.style.height = newMapResolution + "px";
  olMap.updateSize();
  olMap.getView().setCenter(ol.proj.fromLonLat(mapCenter));
  olMap.getView().setResolution(circumferenceEarth / newMapResolution / mapZoomFactor)
  mapdiv.style.display = "none";
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
  let xmlHttp = new XMLHttpRequest();
  xmlHttp.responseType = "text"

  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
      let data = xmlHttp.responseText;
      data = JSON.parse(data);
      console.log(data);
      let distances = data.distances;
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
    let line_color = getCurvatureColor(edges[j].weight);
    let material = new T.LineBasicMaterial({
      color: line_color,
      linewidth: 4,
      linecap: "round"
    });
    for (let pt of paths[i]) {
      points.push(new T.Vector3(pt[0], pt[2], -pt[1]));
    }
    const geometry = new T.BufferGeometry().setFromPoints(points);
    const line = new T.Line(geometry, material);
    scene.add(line);
  }
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
  if (this.value == "") {
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
  if (this.value == "" || isNaN(this.value)) {
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
  if (typeof drawPoint == "undefined") {
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

  if (typeof name == "undefined") {
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

  let showHoverGraph = document.getElementById("show-hover-graph");
  let sprite = getNameSprite(name);
  if (showHoverGraph.checked && drawPoint) {
    sprite.position.set(xPos.value, vertexHeight + 0.1, yPos.value);
    scene.add(sprite);
    newPt.scale.set(0.1, 0.1, 0.1);
    scene.add(newPt);
    gsap.to(newPt.scale, {
      duration: 1.5,
      x: 1,
      y: 1,
      z: 1,
      ease: "elastic"
    });
  }
  vertices[String(vertexCount)] = new VertexObj(vertexCount, name, newPt, sprite, lat, long);
  names[name] = vertexCount;
  vertexCount++;
}

function removeVertex() {
  let parentDiv = this.parentElement;
  let name = parentDiv.childNodes[0].textContent;
  scene.remove(vertices[name].mesh);
  scene.remove(vertices[name].label);
  delete vertices[name];
  parentDiv.remove();
}

function drawEdge(edge, lineMat) {
  let points = [
    new T.Vector3(edge.start.mesh.position.x, vertexHeight, edge.start.mesh.position.z),
    new T.Vector3(edge.end.mesh.position.x, vertexHeight, edge.end.mesh.position.z)
  ];

  let geom = new T.BufferGeometry().setFromPoints(points);

  let mat = new T.LineBasicMaterial({
    color: edgecolor,
    linewidth: 5,
    clippingPlanes: [clipPlaneUp, clipPlaneRight, clipPlaneLeft, clipPlaneBack, clipPlaneFront]
  })
  let line = new T.Line(geom, mat);
  let color = getCurvatureColor(edge.weight);
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
  if (typeof start == "undefined") {
    start = 0;
  }

  if (typeof end == "undefined") {
    end = 0;
  }

  if (typeof weight == "undefined") {
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

function edgeChange() {
  // TODO: Deal with non existent vertices
  if (this.value == "" || isNaN(this.value)) {
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
}

function removeEdge() {
  let parentDiv = this.parentElement;
  let id = parentDiv.childNodes[0].textContent;
  delete edges[id];
  parentDiv.remove();
}

function getNameSprite(name) {
  let canvas = document.createElement("canvas");
  let ctx = canvas.getContext("2d");

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
  constructor(id, start, end, weight) {
    this.id = id;
    this.start = start;
    this.end = end;
    this.weight = weight;
    this.neg_mod = 1;
    this.nrw_mod = 1;
    this.nheight_mod = 1;
    this.mesh = null;
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
  let mapdiv = document.createElement("div");
  mapdiv.id = "map";
  mapdiv.class = "map-div";
  mapdiv.style.width = mapResolution + "px";
  mapdiv.style.height = mapResolution + "px";
  document.body.appendChild(mapdiv);
  let map = new ol.Map({
    target: "map",
    renderer: "canvas",
    layers: [
      new ol.layer.Tile({
        source: new ol.source.OSM(),
      })
    ],
    view: new ol.View({
      projection: 'EPSG:3857',
      center: ol.proj.fromLonLat(mapCenter),
      resolution: circumferenceEarth / mapResolution / mapZoomFactor
    })
  });
  return map;
}

function calculateCurvature() {
  let data = {
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
  for (let id in current_edges) {
    let edge = current_edges[id];
    data.links.push({
      source: edge.start.id,
      target: edge.end.id
    });
  }
  let xmlHttp = new XMLHttpRequest();
  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
      data = JSON.parse(xmlHttp.responseText)
      let current_edges = {
        ...edges
      };
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

function dropOutput(evt) {
  evt.stopPropagation();
  evt.preventDefault();

  let xmlHttp = new XMLHttpRequest();
  xmlHttp.responseType = "text";
  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState != 4 || xmlHttp.status != 200) {
      return;
    }

    let data = JSON.parse(xmlHttp.responseText);

    // Set map
    mapCenter = data.mapCenter;
    mapZoomFactor = data.mapZoomFactor;
    resetView();

    // Set vertex heights
    let heightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0))
    for (let i = 0; i < divisions; i++) {
      for (let j = 0; j < divisions; j++) {
        heightMap[j][divisions - 1 - i] = data.heights[i * divisions + j] * 20.;
      }
    }
    updatePlaneHeights(heightMap);

    // Add vertices and edges
    for (let i = 0; i < data.vertices.length; ++i) {
      let vertex = data.vertices[i];
      addVertex(null, vertex[0], vertex[1], true, i.toString());
    }
    for (let i = 0; i < data.edges.length; ++i) {
      let edge = data.edges[i];
      addEdge(null, edge[0], edge[1], edge[2]);
    }
  }

  let reader = new FileReader();
  reader.onload = function() {
    xmlHttp.open("post", "unpickle");
    xmlHttp.setRequestHeader("Content-Type", "application/octet-stream");
    xmlHttp.send(reader.result);
  }
  reader.readAsArrayBuffer(evt.dataTransfer.files[0]);
}

function dragOver(evt) {
  evt.stopPropagation();
  evt.preventDefault();
  evt.dataTransfer.dropEffect = "copy"; // Explicitly show this is a copy.
}

function convert3JStoVerts(x, y) {
  // Depends on plane geometry
  // Change from -10,10 to 0,49
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

function dist(startPt, endPt) {
  return Math.sqrt((startPt[0] - endPt[0]) ** 2 + (startPt[1] - endPt[1]) ** 2)
}

const CURVATURE_COLORS = [[0.403921568627451,0.0,0.12156862745098039],[0.4154555940023068,0.003690888119953864,0.12341407151095732],[0.4269896193771626,0.007381776239907728,0.12525951557093426],[0.43852364475201844,0.011072664359861591,0.12710495963091117],[0.45005767012687425,0.014763552479815456,0.12895040369088812],[0.4615916955017301,0.01845444059976932,0.13079584775086506],[0.47312572087658594,0.022145328719723183,0.132641291810842],[0.48465974625144176,0.02583621683967705,0.1344867358708189],[0.4961937716262976,0.02952710495963091,0.13633217993079585],[0.5077277970011534,0.03321799307958478,0.13817762399077277],[0.5192618223760093,0.03690888119953864,0.1400230680507497],[0.5307958477508651,0.0405997693194925,0.14186851211072665],[0.5423298731257209,0.044290657439446365,0.1437139561707036],[0.5538638985005767,0.04798154555940023,0.1455594002306805],[0.5653979238754325,0.0516724336793541,0.14740484429065745],[0.5769319492502883,0.05536332179930796,0.14925028835063436],[0.5884659746251442,0.05905420991926182,0.1510957324106113],[0.6,0.06274509803921569,0.15294117647058825],[0.6115340253748558,0.06643598615916955,0.1547866205305652],[0.6230680507497116,0.07012687427912341,0.1566320645905421],[0.6346020761245674,0.07381776239907728,0.15847750865051904],[0.6461361014994232,0.07750865051903114,0.16032295271049596],[0.6576701268742791,0.081199538638985,0.1621683967704729],[0.669204152249135,0.08489042675893888,0.16401384083044984],[0.6807381776239907,0.08858131487889273,0.16585928489042678],[0.6922722029988466,0.09227220299884659,0.1677047289504037],[0.7008073817762399,0.09965397923875433,0.17124183006535948],[0.7063437139561707,0.11072664359861592,0.17647058823529413],[0.7118800461361015,0.12179930795847752,0.18169934640522878],[0.7174163783160322,0.1328719723183391,0.1869281045751634],[0.7229527104959631,0.14394463667820068,0.19215686274509805],[0.7284890426758939,0.15501730103806227,0.1973856209150327],[0.7340253748558246,0.16608996539792387,0.20261437908496732],[0.7395617070357554,0.1771626297577854,0.20784313725490194],[0.7450980392156863,0.18823529411764706,0.2130718954248366],[0.7506343713956171,0.19930795847750865,0.21830065359477124],[0.7561707035755478,0.21038062283737025,0.22352941176470587],[0.7617070357554786,0.22145328719723176,0.2287581699346405],[0.7672433679354095,0.2325259515570934,0.23398692810457516],[0.7727797001153403,0.243598615916955,0.2392156862745098],[0.778316032295271,0.2546712802768166,0.24444444444444444],[0.7838523644752018,0.2657439446366781,0.24967320261437903],[0.7893886966551327,0.2768166089965398,0.2549019607843137],[0.7949250288350634,0.28788927335640135,0.26013071895424833],[0.8004613610149942,0.298961937716263,0.265359477124183],[0.805997693194925,0.3100346020761245,0.2705882352941176],[0.8115340253748559,0.3211072664359862,0.2758169934640523],[0.8170703575547866,0.33217993079584773,0.28104575163398693],[0.8226066897347174,0.34325259515570933,0.28627450980392155],[0.8281430219146482,0.35432525951557087,0.2915032679738562],[0.833679354094579,0.3653979238754325,0.29673202614379085],[0.8392156862745098,0.3764705882352941,0.30196078431372547],[0.8438292964244521,0.3870818915801615,0.3101114955786236],[0.8484429065743945,0.39769319492502875,0.31826220684352163],[0.8530565167243368,0.4083044982698962,0.3264129181084198],[0.8576701268742791,0.41891580161476355,0.3345636293733179],[0.8622837370242215,0.42952710495963087,0.34271434063821604],[0.8668973471741638,0.44013840830449813,0.35086505190311407],[0.8715109573241061,0.4507497116493656,0.35901576316801226],[0.8761245674740484,0.46136101499423293,0.3671664744329104],[0.8807381776239908,0.4719723183391003,0.3753171856978085],[0.8853517877739331,0.48258362168396757,0.38346789696270656],[0.8899653979238754,0.493194925028835,0.3916186082276047],[0.8945790080738177,0.5038062283737024,0.39976931949250283],[0.8991926182237601,0.5144175317185697,0.4079200307574009],[0.9038062283737024,0.5250288350634371,0.41607074202229904],[0.9084198385236447,0.5356401384083043,0.424221453287197],[0.913033448673587,0.5462514417531718,0.43237216455209526],[0.9176470588235294,0.5568627450980391,0.44052287581699334],[0.9222606689734717,0.5674740484429065,0.4486735870818915],[0.926874279123414,0.5780853517877739,0.4568242983467896],[0.9314878892733564,0.5886966551326411,0.4649750096116877],[0.9361014994232987,0.5993079584775085,0.4731257208765858],[0.940715109573241,0.6099192618223759,0.4812764321414839],[0.9453287197231833,0.6205305651672431,0.48942714340638194],[0.9499423298731257,0.6311418685121106,0.49757785467128013],[0.954555940023068,0.641753171856978,0.5057285659361782],[0.9575547866205306,0.6512110726643597,0.515109573241061],[0.9589388696655133,0.659515570934256,0.5257208765859284],[0.960322952710496,0.6678200692041522,0.5363321799307956],[0.9617070357554787,0.6761245674740484,0.546943483275663],[0.9630911188004614,0.6844290657439446,0.5575547866205304],[0.9644752018454441,0.6927335640138407,0.5681660899653976],[0.9658592848904268,0.701038062283737,0.5787773933102651],[0.9672433679354094,0.7093425605536331,0.5893886966551325],[0.9686274509803922,0.7176470588235293,0.5999999999999999],[0.9700115340253749,0.7259515570934255,0.6106113033448672],[0.9713956170703576,0.7342560553633217,0.6212226066897346],[0.9727797001153403,0.7425605536332179,0.631833910034602],[0.9741637831603229,0.7508650519031141,0.6424452133794694],[0.9755478662053056,0.7591695501730102,0.6530565167243365],[0.9769319492502884,0.7674740484429066,0.6636678200692041],[0.9783160322952711,0.7757785467128027,0.6742791234140715],[0.9797001153402538,0.7840830449826989,0.6848904267589389],[0.9810841983852365,0.7923875432525951,0.6955017301038062],[0.9824682814302191,0.8006920415224913,0.7061130334486736],[0.9838523644752019,0.8089965397923875,0.7167243367935409],[0.9852364475201846,0.8173010380622837,0.7273356401384083],[0.9866205305651673,0.8256055363321797,0.7379469434832755],[0.98800461361015,0.833910034602076,0.748558246828143],[0.9893886966551326,0.8422145328719722,0.7591695501730104],[0.9907727797001153,0.8505190311418684,0.7697808535178777],[0.9921568627450981,0.8588235294117647,0.7803921568627451],[0.9912341407151096,0.8631295655517108,0.7877739331026529],[0.9903114186851212,0.867435601691657,0.7951557093425605],[0.9893886966551326,0.8717416378316032,0.8025374855824683],[0.9884659746251442,0.8760476739715493,0.809919261822376],[0.9875432525951557,0.8803537101114955,0.8173010380622837],[0.9866205305651673,0.8846597462514417,0.8246828143021915],[0.9856978085351787,0.8889657823913879,0.8320645905420992],[0.9847750865051903,0.8932718185313341,0.8394463667820069],[0.9838523644752019,0.8975778546712803,0.8468281430219147],[0.9829296424452134,0.9018838908112264,0.8542099192618224],[0.982006920415225,0.9061899269511726,0.8615916955017301],[0.9810841983852365,0.9104959630911187,0.8689734717416377],[0.980161476355248,0.914801999231065,0.8763552479815455],[0.9792387543252595,0.9191080353710112,0.8837370242214533],[0.9783160322952711,0.9234140715109573,0.891118800461361],[0.9773933102652826,0.9277201076509035,0.8985005767012687],[0.9764705882352941,0.9320261437908497,0.9058823529411765],[0.9755478662053056,0.9363321799307959,0.9132641291810842],[0.9746251441753172,0.940638216070742,0.9206459054209919],[0.9737024221453288,0.9449442522106881,0.9280276816608996],[0.9727797001153402,0.9492502883506344,0.9354094579008074],[0.9718569780853518,0.9535563244905806,0.9427912341407151],[0.9709342560553633,0.9578623606305268,0.9501730103806229],[0.9700115340253749,0.9621683967704728,0.9575547866205305],[0.9690888119953864,0.9664744329104191,0.9649365628604383],[0.9657054978854287,0.9672433679354094,0.9680891964628989],[0.9598615916955018,0.9644752018454441,0.9670126874279124],[0.9540176855055748,0.9617070357554787,0.9659361783929258],[0.9481737793156478,0.9589388696655132,0.9648596693579392],[0.942329873125721,0.956170703575548,0.9637831603229527],[0.936485966935794,0.9534025374855825,0.9627066512879662],[0.930642060745867,0.9506343713956171,0.9616301422529796],[0.92479815455594,0.9478662053056517,0.960553633217993],[0.9189542483660131,0.9450980392156864,0.9594771241830066],[0.9131103421760862,0.9423298731257209,0.95840061514802],[0.9072664359861592,0.9395617070357555,0.9573241061130334],[0.9014225297962323,0.9367935409457901,0.956247597078047],[0.8955786236063054,0.9340253748558247,0.9551710880430604],[0.8897347174163783,0.9312572087658594,0.9540945790080738],[0.8838908112264514,0.9284890426758939,0.9530180699730872],[0.8780469050365245,0.9257208765859286,0.9519415609381008],[0.8722029988465976,0.9229527104959632,0.9508650519031142],[0.8663590926566707,0.9201845444059977,0.9497885428681276],[0.8605151864667436,0.9174163783160324,0.9487120338331411],[0.8546712802768167,0.914648212226067,0.9476355247981546],[0.8488273740868899,0.9118800461361016,0.946559015763168],[0.8429834678969628,0.9091118800461362,0.9454825067281815],[0.8371395617070359,0.9063437139561707,0.9444059976931949],[0.8312956555171089,0.9035755478662054,0.9433294886582084],[0.825451749327182,0.90080738177624,0.9422529796232219],[0.8196078431372551,0.8980392156862746,0.9411764705882353],[0.8099192618223763,0.8931180315263362,0.93840830449827],[0.8002306805074973,0.8881968473663977,0.9356401384083045],[0.7905420991926184,0.8832756632064592,0.9328719723183392],[0.7808535178777396,0.8783544790465208,0.9301038062283737],[0.7711649365628607,0.8734332948865823,0.9273356401384084],[0.7614763552479817,0.8685121107266438,0.924567474048443],[0.7517877739331029,0.8635909265667053,0.9217993079584775],[0.742099192618224,0.8586697424067669,0.9190311418685122],[0.7324106113033451,0.8537485582468283,0.9162629757785468],[0.7227220299884662,0.8488273740868898,0.9134948096885814],[0.7130334486735876,0.8439061899269515,0.9107266435986161],[0.7033448673587084,0.8389850057670128,0.9079584775086506],[0.6936562860438296,0.8340638216070744,0.9051903114186852],[0.6839677047289506,0.8291426374471359,0.9024221453287198],[0.6742791234140717,0.8242214532871974,0.8996539792387545],[0.6645905420991929,0.819300269127259,0.896885813148789],[0.654901960784314,0.8143790849673205,0.8941176470588236],[0.645213379469435,0.8094579008073819,0.8913494809688582],[0.6355247981545562,0.8045367166474434,0.8885813148788928],[0.6258362168396773,0.7996155324875049,0.8858131487889275],[0.6161476355247983,0.7946943483275665,0.883044982698962],[0.6064590542099195,0.789773164167628,0.8802768166089966],[0.5967704728950406,0.7848519800076895,0.8775086505190313],[0.5870818915801617,0.779930795847751,0.8747404844290658],[0.5773933102652828,0.7750096116878126,0.8719723183391004],[0.5664744329104193,0.7687043444828915,0.8685121107266437],[0.5543252595155715,0.7610149942329878,0.8643598615916958],[0.5421760861207231,0.7533256439830837,0.8602076124567475],[0.530026912725875,0.7456362937331797,0.8560553633217994],[0.5178777393310268,0.7379469434832758,0.8519031141868513],[0.5057285659361787,0.7302575932333719,0.8477508650519032],[0.4935793925413305,0.7225682429834681,0.8435986159169551],[0.4814302191464823,0.7148788927335642,0.839446366782007],[0.4692810457516342,0.7071895424836603,0.8352941176470589],[0.45713187235678604,0.6995001922337564,0.8311418685121108],[0.4449826989619379,0.6918108419838525,0.8269896193771626],[0.43283352556708976,0.6841214917339487,0.8228373702422146],[0.42068435217224165,0.6764321414840447,0.8186851211072664],[0.4085351787773935,0.6687427912341408,0.8145328719723184],[0.3963860053825453,0.6610534409842369,0.8103806228373702],[0.38423683198769715,0.653364090734333,0.8062283737024222],[0.37208765859284904,0.6456747404844292,0.8020761245674741],[0.3599384851980012,0.6379853902345254,0.7979238754325261],[0.34778931180315276,0.6302960399846214,0.7937716262975778],[0.3356401384083046,0.6226066897347174,0.7896193771626298],[0.3234909650134564,0.6149173394848135,0.7854671280276817],[0.3113417916186083,0.6072279892349096,0.7813148788927335],[0.29919261822376014,0.5995386389850057,0.7771626297577854],[0.287043444828912,0.5918492887351019,0.7730103806228373],[0.27489427143406386,0.584159938485198,0.7688581314878893],[0.2627450980392157,0.5764705882352941,0.7647058823529411],[0.2575163398692811,0.5695501730103806,0.7611687812379854],[0.2522875816993464,0.5626297577854671,0.7576316801230295],[0.24705882352941178,0.5557093425605536,0.7540945790080738],[0.24183006535947713,0.5487889273356401,0.750557477893118],[0.2366013071895425,0.5418685121107266,0.7470203767781622],[0.23137254901960785,0.5349480968858131,0.7434832756632064],[0.2261437908496732,0.5280276816608996,0.7399461745482506],[0.22091503267973872,0.5211072664359864,0.736409073433295],[0.21568627450980393,0.5141868512110727,0.7328719723183391],[0.21045751633986928,0.5072664359861592,0.7293348712033833],[0.20522875816993463,0.5003460207612457,0.7257977700884275],[0.2,0.4934256055363322,0.7222606689734717],[0.1947712418300654,0.4865051903114187,0.718723567858516],[0.1895424836601307,0.47958477508650516,0.7151864667435601],[0.1843137254901961,0.47266435986159167,0.7116493656286044],[0.17908496732026147,0.46574394463667823,0.7081122645136486],[0.17385620915032682,0.45882352941176474,0.7045751633986929],[0.16862745098039217,0.4519031141868512,0.701038062283737],[0.16339869281045752,0.4449826989619377,0.6975009611687812],[0.15816993464052287,0.43806228373702427,0.6939638600538255],[0.15294117647058825,0.4311418685121108,0.6904267589388697],[0.1477124183006536,0.42422145328719724,0.6868896578239139],[0.14248366013071895,0.41730103806228375,0.6833525567089581],[0.13725490196078446,0.4103806228373704,0.6798154555940025],[0.1320261437908497,0.40346020761245677,0.6762783544790466],[0.12725874663590928,0.3958477508650519,0.6687427912341407],[0.1229527104959631,0.3875432525951557,0.6572087658592849],[0.11864667435601693,0.37923875432525955,0.6456747404844291],[0.11434063821607075,0.37093425605536334,0.6341407151095733],[0.11003460207612457,0.36262975778546713,0.6226066897347174],[0.10572856593617841,0.3543252595155709,0.6110726643598616],[0.10142252979623223,0.34602076124567477,0.5995386389850058],[0.09711649365628605,0.33771626297577856,0.58800461361015],[0.09281045751633987,0.3294117647058824,0.5764705882352942],[0.0885044213763937,0.3211072664359862,0.5649365628604384],[0.08419838523644753,0.31280276816609,0.5534025374855824],[0.07989234909650135,0.3044982698961938,0.5418685121107266],[0.07558631295655519,0.29619377162629756,0.5303344867358708],[0.071280276816609,0.2878892733564014,0.518800461361015],[0.06697424067666295,0.2795847750865054,0.5072664359861595],[0.06266820453671666,0.27128027681660905,0.49573241061130335],[0.05836216839677047,0.26297577854671284,0.48419838523644754],[0.054056132256824305,0.2546712802768166,0.4726643598615917],[0.049750096116878126,0.24636678200692042,0.4611303344867359],[0.04544405997693196,0.23806228373702423,0.4495963091118801],[0.04113802383698577,0.22975778546712802,0.4380622837370242],[0.0368319876970396,0.22145328719723184,0.4265282583621684],[0.032525951557093424,0.21314878892733566,0.4149942329873126],[0.02821991541714726,0.20484429065743945,0.40346020761245677],[0.02391387927720108,0.19653979238754324,0.3919261822376009],[0.0196078431372549,0.18823529411764706,0.3803921568627451],[0.0196078431372549,0.18823529411764706,0.3803921568627451]];
function getCurvatureColor(curvature) {
  // TODO: Make this less ridiculous
  let curvature_rgb = CURVATURE_COLORS[math.round(256 * (curvature + 2) / 4)];
  return new T.Color(curvature_rgb[0], curvature_rgb[1], curvature_rgb[2]);
}
