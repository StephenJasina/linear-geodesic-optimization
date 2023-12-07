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
let lineMat = new T.LineBasicMaterial({
  color: edgecolor,
  linewidth: 6,
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
        let lineWidth = 6;
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
    let line_color = new T.Color("hsl(0, 0%, 100%)");
    let endColor;
    if (edges[j].weight >= 0) {
      endColor = new T.Color("hsl(222, 100%, 61%)");
    } else {
      endColor = new T.Color("hsl(356, 74%, 52%)");
    }
    line_color.lerp(endColor, Math.min(Math.abs(edges[j].weight), 1));
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
