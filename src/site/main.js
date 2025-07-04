// // GSAP
// import { gsap } from "gsap";

// OpenLayers
import OLMap from "ol/Map.js";
import TileLayer from "ol/layer/Tile";
import View from "ol/View";
import { fromLonLat } from "ol/proj";
import { OSM } from "ol/source";

// Three.js
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Styling
import { getCurvatureColor, getWeightedColor } from "./modules/colors.js";

// Layers
import { createOLMap, updateOLMap, getCanvasOLMap, makeTextureOLMap, isReadyOLMap } from "./modules/layers/map.js";
import { default as drawEdges } from "./modules/layers/edges.js";
import { default as drawGrid } from "./modules/layers/grid.js";
import { default as drawGeodesics } from "./modules/layers/geodesics.js";
import { resetPlaneHeights, setPlaneHeightsWithInterpolation } from "./modules/layers/heights.js";
import { default as drawHeightChanges } from "./modules/layers/height_changes.js";
import { default as drawOutages } from "./modules/layers/outages.js";
import { default as drawVertices } from "./modules/layers/vertices.js";

// Colors
const COLOR_BACKGROUND = "#f3f3f3";
const COLOR_PLANE = "#ffffff";
const COLOR_VERTEX = "#4caf50";
const COLOR_EDGE = "#000000";
const COLOR_OUTAGE = "#00ff00";

const OUTAGE_BLINK_RATE = 2.;

// Globals tracking the world map
let mapCenter = [0.0, 0.0];
let mapZoomFactor = 1.6;

let canvasResolution = 1000;

// Size of the plane
let planeSideLength = 20.;
let planeXMin = -planeSideLength / 2;
let planeXMax = planeSideLength / 2;
let planeYMin = -planeSideLength / 2;
let planeYMax = planeSideLength / 2;

// Number of points per side
let divisions = 5;

// Globals tracking the manifold shape across time
let times = null;
let heights = null;
let heightsEWMA = null;
let networkVertices = null;
let networkEdges = null;
let networkEdgesAll = null;
let geodesics = null;
let edgeColors = null;

// Globals tracking the animation state
let isPlaying = false;
let timeInitial = null;
let dateTimeInitial = null;
let animationDuration = null;
let canvasNeedsUpdate = true;
let currentNetworkIndex = null;
let previousNetworkIndex = null;

// Scene setup
let scene = new THREE.Scene();
scene.background = new THREE.Color(COLOR_BACKGROUND);

// Camera setup
// let frustumScale = 64;
// let camera = new THREE.OrthographicCamera(
// 	window.innerWidth / -frustumScale, window.innerWidth / frustumScale,
// 	window.innerHeight / frustumScale, window.innerHeight / -frustumScale,
// 	1, 1000
// );
let frustumScale = 2.5 * planeSideLength / Math.max(window.innerWidth, window.innerHeight);
let camera = new THREE.OrthographicCamera(
	-window.innerWidth / 2. * frustumScale, window.innerWidth / 2. * frustumScale,
	window.innerHeight / 2. * frustumScale, -window.innerHeight / 2. * frustumScale,
	1, 1000
);
camera.position.x = -15;
camera.position.z = 20;
camera.position.y = 15;

// Renderer setup
let canvasMain = document.createElement("canvas");
canvasMain.style.zIndex = 50;
let renderer = new THREE.WebGLRenderer({
	canvas: canvasMain
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
let pixelRatio = renderer.getPixelRatio();
document.body.appendChild(renderer.domElement);

// Material setup
const canvasBlank = document.createElement("canvas");
let contextBlank = canvasBlank.getContext("2d");
contextBlank.canvas.width = canvasResolution;
contextBlank.canvas.height = canvasResolution;
contextBlank.fillStyle = COLOR_PLANE;
contextBlank.fillRect(0, 0, contextBlank.canvas.width, contextBlank.canvas.height);
let textureBlank = makeTextureBlank();

let canvasMap = null;
let textureMap = null;
let olMap = createOLMap(canvasResolution, mapCenter, mapZoomFactor);

let textureCurrent = textureBlank;

let planeMaterial = new THREE.MeshStandardMaterial({
	side: THREE.DoubleSide,
	map: textureCurrent,
	transparent: true,
	opacity: 1.0,
})

// Geometry setup
let plane = makePlane(divisions);
scene.add(plane);

// Lighting setup
let lightDirectional = new THREE.DirectionalLight(0xffffff, 1.);
lightDirectional.position.set(1, 1, 1);
scene.add(lightDirectional);
let lightAmbient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(lightAmbient);

// Network drawing setup
const HEIGHT_HOVER_GRAPH = 10;
const SIZE_HOVER_VERTEX = 6;
const SIZE_HOVER_EDGE = 4;

const HOVER_VERTEX_MATERIAL = new THREE.PointsMaterial({
	color: COLOR_VERTEX,
	size: SIZE_HOVER_VERTEX,
});

let hoverVerticesDrawn = [];
let hoverEdgesDrawn = [];

// Weighted arc setup
const MIN_HEIGHT_HOVER_ARC = 5;
const MAX_HEIGHT_HOVER_ARC = 8;
const SIZE_HOVER_ARC_VERTEX = 6;
const SIZE_HOVER_ARC_EDGE = 4;
const HOVER_ARC_DIVISIONS = 31;

const HOVER_ARC_VERTEX_MATERIAL = new THREE.PointsMaterial({
	color: COLOR_VERTEX,
	size: SIZE_HOVER_VERTEX,
});

let hoverArcVerticesDrawn = [];
let hoverArcEdgesDrawn = [];

// Canvas drawing constants
const VERTEX_RADIUS = 4;
const LINE_WIDTH = 6;
const GEODESICS_WIDTH = 3;

// Renderers
let renderPass = new RenderPass(scene, camera);
let composer = new EffectComposer(renderer);
composer.addPass(renderPass);

// Controls setup
let controls = new OrbitControls(camera, renderer.domElement);
controls.enablePan = true;
controls.panSpeed = 1;
controls.enableRotate = true;
controls.enableZoom = true;
controls.minZoom = 1;
window.addEventListener("wheel", wheelEvent, true);

// Interactivity with DOM elements

let divDropOutput = document.getElementById("drop-output");
divDropOutput.addEventListener("dragover", dragOver, false);
divDropOutput.addEventListener("drop", dropOutput, false);

let rangeAnimation = document.getElementById("range-animation");
let checkboxShowHeights = document.getElementById("show-heights");
function setHeights() {
	if (checkboxShowHeights.checked && heights != null) {
		setPlaneHeightsWithInterpolation(plane, heights, rangeAnimation.value, times);
	} else {
		resetPlaneHeights(plane);
	}
}
rangeAnimation.oninput = function() {
	timeInitial = parseFloat(rangeAnimation.value);
	dateTimeInitial = Date.now() / 1000;
	previousNetworkIndex = currentNetworkIndex;
	currentNetworkIndex = getCurrentNetworkIndex();
	setHeights();

	canvasNeedsUpdate = true;
}
checkboxShowHeights.onchange = setHeights;

let checkboxShowHeightChanges = document.getElementById("show-height-changes");
checkboxShowHeightChanges.onchange = function() {
	canvasNeedsUpdate = true;
}

let checkboxShowMap = document.getElementById("show-map");
checkboxShowMap.onchange = function() {
	if (checkboxShowMap.checked) {
		updateOLMap(olMap, getCurrentResolution(), mapCenter, mapZoomFactor);
	} else {
		setCanvasZoom();
	}
	canvasNeedsUpdate = true;
}

let checkboxShowGrid = document.getElementById("show-grid");
checkboxShowGrid.onchange = function() {
	canvasNeedsUpdate = true;
}

let checkboxShowGraph = document.getElementById("show-graph");
checkboxShowGraph.onchange = function() {
	canvasNeedsUpdate = true;
}

let checkboxShowOutages = document.getElementById("show-outages");
let divOutages = document.getElementById("div-outages");
let ulOutages = document.getElementById("ul-outages");
if (checkboxShowOutages.checked) {
	divOutages.style.display = "block";
} else {
	divOutages.style.display = "none";
}
checkboxShowOutages.onchange = function() {
	if (checkboxShowOutages.checked) {
		divOutages.style.display = "block";
	} else {
		divOutages.style.display = "none";
	}

	canvasNeedsUpdate = true;
}

let checkboxShowHoverGraph = document.getElementById("show-hover-graph");
checkboxShowHoverGraph.onchange = function() {
	if (checkboxShowHoverGraph.checked) {
		if (times != null) {
			drawHoverGraph(currentNetworkIndex);
		}
	} else {
		clearHoverGraph();
	}
}

let checkboxShowWeightArcs = document.getElementById("show-weight-arcs");
checkboxShowWeightArcs.onchange = function() {
	if (checkboxShowWeightArcs.checked) {
		drawHoverArcs();
	} else {
		clearHoverArcs();
	}
}

let checkboxShowGeodesics = document.getElementById("show-geodesics");
checkboxShowGeodesics.onchange = function() {
	canvasNeedsUpdate = true;
}

let buttonHelp = document.getElementById("btn-help");
buttonHelp.onclick = helpClick;

let divAnimationControls = document.getElementById("animation-controls");
let buttonPlay = document.getElementById("button-play");
buttonPlay.onclick = function() {
	if (isPlaying) {
		buttonPlay.innerText = "Play";
	} else {
		buttonPlay.innerText = "Pause";
		timeInitial = parseFloat(rangeAnimation.value);
		dateTimeInitial = Date.now() / 1000;
	}
	isPlaying = !isPlaying;
}

/**
 * Draw every layer above the base layer (i.e., excluding the map)
 */
function drawLayers(time) {
	// Figure out which canvas we're using
	let context = null;
	if (checkboxShowMap.checked) {
		plane.material.map = textureMap;
		textureCurrent = textureMap;
		context = canvasMap.getContext("2d");
	} else {
		plane.material.map = textureBlank;
		textureCurrent = textureBlank;
		context = canvasBlank.getContext("2d");
	}

	// Draw grid lines
	if (checkboxShowGrid.checked) {
		drawGrid(context, divisions);
	}

	// Draw the graph, geodesics, and outages onto the canvas
	if (
		networkEdges != null && networkEdges.length > 0
		&& (checkboxShowGraph.checked || checkboxShowGeodesics.checked || checkboxShowOutages.checked)
	) {
		// First, figure out the edges of the current network
		let edges = networkEdges[currentNetworkIndex];

		// Draw the edges
		if (checkboxShowGraph.checked) {
			drawEdges(context, networkVertices, edges);
		}

		// Draw the geodesics
		if (checkboxShowGeodesics.checked) {
			drawGeodesics(context, geodesics[currentNetworkIndex], GEODESICS_WIDTH, edgeColors[currentNetworkIndex]);
		}

		// Deal with outages
		if (checkboxShowOutages.checked) {
			drawOutages(
				context,
				networkVertices, edges, networkEdgesAll,
				currentNetworkIndex != previousNetworkIndex ? ulOutages : null
			);
		}

		// Draw the vertices
		drawVertices(context, networkVertices, VERTEX_RADIUS, COLOR_VERTEX);

	}

	if (checkboxShowHeightChanges.checked) {
		drawHeightChanges(context, heights, heightsEWMA, time, times);
	}

	textureCurrent.needsUpdate = true;
}

function animate() {
	// Update the slider if necessary
	if (isPlaying) {
		let rangeAnimation = document.getElementById("range-animation");
		let timeMin = parseFloat(rangeAnimation.min);
		let timeMax = parseFloat(rangeAnimation.max);
		let timeRange = timeMax - timeMin;

		let dateTimeNow = Date.now() / 1000;
		let alpha = ((dateTimeNow - dateTimeInitial) % animationDuration) / animationDuration;
		let timeNow = (timeInitial - timeMin + alpha * timeRange) % timeRange + timeMin;
		rangeAnimation.value = timeNow;

		// Update the heights if necessary
		if (checkboxShowHeights.checked) {
			setPlaneHeightsWithInterpolation(plane, heights, timeNow, times);
		}

		previousNetworkIndex = currentNetworkIndex;
		currentNetworkIndex = getCurrentNetworkIndex();

		canvasNeedsUpdate = true;
	}

	// Update the canvas if needed
	if (canvasNeedsUpdate || checkboxShowOutages.checked) {
		canvasNeedsUpdate = false;

		// Figure out which canvas we're using
		let context = null;
		if (checkboxShowMap.checked) {
			plane.material.map = textureMap;
			textureCurrent = textureMap;
			context = canvasMap.getContext("2d");
		} else {
			plane.material.map = textureBlank;
			textureCurrent = textureBlank;
			context = canvasBlank.getContext("2d");
		}

		// Draw the bottommost layer
		if (checkboxShowMap.checked) {
			olMap.renderSync();
		} else {
			context.fillStyle = COLOR_PLANE;
			context.fillRect(0, 0, context.canvas.width, context.canvas.height);
		}

		drawLayers((times == null || times.length == 0) ? null : (times.length == 1 ? times[0] : rangeAnimation.value));

		textureCurrent.needsUpdate = true;
	}

	plane.geometry.computeVertexNormals();
	plane.material.needsUpdate = true;

	// Render
	composer.render();
}
// Start the animation loop once the map canvas exists
olMap.once("postrender", function(event) {
	canvasMap = getCanvasOLMap(olMap);
	textureMap = makeTextureOLMap(olMap);

	renderer.setAnimationLoop(animate);
});
// Update the map texture whenever the map changes
olMap.on("postrender", function(event) {
	drawLayers();
	textureMap.needsUpdate = true;
});

window.addEventListener("resize", function() {
	frustumScale = 2.5 * planeSideLength / Math.max(window.innerWidth, window.innerHeight);
	camera.left = -window.innerWidth / 2. * frustumScale;
	camera.right = window.innerWidth / 2. * frustumScale;
	camera.top = window.innerHeight / 2. * frustumScale;
	camera.bottom = -window.innerHeight / 2. * frustumScale;
	camera.updateProjectionMatrix();

	renderer.setSize(window.innerWidth, window.innerHeight);
}, false);

function getCurrentNetworkIndex() {
	let time = parseFloat(rangeAnimation.value);
	if (times == null) {
		return null;
	}

	if (time <= times[0]) {
		return 0;
	}

	if (time >= times[times.length - 1]) {
		return times.length - 1;
	}

	// TODO: Improve this from a linear search
	let index = 0;
	while (times[index] < time) {
		++index;
	}

	if (times[index] == time) {
		return index;
	}

	// If we reach here ts[index - 1] <= t < ts[index]
	let lambda = (time - times[index - 1]) / (times[index] - times[index - 1]);
	if (lambda < 0.5) {
		return index - 1;
	} else {
		return index;
	}
}

function resetView() {
	// gsap.to(camera, {
	// 	duration: 1,
	// 	zoom: 1,
	// 	onUpdate: function() {
	// 		camera.updateProjectionMatrix();
	// 	}
	// });
	// gsap.to(controls.target, {
	// 	duration: 1,
	// 	x: 0,
	// 	y: 0,
	// 	z: 0,
	// 	onUpdate: function() {
	// 		controls.update();
	// 	}
	// });
	// gsap.to(plane.position, {
	// 	duration: 1,
	// 	y: 0,
	// 	onStart: function() {
	// 		plane.visible = true;
	// 	},
	// 	onUpdate: function() {}
	// });

	if (checkboxShowMap.checked) {
		updateOLMap(olMap, canvasResolution, mapCenter, mapZoomFactor);
	} else {
		setCanvasZoom(canvasResolution);
	}

	canvasNeedsUpdate = true;
}

document.addEventListener("keydown", function(ev) {
	if (ev.key == "Escape") {
		resetView();
	}
});

function helpClick(event) {
	let helpDiv = document.getElementById("div-help");
	if (helpDiv.style.display === "none") {
		helpDiv.style.display = "block";
	} else {
		helpDiv.style.display = "none";
	}
}

function makeTextureBlank() {
	let textureBlank = new THREE.CanvasTexture(contextBlank.canvas);
	textureBlank.minFilter = THREE.LinearFilter;
	textureBlank.center = new THREE.Vector2(0.5, 0.5);
	textureBlank.rotation = -Math.PI / 2;
	return textureBlank;
}

function getCurrentResolution() {
	return canvasResolution * Math.min(camera.zoom, 4.);
}

function setCanvasZoom(resolution = null) {
	if (resolution == null) {
		resolution = getCurrentResolution();
	}

	textureBlank.dispose();
	contextBlank.canvas.width = resolution;
	contextBlank.canvas.height = resolution;
	textureBlank = makeTextureBlank();
	plane.material.map = textureBlank;
	textureCurrent = textureBlank;

	canvasNeedsUpdate = true;
}

function wheelEvent(event) {
	if (document.elementFromPoint(event.clientX, event.clientY).tagName != "CANVAS") {
		return;
	}

	if (checkboxShowMap.checked) {
		updateOLMap(olMap, getCurrentResolution(), mapCenter, mapZoomFactor);
	} else {
		setCanvasZoom();
	}
	textureMap = makeTextureOLMap(olMap);

	canvasNeedsUpdate = true;
}

function makePlane(divisions) {
	let planeGeometry = new THREE.PlaneGeometry(
		planeSideLength, planeSideLength,
		divisions - 1, divisions - 1
	);
	let plane = new THREE.Mesh(planeGeometry, planeMaterial);
	plane.rotation.set(-Math.PI / 2, 0, 0);
	return plane;
}

function clearHoverGraph() {
	for (let hoveringVertex of hoverVerticesDrawn) {
		scene.remove(hoveringVertex);
		hoveringVertex.geometry.dispose();
	}
	hoverVerticesDrawn = [];

	for (let hoveringEdge of hoverEdgesDrawn) {
		scene.remove(hoveringEdge);
		hoveringEdge.material.dispose();
		hoveringEdge.geometry.dispose();
	}
	hoverEdgesDrawn = [];
}

function drawHoverGraph(index) {
	clearHoverGraph();

	let vertexArray = [];
	for (let vertex of networkVertices) {
		let vertexGlobal = vertexToGlobalCoordinates(vertex.coordinates);
		vertexArray.push(vertexGlobal[1], HEIGHT_HOVER_GRAPH, vertexGlobal[0]);
	}
	let pointsGeometry = new THREE.BufferGeometry();
	pointsGeometry.setAttribute("position", new THREE.BufferAttribute(
		new Float32Array(vertexArray), 3
	));
	let points = new THREE.Points(pointsGeometry, HOVER_VERTEX_MATERIAL);
	scene.add(points);
	hoverVerticesDrawn.push(points);

	for (let edge of networkEdges[index]) {
		let edgeArray = [];
		let source = vertexToGlobalCoordinates(networkVertices[edge.source].coordinates);
		let target = vertexToGlobalCoordinates(networkVertices[edge.target].coordinates);
		edgeArray.push(source[1], HEIGHT_HOVER_GRAPH, source[0]);
		edgeArray.push(target[1], HEIGHT_HOVER_GRAPH, target[0]);

		let edgesGeometry = new LineSegmentsGeometry();
		edgesGeometry.setPositions(new Float32Array(edgeArray));
		let lineSegment = new LineSegments2(edgesGeometry, new LineMaterial({
			color: getCurvatureColor(edge.curvature),
			linewidth: SIZE_HOVER_EDGE,
		}));

		scene.add(lineSegment);
		hoverEdgesDrawn.push(lineSegment);
	}
}

function clearHoverArcs() {
	for (let hoveringVertex of hoverArcVerticesDrawn) {
		scene.remove(hoveringVertex);
		hoveringVertex.geometry.dispose();
	}
	hoverArcVerticesDrawn = [];

	for (let hoveringEdge of hoverArcEdgesDrawn) {
		scene.remove(hoveringEdge);
		hoveringEdge.material.dispose();
		hoveringEdge.geometry.dispose();
	}
	hoverArcEdgesDrawn = [];
}

function getMeshHeightAtCoordinates(vertex, zs) {
	let i = Math.floor(vertex[0] * (divisions - 1));
	let alpha = (vertex[0] * (divisions - 1)) % 1;
	let j = Math.floor(vertex[1] * (divisions - 1));
	let beta = (vertex[1] * (divisions - 1)) % 1;

	let z00 = zs[i][j];
	let z01 = zs[i][j + 1];
	let z10 = zs[i + 1][j];
	let z11 = zs[i + 1][j + 1];

	return (
		(1. - alpha) * ((1. - beta) * z00 + beta * z01)
		+ alpha * ((1. - beta) * z10 + beta * z11)
	) * planeSideLength;
}

function drawHoverArcsWithInterpolation(networkVertices, networkEdgesLeft, networkEdgesRight, zLeft, zRight, lambda) {
	clearHoverArcs();

	// Compute edge weights
	let edgeWeightsLeft = new Array();
	for (const vertex of networkVertices) {
		edgeWeightsLeft.push(new Map());
	}
	for (const edge of networkEdgesLeft) {
		let source = edge.source;
		let target = edge.target;

		edgeWeightsLeft[source].set(target, edge.throughput);
	}

	let edgeWeights = new Array();
	for (let vertex of networkVertices) {
		edgeWeights.push(new Map());
	}
	for (let edge of networkEdgesRight) {
		let source = edge.source;
		let target = edge.target;

		if (edgeWeightsLeft[source].has(target)) {
			edgeWeights[source].set(target, (1. - lambda) * edgeWeightsLeft[source].get(target) + lambda * edge.throughput);
		} else if (lambda > 0.5) {
				edgeWeights[source].set(target, (2. * lambda - 1.) * edge.throughput);
		}
	}
	for (let edge of networkEdgesLeft) {
		let source = edge.source;
		let target = edge.target;

		if (!edgeWeights[source].has(target) && lambda <= 0.5) {
			edgeWeights[source].set(target, (1. - 2. * lambda) * edge.throughput);
		}
	}

	// Compute network coordinates
	let coordinates = new Array(networkVertices.length);
	for (let i = 0; i < coordinates.length; ++i) {
		let vertex = networkVertices[i].coordinates;

		let coordinatesXY = vertexToGlobalCoordinates(vertex);

		coordinates[i] = new Array(
			coordinatesXY[0],
			coordinatesXY[1],
			(1 - lambda) * getMeshHeightAtCoordinates(vertex, zLeft) + lambda * getMeshHeightAtCoordinates(vertex, zRight)
		);
	}

	// Draw the arcs
	for (let source = 0; source < edgeWeights.length; ++source) {
		let sourceCoordinates = coordinates[source];
		let hSource = sourceCoordinates[2];

		for (let [target, weight] of edgeWeights[source]) {
			let targetCoordinates = coordinates[target];
			let hTarget = targetCoordinates[2];
			let hCenter = MIN_HEIGHT_HOVER_ARC + weight * (MAX_HEIGHT_HOVER_ARC - MIN_HEIGHT_HOVER_ARC);

			let pointsArray = [];

			for (let i = 0; i < HOVER_ARC_DIVISIONS + 1; ++i) {
				let lambda = i / HOVER_ARC_DIVISIONS;
				let p = [
					(1. - lambda) * sourceCoordinates[0] + lambda * targetCoordinates[0],
					(1. - lambda) * sourceCoordinates[1] + lambda * targetCoordinates[1]
				];
				let hp = (
					(2. * hSource - 4. * hCenter + 2. * hTarget) * lambda * lambda
					+ (-3. * hSource + 4. * hCenter - hTarget) * lambda
					+ hSource
				);

				pointsArray.push(p[1], hp, p[0]);
			}

			let edgesGeometry = new LineGeometry();
			edgesGeometry.setPositions(new Float32Array(pointsArray));
			let lineSegment = new Line2(edgesGeometry, new LineMaterial({
				color: getWeightedColor(1. - weight),
				linewidth: SIZE_HOVER_EDGE,
			}));

			scene.add(lineSegment);
			hoverArcEdgesDrawn.push(lineSegment);
		}
	}
}

function drawHoverArcsWithMultipleInterpolation(networkVertices, networkEdges, t, ts, zs) {
	if (ts == null) {
		return;
	}

	if (t <= ts[0]) {
		drawHoverArcsWithInterpolation(networkVertices, networkEdges[0], networkEdges[0], zs[0], zs[0], 0.);
		return;
	}

	if (t >= ts[ts.length - 1]) {
		drawHoverArcsWithInterpolation(networkVertices, networkEdges[ts.length - 1], networkEdges[ts.length - 1], zs[ts.length - 1], zs[ts.length - 1], 0.);
		return;
	}

	// TODO: Improve this from a linear search
	let index = 0;
	while (ts[index] < t) {
		++index;
	}

	if (ts[index] == t) {
		drawHoverArcsWithInterpolation(networkVertices, networkEdges[index], networkEdges[index], zs[index], zs[index], 0.);
		return index;
	}

	// If we reach here ts[index - 1] <= t < ts[index]
	let lambda = (t - ts[index - 1]) / (ts[index] - ts[index - 1]);
	drawHoverArcsWithInterpolation(networkVertices, networkEdges[index - 1], networkEdges[index], zs[index - 1], zs[index], lambda);
}

function drawHoverArcs() {
	if (times != null) {
		drawHoverArcsWithMultipleInterpolation(networkVertices, networkEdges, rangeAnimation.value, times, heights);
	}
}

let dropReader = new FileReader();
dropReader.onload = function() {
	try {
		let data = JSON.parse(dropReader.result);
		let animationData = data.animation;
		let mapData = data.map;

		networkVertices = data.nodes;
		times = new Array(animationData.length);
		heights = new Array(animationData.length);
		networkEdges = new Array(animationData.length);
		geodesics = new Array(animationData.length);
		edgeColors = new Array(animationData.length);
		for (let i = 0; i < animationData.length; ++i) {
			times[i] = animationData[i].time;
			heights[i] = animationData[i].height;
			networkEdges[i] = animationData[i].edges;
			geodesics[i] = animationData[i].geodesics;
			edgeColors[i] = animationData[i].edgeColors;
		}

		animationDuration = (animationData.length - 1) / 2.;
		heightsEWMA = new Array();
		if (animationData.length != 0) {
			let tRange = times[animationData.length - 1] - times[0]
			heightsEWMA[0] = heights[0];
			for (let i = 1; i < animationData.length; ++i) {
				let realtimePassed = animationDuration * (times[i] - times[i - 1]) / tRange;
				let alpha = 1 - Math.exp(-0.5 * realtimePassed);
				let heightEWMA = new Array();
				for (let j = 0; j < heightsEWMA[i - 1].length; ++j) {
					let heightEWMARow = new Array();
					for (let k = 0; k < heightsEWMA[i - 1][j].length; ++k) {
						heightEWMARow.push(alpha * heights[i][j][k] + (1 - alpha) * heightsEWMA[i - 1][j][k]);
					}
					heightEWMA.push(heightEWMARow);
				}
				heightsEWMA.push(heightEWMA);
			}
		}

		if (animationData.length != 0) {
			let nVertices = networkVertices.length;
			networkEdgesAll = new Array(nVertices);
			for (let i = 0; i < nVertices; ++i) {
				networkEdgesAll[i] = new Set();
			}
		}
		for (let i = 0; i < animationData.length; ++i) {
			for (let edge of networkEdges[i]) {
				networkEdgesAll[edge.source].add(edge.target);
			}
		}

		if (heights.length != 0 && heights[0].length != divisions) {
			scene.remove(plane);
			plane = makePlane(heights[0].length);
			scene.add(plane);

			divisions = heights[0].length;
		}

		rangeAnimation.min = animationData[0].time;
		rangeAnimation.max = animationData[animationData.length - 1].time;
		rangeAnimation.step = (rangeAnimation.max - rangeAnimation.min) / 100;
		rangeAnimation.value = rangeAnimation.min;

		buttonPlay.innerText = "Play";

		if (times.length <= 1) {
			divAnimationControls.style.display = "none";
		} else {
			divAnimationControls.style.display = "block";
		}

		if (checkboxShowHoverGraph.checked) {
			drawHoverGraph(0);
		}
		if (checkboxShowWeightArcs.checked) {
			drawHoverArcs();
		}

		setHeights();

		currentNetworkIndex = getCurrentNetworkIndex();

		// TODO: Clear out the list of outages

		mapCenter = mapData.center;
		mapZoomFactor = mapData.zoomFactor;
		updateOLMap(olMap, getCurrentResolution(), mapCenter, mapZoomFactor);
		textureMap = makeTextureOLMap(olMap);

		resetView();

		canvasNeedsUpdate = true;
	} catch (e) {
		console.error("Failed to read JSON");
		console.error(e);
	}
}

function dropOutput(evt) {
	evt.stopPropagation();
	evt.preventDefault();

	if (isPlaying) {
		isPlaying = false;
		timeInitial = null;
		dateTimeInitial = null;
	}

	dropReader.readAsText(evt.dataTransfer.files[0]);
}

function dragOver(evt) {
	evt.stopPropagation();
	evt.preventDefault();
	evt.dataTransfer.dropEffect = "copy"; // Explicitly show this is a copy.
}

function vertexToGlobalCoordinates(vertex) {
	let x = vertex[0] * planeSideLength + planeXMin;
	let y = vertex[1] * planeSideLength + planeYMin;
	return [x, y];
}
