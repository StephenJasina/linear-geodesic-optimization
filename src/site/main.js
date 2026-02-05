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
import { createOLMap, destroyOLMap, updateOLMap, getCanvasOLMap, makeTextureOLMap } from "./modules/layers/map.js";
import { default as drawEdges } from "./modules/layers/edges.js";
import { default as drawGrid } from "./modules/layers/grid.js";
import { default as drawGeodesics } from "./modules/layers/geodesics.js";
import { resetPlaneHeights, setPlaneHeightsWithInterpolation } from "./modules/layers/heights.js";
import { default as drawHeightChanges } from "./modules/layers/height_changes.js";
import { drawOutages } from "./modules/layers/outages.js";
import { default as drawVertices } from "./modules/layers/vertices.js";

// Colors
const COLOR_BACKGROUND = "#f3f3f3";
const COLOR_PLANE = "#ffffff";

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

let spacingBetweenPlanes = 0.45;

// Number of points per side
let divisions = 5;

// Globals tracking the manifold shape across time
let times = null;
let heights = null;
let heightsEWMA = null;
let networkVertices = null;
let networkEdges = null;
let networkEdgesAll = null;
let networkEdgesAdded = null;
let networkEdgesRemoved = null;
let networkBoundaries = null;
let geodesics = null;
let edgeColors = null;
let traffic = null;
let trafficMax = 0.;
let trafficPaths = null;

// Globals tracking the animation state
let isPlaying = false;
let timeInitial = null;
let dateTimeInitial = null;
let animationDuration = null;
let currentNetworkIndex = null;
let previousNetworkIndex = null;

// Scene setup
let scene = new THREE.Scene();
scene.background = new THREE.Color(COLOR_BACKGROUND);

// Camera setup
let frustumScale = 2.5 * planeSideLength / Math.max(window.innerWidth, window.innerHeight);
let camera = new THREE.OrthographicCamera(
	-window.innerWidth / 2. * frustumScale, window.innerWidth / 2. * frustumScale,
	window.innerHeight / 2. * frustumScale, -window.innerHeight / 2. * frustumScale,
	1, 1000
);
camera.position.x = -15.;  // North
camera.position.y = 15.;   // Up
camera.position.z = 20;    // East

// Renderer setup
let canvasMain = document.createElement("canvas");
canvasMain.style.zIndex = 50;
let renderer = new THREE.WebGLRenderer({
	canvas: canvasMain,
	preserveDrawingBuffer: true
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

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
	color: "#4caf50",
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
	color: "#4caf50",
	size: SIZE_HOVER_VERTEX,
});

let hoverArcVerticesDrawn = [];
let hoverArcEdgesDrawn = [];

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
// controls.target = new THREE.Vector3(0., 0., camera.position.z);
// controls.update();
window.addEventListener("wheel", wheelEvent, true);

// Interactivity with DOM elements

// let divDropOutput = document.getElementById("div-drop-output");
let divDropOutput = document.body;
divDropOutput.addEventListener("dragover", dragOver, false);
divDropOutput.addEventListener("drop", dropOutput, false);

let rangeAnimation = document.getElementById("range-animation");
let checkboxShowHeights = document.getElementById("show-heights");
function setHeights() {
	for (let i = 0; i < elementsByTab.length; ++i) {
		let options = elementsByTab[i];
		if (heights != null && options.showHeights) {
			setPlaneHeightsWithInterpolation(options.plane, heights, rangeAnimation.value, times);
		} else {
			resetPlaneHeights(options.plane);
		}
	}
}
rangeAnimation.oninput = function() {
	timeInitial = parseFloat(rangeAnimation.value);
	dateTimeInitial = Date.now() / 1000;
	previousNetworkIndex = currentNetworkIndex;
	currentNetworkIndex = getCurrentNetworkIndex();
	setHeights();

	for (let i = 0; i < elementsByTab.length; ++i) {
		elementsByTab[i].canvasNeedsUpdate = true;
	}

	updateTrafficTable(currentNetworkIndex);
}
checkboxShowHeights.onchange = function() {
	elementsByTab[indexTabCurrent].showHeights = checkboxShowHeights.checked;
	setHeights();
}

let checkboxShowHeightChanges = document.getElementById("show-height-changes");
checkboxShowHeightChanges.onchange = function() {
	elementsByTab[indexTabCurrent].showHeightChanges = checkboxShowHeightChanges.checked;
	elementsByTab[indexTabCurrent].canvasNeedsUpdate = true;
}

let checkboxShowMap = document.getElementById("show-map");
checkboxShowMap.onchange = function() {
	if (checkboxShowMap.checked) {
		updateOLMap(elementsByTab[indexTabCurrent].olMap, getCurrentResolution(), mapCenter, mapZoomFactor);
	} else {
		setCanvasZoom(elementsByTab[indexTabCurrent], getCurrentResolution());
	}
	elementsByTab[indexTabCurrent].showMap = checkboxShowMap.checked;
	elementsByTab[indexTabCurrent].canvasNeedsUpdate = true;
}

let checkboxShowGraph = document.getElementById("show-graph");
checkboxShowGraph.onchange = function() {
	elementsByTab[indexTabCurrent].showGraph = checkboxShowGraph.checked;
	elementsByTab[indexTabCurrent].canvasNeedsUpdate = true;
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
	// TODO: show the div if _any_ of the tabs have this selected
	if (checkboxShowOutages.checked) {
		divOutages.style.display = "block";
	} else {
		divOutages.style.display = "none";
	}

	elementsByTab[indexTabCurrent].showOutages = checkboxShowOutages.checked;
	elementsByTab[indexTabCurrent].canvasNeedsUpdate = true;
}

let checkboxShowHoveringGraph = document.getElementById("show-hover-graph");
checkboxShowHoveringGraph.onchange = function() {
	if (checkboxShowHoveringGraph.checked) {
		if (times != null) {
			drawHoveringGraph(Math.round(currentNetworkIndex));
		}
	} else {
		clearHoveringGraph();
	}
	elementsByTab[indexTabCurrent].showHoveringGraph = checkboxShowHoveringGraph.checked;
}

let checkboxShowWeightArcs = document.getElementById("show-weight-arcs");
checkboxShowWeightArcs.onchange = function() {
	if (checkboxShowWeightArcs.checked) {
		drawHoverArcs();
	} else {
		clearHoverArcs();
	}
	elementsByTab[indexTabCurrent].showWeightArcs = checkboxShowWeightArcs.checked;
}

let checkboxShowGeodesics = document.getElementById("show-geodesics");
checkboxShowGeodesics.onchange = function() {
	elementsByTab[indexTabCurrent].showGeodesics = checkboxShowGeodesics.checked;
	elementsByTab[indexTabCurrent].canvasNeedsUpdate = true;
}

let checkboxShowGrid = document.getElementById("show-grid");
checkboxShowGrid.onchange = function() {
	elementsByTab[indexTabCurrent].showGrid = checkboxShowGrid.checked;
	elementsByTab[indexTabCurrent].canvasNeedsUpdate = true;
}

let buttonHelp = document.getElementById("button-help");
buttonHelp.onclick = helpClick;

let divAnimationControls = document.getElementById("div-animation-controls");
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

let divTabs = document.getElementById("div-tabs");
let elementsByTab = [];
let indexTabCurrent = 0;
function swapToTab(button) {
	// TODO: Make this safer against spamming click
	let indexNew = Number(button.target.id.substring(11));

	// Reset the style of the previous button
	let buttonPrevious = elementsByTab[indexTabCurrent].buttonTab;
	let indexPrevious = Number(buttonPrevious.id.substring(11));
	while (buttonPrevious.hasChildNodes()) {
		buttonPrevious.removeChild(buttonPrevious.lastChild);
	}
	buttonPrevious.appendChild(document.createTextNode(String(indexPrevious + 1)));

	// Determine the current tab index
	for (let indexTab = 0; indexTab < elementsByTab.length; ++indexTab) {
		if (button.target.id == elementsByTab[indexTab].buttonTab.id) {
			indexTabCurrent = indexTab;
			break;
		}
	}

	// Update the style of the next button
	let buttonNew = elementsByTab[indexTabCurrent].buttonTab;
	while (buttonNew.hasChildNodes()) {
		buttonNew.removeChild(buttonNew.lastChild);
	}
	let bNode = document.createElement("b");
	let emNode = document.createElement("em");
	bNode.appendChild(document.createTextNode(String(indexNew + 1)));
	emNode.appendChild(bNode);
	buttonNew.appendChild(emNode);

	// Load the state of the checkboxes
	// TODO: Is this the best way to do this, or should separate copies
	// of each DOM element be stored per-tab?
	checkboxShowHeights.checked = elementsByTab[indexTabCurrent].showHeights;
	checkboxShowHeights.dispatchEvent(new Event("change"));
	checkboxShowHeightChanges.checked = elementsByTab[indexTabCurrent].showHeightChanges;
	checkboxShowHeightChanges.dispatchEvent(new Event("change"));
	checkboxShowMap.checked = elementsByTab[indexTabCurrent].showMap;
	checkboxShowMap.dispatchEvent(new Event("change"));
	checkboxShowGraph.checked = elementsByTab[indexTabCurrent].showGraph;
	checkboxShowGraph.dispatchEvent(new Event("change"));
	checkboxShowOutages.checked = elementsByTab[indexTabCurrent].showOutages;
	checkboxShowOutages.dispatchEvent(new Event("change"));
	checkboxShowHoveringGraph.checked = elementsByTab[indexTabCurrent].showHoveringGraph;
	checkboxShowHoveringGraph.dispatchEvent(new Event("change"));
	checkboxShowWeightArcs.checked = elementsByTab[indexTabCurrent].showWeightArcs;
	checkboxShowWeightArcs.dispatchEvent(new Event("change"));
	checkboxShowGeodesics.checked = elementsByTab[indexTabCurrent].showGeodesics;
	checkboxShowGeodesics.dispatchEvent(new Event("change"));
	checkboxShowGrid.checked = elementsByTab[indexTabCurrent].showGrid;
	checkboxShowGrid.dispatchEvent(new Event("change"));
}
let buttonAddTab = document.getElementById("button-add-tab");
let buttonRemoveTab = document.getElementById("button-remove-tab");
function addTab() {
	buttonRemoveTab.style.display = "inline-block";

	let indexNew = 0;
	if (elementsByTab.length != 0) {
		indexNew = Number(elementsByTab[elementsByTab.length - 1].buttonTab.id.substring(11)) + 1;
	}
	let buttonNew = document.createElement("button");
	buttonNew.type = "button";
	buttonNew.id = "button-tab-" + String(indexNew);
	buttonNew.classList.add("button-tab");
	buttonNew.appendChild(document.createTextNode(String(indexNew + 1)));
	buttonNew.onclick = swapToTab;
	divTabs.insertBefore(buttonNew, buttonAddTab);

	let plane = makePlane(divisions);
	scene.add(plane);

	let contextBlank = document.createElement("canvas").getContext("2d");
	contextBlank.canvas.width = canvasResolution;
	contextBlank.canvas.height = canvasResolution;
	contextBlank.fillStyle = COLOR_PLANE;
	contextBlank.fillRect(0, 0, contextBlank.canvas.width, contextBlank.canvas.height);
	let textureBlank = makeTextureBlank(contextBlank);
	plane.material.map = textureBlank;
	// plane.material.needsUpdate = true;

	let olMap = createOLMap(canvasResolution, mapCenter, mapZoomFactor);

	let options = {
		"plane": plane,
		"contextBlank": contextBlank,
		"textureBlank": textureBlank,
		"contextMap": null,
		"textureMap": null,
		"olMap": olMap,

		"buttonTab": buttonNew,

		"showHeights": true,
		"showHeightChanges": true,
		"showMap": false,
		"showGraph": true,
		"showOutages": false,
		"showHoveringGraph": false,
		"showWeightArcs": false,
		"showGeodesics": false,
		"showGrid": false,

		"canvasNeedsUpdate": true,
	};
	elementsByTab.push(options);

	olMap.once("postrender", function(event) {
		options.contextMap = getCanvasOLMap(olMap).getContext("2d");
		options.textureMap = makeTextureOLMap(olMap);
	});
	// Update the map texture whenever the map changes
	olMap.on("postrender", function(event) {
		options.canvasNeedsUpdate = true;
	});

	setPlaneCenters(true);

	buttonNew.dispatchEvent(new Event("click"));
}
addTab();
buttonAddTab.onclick = addTab;
function removeTab() {
	elementsByTab[indexTabCurrent].buttonTab.remove();

	scene.remove(elementsByTab[indexTabCurrent].plane);
	elementsByTab[indexTabCurrent].textureBlank.dispose();
	elementsByTab[indexTabCurrent].textureMap.dispose();
	destroyOLMap(elementsByTab[indexTabCurrent].olMap);

	elementsByTab.splice(indexTabCurrent, 1);

	if (elementsByTab.length == 0) {
		addTab();
	} else {
		if (indexTabCurrent >= elementsByTab.length) {
			indexTabCurrent = elementsByTab.length - 1;
		}
		elementsByTab[indexTabCurrent].buttonTab.dispatchEvent(new Event("click"));
		setPlaneCenters(true);
	}
}
buttonRemoveTab.onclick = removeTab;

let divTraffic = document.getElementById("div-traffic");
let tableTraffic = document.getElementById("table-traffic");
let tableTrafficCells = new Array(0);
let tableTrafficHover = new Array(0);
function updateTrafficTable(currentNetworkIndex) {
	let indexPrevious = Math.floor(currentNetworkIndex);
	let indexNext = Math.ceil(currentNetworkIndex);
	let alpha = currentNetworkIndex - indexPrevious;

	for (let i = 0; i < networkVertices.length; ++i) {
		for (let j = 0; j < networkVertices.length; ++j) {
			let cell = tableTrafficCells[i][j];
			let trafficAtCell = alpha * traffic[indexNext][i][j] + (1 - alpha) * traffic[indexPrevious][i][j];
			if (trafficMax != 0.) {
				trafficAtCell /= trafficMax;
			}
			let color = getWeightedColor(trafficAtCell);
			cell.style.backgroundColor = color;
		}
	}
}

function getPlaneScaleFactor(n) {
	return 1 / (n + (n - 1) * spacingBetweenPlanes);
}
function setPlaneCenters(setScales = false) {
	let n = elementsByTab.length;
	let scaleFactor = getPlaneScaleFactor(n);
	for (let i = 0; i < n; ++i) {
		// Location of the center from left to right
		let r = (- 1 / 2 + scaleFactor / 2 + i * (1 + spacingBetweenPlanes) * scaleFactor) * planeSideLength;
		// Current direction of looking
		let d = (new THREE.Vector3()).subVectors(controls.target, camera.position);
		let theta = Math.atan2(d.x, d.z);

		elementsByTab[i].plane.position.set(-r * Math.cos(theta), 0, r * Math.sin(theta));

		if (setScales) {
			elementsByTab[i].plane.scale.set(scaleFactor, scaleFactor, scaleFactor);
		}
	}
	setCanvasResolutions();
}

controls.addEventListener("change", function() {
	setPlaneCenters();
});

/**
 * Draw every layer above the base layer (i.e., excluding the map)
 */
function drawLayers(time, options) {
	let plane = options.plane;

	// Figure out which canvas we're using
	let context = null;
	if (options.showMap && options.contextMap != null) {
		context = options.contextMap;
	} else {
		context = options.contextBlank;
	}

	// Draw grid lines
	if (options.showGrid) {
		drawGrid(context, divisions);
	}

	// Draw the graph, geodesics, and outages onto the canvas
	if (
		networkEdges != null && networkEdges.length > 0
		&& (options.showGraph || options.showGeodesics || options.showOutages)
	) {
		// First, figure out the current timing and indexing information
		let currentNetworkIndexInt = Math.round(currentNetworkIndex);

		// Draw the edges
		if (options.showGraph) {
			drawEdges(context, networkVertices, networkEdges[currentNetworkIndexInt], 3);
		}

		// Draw the geodesics
		if (options.showGeodesics) {
			drawGeodesics(
				context, geodesics[currentNetworkIndexInt],
				3, edgeColors[currentNetworkIndexInt]
			);
		}

		let trafficPathsToDraw = new Array();
		let trafficPathsToDrawColors = new Array();
		for (let i = 0; i < networkVertices.length; ++i) {
			for (let j = 0; j < networkVertices.length; ++j) {
				if (tableTrafficHover[i][j] && trafficPaths[currentNetworkIndexInt][i] !== null && trafficPaths[currentNetworkIndexInt][i][j] !== null) {
					trafficPathsToDraw.push(trafficPaths[currentNetworkIndexInt][i][j]);
					trafficPathsToDrawColors.push(new Array(0, 0, 0));
				}
			}
		}
		drawGeodesics(
			context, trafficPathsToDraw,
			6, trafficPathsToDrawColors
		);

		// Deal with outages
		if (options.showOutages && times.length > 1) {
			drawOutages(
				context, networkVertices,
				networkEdgesAdded[Math.floor(currentNetworkIndex)],
				networkEdgesRemoved[Math.floor(currentNetworkIndex)],
				networkEdges[Math.floor(currentNetworkIndex)],
				networkEdgesAll,
				currentNetworkIndex % 1.,
				currentNetworkIndex != previousNetworkIndex ? ulOutages : null
			);
		}
	}

	// Draw the vertices
	if (networkVertices != null) {
		drawVertices(context, networkVertices, 2);
	}

	if (options.showHeightChanges) {
		drawHeightChanges(context, heights, heightsEWMA, time, times, networkBoundaries, .008);
	}
}

function animate() {
	// Update the slider if necessary
	if (isPlaying) {
		let timeMin = parseFloat(rangeAnimation.min);
		let timeMax = parseFloat(rangeAnimation.max);
		let timeRange = timeMax - timeMin;

		let dateTimeNow = Date.now() / 1000;
		let alpha = ((dateTimeNow - dateTimeInitial) % animationDuration) / animationDuration;
		let timeNow = (timeInitial - timeMin + alpha * timeRange) % timeRange + timeMin;
		rangeAnimation.value = timeNow;

		// Update the heights if necessary
		for (let i = 0; i < elementsByTab.length; ++i) {
			if (elementsByTab[i].showHeights) {
				setPlaneHeightsWithInterpolation(elementsByTab[i].plane, heights, timeNow, times);
			}
		}

		previousNetworkIndex = currentNetworkIndex;
		currentNetworkIndex = getCurrentNetworkIndex();

		for (let i = 0; i < elementsByTab.length; ++i) {
			elementsByTab[i].canvasNeedsUpdate = true;
		}

		updateTrafficTable(currentNetworkIndex);
	}

	// Update the canvases if needed
	for (let i = 0; i < elementsByTab.length; ++i) {
		let options = elementsByTab[i];
		let plane = options.plane;
		if (options.canvasNeedsUpdate) {
			options.canvasNeedsUpdate = false;

			// Figure out which canvas we're using
			let context = null;
			let texture = null;
			if (options.showMap) {
				context = options.contextMap;
				texture = options.textureMap;
			} else {
				context = options.contextBlank;
				texture = options.textureBlank;
			}
			plane.material.map = texture;

			// Draw the bottommost layer
			if (options.showMap) {
				options.olMap.renderSync();
				options.canvasNeedsUpdate = false;  // Hack to avoid endlessly rendering
			} else {
				context.fillStyle = COLOR_PLANE;
				context.fillRect(0, 0, context.canvas.width, context.canvas.height);
			}

			drawLayers(
				(times === null || times.length == 0) ? null : (times.length == 1 ? times[0] : rangeAnimation.value),
				options
			);

			texture.needsUpdate = true;
		}
		plane.geometry.computeVertexNormals();
		plane.material.needsUpdate = true;
	}

	// Render
	composer.render();
}
renderer.setAnimationLoop(animate);

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
	if (times === null) {
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
	return index - 1 + lambda;
}

function resetView() {
	// Reset the panning
	controls.reset();

	// Overhead view
	camera.position.set(-Number.EPSILON, 15., 0.);
	camera.zoom = 1.4;

	// Front view
	// camera.position.set(-15., 10., 0.);
	// camera.zoom = 1.7;

	// Diagonal View
	// camera.position.set(-15., 10., -5.);
	// camera.zoom = 1.6;

	// Left view
	// camera.position.set(0., 10., -15.);
	// camera.zoom = 1.7;

	controls.update();
	camera.updateProjectionMatrix();

	for (let i = 0; i < elementsByTab.length; ++i) {
		let options = elementsByTab[i];
		if (options.showMap) {
			updateOLMap(options.olMap, getCurrentResolution(), mapCenter, mapZoomFactor);
			options.textureMap = makeTextureOLMap(options.olMap);
		} else {
			setCanvasZoom(options, getCurrentResolution());
		}
	}

	for (let i = 0; i < elementsByTab.length; ++i) {
		elementsByTab[i].canvasNeedsUpdate = true;
	}
}

document.addEventListener("keydown", function(event) {
	if (event.key == "Escape") {
		resetView();
	} else if (event.key == "h") {
		let elementIDs = ["div-gui", "button-help", "div-traffic"];
		let displayStyles = ["block", "block", "grid"];
		for (let i = 0; i < elementIDs.length; ++i) {
			let element = document.getElementById(elementIDs[i]);
			let displayStyle = displayStyles[i];
			if (element.style.display != "none") {
				element.style.display = "none";
			} else {
				element.style.display = displayStyle;
			}
		}

		document.getElementById("div-help").style.display = "none";
	} else if (event.key == "s") {
		let anchor = document.createElement("a");
		anchor.href = canvasMain.toDataURL();
		anchor.download = "manifold.png";
		anchor.click();
	} else if (event.key == "g") {
		buttonPlay.click();
	} else if (event.key == "-") {
		removeTab();
	} else if (event.key == "=" || event.key == "+") {
		addTab();
	} else if (event.key == "ArrowLeft") {
		if (indexTabCurrent > 0) {
			elementsByTab[indexTabCurrent - 1].buttonTab.dispatchEvent(new Event("click"));
		}
	} else if (event.key == "ArrowRight") {
		if (indexTabCurrent < elementsByTab.length - 1) {
			elementsByTab[indexTabCurrent + 1].buttonTab.dispatchEvent(new Event("click"));
		}
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

function makeTextureBlank(contextBlank) {
	let texture = new THREE.CanvasTexture(contextBlank.canvas);
	texture.minFilter = THREE.LinearFilter;
	texture.center = new THREE.Vector2(0.5, 0.5);
	texture.rotation = -Math.PI / 2;
	return texture;
}

function getCurrentResolution() {
	let planeScaleFactor = getPlaneScaleFactor(elementsByTab.length);
	return canvasResolution * Math.min(camera.zoom * planeScaleFactor, 4.);
}

function setCanvasZoom(options, resolution) {
	options.textureBlank.dispose();
	options.contextBlank.canvas.width = resolution;
	options.contextBlank.canvas.height = resolution;
	options.textureBlank = makeTextureBlank(options.contextBlank);
	options.plane.material.map = options.textureBlank;
}

function wheelEvent(event) {
	if (document.elementFromPoint(event.clientX, event.clientY).tagName != "CANVAS") {
		return;
	}

	setCanvasResolutions();
}
function setCanvasResolutions() {
	for (let i = 0; i < elementsByTab.length; ++i) {
		let options = elementsByTab[i];
		if (options.showMap) {
			updateOLMap(options.olMap, getCurrentResolution(), mapCenter, mapZoomFactor);
			options.textureMap = makeTextureOLMap(options.olMap);
		} else {
			setCanvasZoom(options, getCurrentResolution());
		}
	}

	for (let i = 0; i < elementsByTab.length; ++i) {
		elementsByTab[i].canvasNeedsUpdate = true;
	}
}

function makePlane(divisions) {
	let geometry = new THREE.PlaneGeometry(
		planeSideLength, planeSideLength,
		divisions - 1, divisions - 1
	);
	let material = new THREE.MeshStandardMaterial({
		side: THREE.DoubleSide,
		transparent: true,
		opacity: 1.0,
	});
	let plane = new THREE.Mesh(geometry, material);
	plane.rotation.set(-Math.PI / 2, 0, 0);
	return plane;
}

function clearHoveringGraph() {
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

function drawHoveringGraph(index) {
	clearHoveringGraph();

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
	if (ts === null) {
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
		networkBoundaries = new Array(animationData.length);
		geodesics = new Array(animationData.length);
		edgeColors = new Array(animationData.length);
		traffic = new Array(animationData.length);
		trafficPaths = new Array(animationData.length);
		for (let i = 0; i < animationData.length; ++i) {
			times[i] = animationData[i].time;
			heights[i] = animationData[i].height;
			networkEdges[i] = animationData[i].edges;
			networkBoundaries[i] = animationData[i].boundary;
			geodesics[i] = animationData[i].geodesics;
			edgeColors[i] = animationData[i].edgeColors;
			traffic[i] = animationData[i].traffic;
			trafficPaths[i] = animationData[i].trafficPaths;
		}

		animationDuration = (animationData.length - 1) * 2.;
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
			networkEdgesAll = new Array(networkVertices.length);
			for (let i = 0; i < networkVertices.length; ++i) {
				networkEdgesAll[i] = new Set();
			}
		}
		for (let i = 0; i < animationData.length; ++i) {
			for (let edge of networkEdges[i]) {
				networkEdgesAll[edge.source].add(edge.target);
			}
		}

		if (animationData.length > 1) {
			// Make sets of edges for each network graph
			let networkEdgesSets = new Array(animationData.length);
			for (let i = 0; i < animationData.length; ++i) {
				networkEdgesSets[i] = new Array(networkVertices.length);
				for (let j = 0; j < networkVertices.length; ++j) {
					networkEdgesSets[i][j] = new Set();
				}
				for (let edge of networkEdges[i]) {
					networkEdgesSets[i][edge.source].add(edge.target);
				}
			}

			// Figure out the diff between consecutive pairs of network graphs
			networkEdgesAdded = new Array(animationData.length - 1);
			networkEdgesRemoved = new Array(animationData.length - 1);
			for (let i = 0; i < animationData.length - 1; ++i) {
				networkEdgesAdded[i] = new Array(networkVertices.length);
				networkEdgesRemoved[i] = new Array(networkVertices.length);
				for (let j = 0; j < networkVertices.length; ++j) {
					networkEdgesAdded[i][j] = new Set();
					networkEdgesRemoved[i][j] = new Set();
				}
				for (let edge of networkEdges[i + 1]) {
					if (!networkEdgesSets[i][edge.source].has(edge.target)) {
						networkEdgesAdded[i][edge.source].add(edge.target);
					}
				}
				for (let edge of networkEdges[i]) {
					if (!networkEdgesSets[i + 1][edge.source].has(edge.target)) {
						networkEdgesRemoved[i][edge.source].add(edge.target);
					}
				}
			}
		}

		if (heights.length != 0 && heights[0].length != divisions) {
			for (let i = 0; i < elementsByTab.length; ++i) {
				scene.remove(elementsByTab[i].plane);
				elementsByTab[i].plane = makePlane(heights[0].length);
				scene.add(elementsByTab[i].plane);
			}
			setPlaneCenters(true);

			divisions = heights[0].length;
		}

		rangeAnimation.min = animationData[0].time;
		rangeAnimation.max = animationData[animationData.length - 1].time;
		rangeAnimation.step = (rangeAnimation.max - rangeAnimation.min) / (Math.max(animationData.length * 10, 100));
		rangeAnimation.value = rangeAnimation.min;

		buttonPlay.innerText = "Play";

		if (times.length <= 1) {
			divAnimationControls.style.display = "none";
		} else {
			divAnimationControls.style.display = "block";
		}

		if (checkboxShowHoveringGraph.checked) {
			drawHoveringGraph(0);
		}
		if (checkboxShowWeightArcs.checked) {
			drawHoverArcs();
		}

		setHeights();

		currentNetworkIndex = getCurrentNetworkIndex();

		// TODO: Clear out the list of outages

		tableTrafficHover = new Array(networkVertices.length);
		for (let i = 0; i < networkVertices.length; ++i) {
			let tableTrafficHoverRow = new Array(networkVertices.length);
			for (let j = 0; j < networkVertices.length; ++j) {
				tableTrafficHoverRow[j] = false;
			}
			tableTrafficHover[i] = tableTrafficHoverRow;
		}
		while (tableTraffic.hasChildNodes()) {
			tableTraffic.removeChild(tableTraffic.lastChild);
		}
		tableTrafficCells = new Array(networkVertices.length);
		for (let i = 0; i < networkVertices.length; ++i) {
			let tableRow = document.createElement("tr");
			let tableTrafficCellsRow = new Array(networkVertices.length);
			for (let j = 0; j < networkVertices.length; ++j) {
				let cell = document.createElement("td");
				cell.addEventListener("mouseenter", function(event) {
					tableTrafficHover[i][j] = true;
					for (let k = 0; k < elementsByTab.length; ++k) {
						elementsByTab[k].canvasNeedsUpdate = true;
					}
				});
				cell.addEventListener("mouseleave", function(event) {
					tableTrafficHover[i][j] = false;
					for (let k = 0; k < elementsByTab.length; ++k) {
						elementsByTab[k].canvasNeedsUpdate = true;
					}
				});
				tableRow.appendChild(cell);
				tableTrafficCellsRow[j] = cell;
			}
			tableTraffic.appendChild(tableRow);
			tableTrafficCells[i] = tableTrafficCellsRow;
		}

		trafficMax = 0.;
		for (let trafficMatrix of traffic) {
			for (let trafficMatrixRow of trafficMatrix) {
				for (let trafficMatrixCell of trafficMatrixRow) {
					if (trafficMatrixCell > trafficMax) {
						trafficMax = trafficMatrixCell;
					}
				}
			}
		}
		updateTrafficTable(currentNetworkIndex)

		mapCenter = mapData.center;
		mapZoomFactor = mapData.zoomFactor;
		for (let i = 0; i < elementsByTab.length; ++i) {
			let options = elementsByTab[i];
			if (options.showMap) {
				updateOLMap(options.olMap, getCurrentResolution(), mapCenter, mapZoomFactor);
				options.textureMap = makeTextureOLMap(options.olMap);
			}
		}

		for (let i = 0; i < elementsByTab.length; ++i) {
			elementsByTab[i].canvasNeedsUpdate = true;
		}
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
