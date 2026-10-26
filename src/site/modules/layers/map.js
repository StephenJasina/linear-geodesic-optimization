// OpenLayers
import Tile from "ol/layer/Tile.js";
import Map from "ol/Map.js";
import { fromLonLat } from "ol/proj.js";
import { OSM } from "ol/source.js";
import View from "ol/View.js";

// Three.js
import * as THREE from "three";

const CIRCUMFERENCE_EARTH = 40075016.68557849;

/**
 * Return a View object for a given set of map descriptors.
 *
 * @param {number} resolution
 * @param {Array<number>} mapCenter longitude and latitude
 * @param {number} mapZoomFactor
 */
function getView(resolution, mapCenter, mapZoomFactor) {
	return new View({
		projection: 'EPSG:3857',
		center: fromLonLat(mapCenter),
		resolution: CIRCUMFERENCE_EARTH / resolution / mapZoomFactor
	});
}

/**
 * Return a new OpenLayers map, created as a child of the DOM's body.
 *
 * @param {number} resolution
 * @param {Array<number>} mapCenter longitude and latitude
 * @param {number} mapZoomFactor
 */
function createOLMap(resolution, mapCenter, mapZoomFactor) {
	let divMap = document.createElement("div");
	divMap.id = "map";
	divMap.className = "map-div";
	divMap.style.width = resolution + "px";
	divMap.style.height = resolution + "px";
	// Hide and draw on the bottom
	// TODO: Is this needed?
	divMap.style.visibility = "hidden";
	divMap.style.zIndex = 0;
	document.body.appendChild(divMap);

	let layerTile = new Tile({
		source: new OSM(),
	});

	let olMap = new Map({
		controls: [],
		// Match the blank canvas, so that layers drawn with widths in pixels
		// look the same with and without the map
		pixelRatio: 1,
		interactions: [],
		layers: [layerTile],
		target: divMap,
		view: getView(resolution, mapCenter, mapZoomFactor),
	});

	return olMap;
}

/**
 * @param {Map} olMap
 */
function destroyOLMap(olMap) {
	let divMap = olMap.getTargetElement();
	olMap.setTarget(null);
	olMap.dispose();
	divMap.remove();
}

/**
 * Update an existing OpenLayers map.
 *
 * @param {Map} olMap
 * @param {number} resolution
 * @param {Array<number>} mapCenter longitude and latitude
 * @param {number} mapZoomFactor
 */
function updateOLMap(olMap, resolution, mapCenter, mapZoomFactor) {
	let divMap = olMap.getTargetElement();
	divMap.style.width = resolution + "px";
	divMap.style.height = resolution + "px";
	olMap.updateSize();
	let view = olMap.getView();
	view.setCenter(fromLonLat(mapCenter));
	view.setResolution(CIRCUMFERENCE_EARTH / resolution / mapZoomFactor);
}

/**
 * @param {Map} olMap
 */
function getCanvasOLMap(olMap) {
	return olMap.getTargetElement().getElementsByTagName("canvas")[0];
}

/**
 * @param {Map} olMap
 */
function makeTextureOLMap(olMap) {
	let canvasMap = getCanvasOLMap(olMap);
	let textureMap = new THREE.CanvasTexture(canvasMap);
	textureMap.colorSpace = THREE.SRGBColorSpace;
	textureMap.minFilter = THREE.LinearFilter;
	textureMap.magFilter = THREE.NearestFilter;
	textureMap.center = new THREE.Vector2(0.5, 0.5);
	textureMap.rotation = -Math.PI / 2;
	return textureMap;
}

export { createOLMap, destroyOLMap, updateOLMap, getCanvasOLMap, makeTextureOLMap };
