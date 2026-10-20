import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

import { getCurvatureColor } from "../colors.js";
import { vertexToGlobalCoordinates } from "./utils.js";

const HEIGHT_HOVER_GRAPH = 10;
const SIZE_HOVER_VERTEX = 6;
const SIZE_HOVER_EDGE = 4;

const HOVER_VERTEX_MATERIAL = new THREE.PointsMaterial({
	color: "#4caf50",
	size: SIZE_HOVER_VERTEX,
});

/**
 * Remove a drawn hovering graph from the scene and free its resources.
 *
 * @param {THREE.Scene} scene
 * @param {Array<THREE.Object3D>} objects the objects drawn; emptied in place
 */
export function clearHoveringGraph(scene, objects) {
	for (let object of objects) {
		scene.remove(object);
		object.geometry.dispose();
		if (object.material !== HOVER_VERTEX_MATERIAL) {
			object.material.dispose();
		}
	}
	objects.length = 0;
}

/**
 * Draw a flat copy of the network graph hovering above the plane, replacing
 * the previously drawn one.
 *
 * @param {THREE.Scene} scene
 * @param {Array<THREE.Object3D>} objects the objects drawn; replaced in place
 * @param {Array<{coordinates: Array<number>}>} networkVertices
 * @param {Array<{source: number, target: number, curvature: number}>} edges
 * @param {number} planeSideLength
 */
export function drawHoveringGraph(scene, objects, networkVertices, edges, planeSideLength) {
	clearHoveringGraph(scene, objects);

	let vertexArray = [];
	for (let vertex of networkVertices) {
		let vertexGlobal = vertexToGlobalCoordinates(vertex.coordinates, planeSideLength);
		vertexArray.push(vertexGlobal[1], HEIGHT_HOVER_GRAPH, vertexGlobal[0]);
	}
	let pointsGeometry = new THREE.BufferGeometry();
	pointsGeometry.setAttribute("position", new THREE.BufferAttribute(
		new Float32Array(vertexArray), 3
	));
	let points = new THREE.Points(pointsGeometry, HOVER_VERTEX_MATERIAL);
	scene.add(points);
	objects.push(points);

	for (let edge of edges) {
		let edgeArray = [];
		let source = vertexToGlobalCoordinates(networkVertices[edge.source].coordinates, planeSideLength);
		let target = vertexToGlobalCoordinates(networkVertices[edge.target].coordinates, planeSideLength);
		edgeArray.push(source[1], HEIGHT_HOVER_GRAPH, source[0]);
		edgeArray.push(target[1], HEIGHT_HOVER_GRAPH, target[0]);

		let edgesGeometry = new LineSegmentsGeometry();
		edgesGeometry.setPositions(new Float32Array(edgeArray));
		let lineSegment = new LineSegments2(edgesGeometry, new LineMaterial({
			color: getCurvatureColor(edge.curvature),
			linewidth: SIZE_HOVER_EDGE,
		}));

		scene.add(lineSegment);
		objects.push(lineSegment);
	}
}
