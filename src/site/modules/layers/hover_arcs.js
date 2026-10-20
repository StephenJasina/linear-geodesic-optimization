import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

import { getWeightedColor } from "../colors.js";
import { locateTime } from "../time.js";
import { vertexToGlobalCoordinates } from "./utils.js";

const MIN_HEIGHT_HOVER_ARC = 5;
const MAX_HEIGHT_HOVER_ARC = 8;
const SIZE_HOVER_ARC_EDGE = 4;
const HOVER_ARC_DIVISIONS = 31;

/**
 * Remove drawn arcs from the scene and free their resources.
 *
 * @param {THREE.Scene} scene
 * @param {Array<THREE.Object3D>} objects the objects drawn; emptied in place
 */
export function clearHoverArcs(scene, objects) {
	for (let object of objects) {
		scene.remove(object);
		object.material.dispose();
		object.geometry.dispose();
	}
	objects.length = 0;
}

/**
 * Bilinearly interpolate the mesh height at a vertex in [0, 1]^2.
 *
 * @param {Array<number>} vertex
 * @param {Array<Array<number>>} zs
 * @param {number} planeSideLength
 */
function getMeshHeightAtCoordinates(vertex, zs, planeSideLength) {
	let divisions = zs.length;

	let i, alpha;
	if (vertex[0] === 1.) {
		i = divisions - 2;
		alpha = 1.;
	} else {
		i = Math.floor(vertex[0] * (divisions - 1));
		alpha = (vertex[0] * (divisions - 1)) % 1;
	}
	let j, beta;
	if (vertex[1] === 1.) {
		j = divisions - 2;
		beta = 1.;
	} else {
		j = Math.floor(vertex[1] * (divisions - 1));
		beta = (vertex[1] * (divisions - 1)) % 1;
	}

	let z00 = zs[i][j];
	let z01 = zs[i][j + 1];
	let z10 = zs[i + 1][j];
	let z11 = zs[i + 1][j + 1];

	return (
		(1. - alpha) * ((1. - beta) * z00 + beta * z01)
		+ alpha * ((1. - beta) * z10 + beta * z11)
	) * planeSideLength;
}

function drawHoverArcsWithInterpolation(scene, objects, networkVertices, networkEdgesLeft, networkEdgesRight, zLeft, zRight, lambda, planeSideLength) {
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

		let coordinatesXY = vertexToGlobalCoordinates(vertex, planeSideLength);

		coordinates[i] = new Array(
			coordinatesXY[0],
			coordinatesXY[1],
			(1 - lambda) * getMeshHeightAtCoordinates(vertex, zLeft, planeSideLength)
			+ lambda * getMeshHeightAtCoordinates(vertex, zRight, planeSideLength)
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
				linewidth: SIZE_HOVER_ARC_EDGE,
			}));

			scene.add(lineSegment);
			objects.push(lineSegment);
		}
	}
}

/**
 * Draw arcs above the plane whose heights show each edge's throughput at
 * time t, replacing the previously drawn arcs.
 *
 * @param {THREE.Scene} scene
 * @param {Array<THREE.Object3D>} objects the objects drawn; replaced in place
 * @param {Array<{coordinates: Array<number>}>} networkVertices
 * @param {Array<Array<{source: number, target: number, throughput: number}>>} networkEdges
 * @param {number} t
 * @param {Array<number>} ts
 * @param {Array<Array<Array<number>>>} zs
 * @param {number} planeSideLength
 */
export function drawHoverArcs(scene, objects, networkVertices, networkEdges, t, ts, zs, planeSideLength) {
	clearHoverArcs(scene, objects);

	let { index, lambda } = locateTime(t, ts);
	let indexRight = lambda == 0. ? index : index + 1;
	drawHoverArcsWithInterpolation(
		scene, objects, networkVertices,
		networkEdges[index], networkEdges[indexRight],
		zs[index], zs[indexRight], lambda,
		planeSideLength
	);
}
