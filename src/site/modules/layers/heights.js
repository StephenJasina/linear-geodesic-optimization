import * as THREE from "three";

import { locateTime } from "../time.js";
import { updateCurvatureFrames } from "./hatching.js";

/**
 * @param {THREE.Mesh<THREE.PlaneGeometry>} plane
 */
function resetPlaneHeights(plane) {
	let bufferedHeights = plane.geometry.getAttribute('position');
	let divisions = plane.geometry.parameters.widthSegments + 1;
	for (let i = 0; i < divisions; i++) {
		for (let j = 0; j < divisions; j++) {
			bufferedHeights.setZ(i * divisions + j, 0);
		}
	}
	bufferedHeights.needsUpdate = true;
	plane.geometry.computeVertexNormals();
	updateCurvatureFrames(plane);
}

/**
 * @param {THREE.Mesh<THREE.PlaneGeometry>} plane
 * @param {Array<Array<number>>} zLeft
 * @param {Array<Array<number>>} [zRight]
 * @param {number} [lambda]
 */
function setPlaneHeights(plane, zLeft, zRight = null, lambda = null) {
	let bufferedHeights = plane.geometry.getAttribute('position');
	let divisions = plane.geometry.parameters.widthSegments + 1;

	if (zRight === null) {
		zRight = zLeft;
		lambda = 0.;
	}

	for (let i = 0; i < divisions; i++) {
		for (let j = 0; j < divisions; j++) {
			if (zLeft[i][j] === null || zRight[i][j] === null) {
			} else {
				bufferedHeights.setZ(i * divisions + j, ((1 - lambda) * zLeft[i][j] + lambda * zRight[i][j]) * plane.geometry.parameters.width);
			}
		}
	}
	bufferedHeights.needsUpdate = true;
	plane.geometry.computeVertexNormals();
	updateCurvatureFrames(plane);
}

/**
 * @param {THREE.Mesh<THREE.PlaneGeometry>} plane
 * @param {Array<Array<Array<number>>>} zs
 * @param {number} t
 * @param {Array<number>} ts
 */
function setPlaneHeightsWithInterpolation(plane, zs, t, ts) {
	// Assume ts are sorted. ts and zs are parallel.
	if (ts === null) {
		return;
	}

	let { index, lambda } = locateTime(t, ts);
	if (lambda == 0.) {
		setPlaneHeights(plane, zs[index]);
	} else {
		setPlaneHeights(plane, zs[index], zs[index + 1], lambda);
	}
}

export { resetPlaneHeights, setPlaneHeightsWithInterpolation };
