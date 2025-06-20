import * as THREE from "three";

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

	if (zRight == null) {
		zRight = zLeft;
		lambda = 0.;
	}

	for (let i = 0; i < divisions; i++) {
		for (let j = 0; j < divisions; j++) {
			if (zLeft[i][j] == null || zRight[i][j] == null) {
			} else {
				bufferedHeights.setZ(i * divisions + j, ((1 - lambda) * zLeft[i][j] + lambda * zRight[i][j]) * plane.geometry.parameters.width);
			}
		}
	}
	bufferedHeights.needsUpdate = true;
}

/**
 * @param {THREE.Mesh<THREE.PlaneGeometry>} plane
 * @param {Array<Array<Array<number>>>} zs
 * @param {number} t
 * @param {Array<number>} ts
 */
function setPlaneHeightsWithInterpolation(plane, zs, t, ts) {
	// Assume ts are sorted. ts and zs are parallel.
	if (ts == null) {
		return;
	}

	if (t <= ts[0]) {
		setPlaneHeights(plane, zs[0]);
		return 0;
	}

	if (t >= ts[ts.length - 1]) {
		setPlaneHeights(plane, zs[ts.length - 1]);
		return ts.length - 1;
	}

	// TODO: Improve this from a linear search
	let index = 0;
	while (ts[index] < t) {
		++index;
	}

	if (ts[index] == t) {
		setPlaneHeights(plane, zs[index]);
		return index;
	}

	// If we reach here ts[index - 1] <= t < ts[index]
	let lambda = (t - ts[index - 1]) / (ts[index] - ts[index - 1]);
	setPlaneHeights(plane, zs[index - 1], zs[index], lambda);
}

export { resetPlaneHeights, setPlaneHeightsWithInterpolation };
