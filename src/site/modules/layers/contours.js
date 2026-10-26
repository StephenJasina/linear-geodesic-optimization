import * as THREE from "three";

const COLOR_CONTOUR = "#303030";
const CONTOUR_OPACITY = 0.5;

/**
 * Patch a plane's material so it can draw height isolines. The lines are
 * blended into the surface color before lighting, so they are shaded like
 * the rest of the surface and show over any texture.
 * @param {THREE.MeshStandardMaterial} material
 */
function addContourShader(material) {
	let uniforms = {
		uContourEnabled: { value: false },
		uContourSpacing: { value: 1. },
		uContourColor: { value: new THREE.Color(COLOR_CONTOUR) },
		uContourOpacity: { value: CONTOUR_OPACITY },
	};
	material.userData.contourUniforms = uniforms;

	material.onBeforeCompile = function(shader) {
		Object.assign(shader.uniforms, uniforms);

		shader.vertexShader = shader.vertexShader
			.replace(
				"#include <common>",
				"#include <common>\nvarying float vContourHeight;"
			)
			.replace(
				"#include <begin_vertex>",
				// The plane's local z is the height
				"#include <begin_vertex>\nvContourHeight = position.z;"
			);

		shader.fragmentShader = shader.fragmentShader
			.replace(
				"#include <common>",
				[
					"#include <common>",
					"varying float vContourHeight;",
					"uniform bool uContourEnabled;",
					"uniform float uContourSpacing;",
					"uniform vec3 uContourColor;",
					"uniform float uContourOpacity;",
				].join("\n")
			)
			.replace(
				"#include <map_fragment>",
				[
					"#include <map_fragment>",
					"if (uContourEnabled) {",
					"	float g = vContourHeight / uContourSpacing;",
					"	float w = fwidth(g);",
					// Distance to the nearest isoline, in pixels. Flat regions
					// (w == 0) have no well-defined isoline, so skip them.
					"	float d = abs(fract(g - 0.5) - 0.5) / max(w, 1e-6);",
					"	float line = w > 0.0 ? 1.0 - min(d, 1.0) : 0.0;",
					"	diffuseColor.rgb = mix(diffuseColor.rgb, uContourColor, line * uContourOpacity);",
					"}",
				].join("\n")
			);
	};
}

/**
 * @param {THREE.Mesh} plane
 * @param {boolean} enabled
 * @param {number} spacing Height difference between adjacent isolines, in the plane's local units
 */
function setContours(plane, enabled, spacing) {
	let uniforms = plane.material.userData.contourUniforms;
	uniforms.uContourEnabled.value = enabled && spacing > 0.;
	if (spacing > 0.) {
		uniforms.uContourSpacing.value = spacing;
	}
}

/**
 * Pick an isoline spacing from the range of heights over every frame, so the
 * lines stay put as the animation plays.
 * @param {Array<Array<Array<Array<number>>>>} heightSequences Lists of frames of height grids
 * @param {number} planeSideLength Scale from stored heights to local units
 * @param {number} count Approximate number of isolines across the full range
 */
function getContourSpacing(heightSequences, planeSideLength, count = 20) {
	let zMin = Infinity;
	let zMax = -Infinity;
	for (let sequence of heightSequences) {
		if (sequence === null) {
			continue;
		}
		for (let frame of sequence) {
			for (let row of frame) {
				for (let z of row) {
					if (z !== null) {
						zMin = Math.min(zMin, z);
						zMax = Math.max(zMax, z);
					}
				}
			}
		}
	}
	if (!(zMax > zMin)) {
		return 0.;
	}
	return (zMax - zMin) * planeSideLength / count;
}

export { addContourShader, setContours, getContourSpacing };
