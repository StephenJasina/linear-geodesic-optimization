import * as THREE from "three";

const COLOR_HATCHING = "#303030";
const HATCHING_OPACITY = 0.6;
// Width of a hatch line relative to the width of a contour line
const HATCHING_LINE_WIDTH = 1.25;

const ATTRIBUTE_NAME = "curvatureFrame";

// Values of uHatchingMode
const HATCHING_MODES = { "both": 0, "max": 1, "min": 2 };

/**
 * Compute principal curvature directions of a height field sampled on a
 * square grid laid out like a THREE.PlaneGeometry (row i at y = w/2 - i h,
 * column j at x = j h - w/2).
 *
 * Each principal direction is a line field (d and -d are the same), so it is
 * stored as a doubled angle (cos 2 theta, sin 2 theta), scaled by the
 * anisotropy k1 - k2. That encoding can be interpolated linearly across a
 * triangle without sign flips, and fades out on its own near umbilics. The
 * angles are of the directions projected to the xy plane.
 * @param {ArrayLike<number>} z Heights, indexed by i * n + j (NaN for missing)
 * @param {number} n Number of vertices along a side
 * @param {number} h Grid spacing
 * @param {Float32Array} out Length 4 n^2: (max direction, min direction) per vertex
 */
function computePrincipalFrames(z, n, h, out) {
	out.fill(0.);
	if (n < 3) {
		return;
	}

	for (let i = 0; i < n; ++i) {
		// Boundary vertices take the frame of the nearest interior vertex
		let ic = Math.min(Math.max(i, 1), n - 2);
		for (let j = 0; j < n; ++j) {
			let jc = Math.min(Math.max(j, 1), n - 2);
			let at = (di, dj) => z[(ic + di) * n + (jc + dj)];

			// y decreases as i increases
			let zx = (at(0, 1) - at(0, -1)) / (2. * h);
			let zy = (at(-1, 0) - at(1, 0)) / (2. * h);
			let zxx = (at(0, 1) - 2. * at(0, 0) + at(0, -1)) / (h * h);
			let zyy = (at(1, 0) - 2. * at(0, 0) + at(-1, 0)) / (h * h);
			let zxy = (at(-1, 1) - at(-1, -1) - at(1, 1) + at(1, -1)) / (4. * h * h);

			// First and second fundamental forms of the graph (x, y, z(x, y))
			let E = 1. + zx * zx;
			let F = zx * zy;
			let G = 1. + zy * zy;
			let root = Math.sqrt(1. + zx * zx + zy * zy);
			let L = zxx / root;
			let M = zxy / root;
			let N = zyy / root;

			// Principal curvatures solve det(II - k I) = 0
			let det = E * G - F * F;
			let H = (E * N - 2. * F * M + G * L) / (2. * det);
			let K = (L * N - M * M) / det;
			let delta = Math.sqrt(Math.max(H * H - K, 0.));
			let anisotropy = 2. * delta;
			if (!(anisotropy > 0.)) {
				// Umbilic, flat, or missing data: no preferred direction
				continue;
			}

			let index = 4 * (i * n + j);
			let ks = [H + delta, H - delta];
			for (let f = 0; f < 2; ++f) {
				let k = ks[f];
				// (II - k I) v = 0. Either row gives v; use the better conditioned one.
				let ax = M - k * F, ay = -(L - k * E);
				let bx = N - k * G, by = -(M - k * F);
				let [vx, vy] = (ax * ax + ay * ay >= bx * bx + by * by) ? [ax, ay] : [bx, by];
				let norm2 = vx * vx + vy * vy;
				if (norm2 > 0.) {
					out[index + 2 * f] = anisotropy * (vx * vx - vy * vy) / norm2;
					out[index + 2 * f + 1] = anisotropy * 2. * vx * vy / norm2;
				}
			}
		}
	}
}

/**
 * Recompute the plane's curvature frame attribute from its current heights.
 * @param {THREE.Mesh<THREE.PlaneGeometry>} plane
 */
function updateCurvatureFrames(plane) {
	let geometry = plane.geometry;
	let positions = geometry.getAttribute("position");
	let n = geometry.parameters.widthSegments + 1;
	let h = geometry.parameters.width / geometry.parameters.widthSegments;

	let attribute = geometry.getAttribute(ATTRIBUTE_NAME);
	if (attribute === undefined) {
		attribute = new THREE.BufferAttribute(new Float32Array(4 * n * n), 4);
		geometry.setAttribute(ATTRIBUTE_NAME, attribute);
	}

	let z = new Float64Array(n * n);
	for (let k = 0; k < n * n; ++k) {
		z[k] = positions.getZ(k);
	}
	computePrincipalFrames(z, n, h, attribute.array);
	attribute.needsUpdate = true;
}

/**
 * Patch a plane's material so it can draw hatch lines along the principal
 * curvature directions. Like the contours, the lines are blended into the
 * surface color before lighting. Each line follows its direction locally, so
 * lines shear and break where the direction field turns quickly.
 * @param {THREE.MeshStandardMaterial} material
 */
function addHatchingShader(material) {
	let uniforms = {
		uHatchingEnabled: { value: false },
		uHatchingMode: { value: HATCHING_MODES.both },
		uHatchingSpacing: { value: 1. },
		uHatchingScale: { value: 1. },
		uHatchingColor: { value: new THREE.Color(COLOR_HATCHING) },
		uHatchingOpacity: { value: HATCHING_OPACITY },
		uHatchingLineWidth: { value: HATCHING_LINE_WIDTH },
	};
	material.userData.hatchingUniforms = uniforms;

	let onBeforeCompilePrevious = material.onBeforeCompile;
	material.onBeforeCompile = function(shader, renderer) {
		onBeforeCompilePrevious.call(this, shader, renderer);
		Object.assign(shader.uniforms, uniforms);

		shader.vertexShader = shader.vertexShader
			.replace(
				"#include <common>",
				[
					"#include <common>",
					`attribute vec4 ${ATTRIBUTE_NAME};`,
					"varying vec4 vCurvatureFrame;",
					"varying vec2 vHatchingPosition;",
				].join("\n")
			)
			.replace(
				"#include <begin_vertex>",
				[
					"#include <begin_vertex>",
					`vCurvatureFrame = ${ATTRIBUTE_NAME};`,
					"vHatchingPosition = position.xy;",
				].join("\n")
			);

		shader.fragmentShader = shader.fragmentShader
			.replace(
				"#include <common>",
				[
					"#include <common>",
					"varying vec4 vCurvatureFrame;",
					"varying vec2 vHatchingPosition;",
					"uniform bool uHatchingEnabled;",
					"uniform int uHatchingMode;",
					"uniform float uHatchingSpacing;",
					"uniform float uHatchingScale;",
					"uniform vec3 uHatchingColor;",
					"uniform float uHatchingOpacity;",
					"uniform float uHatchingLineWidth;",
					// Coverage of a family of lines running along the direction
					// encoded by the doubled angle f
					"float hatchingLine(vec2 f) {",
					"	float a = length(f);",
					"	float theta = 0.5 * atan(f.y, f.x + 1e-12);",
					"	vec2 hatchingNormal = vec2(-sin(theta), cos(theta));",
					"	float g = dot(vHatchingPosition, hatchingNormal) / uHatchingSpacing;",
					// Differentiate the position rather than g, since theta
					// jumps by pi where the doubled angle wraps around
					"	vec2 dx = dFdx(vHatchingPosition);",
					"	vec2 dy = dFdy(vHatchingPosition);",
					"	float w = (abs(dot(dx, hatchingNormal)) + abs(dot(dy, hatchingNormal))) / uHatchingSpacing;",
					"	float d = abs(fract(g - 0.5) - 0.5) / max(w * uHatchingLineWidth, 1e-6);",
					"	float line = 1.0 - min(d, 1.0);",
					"	return line * smoothstep(0.0, 1.0, a / uHatchingScale);",
					"}",
				].join("\n")
			)
			.replace(
				"#include <map_fragment>",
				[
					"#include <map_fragment>",
					"if (uHatchingEnabled) {",
					"	float hatch = 0.0;",
					`	if (uHatchingMode != ${HATCHING_MODES.min}) hatch = max(hatch, hatchingLine(vCurvatureFrame.xy));`,
					`	if (uHatchingMode != ${HATCHING_MODES.max}) hatch = max(hatch, hatchingLine(vCurvatureFrame.zw));`,
					"	diffuseColor.rgb = mix(diffuseColor.rgb, uHatchingColor, hatch * uHatchingOpacity);",
					"}",
				].join("\n")
			);
	};
}

/**
 * @param {THREE.Mesh} plane
 * @param {boolean} enabled
 * @param {string} mode "both", "max", or "min"
 * @param {number} spacing Distance between adjacent hatch lines, in the plane's local units
 * @param {number} scale Anisotropy at and above which lines are fully opaque (0 if the surface is flat)
 */
function setHatching(plane, enabled, mode, spacing, scale) {
	let uniforms = plane.material.userData.hatchingUniforms;
	uniforms.uHatchingEnabled.value = enabled && scale > 0.;
	uniforms.uHatchingMode.value = HATCHING_MODES[mode] ?? HATCHING_MODES.both;
	uniforms.uHatchingSpacing.value = spacing;
	if (scale > 0.) {
		uniforms.uHatchingScale.value = scale;
	}
}

/**
 * Pick the anisotropy at which hatch lines become fully opaque from every
 * frame, so the lines don't flicker in strength as the animation plays. The
 * fade is only meant to hide lines where the direction is ill-defined, so by
 * default lines reach full strength at the median anisotropy. A high quantile
 * would let a few sharply curved spots (often near the boundary) wash out
 * everything else.
 * @param {Array<Array<Array<Array<number>>>>} heightSequences Lists of frames of height grids
 * @param {number} planeSideLength Scale from stored heights to local units
 * @param {number} quantile
 */
function getHatchingScale(heightSequences, planeSideLength, quantile = 0.5) {
	let values = [];
	for (let sequence of heightSequences) {
		if (sequence === null) {
			continue;
		}
		for (let frame of sequence) {
			let n = frame.length;
			if (n < 3) {
				continue;
			}
			let z = new Float64Array(n * n);
			for (let i = 0; i < n; ++i) {
				for (let j = 0; j < n; ++j) {
					let value = frame[i][j];
					z[i * n + j] = value === null ? NaN : value * planeSideLength;
				}
			}
			let frames = new Float32Array(4 * n * n);
			computePrincipalFrames(z, n, planeSideLength / (n - 1), frames);
			for (let k = 0; k < n * n; ++k) {
				let a = Math.hypot(frames[4 * k], frames[4 * k + 1]);
				if (a > 0.) {
					values.push(a);
				}
			}
		}
	}
	if (values.length === 0) {
		return 0.;
	}
	values.sort((a, b) => a - b);
	return values[Math.min(Math.floor(quantile * values.length), values.length - 1)];
}

export { addHatchingShader, getHatchingScale, setHatching, updateCurvatureFrames };
