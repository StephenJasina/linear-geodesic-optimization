/**
 * Interpolation of geodesic paths, both across keyframes and toward the
 * routes the traffic takes.
 *
 * Every path is resampled onto a *canonical grid*: the path is split at
 * the hops of its route, and each piece is resampled at `SAMPLES_PER_HOP`
 * evenly spaced arc-length parameters. Paths along the same route then
 * have the same number of points, with corresponding points in
 * corresponding places, so they can be blended pointwise.
 */

export const SAMPLES_PER_HOP = 32;

function distanceSquared(a, b) {
	let dx = a[0] - b[0];
	let dy = a[1] - b[1];
	return dx * dx + dy * dy;
}

/**
 * Resample a polyline at `count` points evenly spaced by arc length,
 * including both endpoints.
 */
function resample(polyline, count) {
	let cumulative = new Array(polyline.length);
	cumulative[0] = 0.;
	for (let i = 1; i < polyline.length; ++i) {
		cumulative[i] = cumulative[i - 1] + Math.sqrt(distanceSquared(polyline[i - 1], polyline[i]));
	}
	let total = cumulative[polyline.length - 1];

	let result = new Array(count);
	let segment = 0;
	for (let i = 0; i < count; ++i) {
		if (total == 0.) {
			result[i] = [polyline[0][0], polyline[0][1]];
			continue;
		}
		let target = total * i / (count - 1);
		while (segment < polyline.length - 2 && cumulative[segment + 1] < target) {
			++segment;
		}
		let length = cumulative[segment + 1] - cumulative[segment];
		let t = length == 0. ? 0. : Math.min(Math.max((target - cumulative[segment]) / length, 0.), 1.);
		let a = polyline[segment];
		let b = polyline[segment + 1];
		result[i] = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
	}
	return result;
}

/**
 * Concatenate pieces resampled at `SAMPLES_PER_HOP + 1` points each,
 * dropping the joint shared by consecutive pieces.
 */
function resamplePieces(pieces) {
	let result = [];
	for (let i = 0; i < pieces.length; ++i) {
		let samples = resample(pieces[i], SAMPLES_PER_HOP + 1);
		result.push(...(i == 0 ? samples : samples.slice(1)));
	}
	return result;
}

/**
 * Get the coordinates of the hops of a route (a list of node indices),
 * merging consecutive repeated nodes. Return `null` if the route has
 * fewer than two distinct hops.
 */
export function getRouteHops(route, networkVertices) {
	let hops = [];
	let previous = null;
	for (let index of route) {
		if (index === previous) {
			continue;
		}
		previous = index;
		hops.push(networkVertices[index].coordinates);
	}
	return hops.length < 2 ? null : hops;
}

/**
 * Split a polyline into one piece per link of the route through `hops`,
 * at the polyline vertices nearest to the interior hops (searched
 * monotonically along the polyline). Return `null` if the split
 * degenerates.
 */
function splitAtHops(polyline, hops) {
	let n = polyline.length;
	let splits = [0];
	for (let h = 1; h < hops.length - 1; ++h) {
		// Leave at least one vertex for each remaining hop
		let start = splits[splits.length - 1] + 1;
		let end = n - 1 - (hops.length - 1 - h);
		if (start > end) {
			return null;
		}
		let best = start;
		let bestDistance = Infinity;
		for (let j = start; j <= end; ++j) {
			let distance = distanceSquared(polyline[j], hops[h]);
			if (distance < bestDistance) {
				bestDistance = distance;
				best = j;
			}
		}
		splits.push(best);
	}
	splits.push(n - 1);

	let pieces = [];
	for (let i = 0; i + 1 < splits.length; ++i) {
		pieces.push(polyline.slice(splits[i], splits[i + 1] + 1));
	}
	return pieces;
}

/**
 * Resample a geodesic onto the canonical grid of the route through
 * `hops`. Return `null` if the geodesic is empty.
 */
export function canonicalizeGeodesic(polyline, hops) {
	if (polyline == null || polyline.length < 2) {
		return null;
	}
	let pieces = splitAtHops(polyline, hops);
	if (pieces === null) {
		return resample(polyline, SAMPLES_PER_HOP * (hops.length - 1) + 1);
	}
	return resamplePieces(pieces);
}

/**
 * The route through `hops` (straight links), on its canonical grid.
 */
export function canonicalizeRoute(hops) {
	let pieces = [];
	for (let i = 0; i + 1 < hops.length; ++i) {
		pieces.push([hops[i], hops[i + 1]]);
	}
	return resamplePieces(pieces);
}

/**
 * Pointwise `(1 - t) * a + t * b` of two polylines of equal length.
 */
export function blendPolylines(a, b, t) {
	if (t == 0.) {
		return a;
	}
	if (t == 1.) {
		return b;
	}
	let result = new Array(a.length);
	for (let i = 0; i < a.length; ++i) {
		result[i] = [
			a[i][0] + t * (b[i][0] - a[i][0]),
			a[i][1] + t * (b[i][1] - a[i][1]),
		];
	}
	return result;
}
