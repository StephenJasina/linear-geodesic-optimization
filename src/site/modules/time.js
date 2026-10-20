/**
 * Locate a time within a sorted list of times.
 *
 * Returns the index of the last time at or before t, along with how far t is
 * towards the next time (0 when t is exactly one of the times). Times outside
 * the range are clamped to its ends.
 *
 * @param {number} t
 * @param {Array<number>} ts
 * @returns {{index: number, lambda: number}}
 */
export function locateTime(t, ts) {
	if (t <= ts[0]) {
		return { index: 0, lambda: 0. };
	}

	if (t >= ts[ts.length - 1]) {
		return { index: ts.length - 1, lambda: 0. };
	}

	// Binary search so that ts[low] <= t < ts[high]
	let low = 0;
	let high = ts.length - 1;
	while (high - low > 1) {
		let middle = Math.floor((low + high) / 2);
		if (ts[middle] <= t) {
			low = middle;
		} else {
			high = middle;
		}
	}

	return { index: low, lambda: (t - ts[low]) / (ts[high] - ts[low]) };
}
