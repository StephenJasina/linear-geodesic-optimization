/**
 * @param {CanvasRenderingContext2D} context
 * @param {number} x
 * @returns {number}
 */
export function xToCanvasCoordinates(context, x) {
	return x * context.canvas.width;
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {number} y
 * @returns {number}
 */
export function yToCanvasCoordinates(context, y) {
	return (1 - y) * context.canvas.height;
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<number>} vertex
 * @returns {Array<number>}
 */
export function vertexToCanvasCoordinates(context, vertex) {
	return [xToCanvasCoordinates(context, vertex[0]), yToCanvasCoordinates(context, vertex[1])];
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {number} length
 * @returns {number}
 */
export function lengthToCanvasCoordinates(context, length) {
	return length * context.canvas.width;
}

/**
 * Convert a vertex in [0, 1]^2 to x and y coordinates on a plane centered
 * at the origin.
 *
 * @param {Array<number>} vertex
 * @param {number} planeSideLength
 * @returns {Array<number>}
 */
export function vertexToGlobalCoordinates(vertex, planeSideLength) {
	return [(vertex[0] - 0.5) * planeSideLength, (vertex[1] - 0.5) * planeSideLength];
}
