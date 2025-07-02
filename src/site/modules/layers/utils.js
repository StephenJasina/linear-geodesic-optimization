/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<number>} x
 * @returns {number}
 */
export function xToCanvasCoordinates(context, x) {
	return x * context.canvas.width;
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<number>} y
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
 * @param {Array<number>} length
 * @returns {number}
 */
export function lengthToCanvasCoordinates(context, length) {
	return length * context.canvas.width;
}
