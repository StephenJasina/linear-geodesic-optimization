/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<number>} vertex
 * @returns {Array<number}
 */
export function vertexToCanvasCoordinates(context, vertex) {
	let x = vertex[0];
	let y = vertex[1];

	return [
		x * context.canvas.width,
		(1 - y) * context.canvas.height,
	];
}
