/**
 * @param {CanvasRenderingContext2D} context
 * @param {number} divisions
 * @param {number} lineWidth
 */
export default function(context, divisions, lineWidth = 2) {
	let canvas = context.canvas;
	let cols = divisions - 1;
	let rows = divisions - 1;

	let width = canvas.width;
	let height = canvas.height;
	let start_i = 0;
	let start_j = 0;

	context.save();

	context.beginPath();
	for (let i = start_i; i <= width; i += (width - start_i) / cols) {
		context.moveTo(i, 0);
		context.lineTo(i, canvas.height);
	}
	for (let j = start_j; j <= height; j += (height - start_j) / rows) {
		context.moveTo(0, j);
		context.lineTo(canvas.width, j);
	}
	context.lineWidth = lineWidth;
	context.strokyStyle = "black";
	context.stroke();

	context.restore();
}
