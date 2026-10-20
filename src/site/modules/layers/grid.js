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

	context.save();

	context.beginPath();
	for (let col = 0; col <= cols; ++col) {
		let i = col * width / cols;
		context.moveTo(i, 0);
		context.lineTo(i, height);
	}
	for (let row = 0; row <= rows; ++row) {
		let j = row * height / rows;
		context.moveTo(0, j);
		context.lineTo(width, j);
	}
	context.lineWidth = lineWidth;
	context.strokeStyle = "black";
	context.stroke();

	context.restore();
}
