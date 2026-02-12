import { vertexToCanvasCoordinates } from "./utils";

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<Array<Array<number>>>} geodesics
 * @param {number} geodesicsWidth
 * @param {Array<Array<number>>} colors
 */
export default function(context, geodesics, geodesicsWidth = 3, colors = null) {
	context.save();

	for (let index = 0; index < geodesics.length; ++index) {
		let geodesic = geodesics[index];
		if (geodesic.length == 0) {
			continue;
		}
		let vertex = vertexToCanvasCoordinates(context, geodesic[0]);
		context.beginPath();
		context.moveTo(vertex[0], vertex[1]);
		if (colors === null) {
			context.strokeStyle = "#000000";
		} else {
			let color = colors[index];
			context.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
		}
		context.lineWidth = geodesicsWidth;
		for (let i = 1; i < geodesic.length; ++i) {
			vertex = vertexToCanvasCoordinates(context, geodesic[i]);
			context.lineTo(vertex[0], vertex[1]);
		}
		context.stroke();
	}

	context.restore();
}
