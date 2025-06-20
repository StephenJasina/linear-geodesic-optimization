import { vertexToCanvasCoordinates } from "./utils";

/**
 * @param {Array<Array<Array<number>>>} geodesics
 * @param {number} geodesicsWidth
 */
export default function(context, geodesics, geodesicsWidth = 3) {
	context.save();

	for (let index = 0; index < geodesics.length; ++index) {
		let geodesic = geodesics[index];
		let vertex = vertexToCanvasCoordinates(context, geodesic[0]);
		context.beginPath();
		context.moveTo(vertex[0], vertex[1]);
		context.strokeStyle = "#000000";
		context.lineWidth = geodesicsWidth;
		for (let i = 1; i < geodesic.length; ++i) {
			vertex = vertexToCanvasCoordinates(context, geodesic[i]);
			context.lineTo(vertex[0], vertex[1]);
		}
		context.stroke();
	}

	context.restore();
}