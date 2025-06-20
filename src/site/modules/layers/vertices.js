import { vertexToCanvasCoordinates } from "./utils";

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<{coordinates: Array<number>}>} networkVertices
 * @param {number} vertexRadius
 * @param {string} colorVertex
 */
export default function(context, networkVertices, vertexRadius = 4, colorVertex = "#4caf50") {
	context.save();

	for (let index = 0; index < networkVertices.length; ++index) {
		let vertex = vertexToCanvasCoordinates(context, networkVertices[index].coordinates);
		context.fillStyle = "#000000";
		context.beginPath();
		context.arc(vertex[0], vertex[1], vertexRadius + 5, 0, 2 * Math.PI);
		context.fill();
	}
	// Then the actual vertices
	for (let indexVertex = 0; indexVertex < networkVertices.length; ++indexVertex) {
		let vertex = vertexToCanvasCoordinates(context, networkVertices[indexVertex].coordinates);
		context.fillStyle = colorVertex;
		context.beginPath();
		context.arc(vertex[0], vertex[1], vertexRadius, 0, 2 * Math.PI);
		context.fill();
	}

	context.restore();
}