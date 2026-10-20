import { vertexToCanvasCoordinates } from "./utils";

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<{coordinates: Array<number>}>} networkVertices
 * @param {number} vertexRadius
 * @param {string} colorVertex
 */
export default function(context, networkVertices, vertexRadius = 4, colorVertex = "#4caf50") {
	context.save();

	// Borders first. Each layer is a single path since the circles share a
	// style; moveTo keeps the circles from being joined.
	context.beginPath();
	for (let index = 0; index < networkVertices.length; ++index) {
		let vertex = vertexToCanvasCoordinates(context, networkVertices[index].coordinates);
		context.moveTo(vertex[0] + vertexRadius + 5, vertex[1]);
		context.arc(vertex[0], vertex[1], vertexRadius + 5, 0, 2 * Math.PI);
	}
	context.fillStyle = "#000000";
	context.fill();
	// Then the actual vertices
	context.beginPath();
	for (let indexVertex = 0; indexVertex < networkVertices.length; ++indexVertex) {
		let vertex = vertexToCanvasCoordinates(context, networkVertices[indexVertex].coordinates);
		context.moveTo(vertex[0] + vertexRadius, vertex[1]);
		context.arc(vertex[0], vertex[1], vertexRadius, 0, 2 * Math.PI);
	}
	context.fillStyle = colorVertex;
	context.fill();

	context.restore();
}