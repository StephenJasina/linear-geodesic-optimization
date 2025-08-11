import { vertexToCanvasCoordinates } from "./utils";
import { getCurvatureColor } from "../colors.js";

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<{coordinates: Array<number>}>} networkVertices
 * @param {Array<{source: number, target: number, curvature: number}>} edges
 * @param {number} lineWidth
 */
export default function(context, networkVertices, edges, lineWidth = 6, colorEdge = "#000000") {
	context.save();

	// Borders first
	for (let indexEdge = 0; indexEdge < edges.length; ++indexEdge) {
		let edge = edges[indexEdge];
		let source = vertexToCanvasCoordinates(context, networkVertices[edge.source].coordinates);
		let target = vertexToCanvasCoordinates(context, networkVertices[edge.target].coordinates);

		// Draw the edge
		context.beginPath();
		context.moveTo(source[0], source[1]);
		context.lineTo(target[0], target[1]);
		context.strokeStyle = colorEdge;
		context.lineWidth = lineWidth + 5;
		context.stroke();
	}
	// Then the actual edges
	for (let indexEdge = 0; indexEdge < edges.length; ++indexEdge) {
		let edge = edges[indexEdge];
		let source = vertexToCanvasCoordinates(context, networkVertices[edge.source].coordinates);
		let target = vertexToCanvasCoordinates(context, networkVertices[edge.target].coordinates);

		// Draw the edge
		context.beginPath();
		context.moveTo(source[0], source[1]);
		context.lineTo(target[0], target[1]);
		// ctx.strokeStyle = getWeightedColor(edge.throughput);
		context.strokeStyle = getCurvatureColor(edge.curvature);
		context.lineWidth = lineWidth;
		context.stroke();
	}

	context.restore();
}
