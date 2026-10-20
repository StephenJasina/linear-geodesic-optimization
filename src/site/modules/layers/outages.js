import { vertexToCanvasCoordinates } from "./utils";

/**
 * Draw the edges added (fading in) and removed (fading out) between two
 * consecutive network graphs.
 *
 * @param {CanvasRenderingContext2D} context
 * @param {Array<{coordinates: Array<number>}>} networkVertices
 * @param {Array<Set<number>>} edgesAdded
 * @param {Array<Set<number>>} edgesRemoved
 * @param {number} lambda
 * @param {number} [lineWidth]
 */
export function drawOutages(context, networkVertices, edgesAdded, edgesRemoved, lambda, lineWidth = 6) {
	if (lambda == 0) {
		return;
	}

	context.save();

	let edgesToDrawAll = new Array();
	let edgesToDrawAdded = new Array();
	let edgesToDrawRemoved = new Array();
	for (let source = 0; source < networkVertices.length; ++source) {
		for (let target of edgesAdded[source]) {
			let edge = {
				'source': source,
				'target': target,
				'lambda': lambda,
			};
			edgesToDrawAll.push(edge);
			edgesToDrawAdded.push(edge);
		}
		for (let target of edgesRemoved[source]) {
			let edge = {
				'source': source,
				'target': target,
				'lambda': 1 - lambda,
			};
			edgesToDrawAll.push(edge);
			edgesToDrawRemoved.push(edge);
		}
	}

	// Borders first
	for (let edge of edgesToDrawAll) {
		let coordinatesSource = vertexToCanvasCoordinates(context, networkVertices[edge.source].coordinates);
		let coordinatesTarget = vertexToCanvasCoordinates(context, networkVertices[edge.target].coordinates);
		// Draw the edge
		context.beginPath();
		context.moveTo(coordinatesSource[0], coordinatesSource[1]);
		context.lineTo(coordinatesTarget[0], coordinatesTarget[1]);
		context.strokeStyle = `rgba(0, 0, 0, ${edge.lambda})`;
		context.lineWidth = lineWidth + 5;
		context.stroke();
	}
	// Then the actual edges
	for (let edge of edgesToDrawAdded) {
		let coordinatesSource = vertexToCanvasCoordinates(context, networkVertices[edge.source].coordinates);
		let coordinatesTarget = vertexToCanvasCoordinates(context, networkVertices[edge.target].coordinates);
		// Draw the edge
		context.beginPath();
		context.moveTo(coordinatesSource[0], coordinatesSource[1]);
		context.lineTo(coordinatesTarget[0], coordinatesTarget[1]);
		context.strokeStyle = `rgba(0, 255, 0, ${edge.lambda})`;
		context.lineWidth = lineWidth;
		context.stroke();
	}
	for (let edge of edgesToDrawRemoved) {
		let coordinatesSource = vertexToCanvasCoordinates(context, networkVertices[edge.source].coordinates);
		let coordinatesTarget = vertexToCanvasCoordinates(context, networkVertices[edge.target].coordinates);
		// Draw the edge
		context.beginPath();
		context.moveTo(coordinatesSource[0], coordinatesSource[1]);
		context.lineTo(coordinatesTarget[0], coordinatesTarget[1]);
		context.strokeStyle = `rgba(255, 0, 0, ${edge.lambda})`;
		context.lineWidth = lineWidth;
		context.stroke();
	}

	context.restore();
}

/**
 * Fill the outage list with every edge that appears at some point in the
 * animation but is missing from the given network graph.
 *
 * @param {HTMLUListElement} ulOutages
 * @param {Array<{label: string}>} networkVertices
 * @param {Array<{source: number, target: number}>} edges
 * @param {Array<Set<number>>} edgesAll
 */
export function updateOutageList(ulOutages, networkVertices, edges, edgesAll) {
	// Clear out the list first
	while (ulOutages.firstChild) {
		ulOutages.removeChild(ulOutages.firstChild);
	}

	// Add the missing edges
	let currentEdges = new Array(networkVertices.length);
	for (let source = 0; source < networkVertices.length; ++source) {
		currentEdges[source] = new Set();
	}
	for (let edge of edges) {
		currentEdges[edge.source].add(edge.target);
	}
	for (let source = 0; source < networkVertices.length; ++source) {
		for (let target of edgesAll[source]) {
			if (!currentEdges[source].has(target)) {
				let li = document.createElement("li");
				li.appendChild(document.createTextNode(networkVertices[source].label + " ↔ " + networkVertices[target].label));
				ulOutages.appendChild(li);
			}
		}
	}
}
