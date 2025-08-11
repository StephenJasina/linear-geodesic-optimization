import { vertexToCanvasCoordinates } from "./utils";

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<{coordinates: Array<number>}>} networkVertices
 * @param {Array<{source: number, target: number}>} edges
 * @param {Array<Array<number>>} edgesAll
 * @param {HTMLUListElement} [ulOutages]
 * @param {number} [outageBlinkRate]
 * @param {number} [lineWidth]
 * @param {string} [colorOutage]
 *
 * @returns {null}
 */
export function drawOutagesOld(context, networkVertices, edges, edgesAll, ulOutages = null, lineWidth = 6, colorOutage = "#00ff00") {
	context.save();

	let currentEdges = new Array(networkVertices.length);
	for (let source = 0; source < networkVertices.length; ++source) {
		currentEdges[source] = new Set();
	}
	for (let edge of edges) {
		currentEdges[edge.source].add(edge.target);
	}

	// Borders first
	for (let source = 0; source < networkVertices.length; ++source) {
		for (let target of edgesAll[source]) {
			if (!currentEdges[source].has(target)) {
				let coordinatesSource = vertexToCanvasCoordinates(context, networkVertices[source].coordinates);
				let coordinatesTarget = vertexToCanvasCoordinates(context, networkVertices[target].coordinates);
				// Draw the edge
				context.beginPath();
				context.moveTo(coordinatesSource[0], coordinatesSource[1]);
				context.lineTo(coordinatesTarget[0], coordinatesTarget[1]);
				context.strokeStyle = "#000000";
				context.lineWidth = lineWidth + 5;
				context.stroke();
			}
		}
	}
	// Then the actual edges
	for (let source = 0; source < networkVertices.length; ++source) {
		for (let target of edgesAll[source]) {
			if (!currentEdges[source].has(target)) {
				let coordinatesSource = vertexToCanvasCoordinates(context, networkVertices[source].coordinates);
				let coordinatesTarget = vertexToCanvasCoordinates(context, networkVertices[target].coordinates);
				// Draw the edge
				context.beginPath();
				context.moveTo(coordinatesSource[0], coordinatesSource[1]);
				context.lineTo(coordinatesTarget[0], coordinatesTarget[1]);
				context.strokeStyle = colorOutage;
				context.lineWidth = lineWidth;
				context.stroke();
			}
		}
	}

	context.restore();

	// Update the info box
	if (ulOutages != null) {
		// Clear out the list first
		while (ulOutages.firstChild) {
			ulOutages.removeChild(ulOutages.firstChild);
		}

		// Add the new edges
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
}

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<{coordinates: Array<number>}>} networkVertices
 * @param {Array<Set<number>>} edgesAdded
 * @param {Array<Set<number>>} edgesRemoved
 * @param {number} lambda
 * @param {HTMLUListElement} [ulOutages]
 * @param {number} [outageBlinkRate]
 * @param {number} [lineWidth]
 * @param {string} [colorOutage]
 *
 * @returns {null}
 */
export function drawOutages(context, networkVertices, edgesAdded, edgesRemoved, lambda, ulOutages = null, lineWidth = 6, colorOutage = "#00ff00") {
	if (lambda == 0) {
		return;
	}

	context.save();

	let edgesToDrawAll = new Array();
	let edgesToDrawAdded = new Array();
	let edgesToDrawRemoved = new Array();
	for (let source = 0; source < networkVertices.length; ++source) {
		for (let target of edgesAdded[source]) {
			edgesToDrawAll.push({
				'source': source,
				'target': target,
				'lambda': lambda,
			});
			edgesToDrawAdded.push({
				'source': source,
				'target': target,
				'lambda': lambda,
			});
		}
		for (let target of edgesRemoved[source]) {
			edgesToDrawAll.push({
				'source': source,
				'target': target,
				'lambda': 1 - lambda,
			});
			edgesToDrawRemoved.push({
				'source': source,
				'target': target,
				'lambda': 1 - lambda,
			});
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
	context.save();
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

	context.restore();

	// Update the info box
	// if (ulOutages != null) {
	// 	// Clear out the list first
	// 	while (ulOutages.firstChild) {
	// 		ulOutages.removeChild(ulOutages.firstChild);
	// 	}

	// 	// Add the new edges
	// 	for (let source = 0; source < networkVertices.length; ++source) {
	// 		for (let target of edgesAll[source]) {
	// 			if (!currentEdges[source].has(target)) {
	// 				let li = document.createElement("li");
	// 				li.appendChild(document.createTextNode(networkVertices[source].label + " ↔ " + networkVertices[target].label));
	// 				ulOutages.appendChild(li);
	// 			}
	// 		}
	// 	}
	// }
}
