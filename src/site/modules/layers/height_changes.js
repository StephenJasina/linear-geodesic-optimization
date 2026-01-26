import { xToCanvasCoordinates, yToCanvasCoordinates, lengthToCanvasCoordinates } from "./utils.js"

/**
 * @param {CanvasRenderingContext2D} context
 * @param {Array<Array<Array<number>>>} zs
 * @param {Array<Array<Array<number>>>} zsEWMA
 * @param {number} t
 * @param {Array<number>} ts
 * @param {number} [sensitivity]
 */
export default function(context, zs, zsEWMA, t, ts, boundaries, sensitivity = 0.02) {
	if (t === null || ts === null || t <= ts[0] || t > ts[ts.length - 1]) {
		return;
	}

	let index = 0;
	while (ts[index] < t) {
		++index;
	}
	if (t - ts[index - 1] < 0.5 * (ts[index] - ts[index - 1])) {
		--index;
	}

	let divisionsX = zs[index].length;
	let divisionsY = zs[index][0].length;
	let r = lengthToCanvasCoordinates(context, 1. / 2 / (divisionsX - 1));

	context.save();
	context.fillStyle = "#FF0000";
	context.globalAlpha = 0.5;

	context.beginPath();

	let hasChange = new Array();
	// Circles
	for (let i = 0; i < divisionsX; ++i) {
		let x = xToCanvasCoordinates(context, i / (divisionsX - 1));
		let hasChangeRow = new Array();
		for (let j = 0; j < divisionsY; ++j) {
			let y = yToCanvasCoordinates(context, j / (divisionsY - 1));

			if (Math.abs(zs[index][i][j] - zsEWMA[index][i][j]) > sensitivity) {
				context.moveTo(x, y);
				context.arc(x, y, r, 0, 2 * Math.PI);

				hasChangeRow.push(true);
			} else {
				hasChangeRow.push(false);
			}
		}
		hasChange.push(hasChangeRow);
	}

	// Rectangles between horizontal pairs of circles
	for (let i = 0; i < divisionsX - 1; ++i) {
		let x = xToCanvasCoordinates(context, i / (divisionsX - 1));
		for (let j = 0; j < divisionsY; ++j) {
			let y = yToCanvasCoordinates(context, (j + 0.5) / (divisionsY - 1));

			if (hasChange[i][j] && hasChange[i + 1][j]) {
				context.rect(x, y, 2 * r, 2 * r);
			}
		}
	}

	// Rectangles between vertical pairs of circles
	for (let i = 0; i < divisionsX; ++i) {
		let x = xToCanvasCoordinates(context, (i - 0.5) / (divisionsX - 1));
		for (let j = 0; j < divisionsY - 1; ++j) {
			let y = yToCanvasCoordinates(context, (j + 1) / (divisionsY - 1));

			if (hasChange[i][j] && hasChange[i][j + 1]) {
				context.rect(x, y, 2 * r, 2 * r);
			}
		}
	}

	// Rectangles between major diagonal pairs of circles
	for (let i = 0; i < divisionsX - 1; ++i) {
		let x = xToCanvasCoordinates(context, i / (divisionsX - 1));
		for (let j = 0; j < divisionsY - 1; ++j) {
			let y = yToCanvasCoordinates(context, j / (divisionsY - 1));

			if (hasChange[i][j] && hasChange[i + 1][j + 1]) {
				context.save();
				context.translate(x, y);
				context.rotate(Math.PI / 4);
				context.moveTo(0, 0);
				context.rect(-r, -2 * Math.sqrt(2) * r, 2 * r, 2 * Math.sqrt(2) * r);
				context.restore();
			}
		}
	}

	// Rectangles between minor diagonal pairs of circles
	for (let i = 0; i < divisionsX - 1; ++i) {
		let x = xToCanvasCoordinates(context, (i + 1) / (divisionsX - 1));
		for (let j = 0; j < divisionsY - 1; ++j) {
			let y = yToCanvasCoordinates(context, j / (divisionsY - 1));

			if (hasChange[i + 1][j] && hasChange[i][j + 1]) {
				context.save();
				context.translate(x, y);
				context.rotate(Math.PI / 4);
				context.rect(-2 * Math.sqrt(2) * r, -r, 2 * Math.sqrt(2) * r, 2 * r);
				context.restore()
			}
		}
	}

	// context.fillStyle = "rgba(0, 0, 0, 0)";
	// context.fill();

	// context.save()
	// context.globalCompositeOperation = "source-in";
	// context.rect(0, 0, 500, 500);
	// context.fillStyle = "#0000FF";
	context.fill();
	// context.restore();

	context.restore();
}
