// Three.js
import * as THREE from "three";

import { createOLMap, updateOLMap, makeTextureMap } from "./layers/map.js";

class MatissePlane {
	#plane;
	#olMap;

	/**
	 * Create a plane.
	 *
	 * @param {number} divisions
	 */
	constructor(divisions, olMap, ) {
		this.#plane = this.#makePlane(divisions);
		this.#olMap = olMap;
	}

	/**
	 * Create a three.js plane with a certain refinement.
	 *
	 * @param {number} divisions
	 */
	#makePlane(divisions) {
		let planeGeometry = new THREE.PlaneGeometry(
			planeWidth, planeHeight,
			divisions - 1, divisions - 1
		);
		let plane = new THREE.Mesh(planeGeometry, planeMaterial);
		plane.rotation.set(-Math.PI / 2, 0, 0);
		return plane;
	}

	#makeTextureBlank() {
		let textureBlank = new THREE.CanvasTexture(contextBlank.canvas);
		textureBlank.minFilter = THREE.LinearFilter;
		textureBlank.center = new THREE.Vector2(0.5, 0.5);
		textureBlank.rotation = -Math.PI / 2;
		return textureBlank;
	}
}
