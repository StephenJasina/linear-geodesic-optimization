window.onload = function() {
  const ctx = document.createElement('canvas').getContext('2d');
  document.body.append(ctx.canvas)

  ctx.canvas.width = 1000;
  ctx.canvas.height = 1000;
  ctx.fillStyle = "#ffff00";
  ctx.fillRect(10, 10, 200, 200);
  // const texture = new THREE.CanvasTexture(ctx.canvas);
  // texture.minFilter = THREE.LinearFilter;
}
