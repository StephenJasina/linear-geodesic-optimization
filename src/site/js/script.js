import { EffectComposer } from './scripts/EffectComposer.js';
import { RenderPass } from './scripts/RenderPass.js';
import { ShaderPass } from './scripts/ShaderPass.js';
import { CopyShader } from './scripts/CopyShader.js';
import { FXAAShader } from './scripts/FXAAShader.js';
import { LineGeometry } from './scripts/LineGeometry.js';
import { Line2 } from './scripts/Line2.js';
import { LineMaterial } from './scripts/LineMaterial.js';
import { SelectionBox } from './scripts/SelectionBox.js';
import { SelectionHelper } from './scripts/SelectionHelper.js';
import RBush from './scripts/rbush/index.js'
import quickselect from './scripts/quickselect/index.js'
import Draw from './scripts/ol/interaction/Draw.js';
import {transformExtent} from './scripts/ol/proj.js';


// import { GreatCircle } from './GreatCircle/GreatCircle.js';


let bgcolor = 0xf3f3f3
let graphcolor = 0xffffff
let vertexcolor = 0x4CAF50
let edgecolor = 0x21bf73
let edgecolor_sec = 0x4cd995
let canvascolor = "#ffffff" // #c7c7c7
let contcolor = 0xff0000

// Extra colors
/*
// let bgcolor = 0xf3f3f3
// let graphcolor = 0xebe6e6
// let vertexcolor = 0x4CAF50
// let edgecolor = 0x21bf73
// let canvascolor = "#c7c7c7"

// let bgcolor = 0x512b58
// let graphcolor = 0x2c003e
// let vertexcolor = 0xfe346e
// let edgecolor = 0xd2fafb
// let canvascolor = "#2c003e"


// let bgcolor = 0xffc2c2
// let graphcolor = 0xff9d9d
// let vertexcolor = 0xff2e63
// let edgecolor = 0x010a43
// let canvascolor = "#ff9d9d"
*/

// TODO: Resize graph based on highest weight
// TODO: Optimization move lines instead of redrawing?
// TODO: Smoothen graph, check vertices very far from numbers, take average if so
// TODO: Discard opacityMap, redundant?

let T = THREE

let vertexCount = 0
let edgeCount = 0

let zoomWidths = []
let zoomHeights = []
let zoomLevels = []

let planeXMin = -10, planeXMax = 10
let planeYMin = -10, planeYMax = 10
let planeW = planeXMax - planeXMin
let planeH = planeYMax - planeYMin
let divisions = 20 // Was 150
let heightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
let opacityMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
let curvMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
let refine_data = {}

let calcHeightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));

let vertexHeight = 3

let time = Date.now()

let dataSent = false

// Cycle variables for threshold cycle
let cycle = false
let last_cycle = Date.now()

// Flag to indicate threshold change -> update heights when thresh changed
let thresh_change = false

// ThreeJS Scene Setup

var scene = new T.Scene()
// var camera = new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 )
// var camera = new THREE.PerspectiveCamera( 25, window.innerWidth/window.innerHeight, 0.1, 1000 )
let div = 128
var camera = new THREE.OrthographicCamera( window.innerWidth/-div, window.innerWidth/div, window.innerHeight/div, window.innerHeight/-div, 0.1, 1000 )
camera.position.x = -15
camera.position.z = 20
camera.position.y = 15

var clock = new T.Clock()

var renderer = new THREE.WebGLRenderer( { logarithmicDepthBuffer: true, antialias: false } )
renderer.setSize( window.innerWidth, window.innerHeight )
renderer.autoClear = false;
renderer.setPixelRatio( window.devicePixelRatio )
// renderer.shadowMap.enabled = true;
// renderer.setClearColor()
document.body.appendChild( renderer.domElement )

scene.background = new THREE.Color(bgcolor)
var controls = new T.OrbitControls( camera, renderer.domElement );

const olMap = createMap()
const canv = document.createElement('canvas')
canv.id = "canvas-texture"
let ctx = canv.getContext('2d');
ctx.canvas.width = 4000 //2000;
ctx.canvas.height = 4000 //2000;
ctx.fillStyle = canvascolor;
ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
var texture = new T.CanvasTexture(ctx.canvas);
texture.minFilter = THREE.LinearFilter;
texture.center = new T.Vector2(0.5, 0.5)
texture.rotation = -Math.PI/2

// let background = new Image()
// background.src = "./images/grayworld2.jpg"
// background.onload = function() {
//   ctx.drawImage(background,0,0)
//
// }

let ptGeom = new T.SphereGeometry(0.02, 32, 32) // 0.15 // 0.04 // 0.10 // 0.02
let ptMat = new T.MeshBasicMaterial({color: vertexcolor})


let p1 = new T.Vector3(-1, 0, 3.5)
let p2 = new T.Vector3(1, 0, 3.5)
let p3 = new T.Vector3(0, -1, 2.4)

// let p1 = new T.Vector3(-1, -1.2, 3)
// let p2 = new T.Vector3(1, -1.2, 3)
// let p3 = new T.Vector3(0, -1.8, 2.4)

let newPt = new T.Mesh(ptGeom, ptMat)
newPt.position.y = p1.y
newPt.position.x = p1.x
newPt.position.z = p1.z
// scene.add(newPt)

newPt = new T.Mesh(ptGeom, ptMat)
newPt.position.y = p2.y
newPt.position.x = p2.x
newPt.position.z = p2.z
// scene.add(newPt)

newPt = new T.Mesh(ptGeom, ptMat)
newPt.position.y = p3.y
newPt.position.x = p3.x
newPt.position.z = p3.z
// scene.add(newPt)

var clipPlane = new T.Plane().setFromCoplanarPoints(p1, p2, p3)

let p4 = new T.Vector3(-1, 0, -3.5)
let p5 = new T.Vector3(1, 0, -3.5)
let p6 = new T.Vector3(0, -1, -3)

// let p1 = new T.Vector3(-1, -1.5, 3)
// let p2 = new T.Vector3(1, -1.5, 3)
// let p3 = new T.Vector3(0, -2, 2.4)

newPt = new T.Mesh(ptGeom, ptMat)
newPt.position.y = p4.y
newPt.position.x = p4.x
newPt.position.z = p4.z
// scene.add(newPt)

newPt = new T.Mesh(ptGeom, ptMat)
newPt.position.y = p5.y
newPt.position.x = p5.x
newPt.position.z = p5.z
// scene.add(newPt)

newPt = new T.Mesh(ptGeom, ptMat)
newPt.position.y = p6.y
newPt.position.x = p6.x
newPt.position.z = p6.z
// scene.add(newPt)

var clipPlane2 = new T.Plane().setFromCoplanarPoints(p6, p5, p4)

var clipPlane = new T.Plane(new T.Vector3(0, 2, 0), 2)
var clipPlane2 = new T.Plane(new T.Vector3(1, 0, 0), 7) // 5.5
var clipPlane3 = new T.Plane(new T.Vector3(-1, 0, 0), 7) // 6 // 7
var clipPlane4 = new T.Plane(new T.Vector3(0, 0, 1), 10) // 3.3
var clipPlane5 = new T.Plane(new T.Vector3(0, 0, -1), 10) // 3.3

var loader = new T.ImageLoader()
var aMap = loader.load("images/grayscale.png")


var geometry = new T.PlaneGeometry(planeW, planeH, divisions-1, divisions-1)
var contGeom = new T.PlaneGeometry(planeW/2, planeH/2, divisions-1, divisions-1)
var material = new T.MeshBasicMaterial( { color: graphcolor, side: T.DoubleSide} )
var contMat = new T.MeshBasicMaterial( { color: contcolor, side: T.DoubleSide} )
//TODO: change back
var planeMat = new THREE.MeshPhongMaterial( { color: graphcolor, clippingPlanes: [clipPlane2, clipPlane3, clipPlane4, clipPlane5],
  vertexColors: T.VertexColors, side: THREE.DoubleSide,  flatShading: false, shininess: 0,
  wireframe: false, map: texture, transparent: true, opacity: 1.0}) //, alphaMap: T.ImageUtils.loadTexture("images/grayscale.png")} )
let transparentMat = new T.MeshLambertMaterial({visible: false})
var transparentPlaneMat = new THREE.MeshPhongMaterial( { color: graphcolor, clippingPlanes: [clipPlane2, clipPlane3, clipPlane4, clipPlane5], vertexColors: T.VertexColors, side: THREE.DoubleSide,  flatShading: false, shininess: 0, wireframe: false, map: texture, transparent: true, opacity: 0.5} )
let mMat = [planeMat, transparentMat, transparentPlaneMat]
var plane = new T.Mesh( geometry, mMat )

// for (let i = -1 ; i < 1 ; i+= 0.4) {
//   let contourPlane = new T.Mesh(contGeom, contMat)
//   contourPlane.rotation.set(-1.57, 0, 0)
//   contourPlane.position.set(0, i, 0)
//   scene.add(contourPlane)
// }
// plane.receiveShadow = true
// plane.castShadow = true
plane.rotation.set(-Math.PI/2, 0, 0.)
scene.add( plane )

controls.update();


let light = new T.PointLight( 0xffffff, 0.5)
light.position.set( -7, 8, 0)  // -7, 10, 0 // 7 3 -5
scene.add(light)

const directionalLight = new T.DirectionalLight( 0xffffff, 0.6 );
directionalLight.position.set(1.5, 0.1 , 10)
scene.add( directionalLight );

// const pointLightHelper = new T.PointLightHelper( light, 1 );
// scene.add( pointLightHelper );

var alight = new THREE.AmbientLight( 0x404040 ); // soft white light
scene.add( alight );


// Extra lights
/**
// s1 - (0, 10, 0)  | intensity 1   | 0xebe6e6
// s4 - (0, 10, 0)  | intensity 1   | white
// s5 - (0, 10, 0)  | intensity 1.1 | white
// s6 - (-7, 10, 0) | intensity 1.1 | white
// s7 - (0, 10, 5)  | intensity 1.1 | white
// s8 - (-7, 10, 0) | intensity 1   | white

// var alight = new THREE.AmbientLight( 0x404040 ); // soft white light
// scene.add( alight );

// const dlight = new THREE.DirectionalLight(0xffffff, 1);
// dlight.castShadow = true;
// dlight.position.set(0, 5, -20);
// dlight.target.position.set(0, 0, 0);
// scene.add(dlight);
// scene.add(dlight.target);

// var directionalLight = new THREE.DirectionalLight( 0xffffff, 1 );
// directionalLight.castShadow = true
// directionalLight.position.set(0, 2,3)
// scene.add( directionalLight );


// let light2 = new T.PointLight( 0xffffff, 3.5)
// light2.position.set(0, -2, 20)
// scene.add(light2)


// let light3 = new T.PointLight( 0xffffff, 1, 100)
// light3.position.set(0, 10, -10)
// scene.add(light3)
//
// let light4 = new T.PointLight( 0xffffff, 1, 100)
// light4.position.set(-10, 10, 10)
// scene.add(light4)
//
// let light2 = new T.PointLight( 0xffffff, 1, 100)
// light2.position.set(0, -10, 10)
// scene.add(light2)
**/

let vertices = {}
let edges = {}
// let edgeCollection = [] // TODO: Remove
let graphs = []
let names = {}
let linesDrawn = []
let linesCleared = true
let subPlanes = []
let current_edges = {}

// edgecolor_sec = 0x6decaf
edgecolor_sec = 0x2cc57c
edgecolor = 0x178e51
// lineWidth = 6 //
var lineMat = new T.LineBasicMaterial({color: edgecolor, linewidth: 6, clippingPlanes: [clipPlane] })
// var lineMatSec = new T.LineBasicMaterial({color: edgecolor, linewidth: 4, opacity: 0.3, transparent: true})
var lineMatSec = new T.LineBasicMaterial({color: edgecolor_sec, linewidth: 1.5, depthFunc: T.LessDepth})
// var matLine

var contourMeshLines = []
var contourCount = -1

// drawGrid(canv)



plane.geometry.dynamic = true

// Extended plane for wraparound distance calculation
var extendGeom = new T.PlaneGeometry(planeW, planeH*2, (divisions-1), (divisions*2)-1)
var ePlane = new T.Mesh( extendGeom, new THREE.MeshPhongMaterial( {color: 0x00ff00} ))
ePlane.rotation.set(-Math.PI/2, 0, 0.)
ePlane.position.z += 10
// scene.add( ePlane )

for (let face of plane.geometry.faces) {
  face.vertexColors[0] = new T.Color(0xffffff)
  face.vertexColors[1] = new T.Color(0xffffff)
  face.vertexColors[2] = new T.Color(0xffffff)
}

var renderPass = new RenderPass( scene, camera );

var contX = []
var contY = []
for (let i = 0 ; i < heightMap.length ; i++) {
  contX.push(i)
  contY.push(i)
}

var ref_used_cur = false
var ref_used = false

// Composers and FXAA shader
{
  var composer1, composer2, fxaaPass;

  fxaaPass = new ShaderPass( FXAAShader );
  fxaaPass.renderToScreen = false

  var pixelRatio = renderer.getPixelRatio();

  fxaaPass.material.uniforms[ 'resolution' ].value.x = 1 / ( window.innerWidth * pixelRatio );
  fxaaPass.material.uniforms[ 'resolution' ].value.y = 1 / ( window.innerHeight * pixelRatio );

  composer1 = new EffectComposer( renderer );
  composer1.addPass( renderPass );
  composer1.addPass( fxaaPass );

  //

  var copyPass = new ShaderPass( CopyShader );

  composer2 = new EffectComposer( renderer );
  composer2.addPass( renderPass );
  composer2.addPass( copyPass );
}


window.onload = function() {



  var mapCanvas = document.getElementById('map').getElementsByTagName('canvas')[0]

  // mapCanvas.style.transform = "rotate(90deg)"
  let ctx = mapCanvas.getContext('2d')
  // ctx.canvas.width = 2000;
  // ctx.canvas.height = 2000;
  // ctx.fillStyle = canvascolor;
  // ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const customTexture = texture
  const mapTexture = new T.CanvasTexture(ctx.canvas)
  mapTexture.minFilter = THREE.LinearFilter
  mapTexture.magFilter = THREE.NearestFilter
  mapTexture.center = new T.Vector2(0.5, 0.5)
  mapTexture.rotation = -Math.PI/2
  plane.material[2].map = mapTexture

  var mapdiv = document.getElementById("map")

  mapdiv.style.display = "none"

  // TODO: Stack for -ve zoom levels for consistency

  // TODO: enable
  window.addEventListener('wheel', wheelEvent, true)

  var dropNodes = document.getElementById('drop-nodes');
  dropNodes.addEventListener('dragover', dragOver, false);
  dropNodes.addEventListener('drop', fileSelectNodes, false);

  var dropEdges = document.getElementById('drop-edges');
  dropEdges.addEventListener('dragover', dragOver, false);
  dropEdges.addEventListener('drop', fileSelectEdges, false);

  let btnAddVertex = document.getElementById("btn-add-vertex")
  btnAddVertex.onclick = addVertex

  let btnAddEdge  = document.getElementById("btn-add-edge")
  btnAddEdge.onclick = addEdge

  document.getElementById("btn-refine").onclick = refine

  document.getElementById("btn-calc-dist").onclick = function() {


    if (subPlanes.length != 0)
      calcDistanceOnSurface(subPlanes[0].plane, vertices, current_edges)
    else
      calcDistanceOnSurface(ePlane, vertices, current_edges)
  }

  document.getElementById("heatmap-div").style.display = "none"

  let hideSurface = document.getElementById("hide-surface")
  // hideSurface.style.visibility = "hidden"
  document.getElementById("hide-surface-label").style.visibility = "hidden"
  let chkCalcSurface = document.getElementById("use-calc-surface")
  let useTransp = document.getElementById("use-transparency")
  let showMap = document.getElementById("show-map")
  let showGraph = document.getElementById("show-graph")

  showGraph.onchange = function() {
    if (vertices[0] != undefined) {
      for (let id in vertices) {
        if (!showGraph.checked) {
          scene.remove(vertices[id].mesh)
          scene.remove(vertices[id].label)
        } else {
          scene.add(vertices[id].mesh)
          scene.add(vertices[id].label)
        }
      }
    }
  }

  document.getElementById("threshold-slider").onchange = function() {
    thresh_change = true
  }


  let vertexControlDiv = document.getElementById("div-vertex")
  // vertexControlDiv.style.display = "none"

  let edgeControlDiv = document.getElementById("div-edge")
  // edgeControlDiv.style.display = "none"

  let btnGenGraph = document.getElementById("btn-gen-graph")
  btnGenGraph.onclick = generateGraph

  let btnGenGraphEmpty = document.getElementById("btn-gen-graph-empty")
  btnGenGraphEmpty.onclick = generateGraphNoWeights
  // btnGenGraphEmpty.style.display = "none"

  let btnCalcCurv = document.getElementById("btn-calc-curv")
  btnCalcCurv.onclick = calculateCurvature

  let btnHelp = document.getElementById("btn-help")
  btnHelp.onclick = helpClick

  document.getElementById("btn-cycle-thresholds").onclick = cycleThresholds


  document.getElementById("btn-calc-surface").onclick = calcSurface


  controls.enablePan = true
  controls.panSpeed = 1
  controls.enableRotate = true
  controls.enableZoom = true
  controls.minZoom = 1
  controls.update()



  var animate = function () {

    updateSliderVals()

    if (showMap.checked) {
      plane.material[0].map = mapTexture
      texture = mapTexture
      ctx = mapCanvas.getContext("2d")
    } else {
      plane.material[0].map = customTexture
      texture = customTexture
      ctx = canv.getContext("2d")
    }
    plane.material[0].needsUpdate = true
    plane.material[2].needsUpdate = true

    curvMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));

  	requestAnimationFrame( animate )

    controls.update()
    // controls.enabled = false

    // Update thresholds if cycling
    if (cycle && (Date.now()-last_cycle)/1000 > 2) {
      let slider = document.getElementById("threshold-slider")
      let value = parseInt(slider.value)
      value += 1
      value %= (parseInt(slider.max)+1)
      slider.value = value
      last_cycle = Date.now()
      thresh_change = true
    }



    // Clear lines, heights, reset textures
    // for (let line of linesDrawn) {
    //   scene.remove(line)
    //   line = null
    // }
    // linesDrawn = []
    // linesCleared = true



    ctx.fillStyle = canvascolor

    // if (!showMap.checked) {
    //   ctx.drawImage(background,0,0,697,998,250,-10,1650,2080) // 696x995
    // }




    let viewSeperate = false
    // let vertices_visual = vertices, edges_visual = edges
    // if (viewSeperate) {
    //   vertices_visual = vertices2
    //   edges_visual = edges2
    // }
    current_edges = {...edges}
    if (graphs.length > 0)
      current_edges = {...graphs[document.getElementById("threshold-slider").value].edges}

    // let logical_edges = [[1, 6, null, null], [4, 7, 1, -0.5], [0, 8, 2, -0.8], [2, 8, 2, -0.7], [2, 5, 0.5, -0.8]]
    // let logical_edges = [[1, 6, null, null], [4, 7, 1.2, -0.3], [0, 8, 1.6, -0.3], [2, 8, 1.6, -0.3], [2, 5, 1.2, -0.3]]
    // let logical_edges = [[1, 6, null, null], [4, 7, 0.2, -0.3], [0, 8, 0.6, -0.3], [2, 8, 0.6, -0.3], [2, 5, 0.2, -0.3]]
    let logical_edges = []

    if (!showMap.checked) {
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    }


    // Draws grid lines
    // drawGrid(ctx.canvas)



    ctx.setLineDash([])

    // Draw physical graph edge, texture edge
    for (let id in current_edges) {
      let lineWidth =  2  //12 / 6 / 2 /
      let borders = true
      let edge = current_edges[id]
      // console.log(`${edge.start.lat}, ${edge.start.long}, ${edge.end.lat}, ${edge.end.long}`)
      // console.log(edge.start)
      // if (edge.start.name != "Tokyo")
      //   continue
      // Draw graph edge
      if (edge.split) {
        // Create split edges
        let eSize = Object.keys(current_edges).length
        let end1 = {name: edge.end.name, mesh: {position: {x: parseFloat(edge.startSplit[0]*10/155), z: parseFloat(edge.startSplit[1]*10/180)}}}
        let edge1 = new EdgeObj(eSize, edge.start, end1, edge.weight)
        if (showGraph.checked)
          drawEdge(edge1, lineMat)
        let startPt = [parseFloat(edge1.start.mesh.position.x), parseFloat(edge1.start.mesh.position.z)]
        let endPt = [parseFloat(edge1.end.mesh.position.x), parseFloat(edge1.end.mesh.position.z)]

        startPt = [(1 - (startPt[0] - planeXMin) / planeW) * ctx.canvas.width, (startPt[1] - planeYMin) * ctx.canvas.height / planeH]
        endPt = [(1 - (endPt[0] - planeXMin) / planeW) * ctx.canvas.width, (endPt[1] - planeYMin) * ctx.canvas.height / planeH]
        ctx.save()
        ctx.globalAlpha = 1
        ctx.beginPath();

        if (borders) {
          ctx.moveTo(startPt[1], startPt[0])
          ctx.lineTo(endPt[1], endPt[0])
          ctx.strokeStyle = "#000000" // #2cacc9 // #40bad5
          ctx.lineWidth = lineWidth+1
          ctx.stroke()
        }

        ctx.moveTo(startPt[1], startPt[0])
        ctx.lineTo(endPt[1], endPt[0])
        let color = new T.Color()
        if (edge.weight >= 0)
          var endColor = new T.Color("hsl(145, 98%, 40%)") // "hsl(145, 98%, 40%)"
        else
          var endColor = new T.Color("hsl(0, 76%, 43%)") // "hsl(0, 76%, 43%)"
        color.lerpHSL(endColor, Math.min(Math.abs(edge.weight), 1))
        ctx.strokeStyle = "#" + color.getHexString() // #2cacc9 // #40bad5
        ctx.lineWidth = lineWidth
        ctx.stroke()
        ctx.restore()

        let start2 = {name: edge.start.name, mesh: {position: {x: parseFloat(edge.endSplit[0]*10/155), z: parseFloat(edge.endSplit[1]*10/180)}}}
        let edge2 = new EdgeObj(eSize+1, start2, edge.end, edge.weight)
        if (showGraph.checked)
          drawEdge(edge2, lineMat)
        startPt = [parseFloat(edge2.start.mesh.position.x), parseFloat(edge2.start.mesh.position.z)]
        endPt = [parseFloat(edge2.end.mesh.position.x), parseFloat(edge2.end.mesh.position.z)]

        startPt = [(1 - (startPt[0] - planeXMin) / planeW) * ctx.canvas.width, (startPt[1] - planeYMin) * ctx.canvas.height / planeH]
        endPt = [(1 - (endPt[0] - planeXMin) / planeW) * ctx.canvas.width, (endPt[1] - planeYMin) * ctx.canvas.height / planeH]
        ctx.save()
        ctx.globalAlpha = 1
        ctx.beginPath();

        if (borders) {
          ctx.moveTo(startPt[1], startPt[0])
          ctx.lineTo(endPt[1], endPt[0])
          ctx.strokeStyle = "#000000" // #2cacc9 // #40bad5
          ctx.lineWidth = lineWidth+1
          ctx.stroke()
        }
        ctx.moveTo(startPt[1], startPt[0])
        ctx.lineTo(endPt[1], endPt[0])
        color = new T.Color()
        if (edge.weight >= 0)
          var endColor = new T.Color("hsl(145, 98%, 40%)")
        else
          var endColor = new T.Color("hsl(0, 76%, 43%)")
        color.lerpHSL(endColor, Math.min(Math.abs(edge.weight), 1))
        ctx.strokeStyle = "#" + color.getHexString() // #2cacc9 // #40bad5
        ctx.lineWidth = lineWidth
        ctx.stroke()
        ctx.restore()
        current_edges[eSize] = edge1
        current_edges[eSize+1] = edge2

        continue

        // Add split edge to current edges
        // Skip split edges in setHeights
      }
      if (showGraph.checked && (edge.mesh == null || linesCleared))
        edge.mesh = drawEdge(edge, lineMat)

      let startPt = [parseFloat(edge.start.mesh.position.x), parseFloat(edge.start.mesh.position.z)]
      let endPt = [parseFloat(edge.end.mesh.position.x), parseFloat(edge.end.mesh.position.z)]
      // if ((edge.start.name[0] == "F" || edge.end.name[0] == "F") && edge.weight >= 0)
      //   continue
      // if ((edge.start.name[0] == "N" && edge.end.name[0] == "M") || (edge.start.name[0] == "M" && edge.end.name[0] == "N"))
      //   continue

      // Draw texture edge // TODO: undo
      startPt = [(1 - (startPt[0] - planeXMin) / planeW) * ctx.canvas.width, (startPt[1] - planeYMin) * ctx.canvas.height / planeH]
      endPt = [(1 - (endPt[0] - planeXMin) / planeW) * ctx.canvas.width, (endPt[1] - planeYMin) * ctx.canvas.height / planeH]
      ctx.save()
      ctx.globalAlpha = 1.0
      // ctx.globalCompositeOperation = "color-dodge";
      ctx.beginPath();

      if (borders) {
        ctx.moveTo(startPt[1], startPt[0])
        ctx.lineTo(endPt[1], endPt[0])
        ctx.strokeStyle = "#000000" // #2cacc9 // #40bad5
        ctx.lineWidth = lineWidth+1
        ctx.stroke()
      }

      ctx.moveTo(startPt[1], startPt[0])
      ctx.lineTo(endPt[1], endPt[0])
      let color = new T.Color()
      if (edge.weight >= 0)
        var endColor = new T.Color("hsl(222, 100%, 61%)")
      else
        var endColor = new T.Color("hsl(356, 74%, 52%)")
      color.lerp(endColor, Math.min(Math.abs(edge.weight), 1))
      ctx.strokeStyle = "#" + color.getHexString() // #2cacc9 // #40bad5
      ctx.lineWidth = lineWidth
      ctx.stroke()
      ctx.restore()
    }

    linesCleared = false

    // Draw logical edges into graph, Draw logical edges into texture
    for (let ids of logical_edges) {
      // for (let id2 in vertices_visual) {
      //   if (id < id2) {
      //     continue
      //   }
      let id = ids[0]
      let id2 = ids[1]

      let startPt = [parseFloat(vertices_visual[id].mesh.position.x), parseFloat(vertices_visual[id].mesh.position.z)]
      let endPt = [parseFloat(vertices_visual[id2].mesh.position.x), parseFloat(vertices_visual[id2].mesh.position.z)]

      startPt = [(startPt[0] - planeXMin) * ctx.canvas.width / planeW, (startPt[1] - planeYMin) * ctx.canvas.height / planeH]
      endPt = [(endPt[0] - planeXMin) * ctx.canvas.width / planeW, (endPt[1] - planeYMin) * ctx.canvas.height / planeH]

      ctx.setLineDash([])
      ctx.beginPath();
      ctx.moveTo(startPt[0], startPt[1])
      if (ids[2] == null) {
        ctx.lineTo(endPt[0], endPt[1])
        ctx.strokeStyle = "#2cacc9" //  #235789 // #68c8de // #5ecfe2  // #9f9f9f
        ctx.lineWidth = 4
        ctx.stroke()
      } else {
        let ctrlPt = [ids[2], ids[3]]
        ctrlPt = [(ctrlPt[0] - planeXMin) * ctx.canvas.width / planeW, (ctrlPt[1] - planeYMin) * ctx.canvas.height / planeH]
        // ctx.quadraticCurveTo(ctrlPt[0], ctrlPt[1], endPt[0], endPt[1])
        ctx.bezierCurveTo(ctrlPt[0], ctrlPt[1], ctrlPt[0], ctrlPt[1], endPt[0], endPt[1])

        ctx.strokeStyle = "#2cacc9" //  #235789 // #68c8de // #5ecfe2  // #9f9f9f
        ctx.lineWidth = 4
        ctx.stroke()

        // ctjcmathews2 ;
      }





      drawEdge(new EdgeObj(null, vertices_visual[id], vertices_visual[id2], null), lineMatSec)
      // }
    }

    // Set plane vertices' height
    heightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.))


    // Set height map for +ve edges
    for (let id in current_edges) {
      let edge = current_edges[id]
      if (edge.weight < 0 || edge.split)
        continue

      let startPt = [parseFloat(edge.start.mesh.position.x), parseFloat(edge.start.mesh.position.z)]
      let endPt = [parseFloat(edge.end.mesh.position.x), parseFloat(edge.end.mesh.position.z)]

      let midPt = [(startPt[0] + endPt[0]) / 2, (startPt[1] + endPt[1]) / 2]
      midPt[0] = (midPt[0] - planeXMin)// Change from (min,max) to (0, newmax)
      midPt[1] = (midPt[1] - planeYMin)// Change from (min,max) to (0, newmax)

      midPt[0] = Math.round((midPt[0] / planeW) * divisions) // Change from (0, planeWidth) to (0, divisions)
      midPt[1] = Math.round((midPt[1] / planeH) * divisions) // Change from (0, planeHeight) to (0, divisions)

      let newMidPt = {x: 0, y: 0}
      newMidPt.x = midPt[0]
      newMidPt.y = midPt[1]

      let newEndPt = {x: 0, y: 0}
      newEndPt.x = endPt[0]
      newEndPt.y = endPt[1]
      newEndPt.x = (newEndPt.x - planeXMin)// Change from (min,max) to (0, newmax)
      newEndPt.y = (newEndPt.y - planeYMin)// Change from (min,max) to (0, newmax)

      newEndPt.x = Math.round((newEndPt.x / planeW) * divisions) // Change from (0, planeWidth) to (0, divisions)
      newEndPt.y = Math.round((newEndPt.y / planeH) * divisions) // Change from (0, planeHeight) to (0, divisions)

      let newStartPt = {x: 0, y: 0}
      newStartPt.x = startPt[0]
      newStartPt.y = startPt[1]
      newStartPt.x = (newStartPt.x - planeXMin)// Change from (min,max) to (0, newmax)
      newStartPt.y = (newStartPt.y - planeYMin)// Change from (min,max) to (0, newmax)

      newStartPt.x = Math.round((newStartPt.x / planeW) * divisions) // Change from (0, planeWidth) to (0, divisions)
      newStartPt.y = Math.round((newStartPt.y / planeH) * divisions) // Change from (0, planeHeight) to (0, divisions)


      // Set heightmap
      if (!chkCalcSurface.checked) {
        setHeights(newStartPt, newMidPt, newEndPt, edge.weight, heightMap)
      }
    }


    //TODO: change two back
    // console.log(negative_edges)

    // Set height map for -ve edges
    for (let id in current_edges) {
      let edge = current_edges[id]
      if (edge.weight >= 0 || edge.split)
        continue

      let startname = Math.min(edge.start.name, edge.end.name)
      let endname = Math.max(edge.start.name, edge.end.name)
      if (isNaN(startname)) {
        if (edge.start.name.localeCompare(edge.end.name) < 0) {
          startname = edge.start.name
          endname = edge.end.name
        } else {
          startname = edge.end.name
          endname = edge.start.name
        }
      }

      // console.log(startname, endname)
      let neg_mod = edge.neg_mod
      let nrw_mod = edge.nrw_mod
      let nheight_mod = edge.nheigt_mod
      if (!ref_used && refine_data[startname] && refine_data[startname][endname]) {
        ref_used_cur = true
        let ref = refine_data[startname][endname]
        if (ref > 0) {
          neg_mod = 1.2*neg_mod
          nrw_mod = 1.2*nrw_mod
          nheight_mod = 1.2*nheight_mod
          console.log("refine")
        } else if (ref < -1) {
          neg_mod = 0.8*neg_mod
          nrw_mod = 0.8*nrw_mod
          console.log("refine")
        }
        edge.neg_mod = neg_mod
        edge.nrw_mod = nrw_mod
        edge.nheight_mod = nheight_mod
      }
      let startPt = [parseFloat(edge.start.mesh.position.x), parseFloat(edge.start.mesh.position.z)]
      let endPt = [parseFloat(edge.end.mesh.position.x), parseFloat(edge.end.mesh.position.z)]

      let midPt = [(startPt[0] + endPt[0]) / 2, (startPt[1] + endPt[1]) / 2]
      midPt[0] = (midPt[0] - planeXMin)// Change from (min,max) to (0, newmax)
      midPt[1] = (midPt[1] - planeYMin)// Change from (min,max) to (0, newmax)

      midPt[0] = Math.round((midPt[0] / planeW) * divisions) // Change from (0, planeWidth) to (0, divisions)
      midPt[1] = Math.round((midPt[1] / planeH) * divisions) // Change from (0, planeHeight) to (0, divisions)

      let newMidPt = {x: 0, y: 0}
      newMidPt.x = midPt[0]
      newMidPt.y = midPt[1]

      let newEndPt = {x: 0, y: 0}
      newEndPt.x = endPt[0]
      newEndPt.y = endPt[1]
      newEndPt.x = (newEndPt.x - planeXMin)// Change from (min,max) to (0, newmax)
      newEndPt.y = (newEndPt.y - planeYMin)// Change from (min,max) to (0, newmax)

      newEndPt.x = Math.round((newEndPt.x / planeW) * divisions) // Change from (0, planeWidth) to (0, divisions)
      newEndPt.y = Math.round((newEndPt.y / planeH) * divisions) // Change from (0, planeHeight) to (0, divisions)

      let newStartPt = {x: 0, y: 0}
      newStartPt.x = startPt[0]
      newStartPt.y = startPt[1]
      newStartPt.x = (newStartPt.x - planeXMin)// Change from (min,max) to (0, newmax)
      newStartPt.y = (newStartPt.y - planeYMin)// Change from (min,max) to (0, newmax)

      newStartPt.x = Math.round((newStartPt.x / planeW) * divisions) // Change from (0, planeWidth) to (0, divisions)
      newStartPt.y = Math.round((newStartPt.y / planeH) * divisions) // Change from (0, planeHeight) to (0, divisions)

      if (!chkCalcSurface.checked) {
        setHeights(newStartPt, newMidPt, newEndPt, edge.weight, heightMap, 1, neg_mod, nrw_mod, nheight_mod)
      }
    }
    if (ref_used_cur)
      ref_used = true
    // refine_data = []



    // smoothHeightMap(heightMap)
    // smoothHeightMap(heightMap)
    smoothHeightMap(heightMap)
    smoothHeightMap(heightMap)

    // TODO: doesn't work / use BufferGeom?


    // Draw point on surface texture
    for (let id in vertices) {
      let radius = 2 // 5 / 3
      let vertex = vertices[id]
      let point = [parseFloat(vertex.mesh.position.x), parseFloat(vertex.mesh.position.z)]
      point = [(1 - (point[0] - planeXMin) / planeW) * ctx.canvas.width, (point[1] - planeYMin) * ctx.canvas.height / planeH]
      ctx.fillStyle = "#FF5C5C" // #035aa6

      ctx.beginPath();
      ctx.arc(point[1], point[0], radius, 0, 2 * Math.PI);
      ctx.fill();
    }


    let map = heightMap
    let useTransp = document.getElementById("use-transparency")


    if (document.getElementById("use-calc-surface").checked)
      if (graphs.length > 0)
        map = graphs[document.getElementById("threshold-slider").value].heightmap
      else
        map = calcHeightMap
    if (thresh_change) {
      console.log("threshold changed")
      thresh_change = false
      updatePlaneHeights(map)
    }

    if (map == heightMap)
      updatePlaneHeights(map)

    if (showMap.checked) {

      opacityMap = Array(100).fill().map(() => Array(100).fill(0.0));

      calcOpacityMap(opacityMap, vertices, current_edges, map)
      aMap = createAndUpdateAlphaMapD3(opacityMap)


      // Createadn set alphaMap image for transparency from opacityMap
      // aMap = createAlphaMap(opacityMap)
      plane.material[0].alphaMap = aMap
    }

    for (let p of subPlanes) {
      for (let i=p.start[0]; i<p.end[0] ; i++) {
        for (let j=p.start[1]; j<p.end[1] ; j++) {
          plane.geometry.vertices[j*divisions+i].z = 0
        }
      }

      // COlor no hieght surface black

      // for (let face of p.plane.geometry.faces) {
      //   const black = new THREE.Color(0x000000) // 0xef626c
      //   let z1 = Math.abs(p.plane.geometry.vertices[face.a].z) < 0.001
      //   let z2 = Math.abs(p.plane.geometry.vertices[face.b].z) < 0.001
      //   let z3 = Math.abs(p.plane.geometry.vertices[face.c].z) < 0.001
      //   // console.log(z1)
      //   // console.log(z2)
      //   // console.log(z3)
      //
      //   // if (face.vertexColors[0] == undefined) {
      //   //   face.vertexColors[0] = new THREE.Color( 1, 1, 1 );
      //   //   face.vertexColors[1] = new THREE.Color( 1, 1, 1 );
      //   //   face.vertexColors[2] = new THREE.Color( 1, 1, 1 );
      //   // }
      //
      //   if (z1 && z2 && z3) {
      //     face.vertexColors[0] = new THREE.Color( 0, 0, 0 );
      //     face.vertexColors[1] = new THREE.Color( 0, 0, 0 );
      //     face.vertexColors[2] = new THREE.Color( 0, 0, 0 );
      //   }
      // }
      // if (showMap.checked) {
        // p.plane.material.map = mapTexture
      // } else {
      //   p.plane.material.map = customTexture
      // }
      // colorCurvature(p.plane)
      p.plane.material.alphaMap = aMap
      p.plane.material.map.needsUpdate = true
      p.plane.geometry.colorsNeedUpdate = true

    }


    // colorCurvature(plane)

    plane.material[0].needsUpdate = true
    texture.needsUpdate = true




    // Set materials for plane faces, to hide unwanted





    contourCount++

    // CONTOURS
    // if (chkCalcSurface.checked)
    //   if (graphs.length > 0)
    //     calcContours(100, 100, graphs[document.getElementById("threshold-slider").value].heightmap)
    //   else
    //     calcContours(100, 100, calcHeightMap)
    // else
    //   calcContours(100, 100, heightmap)

    plane.geometry.groupsNeedUpdate = true
    plane.geometry.verticesNeedUpdate = true
    plane.geometry.colorsNeedUpdate = true
    plane.geometry.computeVertexNormals()


    // Render
    renderer.localClippingEnabled = hideSurface.checked





    olMap.render()

    // Set up opacity map for hiding surface
    // calcCurvMap(curvMap, vertices, current_edges)
    // updateShading(curvMap)



    // matLine.resolution.set( window.innerWidth, window.innerHeight );
    // renderer.setViewport( 0, 0, window.innerWidth/2, window.innerHeight );
    composer1.render()
    // renderer.render( scene, camera )
    // renderer.setViewport( window.innerWidth/2, 0, window.innerWidth/2, window.innerHeight );
    // composer2.render()

  };

  animate();
}

var selectionBox = new SelectionBox( camera, scene )
var helper = new SelectionHelper( selectionBox, renderer, 'selectBox' )

document.addEventListener("keydown", function (event) { // Shift click for select
  if (event.shiftKey) {
    controls.enablePan = false
    controls.update()
    document.body.style.cursor = "crosshair"
    document.addEventListener( 'pointerdown', pointerDown )
    document.addEventListener( 'pointermove', pointerMove )
    document.addEventListener( 'pointerup', pointerUp )
    console.log("select")
  } else if (event.which == 27) {
    console.log("clear selected")
    for (let i in subPlanes) {
      var sPlane = subPlanes[i].plane
      gsap.to( sPlane.scale, {
            duration: 1,
            x: 0.1,
            y: 0.1,
            z: 0.1,
            onComplete: function() {
              scene.remove(sPlane)
            }
      })
    }
    // TODO: zoom widthd, heights, levels reset

    gsap.to( camera, {
  				duration: 1,
  				zoom: 1,
  				onUpdate: function () {
  					camera.updateProjectionMatrix();
  				}
  	})
    gsap.to( controls.target, {
  				duration: 1,
  				x: 0,
  				y: 0,
  				z: 0,
  				onUpdate: function () {
  					controls.update();
  				}
  	})
    gsap.to( plane.position, {
  				duration: 1,
  				y: 0,
          onStart: function() {
            plane.visible = true
          },
  				onUpdate: function () {
  				}
  	})

    olMap.getView().setZoom(0)
    subPlanes = []
  }
})

document.addEventListener("keyup", function(event) {
  if (event.which == 16) {
    document.body.style.cursor = "auto"
    controls.enablePan = true
    controls.update()
    document.removeEventListener( 'pointerdown', pointerDown )
    document.removeEventListener( 'pointermove', pointerMove )
    document.removeEventListener( 'pointerup', pointerUp )
  }
})

function refine() {
  console.log("refine")
  var xmlHttp = new XMLHttpRequest();
  // xmlHttp.responseType = "arraybuffer"
  xmlHttp.responseType = "text"

  xmlHttp.onreadystatechange = function()
  {
      if(xmlHttp.readyState == 4 && xmlHttp.status == 200)

      {
          let data = xmlHttp.responseText
          // data = data.substring(data.indexOf('['))
          data = JSON.parse(data)
          console.log("data recv")
          console.log(data)
          refine_data = data
      }
  }
  xmlHttp.open("get", "refine");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xmlHttp.send();
  console.log("refine get sent")
}

function updateSliderVals() {
  document.getElementById("xspread-slider-val").innerHTML = parseFloat(document.getElementById("xspread-slider").value).toFixed(2)
  document.getElementById("yspread-slider-val").innerHTML = parseFloat(document.getElementById("yspread-slider").value).toFixed(2)
  document.getElementById("xlimit-slider-val").innerHTML = parseFloat(document.getElementById("xlimit-slider").value).toFixed(2)
  document.getElementById("ylimit-slider-val").innerHTML = parseFloat(document.getElementById("ylimit-slider").value).toFixed(2)
  document.getElementById("height-slider-val").innerHTML = parseFloat(document.getElementById("height-slider").value).toFixed(2)
  document.getElementById("posrange-slider-val").innerHTML = parseFloat(document.getElementById("posrange-slider").value).toFixed(2)
  document.getElementById("posheight-slider-val").innerHTML = parseFloat(document.getElementById("posheight-slider").value).toFixed(2)
  document.getElementById("amp-slider-val").innerHTML = parseFloat(document.getElementById("amp-slider").value).toFixed(2)
  document.getElementById("rotation-slider-val").innerHTML = parseFloat(document.getElementById("rotation-slider").value).toFixed(2)

}

function cycleThresholds() {
  let btnCycle = document.getElementById("btn-cycle-thresholds")
  cycle = !cycle
  if (!cycle) {
    btnCycle.innerHTML = "Cycle Thresholds"
  } else {
    btnCycle.innerHTML = "Stop Cycle"
  }
  last_cycle = Date.now()

}

function colorCurvature(plane) {
  // CURVATURE calc
  let max_curv = 0
  let min_curv = 1
  for (let i = 9 ; i < 39 ; i++) {
    for (let j = 9 ; j < 39 ; j++) {
      // console.log(plane.geometry.vertices[i].x + " " + plane.geometry.vertices[i].y)
      let o = [plane.geometry.vertices[i*50 + j].x, plane.geometry.vertices[i*50 + j].y, plane.geometry.vertices[i*50 + j].z]
      let a = [plane.geometry.vertices[(i-1)*50 + j-1].x, plane.geometry.vertices[(i-1)*50 + j-1].y, plane.geometry.vertices[(i-1)*50 + j-1].z]
      let b = [plane.geometry.vertices[(i-1)*50 + j].x, plane.geometry.vertices[(i-1)*50 + j].y, plane.geometry.vertices[(i-1)*50 + j].z]
      let c = [plane.geometry.vertices[i*50 + j-1].x, plane.geometry.vertices[i*50 + j-1].y, plane.geometry.vertices[i*50 + j-1].z]
      let d = [plane.geometry.vertices[i*50 + j+1].x, plane.geometry.vertices[i*50 + j+1].y, plane.geometry.vertices[i*50 + j+1].z]
      let e = [plane.geometry.vertices[(i+1)*50 + j].x, plane.geometry.vertices[(i+1)*50 + j].y, plane.geometry.vertices[(i+1)*50 + j].z]
      let f = [plane.geometry.vertices[(i+1)*50 + j+1].x, plane.geometry.vertices[(i+1)*50 + j+1].y, plane.geometry.vertices[(i+1)*50 + j+1].z]
      let vxs = [a, b, d, f, e, c]
      let angles = []

      for (let k = 0 ; k < vxs.length; k++) {
        let posB = vxs[k]
        let posC = vxs[(k+1)%vxs.length]
        let vec1 = [posB[0] - o[0], posB[1] - o[1], posB[2] - o[2]]
        let vec2 = [posC[0] - o[0], posC[1] - o[1], posC[2] - o[2]]
        let dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
        let norm_v1 = Math.sqrt(Math.pow(vec1[0], 2) + Math.pow(vec1[1], 2) + Math.pow(vec1[2], 2))
        let norm_v2 = Math.sqrt(Math.pow(vec2[0], 2) + Math.pow(vec2[1], 2) + Math.pow(vec2[2], 2))
        let angle1 = Math.acos(dot/(norm_v1*norm_v2))
        angles.push(angle1)
      }
      let curvature = 2*Math.PI - angles.reduce((a,b) => a+b)
      plane.geometry.vertices[i*50 + j].curvature = curvature
      max_curv = Math.max(max_curv, curvature)
      min_curv = Math.min(min_curv, curvature)

      // if (Math.random() > 0.99999) {
      //   console.log(curvature)
      //   console.log(plane.geometry.vertices[i*50 + j])
      // }
      // let face = plane.geometry.faces[i*50 + Math.floor(j/2)]
      // face.vertexColors[0].setHSL(Math.random(), 0.5, 0.5)
      // face.vertexColors[1].setHSL(Math.random(), 0.5, 0.5)
      // face.vertexColors[2].setHSL(Math.random(), 0.5, 0.5)
    }
  }

  for (let face of plane.geometry.faces) {
    if (Math.random() > 0.99999) {
      // console.log(min_curv, max_curv)
      // if (plane.geometry.vertices[face.a].curvature != 0)
        // console.log(plane.geometry.vertices[face.a].curvature)
    }
    const red = new THREE.Color(0xdf2935) // 0xef626c
    const blue = new THREE.Color(0x3772ff) // 0x118ab2
    let curv = plane.geometry.vertices[face.a].curvature
    let scale = 2
    if (face.vertexColors[0] == undefined) {
      face.vertexColors[0] = new THREE.Color( 1, 1, 1 );
      face.vertexColors[1] = new THREE.Color( 1, 1, 1 );
      face.vertexColors[2] = new THREE.Color( 1, 1, 1 );
    }
    face.vertexColors[0].setRGB( 1, 1, 1);
    if (curv > 0.01) {
      face.vertexColors[0].lerp(blue, Math.ceil(scale*curv/max_curv)/scale)
    }
    else if (curv < -0.01)
      face.vertexColors[0].lerp(red, Math.ceil(scale*curv/min_curv)/scale)

    curv = plane.geometry.vertices[face.b].curvature
    face.vertexColors[1].setRGB( 1, 1, 1);
    if (curv > 0.01)
      face.vertexColors[1].lerp(blue, Math.ceil(scale*curv/max_curv)/scale)
    else if (curv < -0.01)
      face.vertexColors[1].lerp(red, Math.ceil(scale*curv/min_curv)/scale)

    curv = plane.geometry.vertices[face.c].curvature
    face.vertexColors[2].setRGB( 1, 1, 1);
    if (curv > 0.01)
      face.vertexColors[2].lerp(blue, Math.ceil(scale*curv/max_curv)/scale)
    else if (curv < -0.01)
      face.vertexColors[2].lerp(red, Math.ceil(scale*curv/min_curv)/scale)
  }
}

function drawGrid(canvas) {
  let ctx = canvas.getContext("2d")
  let cols = 49
  let rows = 49
  let width = canvas.width //canvas.width
  let height = canvas.height
  let start_i = 0, start_j = 0
  // 925 2198 725 475
  if (subPlanes.length > 0) {
    let sp = subPlanes[0]

    start_i = (sp.ystart3JS+10)/20*canvas.width
    width = (sp.ystart3JS + sp.height+10)/20*canvas.width
    start_j = (-sp.xstart3JS+10)/20*canvas.height
    height = (-sp.xstart3JS-sp.width+10)/20*canvas.height
    if (height < start_j) {
      let temp = height
      height = start_j
      start_j = temp
    }
    // console.log(sp)
    // console.log(start_i, width, start_j, height)
  }
  ctx.beginPath()
  for (let i = start_i; i <= width; i += (width-start_i)/cols) {
    ctx.moveTo(i, 0)
    ctx.lineTo(i, canvas.height)
    // console.log(i)
  }
  for (let j = start_j; j <= height ; j += (height-start_j)/rows) {
    ctx.moveTo(0, j)
    ctx.lineTo(canvas.width, j)
  }
  ctx.lineWidth = 4 // 2
  ctx.strokyStyle = "black"
  ctx.stroke()
}

function updatePlaneHeights(map) {
  let useTransp = document.getElementById("use-transparency")

  let ex = 0.3
  let direction = new T.Vector3(0, 1, 0)
  for (let i=0; i<divisions ; i++) {
    for (let j=0; j < divisions ; j++) {
      if (i < 2) {
        // plane.geometry.vertices[i*divisions+j].z =  map[3][j]
        gsap.to(plane.geometry.vertices[i*divisions+j],
          { duration: 0.25,
            z: map[3][j],
          }
        )
      } else if (i >= divisions-2) {
        // plane.geometry.vertices[i*divisions+j].z =  map[divisions-3][j]
        gsap.to(plane.geometry.vertices[i*divisions+j],
          { duration: 0.25,
            z: map[divisions-3][j],
          }
        )
      } else {
        // plane.geometry.vertices[i*divisions+j].z =  0
        gsap.to(plane.geometry.vertices[i*divisions+j],
          { duration: 0.25,
            z: map[i][j],
          }
        )
      }
    }
  }

  // Set extended plane


  for (let i=0; i<divisions ; i++) {
    for (let j=0; j < divisions ; j++) {
      if (i < 2) {
        ePlane.geometry.vertices[(i+divisions)*divisions+j].z = map[3][j]
        ePlane.geometry.vertices[i*divisions+j*divisions].z = map[3][j]
      } else if (i >= divisions-2) {
        ePlane.geometry.vertices[(i+divisions)*divisions+j].z = map[divisions-3][j]
        ePlane.geometry.vertices[i*divisions+j].z = map[divisions-3][j]
      } else {
        ePlane.geometry.vertices[(i+divisions)*divisions+j].z = map[i][j]
        ePlane.geometry.vertices[i*divisions+j].z =  map[i][j]
      }
    }
  }

  ePlane.geometry.groupsNeedUpdate = true
  ePlane.geometry.verticesNeedUpdate = true
  ePlane.geometry.colorsNeedUpdate = true
  ePlane.geometry.computeVertexNormals()
  // for (let i = 0 ; i < divisions*divisions - 2 ; i++) {
  //   gsap.to(plane.geometry.vertices[i],
  //     { duration: 10,
  //       z: 10+Math.random()*10,
  //       // onUpdate: function() {
  //       // //   plane.geometry.groupsNeedUpdate = true
  //       //   plane.geometry.verticesNeedUpdate = true
  //       //   // console.log("increase height" + plane.geometry.vertices[i].z)
  //       // //   plane.geometry.colorsNeedUpdate = true
  //       // //   plane.geometry.computeVertexNormals()
  //       // }
  //     }
  //   )
  // }



  // for (let i = 0 ; i < plane.geometry.faces.length ; i++) {
  //   let face = plane.geometry.faces[i]
  //   let posA = [plane.geometry.vertices[face.a].x, plane.geometry.vertices[face.a].y, plane.geometry.vertices[face.a].z]
  //   let posB = [plane.geometry.vertices[face.b].x, plane.geometry.vertices[face.b].y, plane.geometry.vertices[face.b].z]
  //   let posC = [plane.geometry.vertices[face.c].x, plane.geometry.vertices[face.c].y, plane.geometry.vertices[face.c].z]
  //
  //   let vec1 = [posB[0] - posA[0], posB[1] - posA[1], posB[2] - posA[2]]
  //   let vec2 = [posC[0] - posA[0], posC[1] - posA[1], posC[2] - posA[2]]
  //   let dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
  //   let norm_v1 = Math.sqrt(Math.pow(vec1[0], 2) + Math.pow(vec1[1], 2) + Math.pow(vec1[2], 2))
  //   let norm_v2 = Math.sqrt(Math.pow(vec2[0], 2) + Math.pow(vec2[1], 2) + Math.pow(vec2[2], 2))
  //   let angle1 = Math.acos(dot/norm_v1*norm_v2)
  //
  //   vec1 = [-1*vec1[0], -1*vec1[1], -1*vec1[2]]
  //   vec2 = [posC[0] - posB[0], posC[1] - posB[1], posC[2] - posB[2]]
  //   dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
  //   norm_v1 = Math.sqrt(Math.pow(vec1[0], 2) + Math.pow(vec1[1], 2) + Math.pow(vec1[2], 2))
  //   norm_v2 = Math.sqrt(Math.pow(vec2[0], 2) + Math.pow(vec2[1], 2) + Math.pow(vec2[2], 2))
  //   let angle2 = Math.acos(dot/norm_v1*norm_v2)
  //
  //   vec1 = [-1*vec2[0], -1*vec2[1], -1*vec2[2]]
  //   vec2 = [posA[0] - posC[0], posA[1] - posC[1], posA[2] - posC[2]]
  //   dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
  //   norm_v1 = Math.sqrt(Math.pow(vec1[0], 2) + Math.pow(vec1[1], 2) + Math.pow(vec1[2], 2))
  //   norm_v2 = Math.sqrt(Math.pow(vec2[0], 2) + Math.pow(vec2[1], 2) + Math.pow(vec2[2], 2))
  //   let angle3 = Math.acos(dot/norm_v1*norm_v2)
  //
  //   if (Math.random() > 0.99999) {
  //     console.log(angle1)
  //     console.log(angle2)
  //     console.log(angle3)
  //     console.log("c = " + (Math.PI - angle1 - angle2 - angle3))
  //   }
  // }


  for (let face of plane.geometry.faces) {
    let z1 = plane.geometry.vertices[face.a].z
    let z2 = plane.geometry.vertices[face.b].z
    let z3 = plane.geometry.vertices[face.c].z
    let hide = false
    let v = face.a
    let i = Math.floor(v/divisions)
    let j = v%divisions

    if (Math.random() > 0.99999) {
      // if (plane.geometry.vertices[face.a].curvature != 0)
        // console.log(plane.geometry.vertices[face.a].curvature)
    }
    // if (z1 > 1)
    //   face.vertexColors[0].setHSL( 1, 0, 0.5);
    // if (z2 > 1)
    //   face.vertexColors[1].setHSL( 1, 0, 0.5);
    // if (z3 > 1)
    //   face.vertexColors[2].setHSL( 1, 0, 0.5);
    // face.vertexColors[0].setHSL( 1, 0, Math.floor(z1*10)/10);
    // face.vertexColors[1].setHSL( 1, 0, Math.floor(z2*10)/10);
    // face.vertexColors[2].setHSL( 1, 0, Math.floor(z3*10)/10);


    // face.vertexColors[0].setHSL(Math.random(), 0.5, 0.5)
    // face.vertexColors[1].setHSL(Math.random(), 0.5, 0.5)
    // face.vertexColors[2].setHSL(Math.random(), 0.5, 0.5)

    // face.vertexColors[0] = new T.Color( 0xff00ff )
    // if ((i < xlimit || i > heightMap.length - xlimit) || (j < ylimit || j > heightMap[0].length - ylimit))
    //   hide = true
    let transparent = true
    let points = []
    points.push([Math.floor(i), Math.floor(j)])
    points.push([Math.ceil(i), Math.ceil(j)])
    points.push([Math.floor(i), Math.ceil(j)])
    points.push([Math.ceil(i), Math.floor(j)])
    // if (opacityMap[Math.floor(i)][Math.floor(j)] == 1)
    //   transparent = false
    v = face.b
    i = v/divisions
    j = v%divisions
    points.push([Math.floor(i), Math.floor(j)])
    points.push([Math.ceil(i), Math.ceil(j)])
    points.push([Math.floor(i), Math.ceil(j)])
    points.push([Math.ceil(i), Math.floor(j)])
    // if (opacityMap[Math.floor(i)][Math.floor(j)] == 1)
    //   transparent = false
    // if ((i < xlimit || i > heightMap.length - xlimit) || (j < ylimit || j > heightMap[0].length - ylimit))
    //   hide = true
    v = face.c
    i = v/divisions
    j = v%divisions
    points.push([Math.floor(i), Math.floor(j)])
    points.push([Math.ceil(i), Math.ceil(j)])
    points.push([Math.floor(i), Math.ceil(j)])
    points.push([Math.ceil(i), Math.floor(j)])
    // if (opacityMap[Math.floor(i)][Math.floor(j)] == 1)
    //   transparent = false
    for (let p of points) {
      if (0 <= p[0] && p[0] < divisions && 0 <= p[1] && p[1] < divisions)
        if (opacityMap[p[0]][p[1]] == 1 || Math.abs(map[p[0]][p[1]]) > 0.5)
          transparent = false
    }
    // if ((i < xlimit || i > heightMap.length - xlimit) || (j < ylimit || j > heightMap[0].length - ylimit))
    //   hide = true
    if (false && hideSurface.checked && Math.abs(z1) == 0 && Math.abs(z2) == 0 && Math.abs(z3) == 0) {
      face.materialIndex = 1 // Transparent
    } else if (false && hideSurface.checked && (Math.abs(z2-z1) > 0.5 || Math.abs(z3-z1) > 0.5)) { // Extra condition for tests
      face.materialIndex = 1
    } else if (false && hideSurface.checked && (Math.abs(z2-z1) + Math.abs(z3-z1) + Math.abs(z3-z2) > 0.8)) { // Extra condition for tests
      face.materialIndex = 1
    } else if (false && hideSurface.checked && (z1 == 0 || z2 == 0 || z3 == 0)) { // Extra condition for tests // Was true
      face.materialIndex = 1
    } else if (false && hide && hideSurface.checked) {
      face.materialIndex = 1
    } else if (false && hideSurface.checked && (z1 < -2.4 || z2 < -2.4 || z3 < -2.4)) { // Inward edge
      face.materialIndex = 1
    } else if (transparent && useTransp.checked) {
      face.materialIndex = 2
    } else {
      face.materialIndex = 0
    }
  }
}

function helpClick(event) {
  let helpDiv = document.getElementById("div-help")
  if (helpDiv.style.display === "none") {
    helpDiv.style.display = "block";
  } else {
    helpDiv.style.display = "none";
  }
}

function pointerDown(event) {
  if (event.which != 1)
    return
  for ( var item of selectionBox.collection ) {
    if (Array.isArray(item.material))
      continue
    item.material = item.material.clone()
    // item.material.color.set( 0x4CAF50 )
  }

  selectionBox.startPoint.set(
  	( event.clientX / window.innerWidth ) * 2 - 1,
  	- ( event.clientY / window.innerHeight ) * 2 + 1,
  	0.5 )
}

function pointerMove(event) {
  if ( helper.isDown ) {
    for ( var i = 0; i < selectionBox.collection.length; i ++ ) {
      if (Array.isArray(selectionBox.collection[i].material))
        continue
      selectionBox.collection[i].material = selectionBox.collection[i].material.clone()
      // selectionBox.collection[i].material.color.set( 0x4CAF50 );

    }

    selectionBox.endPoint.set(
      ( event.clientX / window.innerWidth ) * 2 - 1,
      - ( event.clientY / window.innerHeight ) * 2 + 1,
      0.5 );

    var allSelected = selectionBox.select();
    for ( var i = 0; i < allSelected.length; i ++ ) {
      if (Array.isArray(allSelected[ i ].material))
        continue
      allSelected[ i ].material = allSelected[ i ].material.clone()
      // allSelected[ i ].material.color.set( 0xffffff );

    }

  }
}

function pointerUp(event) {
  if (event.which != 1)
    return
  selectionBox.endPoint.set(
    ( event.clientX / window.innerWidth ) * 2 - 1,
    - ( event.clientY / window.innerHeight ) * 2 + 1,
    0.5 )

  var allSelected = selectionBox.select();
  for ( var i = 0; i < allSelected.length; i ++ ) {
    if (Array.isArray(allSelected[ i ].material))
      continue
    // allSelected[ i ].material.color.set( 0xffffff );

  }
  subgraphSelect(allSelected)
}

function wheelEvent(event) {
  if (document.elementFromPoint(event.clientX, event.clientY).tagName != 'CANVAS')
    return
  var mapdiv = document.getElementById("map")
  mapdiv.style.display = "block"
  if (event.deltaY > 0) { // zoom out
    if (zoomLevels.length != 0) {
      mapdiv.style.width = zoomWidths.pop() + 'px'
      mapdiv.style.height = zoomHeights.pop() + 'px'
      olMap.updateSize()
      olMap.getView().setZoom(zoomLevels.pop())
    }

    // olMap.getView().setCenter(ol.proj.fromLonLat([87.6, 41.8]))
  } else { // zoom in

    zoomWidths.push(mapdiv.offsetWidth)
    zoomHeights.push(mapdiv.offsetHeight)
    zoomLevels.push(olMap.getView().getZoom())
    olMap.getView().setZoom(olMap.getView().getZoom()+0.05)
    mapdiv.style.width = mapdiv.offsetWidth*1.05 + 'px'
    mapdiv.style.height = mapdiv.offsetHeight*1.05 + 'px'
    olMap.updateSize()
    let p1 = ol.proj.fromLonLat([0, 0])
    let p2 = ol.proj.fromLonLat([100/(2**zoomLevels.length), 100/(2**zoomLevels.length)])
    let extents = [p1[0], p1[1], p2[0], p2[1]]
    // console.log(olMap.getView().getResolution())
    // olMap.getLayers().array_[0].setExtent(extents)
    // olMap.getView().setCenter(ol.proj.fromLonLat([87.6, 41.8]))
  }
  mapdiv.style.display = "none"
}

function calcOpacityMap(opacityMap, vertices, edges, heightMap) {
  let divisions = 100
  for (let id in edges) {
    if (edges[id].split)
      continue
    let startPt = [edges[id].start.mesh.position.x, edges[id].start.mesh.position.z]
    let endPt = [edges[id].end.mesh.position.x, edges[id].end.mesh.position.z]

    startPt = convert3JStoOM(startPt, divisions)
    endPt = convert3JStoOM(endPt, divisions)
    for (let i = 0 ; i < divisions ; i++) {
      for (let j = 0 ; j < divisions ; j++) {
        // if (distanceToLine(startPt, endPt, [i, j]) < 0.3)
        //   opacityMap[j][i] = 1
        if (Math.abs(dist(startPt, [i, j]) + dist([i, j], endPt) - dist(startPt, endPt)) < 0.1)
          opacityMap[j][i] = 1
        if (heightMap[Math.floor(j/2)][Math.floor(i/2)] > 0.3) {
          opacityMap[j][i] = 1
        }
      }
    }
  }
}

function calcDistanceOnSurface(plane, vertices, edges) {
  console.log("Calc distance")
  let verts = []
  let faces = []
  let nodes = [] // Array of mesh positions of nodes/vertices of graphs
  let send_edges = []
  for (let vert of plane.geometry.vertices) {
    verts.push([vert.x, vert.y, vert.z])
  }
  for (let face of plane.geometry.faces) {
    faces.push([face.a, face.b, face.c])
  }
  console.log(verts)
  console.log(faces)
  for (let vert in vertices) {
    // Convert Graph node position to specific vertex
    let cur_node = vertices[vert]
    // let converted = convert3JStoVertsExtended(cur_node.mesh.position.x, cur_node.mesh.position.z)
    // nodes.push(converted[0])
    // nodes.push(converted[1])

    let converted = convert3JStoVerts(cur_node.mesh.position.x, cur_node.mesh.position.z)
    nodes.push(converted)
    // TODO push
    console.log(converted)
    console.log(cur_node.name + " " + convert3JStoLatLong(cur_node.mesh.position.x, cur_node.mesh.position.z))
  }
  plane.geometry.verticesNeedUpdate = true
  for (let id in edges) {
    if (edges[id].split)
      continue
    let start = edges[id].start
    let end = edges[id].end
    let startNode = convert3JStoVerts(start.mesh.position.x, start.mesh.position.z)
    let endNode = convert3JStoVerts(end.mesh.position.x, end.mesh.position.z)
    send_edges.push([startNode, endNode])
  }

  let send_data = {verts: verts, faces: faces, nodes: nodes, edges: send_edges}
  var xmlHttp = new XMLHttpRequest();
  // xmlHttp.responseType = "arraybuffer"
  xmlHttp.responseType = "text"


  xmlHttp.onreadystatechange = function()
  {
      if(xmlHttp.readyState == 4 && xmlHttp.status == 200)

      {
          let data = xmlHttp.responseText
          // data = data.substring(data.indexOf('['))
          data = JSON.parse(data)
          console.log("data recv")
          console.log(data)
          let distances = data.distances
          let grads = data.grads
          let paths = data.paths
          drawSurfacePathsFlip(paths, edges)

          // const rows = [
          //     ["name1", "city1", "some other info"],
          //     ["name2", "city2", "more info"]
          // ];
          let csv_data = []
          let csv_header = [""]
          for (let vert in vertices) {
            csv_header.push(vertices[vert].name)
          }
          csv_data.push(csv_header)
          let length = Object.keys(vertices).length
          for (let i = 0 ; i < length ; i++) {
            // csv_header.push(vertices[vert].name)
            let csv_row = [vertices[i].name]
            console.log(csv_row)
            for (let j = 0 ; j < length ; j++) {
              if (nodes[i] < nodes[j])
                csv_row.push(distances[i][nodes[j]])
              else
                csv_row.push(distances[j][nodes[i]])
              // Min of dist b/w (A and B) and (A and alt B)
              // TODO: change for full plane
              // let min_dist = Math.min(distances[2*i][nodes[2*j]], distances[2*i][nodes[2*j+1]])

              // Min of current min and dist(B and A)
              // min_dist = Math.min(min_dist, distances[2*j][nodes[2*i]])

              // Min of current min and dist(B and alt A)
              // min_dist = Math.min(min_dist, distances[2*j][nodes[2*i+1]])
              // csv_row.push(min_dist)
              // console.log(csv_row)
            }
            csv_data.push(csv_row)
          }
          console.log(csv_data)
          let csvContent = "data:text/csv;charset=utf-8,"
              + csv_data.map(e => e.join(",")).join("\n")
          var encodedUri = encodeURI(csvContent)
          // console.log(csvContent)
          window.open(encodedUri)

          console.log("DOWNLOAD")

          // for (let id in edges) {
          //   if (edges[id].split)
          //     continue
          //   let start = edges[id].start.id
          //   let end = edges[id].end.id
          //   // console.log(vertices[start].name + "," + vertices[end].name + "," + distances[start][nodes[end]])
          //   setTimeout (console.log.bind (console, vertices[start].name + "," + vertices[end].name + "," + distances[start][nodes[end]]))
          //   drawSurfacePath(distances[start], nodes[end])
          //   // drawSurfacePathGrads(grads[start], distances[start], nodes[end])
          //
          // }
          // for (let i = 0 ; i < data.length ; i++) {
          //   for (let j = 0 ; j < data.length ; j++) {
          //     // console.log(i)
          //     // console.log(j)
          //     // console.log(data[i][nodes[j]])
          //     console.log(i + " to " + j + ": " + data[i][nodes[j]])
          //   }
          //   console.log(" ")
          // }

      }
  }
  xmlHttp.open("post", "calc-distance");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xmlHttp.send(JSON.stringify(send_data));
  console.log("data sent")
  console.log(send_data)
  dataSent = true
}

function drawSurfacePathsFlip(paths, edges) {

  for (let i=0, j=0 ; i < paths.length ; i++, j++) {
    // if (edges[j].split) {
    //   i--
    //   continue
    // }
    let points = []
    let line_color = new T.Color("hsl(0, 0%, 100%)")
    if (edges[j].weight >= 0)
      var endColor = new T.Color("hsl(222, 100%, 61%)")
    else
      var endColor = new T.Color("hsl(356, 74%, 52%)")
    line_color.lerp(endColor, Math.min(Math.abs(edges[j].weight), 1))
    // if (edges[j].weight < 0) {
    //   line_color = new T.Color("rgb(255, 0, 0)")
    // }
    let material = new T.LineBasicMaterial({
  	   color: line_color, linewidth: 4, linecap: 'round'
    })
    for (let pt of paths[i]) {
      if (subPlanes.length != 0) {
        pt[2] += subPlanes[0].plane.position.y + 0.01
        pt[0] += subPlanes[0].plane.position.x
        pt[1] -= subPlanes[0].plane.position.z
      }
      points.push(new T.Vector3(pt[0], pt[2], -pt[1]))
    }
    const geometry = new T.BufferGeometry().setFromPoints( points )
    const line = new T.Line( geometry, material )
    scene.add( line )
  }

}

function drawSurfacePath(distances, start) {
  let dist = distances[start]
  let i = Math.floor(start/50)
  let j = start%50
  let min_i = -1
  let min_j = -1

  let distCopy = [...distances]
  let distMat = []
  while(distCopy.length)
    distMat.push(distCopy.splice(0, 50))
 // 3a86ff
  const material = new T.LineBasicMaterial({
	   color: 0xFDCA40, linewidth: 4, linecap: 'round'
  })

  let heightmap = heightMap

  if (subPlanes.length != 0)
    heightmap = subPlanes[0].heightmap

  let converted = convertVertsto3JS(i, j)
  const points = [new T.Vector3(converted[0], 2.01 + heightmap[i][j], converted[1])]

  let t = 0
  while(dist && t < 50) {
    t++
    for (let k = -1 ; k <= 1 ; k++)
      for (let l = -1 ; l <= 1 ; l++) {
        if (typeof distMat[i+k] != 'undefined' && typeof distMat[i+k][j+l] != 'undefined' && distMat[i+k][j+l] < dist) {
          dist = distMat[i+k][j+l]
          min_i = i+k
          min_j = j+l
        }
      }

    i = min_i
    j = min_j
    converted = convertVertsto3JS(i, j)
    points.push( new T.Vector3(converted[0], 2.01 + heightmap[i][j], converted[1]))
  }

  const geometry = new T.BufferGeometry().setFromPoints( points )
  const line = new T.Line( geometry, material )

  // const geom = new T.BufferGeometry()
  // geom.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( 4 * 3 ), 3 ) );
  // let curve = new THREE.CatmullRomCurve3( points )
	// curve.curveType = 'catmullrom'
	// curve.mesh = new THREE.Line( geom.clone(), new THREE.LineBasicMaterial( {
	// 	color: 0xff0000
	// } ) );
  //
  // scene.add(curve.mesh)
  scene.add( line )
}

function drawSurfacePathGrads(grads, distances, start) {
  // Doesn't work yet
  let grad = grads[start]
  let dist = distances[start]
  let i = Math.floor(start/50)
  let j = start%50
  let min_i = -1
  let min_j = -1

  let distCopy = [...distances]
  let distMat = []
  while(distCopy.length)
    distMat.push(distCopy.splice(0, 50))

  let gradCopy = [...grads]
  let gradMat = []
  while(gradCopy.length)
    gradMat.push(gradCopy.splice(0, 50))

  const material = new T.LineBasicMaterial({
	   color: 0x0000ff, linewidth: 2, linecap: 'round'
  })

  let converted = convertVertsto3JS(i, j)
  const points = [new T.Vector3(converted[1], 0.05 + heightMap[i][j], converted[0])]

  let t = 0
  while(dist && t < 50) {
    t++
    for (let k = -1 ; k <= 1 ; k++)
      for (let l = -1 ; l <= 1 ; l++) {
        if (typeof gradMat[i+k] != 'undefined' && typeof gradMat[i+k][j+l] != 'undefined' && gradMat[i+k][j+l] < grad) {
          dist = distMat[i+k][j+l]
          grad = gradMat[i+k][j+l]
          min_i = i+k
          min_j = j+l
        }
      }

    i = min_i
    j = min_j
    converted = convertVertsto3JS(i, j)
    points.push( new T.Vector3(converted[1], 0.05 + heightMap[i][j], converted[0]))
  }
  console.log(dist)

  const geometry = new T.BufferGeometry().setFromPoints( points )
  const line = new T.Line( geometry, material )

  // const geom = new T.BufferGeometry()
  // geom.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( 4 * 3 ), 3 ) );
  // let curve = new THREE.CatmullRomCurve3( points )
	// curve.curveType = 'catmullrom'
	// curve.mesh = new THREE.Line( geom.clone(), new THREE.LineBasicMaterial( {
	// 	color: 0xff0000
	// } ) );
  //
  // scene.add(curve.mesh)
  scene.add( line )
}

function subgraphSelect(selected) {
  console.log(selected.length)
  console.log(vertexCount + edgeCount + 1)
  if (selected.length <= 1) {
    console.log("return / empty select")
    return
  }
  var data = {nodes: [], links: []}
  var nodes = {}
  var edges = []
  let ids = {}
  let xRange = [10, -10]
  let yRange = [10, -10]

  // Get nodes and edges from selected shapes
  console.log(refine_data)
  for (let id in selected) {
    if (selected[id].geometry.type == "SphereGeometry") {
      // console.log(`Name: ${selected[id].name}, Lat: ${selected[id].position.x}, Lon: ${selected[id].position.z}`)
      xRange[0] = Math.min(xRange[0], selected[id].position.x)
      xRange[1] = Math.max(xRange[1], selected[id].position.x)
      yRange[0] = Math.min(yRange[0], selected[id].position.z)
      yRange[1] = Math.max(yRange[1], selected[id].position.z)

      data.nodes.push({id: parseInt(id), city: selected[id].name, lat: parseFloat(selected[id].position.x), long: parseFloat(selected[id].position.z)})
      ids[selected[id].name] = parseInt(id)
      nodes[selected[id].name] = {lat: parseFloat(selected[id].position.x), long: parseFloat(selected[id].position.z), name: selected[id].name}

    } else if (selected[id].geometry.type == "BufferGeometry") {
      // console.log(ids)
      let start = selected[id].name.split('/')[0]
      let end = selected[id].name.split('/')[1]
      let weight = selected[id].userData.weight
      let neg_mod = selected[id].userData.neg_mod
      let nrw_mod = selected[id].userData.nrw_mod
      let nheight_mod = selected[id].userData.nheight_mod
      // console.log(selected[id].name)
      // console.log(selected[id].name.split('/'))
      // console.log(start)
      // console.log(`Start: ${start}(${ids[start]}), End: ${end}(${ids[end]}), Weight: ${weight}`)
      if (ids[start] == undefined || ids[end] == undefined)
        continue

      let startname = Math.min(nodes[start].name, nodes[end].name)
      let endname = Math.max(nodes[start].name, nodes[end].name)
      console.log(startname, endname)

      if (refine_data[startname] && refine_data[startname][endname]) {
        console.log("refine start")
        let ref = refine_data[startname][endname]
        if (weight < 0)
          console.log(startname, endname, ref)

        if (ref > 0) {
          neg_mod = 1.2*neg_mod
          nrw_mod = 1.2*nrw_mod
          nheight_mod = 1.2*nheight_mod
          console.log("subrefine")
        } else if (ref < -1) {
          neg_mod = 0.8*neg_mod
          nrw_mod = 0.8*nrw_mod
          console.log("subrefine")

        }
        selected[id].userData.neg_mod = neg_mod
        selected[id].userData.nrw_mod = nrw_mod
        selected[id].userData.nheight_mod = nheight_mod
        // console.log(selected[id].userData)
      }
      let edge = {start: nodes[start], end: nodes[end], weight: weight, nrw_mod: nrw_mod, neg_mod: neg_mod, nheight_mod: nheight_mod}
      data.links.push({source: ids[start], target: ids[end], ricciCurvature: weight})
      edges.push(edge)
    }
  }

  refine_data = []

  let padding = 0.25
  xRange[0] = Math.max(-7, xRange[0] - padding)
  xRange[1] = Math.min(7, xRange[1] + padding)
  yRange[0] = Math.max(-10, yRange[0] - padding)
  yRange[1] = Math.min(10, yRange[1] + padding)

  data.nodes.push({id: data.length, city: "edge1", lat: parseFloat(xRange[0]), long: parseFloat(yRange[0])})
  data.nodes.push({id: data.length, city: "edge2", lat: parseFloat(xRange[1]), long: parseFloat(yRange[1])})


  let mid = [(xRange[0]+xRange[1])/2, (yRange[0]+yRange[1])/2]
  let width = xRange[1] - xRange[0]
  let height = yRange[1] - yRange[0]
  // --- DRAW PLANE ---
  let ctx = canv.getContext('2d')
  if (document.getElementById("show-map").checked) {
    let mapCanvas = document.getElementById('map').getElementsByTagName('canvas')[0]
    ctx = mapCanvas.getContext('2d')
  }

  var subTexture = new T.CanvasTexture(ctx.canvas)
  subTexture.minFilter = THREE.LinearFilter
  subTexture.center = new T.Vector2(0.5, 0.5)
  subTexture.rotation = -Math.PI/2
  subTexture.repeat.y = (xRange[1] - xRange[0]) / 20
  subTexture.repeat.x = (yRange[1] - yRange[0]) / 20
  subTexture.offset.x = mid[1] / 20
  subTexture.offset.y = mid[0] / 20

  var subGeom = new T.PlaneGeometry(width, height, divisions-1, divisions-1)
  // var material = new T.MeshBasicMaterial( { color: graphcolor, side: T.DoubleSide} )
  var subMat = new THREE.MeshPhongMaterial( { color: graphcolor, clippingPlanes: [clipPlane2, clipPlane3, clipPlane4, clipPlane5], vertexColors: T.VertexColors, side: THREE.DoubleSide,  flatShading: false, shininess: 0, wireframe: false, map: subTexture} )
  var subPlane = new T.Mesh( subGeom, subMat )

  // TODO: change
  subPlane.position.set(mid[0], 2, mid[1])
  subPlane.rotation.set(-Math.PI/2, 0., 0.)
  let spObj = {}
  spObj.xstart3JS = xRange[0]
  spObj.width = width
  spObj.ystart3JS = yRange[0]
  spObj.height = height
  spObj.start = [Math.floor((xRange[0] + 10) * divisions / 20), Math.floor((yRange[0] + 10) * divisions / 20)]
  spObj.end = [Math.ceil((xRange[1] + 10) * divisions / 20), Math.ceil((yRange[1] + 10) * divisions / 20)]
  spObj.plane = subPlane
  subPlane.scale.set(0.1, 0.1, 0.1)
  scene.add( subPlane )
  subPlanes.push(spObj)
  let scale = Math.min(width, height) / divisions
  scale *= 10 // 10
  console.log(scale)

  // plane.position.set(0, -10, 0)

//TODO: change back

  gsap.to( camera, {
				duration: 1,
				zoom: 2,
				onUpdate: function () {
					camera.updateProjectionMatrix();
				}
	})

  gsap.to( controls.target, {
				duration: 1,
				x: subPlane.position.x,
				y: subPlane.position.y,
				z: subPlane.position.z,
				onUpdate: function () {
					controls.update();
				}
	})

  gsap.to( plane.position, {
        duration: 1,
        y: -20,
        onUpdate: function () {
        },
        onComplete: function() {
          plane.visible = false
        }
  })

  gsap.to( subPlane.scale, {
        duration: 1,
        x: 1,
        y: 1,
        z: 1,
  })

  // let btn = document.createElement("button")
  // btn.innerHTML = "Zoom Out"
  // btn.id = "zoom-out"
  // document.body.appendChild(btn)
  // plane.material.transparent = true
  // plane.material.opacity = 0.5f


  // controls.target = subPlane.position
  // camera.zoom = 2
  // controls.update()
  // camera.updateProjectionMatrix()
  // ---GENERATE SURFACE
  let chkCalcSurface = document.getElementById("use-calc-surface")
  let newHeightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));

  let modifier = Math.max(width, height) / 20
  console.log(modifier, width, height)
  // modifier = 0.4

  if (!chkCalcSurface.checked) {
    let planeXMin = xRange[0]
    let planeWidth = xRange[1] - xRange[0]
    let planeYMin = yRange[0]
    let planeHeight = yRange[1] - yRange[0]

    for (let edge of edges) {
      if (edge.weight < 0)
        continue
      // let startname = Math.min(edge.start.name, edge.end.name)
      // let endname = Math.max(edge.start.name, edge.end.name)
      // console.log(startname, endname)
      // if (refine_data[startname] && refine_data[startname][endname]) {
      //   console.log(startname, endname, refine_data[startname][endname])
      // }
      let startPt = convert3JStoHMgeneric([edge.start.lat, edge.start.long], planeXMin, planeYMin, planeWidth, planeHeight)
      let endPt = convert3JStoHMgeneric([edge.end.lat, edge.end.long], planeXMin, planeYMin, planeWidth, planeHeight)
      let start = {x: startPt[0], y: startPt[1]}
      let end = {x: endPt[0], y: endPt[1]}
      let mid = {x: Math.floor((startPt[0] + endPt[0])/2), y: Math.floor((startPt[1] + endPt[1])/2)}
      setHeights(start, mid, end, edge.weight, newHeightMap, modifier)
    }



    for (let edge of edges) {
      if (edge.weight > 0)
        continue
        console.log(edge)
      let startname = Math.min(edge.start.name, edge.end.name)
      let endname = Math.max(edge.start.name, edge.end.name)
      let neg_mod = edge.neg_mod
      console.log(neg_mod)
      let nrw_mod = edge.nrw_mod
      let nheight_mod = edge.nheight_mod

      let startPt = convert3JStoHMgeneric([edge.start.lat, edge.start.long], planeXMin, planeYMin, planeWidth, planeHeight)
      let endPt = convert3JStoHMgeneric([edge.end.lat, edge.end.long], planeXMin, planeYMin, planeWidth, planeHeight)
      //TODO: remove
      startPt[1] += 1
      endPt[1] += 1
      let start = {x: startPt[0], y: startPt[1]}
      let end = {x: endPt[0], y: endPt[1]}
      let mid = {x: Math.floor((startPt[0] + endPt[0])/2), y: Math.floor((startPt[1] + endPt[1])/2)}
      setHeights(start, mid, end, edge.weight, newHeightMap, modifier, neg_mod, nrw_mod, nheight_mod)
    }

    // smoothHeightMap(newHeightMap)
    // smoothHeightMap(newHeightMap)
    smoothHeightMap(newHeightMap)
    smoothHeightMap(newHeightMap)
    setPlaneHeights(subPlane, newHeightMap)
    spObj.heightmap = newHeightMap


    return
  }

  var xmlHttp = new XMLHttpRequest();
  xmlHttp.responseType = "text"
  var smooth_pen = document.getElementById("input-smooth").value
  var niter = document.getElementById("input-niter").value
  var send_data = {graph: data, smooth_pen: smooth_pen, niter: niter}

  xmlHttp.onreadystatechange = function() {
    if(xmlHttp.readyState == 4 && xmlHttp.status == 200) {
      dataSent = false
      data = xmlHttp.responseText
      data = data.substring(data.indexOf('['))
      data = JSON.parse(data)
      console.log(data)
      console.log("data recv")
      let newHeightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
      for (let i = 0 ; i < divisions ; i++) {
        for (let j = 0 ; j < divisions ; j++) {
          newHeightMap[j][49-i] = data[i*divisions + j]*scale
        }
      }
      console.log(newHeightMap)
      setPlaneHeights(subPlane, newHeightMap)
    }
  }
  xmlHttp.open("post", "calc-surface");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

  xmlHttp.send(JSON.stringify(send_data));
  console.log("data sent")
  dataSent = true
}

function setPlaneHeights(plane, map) {
  for (let i=0; i<divisions ; i++) {
    for (let j=0; j < divisions ; j++) {
      plane.geometry.vertices[i*divisions+j].z =  map[i][j]
    }
  }
  plane.geometry.groupsNeedUpdate = true
  plane.geometry.verticesNeedUpdate = true
  plane.geometry.colorsNeedUpdate = true
  plane.geometry.computeVertexNormals()
}

function calcContours(xlimit, ylimit, heightMap) {
  xlimit = 0
  ylimit = 0
  if (contourMeshLines.length != 0) {
    for (let line of contourMeshLines)
      scene.remove(line)
  }

  contcolor = 0x000000 // ffffff // 707070
  // lineWidth = 4
  var lineMat = new T.LineBasicMaterial({color: contcolor, linewidth: 4, depthFunc: T.LessEqualDepth, transparent: true, opacity: 0.5, clippingPlanes: [clipPlane, clipPlane2]})
  var conrec = new Conrec
  let nLevels = 26
  let levels = []
  let min = -2.2
  let max = 2.2
  for (let i = min; i < max ; i+=(max-min)/nLevels) {
    levels.push(i)
  }

  // let levels = [-2.4, -2.2, -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
  //-- CONTOUT LINES --//
  // conrec.contour(heightMap, xlimit, heightMap.length - xlimit - 1, ylimit, heightMap[0].length - 1 - ylimit, contX, contY, levels.length, levels)
  conrec.contour(heightMap, xlimit, heightMap.length - xlimit - 1, ylimit, heightMap[0].length - 1 - ylimit, contX, contY, levels.length, levels)

  let lines = conrec.contourList()

  // LIMITS OF LINES IN HERE
  for (let line of lines) {
    let points = []
    // console.log(line)
    // console.log(line)
    for (let pt of line) {
      pt.x = (pt.x*(planeW/49)) - (planeW/2)// (0, 149) to (planeXMin, planeXMax)
      pt.y = (pt.y*(planeH/49)) - (planeH/2)// (0, 149) to (planeYMin, planeYMax)
      // console.log(pt.x)
      let limits = 5
      // if (pt.x >= -5 && pt.x <= 5 && pt.y >= -7 && pt.y <= 7)
      //   points.push(new T.Vector3(pt.y, line.level+0.01, pt.x))
      // if (pt.x >= -5 && pt.x <= 5 && pt.y >= -7 && pt.y <= 7)
      //   points.push(new T.Vector3(pt.y, line.level+0.01, pt.x))
      if (pt.x > -10 && pt.x < 10 && pt.y > -7 && pt.y < 7)
        points.push(new T.Vector3(pt.y, line.level+0.01, pt.x))
    }

    let geom = new T.BufferGeometry().setFromPoints(points)
    let lineMesh = new T.Line(geom, lineMat)
    contourMeshLines.push(lineMesh)
    scene.add(lineMesh)
  }




    // if (lineMat != lineMatSec) {
    //   points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight+0.0001, edge.start.mesh.position.z))
    //   points.push(new T.Vector3(edge.end.mesh.position.x, vertexHeight+0.0001, edge.end.mesh.position.z))
    // } else {
    //   points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight, edge.start.mesh.position.z))
    //   points.push(new T.Vector3(edge.end.mesh.position.x, vertexHeight, edge.end.mesh.position.z))
    // }
    // points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight+0.0001, edge.start.mesh.position.z))
    //
    // let geom = new T.BufferGeometry().setFromPoints(points)
    //
    // let line = new T.Line(geom, lineMat)
    //
    //
    // scene.add( line );
    // linesDrawn.push(line)
}

function raiseHeightMap() {
  xLimit = 55
  yLimit = 35
  for (let i = 0 ; i < xLimit ; i++) {
    for (let j = 0 ; j < (heightMap[0].length-1)/2; j++) {
      heightMap[i][j] = 0
      heightMap[i][heightMap[0].length-1-j] = 0
      heightMap[heightMap.length-1-i][j] = 0
      heightMap[heightMap.length-1-i][heightMap[0].length-1-j] = 0
    }
  }
  for (let i = xLimit ; i < heightMap.length - xLimit ; i++) {
    for (let j = 0 ; j < yLimit; j++) {
      heightMap[i][j] = 0
      heightMap[i][heightMap[0].length-1-j] = 0
      heightMap[heightMap.length-1-i][j] = 0
      heightMap[heightMap.length-1-i][heightMap[0].length-1-j] = 0
    }
  }
  return 0
}

function smoothHeightMap(heightMap) {
  for (let j = 0 ; j < heightMap[0].length ; j++) {
    heightMap[0][j] = heightMap[2][j]
    heightMap[1][j] = heightMap[2][j]
    heightMap[heightMap.length-2][j] = heightMap[heightMap.length-3][j]
    heightMap[heightMap.length-1][j] = heightMap[heightMap.length-2][j]


  }
  for (let i = 2 ; i < heightMap.length-2 ; i++) {
    for (let j = 2 ; j < heightMap[0].length-2; j++) {
      // if (heightMap[i][j] == 0) {
      //   if (heightMap[i+1][j] * heightMap[i-1][j] *
      //     heightMap[i][j+1] * heightMap[i][j-1] != 0 ) {// If all neighbours are non zero
      //     heightMap[i][j] = (heightMap[i+1][j] + heightMap[i-1][j] +
      //       heightMap[i][j+1] + heightMap[i][j-1]) / 4
      //
      //   } else {
      //     continue
      //   }
      // }
      // let neighbours = [heightMap[i+1][j], heightMap[i-1][j],
      //   heightMap[i][j+1], heightMap[i][j-1], heightMap[i+1][j+1],
      //   heightMap[i+1][j-1], heightMap[i-1][j-1], heightMap[i-1][j+1],
      //   heightMap[i+2][j], heightMap[i-2][j],
      //   heightMap[i][j+2], heightMap[i][j-2], heightMap[i+2][j+2],
      //   heightMap[i+2][j-2], heightMap[i-2][j-2], heightMap[i-2][j+2]]

      let neighbours = [heightMap[i+1][j], heightMap[i-1][j],
        heightMap[i][j+1], heightMap[i][j-1], heightMap[i+1][j+1],
        heightMap[i+1][j-1], heightMap[i-1][j-1], heightMap[i-1][j+1]]

      // if (Math.min(neighbours) < 0 && Math.max(neighbours) > 0)
      //   continue

      let sum = neighbours.reduce((a, b) => a + b, 0)
      let count = neighbours.length
      sum = 0
      count = 0
      let useZero = true
      if (heightMap[i][j] < -0.1)
        useZero = false
      for(let i = 0 ; i < neighbours.length ; i++) {
        if (neighbours[i] == 0 && !useZero) { // If neighbour is 0 and don't use zero flag is set
          continue
        }
        sum += neighbours[i]
        count++
      }

      heightMap[i][j] += sum
      heightMap[i][j] /= count + 1
      // if (Math.abs(heightMap[i][j]) <= 0.0003) {
      //   heightMap[i][j] = 0
      // }

    }
  }
}

// TODO: Make the percent dropoff more quadratic
// TODO: Deal with clashing heights - Add heights of multiple edges together -> Deal with huge towers

function setHeights(start, mid, end, weight, heightMap, modifier = 1, neg_modifier=1, narw_modifier=1, nheight_modifier=1) {

  if (weight >= 0) {
    // --- Gaussian heights ---
    // TODO: Only iterate through local points for speedup instead of whole 2d array
    let x = mid.y
    let y = mid.x
    let amp = document.getElementById("amp-slider").value // Def 1000
    // weight = 2.5*weight
    let spread = document.getElementById("posrange-slider").value
    let xSpread =  spread // Use divisions variable instead of hard coding spread
    let ySpread = spread
    for (let i = 0 ; i < heightMap.length ; i++) {
      for (let j = 0 ; j < heightMap[0].length ; j++) {
        if ((i-x)**2 + (j-y)**2 > spread)
          continue
        let xTerm = Math.pow(i - x, 2) / (2.0*Math.pow(xSpread, 2))
        let yTerm = Math.pow(j - y, 2) / (2.0*Math.pow(ySpread, 2))
        let newHeight = weight*amp*Math.exp(-1.0*(xTerm + yTerm))*modifier
        // if (Math.abs(newHeight) <= 0.01) {
        //   newHeight = 0
        // }
        if (heightMap[i][j] * newHeight > 0) { // Both in same direction, then choose highest magnitude
          if (newHeight >= 0) {
            heightMap[i][j] = Math.max(heightMap[i][j], newHeight)
          } else {
            heightMap[i][j] = Math.min(heightMap[i][j], newHeight)
          }
        } else { // Else average
          if (Math.abs(heightMap[i][j]) < Math.abs(newHeight)) {
            heightMap[i][j] = newHeight
          }
        }
      }
    }
  } else {
    // TODO: Improve scaling [add 0.2?] -> -0.1 doesnt do much
    // TODO: Rotate
    // TODO: Rotate checking max heights
    // TODO: Point the ends of the saddle curve
    // TODO: Change ySpread and yLimit based on edge distance and heights
    // TODO: Left and right sides of curve have different yLimits to line up with heights
    // --- Saddle Heights ---
    let slope = (start.y - end.y) / (start.x - end.x)
    let angle = Math.atan(slope) + parseFloat(document.getElementById("rotation-slider").value)
    let calcdist = calcDist(start, end)

    let xSpread = Math.max(20, calcdist*0.56)*parseFloat(document.getElementById("xspread-slider").value) // length // Def 26 // Slider def 0.5
    xSpread = (0.58*calcdist) + parseFloat(document.getElementById("xspread-slider").value) // length // Def 26 // Slider def 0.5
    // xLimits * weight
    let ySpread = (1.875) + parseFloat(document.getElementById("yspread-slider").value) // 2.5 // width TODO: Multiply with edge length
    let xLimit = ((2*Math.min(-weight+0.5, 1))/(xSpread)) * parseFloat(document.getElementById("xlimit-slider").value) * neg_modifier // Def 1000// height along length Def 0.05
    let xLimit2 = ((2*Math.min(-weight+0.5, 1))/(xSpread)) * parseFloat(document.getElementById("xlimit2-slider").value) * neg_modifier
    // let xLimit = ((1.25*2*weight)/(xSpread)) * parseFloat(document.getElementById("xlimit-slider").value) // Def 1000// height along length Def 0.05
    // let xLimit2 = ((1.25*2*weight)/(xSpread)) * parseFloat(document.getElementById("xlimit2-slider").value)
    let yLimit = (0.1*0.7)  * parseFloat(document.getElementById("ylimit-slider").value) * narw_modifier// 0.7 // 0.55 // depth along width TODO: Change based on edge length
    let addHeight = (-0.5) + parseFloat(document.getElementById("height-slider").value)
    // console.log(addHeight)
    // console.log(document.getElementById("height-slider").value)
    // addHeight += document.getElementById("height-slider").value
    // console.log(distanceToLine([0,0], [1,1]))

    for (let i = mid.x - xSpread ; i <= mid.x + xSpread ; i++) {
      for (let j = mid.y - ySpread; j <= mid.y + ySpread ; j++) {
        let newHeight = 0
        if (Math.abs(i - mid.x) < Math.abs(j - mid.y) && Math.abs(i - mid.x) > 1) {
          // console.log("CONT")
          continue
        }
        if (dist([start.x, start.y], [i, j]) > dist([end.x, end.y], [i, j]))
          newHeight = ((i-mid.x)*xLimit)**2 - ((j-mid.y)*yLimit)**2
        else
          newHeight = ((i-mid.x)*xLimit2)**2 - ((j-mid.y)*yLimit)**2
        // newHeight *= -1
        newHeight += addHeight
        newHeight *= modifier
        // newHeight = 1*modifier
        let x_pos = j
        let y_pos = i

        // X and Y coordinate calculations are switched
        x_pos = Math.floor((i-mid.x)*Math.sin(angle) + (j-mid.y)*Math.cos(angle)) + mid.y
        y_pos = Math.floor((i-mid.x)*Math.cos(angle) - (j-mid.y)*Math.sin(angle)) + mid.x
        if (!(x_pos >= heightMap.length || y_pos >= heightMap.length || x_pos < 0 || y_pos < 0 || isNaN(x_pos) || isNaN(y_pos)))
          heightMap[x_pos][y_pos] = newHeight

        // Check closest to which pt
        // if (newHeight > heightMap[x_pos][y_pos]) {
        //   if (heightMap[x_pos][y_pos] != 0)
        //     continue
        //   newHeight = 0.1
        // }

        // if (newHeight < heightMap[x_pos][y_pos] && heightMap[x_pos][y_pos] > 0.1) {
        //   continue
        // }

        x_pos = Math.ceil((i-mid.x)*Math.sin(angle) + (j-mid.y)*Math.cos(angle)) + mid.y
        y_pos = Math.ceil((i-mid.x)*Math.cos(angle) - (j-mid.y)*Math.sin(angle)) + mid.x
        if (!(x_pos >= heightMap.length || y_pos >= heightMap.length || x_pos < 0 || y_pos < 0 || isNaN(x_pos) || isNaN(y_pos)))
          heightMap[x_pos][y_pos] = newHeight

        x_pos = Math.floor((i-mid.x)*Math.sin(angle) + (j-mid.y)*Math.cos(angle)) + mid.y
        y_pos = Math.ceil((i-mid.x)*Math.cos(angle) - (j-mid.y)*Math.sin(angle)) + mid.x
        if (!(x_pos >= heightMap.length || y_pos >= heightMap.length || x_pos < 0 || y_pos < 0 || isNaN(x_pos) || isNaN(y_pos)))
          heightMap[x_pos][y_pos] = newHeight

          x_pos = Math.ceil((i-mid.x)*Math.sin(angle) + (j-mid.y)*Math.cos(angle)) + mid.y
          y_pos = Math.floor((i-mid.x)*Math.cos(angle) - (j-mid.y)*Math.sin(angle)) + mid.x
          if (!(x_pos >= heightMap.length || y_pos >= heightMap.length || x_pos < 0 || y_pos < 0 || isNaN(x_pos) || isNaN(y_pos)))
            heightMap[x_pos][y_pos] = newHeight


        // if (heightMap[x_pos][y_pos] * newHeight > 0) { // Both in same direction, then choose highest magnitude
        //   if (newHeight >= 0) {
        //     heightMap[x_pos][y_pos] = Math.max(heightMap[x_pos][y_pos], newHeight)
        //   } else {
        //     heightMap[x_pos][y_pos] = Math.min(heightMap[x_pos][y_pos], newHeight)
        //   }
        // } else { // Else highest magnitude
        //   if (Math.abs(heightMap[x_pos][y_pos]) < Math.abs(newHeight)) {
        //     heightMap[x_pos][y_pos] = newHeight
        //   }
        // }

        // distStart = calcDist(start, {x: i, y: j})
        // distEnd = calcDist(end, {x: i, y: j})
        // if (distStart < distEnd) {
        //   if (newHeight > heightMap[start.y][start.x])
        //     console.log("true1")
        //   heightMap[i][j] = Math.min(newHeight, heightMap[start.y][start.x])
        // } else {
        //   if (newHeight > heightMap[end.y][end.x])
        //     console.log("true2")
        //   heightMap[i][j] = Math.min(newHeight, heightMap[end.y][end.x])
        //
        // }
      }
    }

    // heightMap[start.y][start.x] = 10
    // heightMap[end.y][end.x] = 20

  }

  // heightMap[x][y] = weight

  /* --- Sine heights ---
  levels = (divisions/10) * Math.abs(weight)  // Make levels dependant on height (* weight)
  percent = 1/levels
  for (let i = 0, j = levels+2 ; j >= 0 ; i += 2/levels, j--) {
    for (let angle = 0 ; angle < 360 ; angle++) {
      new_x = Math.round(x + (j)*Math.cos(angle))
      new_y = Math.round(y + (j)*Math.sin(angle))
      percent = ((0.5 * Math.sin(1.4*(i - 1.1))) + 0.5)
      if (percent <= 0 && i > 1)
        percent = 1;
      new_val = weight * percent  // 3rd level -> 25%, 2nd level -> 50% ... etc
      if (new_x < divisions && new_y < divisions && new_x >= 0 && new_y >= 0) {
        if (new_val * heightMap[new_x][new_y] < 0) // They have opposite sign
          heightMap[new_x][new_y] = (heightMap[new_x][new_y] + new_val) / 2 // Take average
        else if (Math.abs(new_val) > 0 && Math.abs(new_val) > Math.abs(heightMap[new_x][new_y])) // Else one is bigger than the other
          heightMap[new_x][new_y] = new_val
      }
    }

  }
  */

  /* --- Radial heights ---
  for (levels -= 0 ; levels >= 0 ; levels--) {
    for (let angle = 0 ; angle < 360 ; angle++) {
      new_x = Math.round(x + (levels+1)*Math.cos(angle))
      new_y = Math.round(y + (levels+1)*Math.sin(angle))
      new_val = weight * (1 - levels * percent)  // 3rd level -> 25%, 2nd level -> 50% ... etc
      if (new_x < divisions && new_y < divisions && new_x >= 0 && new_y >= 0)
        if (new_val * heightMap[new_x][new_y] < 0) // They have opposite sign
          heightMap[new_x][new_y] = (heightMap[new_x][new_y] + new_val) / 2 // Take average
        if (Math.abs(new_val) > 0 && Math.abs(new_val) > Math.abs(heightMap[new_x][new_y])) // Else one is bigger than the other
          heightMap[new_x][new_y] = new_val
    }
  }
  */


}

function calcDist(pt1, pt2) {
  return Math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)
}

function vertexNameChange() {
  // TODO: Deal with duplicate names
  if (this.value == '')
    return
  let parentDiv = this.parentElement
  let id = parentDiv.childNodes[0].textContent
  let pt = vertices[id]
  let oldName = pt.name
  pt.name = this.value
  scene.remove(pt.label)
  pt.label = getNameSprite(this.value)
  pt.label.position.set(pt.mesh.position.x, vertexHeight + 0.5, pt.mesh.position.z)
  scene.add(pt.label)
  delete names[oldName]
  names[pt.name] = parseInt(id)

}

function vertexPositionChange() {
  if (this.value == '' || isNaN(this.value))
    return
  // console.log("Postion Change")
  let parentDiv = this.parentElement
  let id = parentDiv.childNodes[0].textContent
  let pt = vertices[id]
  if (this.className == "xPos") {
    gsap.to( pt.mesh.position, {
  				duration: 0.25,
  				x: this.value,
          onUpdate: function() {
            olMap.render()
          }
  	})
    gsap.to( pt.label.position, {
  				duration: 0.25,
  				x: this.value,
          onUpdate: function() {
            olMap.render()
          }
  	})
    // pt.mesh.position.x = this.value
    // pt.label.position.x = this.value
    pt.lat = this.value*(90/7)

  } else {
    gsap.to( pt.mesh.position, {
  				duration: 0.25,
  				z: this.value,
          onUpdate: function() {
            olMap.render()
          }
  	})
    gsap.to( pt.label.position, {
  				duration: 0.25,
  				z: this.value,
          onUpdate: function() {
            olMap.render()
          }
  	})
    // pt.mesh.position.z = this.value
    // pt.label.position.z = this.value
    pt.long = this.value*(180/10)

  }

}

function addVertex(obj, x, y, drawPoint, name, lat=null, long=null) {
  if (typeof drawPoint == 'undefined')
    drawPoint = true

  if (x == undefined) {
    if (lat != null)
      x = lat*10/155
    else
      // x = getRandomArbitrary(-6, 6).toFixed(2)
      x = 0

  }
  if (y == undefined) {
    if (long != null)
      y = long*10/180
    else
      // y = getRandomArbitrary(-9, 9).toFixed(2)
      if (vertexCount == 0)
        y = 5
      else
        y = -5
  }
  if (lat == null) {
    lat = x*155/10
  }
  if (long == null) {
    long = y*180/10
  }


  let vDiv = document.createElement("div")
  vDiv.id = "vertex" + vertexCount
  vDiv.className = "form-box"

  let idLbl = document.createElement("label")
  idLbl.setAttribute("for", "id")
  idLbl.textContent = vertexCount

  if (typeof name == 'undefined')
    name = vertexCount

  let nameLbl = document.createElement("label")
  nameLbl.setAttribute("for", "name")
  nameLbl.textContent = "Name:"

  let nameInput = document.createElement("input")
  nameInput.className = "name"
  nameInput.setAttribute("type", "text")
  nameInput.defaultValue = name
  nameInput.onchange = vertexNameChange

  let xPosLbl = document.createElement("label")
  xPosLbl.setAttribute("for", "xPos")
  xPosLbl.textContent = "x:"

  let xPos = document.createElement("input")
  xPos.className = "xPos"
  xPos.setAttribute("type", "text")
  xPos.defaultValue = x
  xPos.oninput = vertexPositionChange

  let yPosLbl = document.createElement("label")
  yPosLbl.setAttribute("for", "yPos")
  yPosLbl.textContent = "y:"


  let yPos = document.createElement("input")
  yPos.className = "yPos"
  yPos.setAttribute("type", "text")
  yPos.defaultValue = y
  yPos.oninput = vertexPositionChange

  let del = document.createElement("button")
  del.className = "btn-delete"
  del.innerHTML = "X";
  del.onclick = removeVertex

  vDiv.appendChild(idLbl)
  vDiv.appendChild(nameLbl)
  vDiv.appendChild(nameInput)
  vDiv.appendChild(xPosLbl)
  vDiv.appendChild(xPos)
  vDiv.appendChild(yPosLbl)
  vDiv.appendChild(yPos)
  vDiv.appendChild(del)
  document.getElementById("div-vertex").appendChild(vDiv)

  xPos.select()


  let newPt = new T.Mesh(ptGeom, ptMat)
  newPt.position.y = vertexHeight
  newPt.position.x = xPos.value
  newPt.position.z = yPos.value
  newPt.name = name

  // console.log(newPt.position.y)

  let sprite = getNameSprite(name)
  sprite.position.set(xPos.value, vertexHeight + 0.1, yPos.value)  //  + 0.2 + Math.random()*0.2

  if (drawPoint) {
    scene.add(sprite)
    newPt.scale.set(0.1, 0.1, 0.1)
    scene.add(newPt)
    gsap.to( newPt.scale, {
          duration: 1.5,
          x: 1,
          y: 1,
          z: 1,
          ease: 'elastic'
    })
  }
  vertices[String(vertexCount)] = new VertexObj(vertexCount, name, newPt, sprite, lat, long)
  names[name] = vertexCount
  vertexCount++
}

function addVertexSec(obj, x, y, vertices, drawPoint=false) {
  let newPt = new T.Mesh(ptGeom, ptMat)
  newPt.position.y = vertexHeight
  newPt.position.x = x
  newPt.position.z = y

  length = Object.keys(vertices).length

  let sprite = getNameSprite(length)
  sprite.position.set(x, vertexHeight + 0.5, y)

  if (drawPoint) {
    scene.add(sprite)
    newPt.scale.set(0.1, 0.1, 0.1)
    scene.add(newPt)
    gsap.to( newPt.scale, {
          duration: 1,
          x: 1,
          y: 1,
          z: 1,
          ease: 'elastic'
    })
    // console.log(length)
  }

  vertices[String(length)] = new VertexObj(length, length, newPt, sprite)
}

function removeVertex() {
  let parentDiv = this.parentElement
  let name = parentDiv.childNodes[0].textContent
  scene.remove(vertices[name].mesh)
  scene.remove(vertices[name].label)
  delete vertices[name]
  parentDiv.remove()
}

function generateGraph() {
  addVertex(null, -3.15, 4.5, true, name="A");
  addVertex(null, -1.7213114754098358, 0.0, true, name="B");
  addVertex(null, -3.15, -4.5, true, name="C");
  addVertex(null, 3.15, -4.5, true, name="D");
  addVertex(null, 1.7213114754098358, 0.0, true, name="E");
  addVertex(null, 3.15, 4.5, true, name="F");

  addEdge(null, 0, 1, 1.0);
  addEdge(null, 1, 2, 1.0);
  addEdge(null, 2, 0, 1.0);
  addEdge(null, 3, 4, 1.0);
  addEdge(null, 4, 5, 1.0);
  addEdge(null, 5, 3, 1.0);
  addEdge(null, 1, 4, -2.0);
}

function generateGraphNoWeights() {
  // Graph 1 & 2
  {
    // Graph 1
    addVertex(null, -5, 0)
    addVertex(null, -4, -1.73)
    addVertex(null, -3, -0.5)
    addVertex(null, 3, -0.5)
    addVertex(null, 4, 1.73)
    addVertex(null, 5, 0)

    addEdge(null, 2, 3, 0)
    addEdge(null, 0, 1, 0)
    addEdge(null, 0, 2, 0)
    addEdge(null, 1, 2, 0)
    addEdge(null, 3, 4, 0)
    addEdge(null, 3, 5, 0)
    addEdge(null, 4, 5, 0)

    // Graph 2
    // 0-A - -5, 0
    // 1-B - -4.5, -1
    // 2-C - -3.5, 0.5
    // 3-E - -3, 0 // Skip D
    // 4-F - -1.5, 0
    // 5-G - 2.4, 0.5
    // 6-H - 2.4, -0.5
    // 7-I - 3.5, 0.8
    // 8-J - 4, 3
  }
}

function drawEdge(edge, lineMat) {
  // console.log(edge.end.mesh.position.x + " " + edge.end.mesh.position.z)
  let points = []
  if (lineMat != lineMatSec) {
    points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight+0.0001, edge.start.mesh.position.z))
    points.push(new T.Vector3(edge.end.mesh.position.x, vertexHeight+0.0001, edge.end.mesh.position.z))
  } else {
    points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight, edge.start.mesh.position.z))
    points.push(new T.Vector3(edge.end.mesh.position.x, vertexHeight, edge.end.mesh.position.z))
  }
  points.push(new T.Vector3(edge.start.mesh.position.x, vertexHeight+0.0001, edge.start.mesh.position.z))

  let geom = new T.BufferGeometry().setFromPoints(points)

  // New Line //
  // geom = new LineGeometry()
  // geom.setPositions(points)
  //
  // var colors = []
  // var color = new THREE.Color();
  // color.setHSL( 1, 1.0, 0.5 );
  // colors.push( color.r, color.g, color.b );
  //
  // geom.setColors( colors );
  //
  // matLine = new LineMaterial( {
  //
	// 				color: 0xff0000,
	// 				linewidth: 5, // in pixels
	// 				vertexColors: false,
	// 				//resolution:  // to be set by renderer, eventually
	// 				dashed: false
  //
	// 			} );
  //
  // let line = new Line2(geom, matLine)
  // linewidth = 4
  // 0x2cc57c
  let mat = new T.LineBasicMaterial({color: contcolor, linewidth: 4, depthFunc: T.LessEqualDepth, transparent: true, opacity: 0.05, clippingPlanes: [clipPlane, clipPlane2]})
  mat = new T.LineBasicMaterial({color: edgecolor, linewidth: 5, clippingPlanes: [clipPlane, clipPlane2, clipPlane3, clipPlane4, clipPlane5, ] })
  let line = new T.Line(geom, mat)
  let color = new T.Color("hsl(0, 0%, 100%)")
  if (edge.weight >= 0)
    var endColor = new T.Color("hsl(222, 100%, 61%)")
  else
    var endColor = new T.Color("hsl(356, 74%, 52%)")
  color.lerp(endColor, Math.min(Math.abs(edge.weight), 1))
  line.material.color.set(color)
  line.name = edge.start.name + "/" + edge.end.name
  line.userData = {weight: edge.weight, neg_mod: edge.neg_mod, nrw_mod: edge.nrw_mod, nheight_mod: edge.nheight_mod}

  scene.add( line );
  linesDrawn.push(line)
  return line
}

function addEdge(obj, start, end, weight) {
  if (typeof start == 'undefined') {
    start = 0
  }

  if (typeof end == 'undefined') {
    end = 0
  }

  if (typeof weight == 'undefined') {
    weight = 0
  }

  let vDiv = document.createElement("div")
  vDiv.id = "edge" + edgeCount
  vDiv.className = "form-box"

  let nameLbl = document.createElement("label")
  nameLbl.setAttribute("for", "name")
  nameLbl.textContent = edgeCount

  let startLbl = document.createElement("label")
  startLbl.setAttribute("for", "start")
  startLbl.textContent = "start:"

  let startText = document.createElement("input")
  startText.className = "start"
  startText.setAttribute("type", "text")
  startText.defaultValue = start
  startText.oninput = edgeChange

  let endLbl = document.createElement("label")
  endLbl.setAttribute("for", "start")
  endLbl.textContent = "end:"

  let endText = document.createElement("input")
  endText.className = "end"
  endText.setAttribute("type", "text")
  endText.defaultValue = end
  endText.oninput = edgeChange

  let weightLbl = document.createElement("label")
  weightLbl.setAttribute("for", "weight")
  weightLbl.textContent = "weight:"

  let weightText = document.createElement("input")
  weightText.className = "weight"
  weightText.setAttribute("type", "text")
  weightText.defaultValue = weight
  weightText.oninput = edgeChange

  let del = document.createElement("button")
  del.className = "btn-delete"
  del.innerHTML = "X";
  del.onclick = removeEdge



  vDiv.appendChild(nameLbl)
  vDiv.appendChild(startLbl)
  vDiv.appendChild(startText)
  vDiv.appendChild(endLbl)
  vDiv.appendChild(endText)
  vDiv.appendChild(weightLbl)
  vDiv.appendChild(weightText)
  vDiv.appendChild(del)
  document.getElementById("div-edge").appendChild(vDiv)


  let size = Object.keys(vertices).length

  let s = parseInt(startText.value)
  let e = parseInt(endText.value)
  // console.log("s: " + s + " e: " + e)

  weight = parseFloat(weightText.value)

  let startPt = vertices[s]
  let endPt = vertices[e]
  if (startPt == endPt) {
    // TODO: Deal with this
  }

  let edge = new EdgeObj(edgeCount, startPt, endPt, weight)
  edges[edgeCount] = edge
  edgeCount++
}

function addEdgeSec(obj, start, end, weight, vertices, edges) {
  let vSize = Object.keys(vertices).length
  let eSize = Object.keys(edges).length

  let s = parseInt(start)
  let e = parseInt(end)
  // console.log("s: " + s + " e: " + e)

  weight = parseFloat(weight)

  let startPt = vertices[s]
  let endPt = vertices[e]
  if (startPt == endPt) {
    // TODO: Deal with this
  }

  let edge = new EdgeObj(eSize, startPt, endPt, weight)
  edges[eSize] = edge
}

function edgeChange() {
  // TODO: Deal with non existent vertices
  if (this.value == '' || isNaN(this.value))
    return
  let parentDiv = this.parentElement
  let startId = parentDiv.childNodes[2].value
  let endId = parentDiv.childNodes[4].value
  let weight = parseFloat(parentDiv.childNodes[6].value)
  let id = parentDiv.childNodes[0].textContent
  // console.log(startId + " " + endId + " " + weight)
  let edge = edges[id]
  edge.start = vertices[startId]
  edge.end = vertices[endId]
  edge.weight = weight
  edge.checkSplit()
}

function removeEdge() {
  //TODO: Remove edge
  console.log("Remove edge")
  let parentDiv = this.parentElement
  let id = parentDiv.childNodes[0].textContent
  delete edges[id]
  parentDiv.remove()
}

function getNameSprite(name) {
  // if (name < 3)
  //   name = String.fromCharCode(65 + name)
  // else
  //   name = String.fromCharCode(65 + name + 1)

  let canvas = document.createElement('canvas')
  let ctx = canvas.getContext('2d')


  // ctx.fillStyle = "#ffff00";
  // ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  let metrics = ctx.measureText( name );
  let textWidth = metrics.width;
  let textHeight = metrics.height
  // console.log(metrics.width)

  ctx.canvas.width = textWidth*30+30;
  ctx.canvas.height = textWidth*30+10;

  ctx.font="20px Roboto Mono" // 120px // 40px // 20px
  ctx.fillStyle = "#000000"






  ctx.fillText(name, ctx.canvas.width/2 - textWidth/2, ctx.canvas.height/2)

  let texture = new T.CanvasTexture(ctx.canvas)
  texture.needsUpdate = true

  let spriteMat = new T.SpriteMaterial({map: texture, alphaTest: 0.1})
  let sprite = new T.Sprite(spriteMat)
  sprite.scale.set(0.05*textWidth, 0.05*textWidth, 0.05*textWidth)
  return sprite
}

let VertexObj = class {
  start = [] // Edges starting at this vertex
  end = [] // Edges ending at this vertex

  constructor(id, name, mesh, label, lat=0, long=0, start=[], end=[]) {
    this.id = id
    this.name = name
    this.mesh = mesh
    this.label = label
    this.start = start
    this.end = end
    this.lat = lat
    this.long = long
  }
}

let EdgeObj = class {
  constructor(id, start, end, weight) {
    this.id = id
    this.start = start
    this.end = end
    this.weight = weight
    this.bearing = GreatCircle.bearing(start.lat, start.long, end.lat, end.long)
    this.split = false
    this.neg_mod = 1
    this.nrw_mod = 1
    this.nheight_mod = 1
    this.mesh = null
    // console.log(`${start.long}, ${end.long}, ${this.bearing}`)
    if (start.long > end.long && this.bearing <= 180) {
      this.split = true
      let p1 = [start.mesh.position.x, start.mesh.position.z]
      // console.log(`${end.mesh.position.x}, ${end.mesh.position.z}`)
      this.startSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z)], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)+20], [10, 10], [-10, 10])
      // console.log([parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)+20])
      this.endSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z)-20], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)], [10, -10], [-10, -10])
      // console.log([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z-20)])
      this.startSplit = [this.startSplit[0]*155/10, this.startSplit[1]*180/10]
      this.endSplit = [this.endSplit[0]*155/10, this.endSplit[1]*180/10]
      console.log(this.startSplit)
      console.log(this.endSplit)
    } else if (start.long < end.long && this.bearing >= 180) {
      this.split = true
      let p1 = [start.mesh.position.x, start.mesh.position.z]
      // console.log(`${start.mesh.position.x}, ${start.mesh.position.z}`)
      this.startSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z)], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)-20], [10, -10], [-10, -10])
      // console.log([parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z-20)])
      this.endSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z)+20], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)], [10, 10], [-10, 10])
      // console.log([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z+20)])
      this.startSplit = [this.startSplit[0]*155/10, this.startSplit[1]*180/10]
      this.endSplit = [this.endSplit[0]*155/10, this.endSplit[1]*180/10]
      console.log(this.startSplit)
      console.log(this.endSplit)
    }

  }

  checkSplit() {
    let start = this.start
    let end = this.end
    console.log('checkSplit')
    if (start.long > end.long && this.bearing <= 180) {
      console.log('split exists')
      this.split = true
      console.log(start.mesh.position.x)
      this.startSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z)], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z+20)], [7, 10], [-7, 10])
      this.endSplit = math.intersect([parseFloat(start.mesh.position.x), parseFloat(start.mesh.position.z-20)], [parseFloat(end.mesh.position.x), parseFloat(end.mesh.position.z)], [7, -10], [-7, -10])
      console.log(start.long)
      console.log(end.long)
    }
  }
}

function getIntersection(p1, p2, p3, p4) {
  // Get intersection formed by line p1-p2 and line p3-p4


}

let GraphObj = class {
  constructor(vertices, edges, heightmap) {
    this.vertices = vertices
    this.heightmap = heightmap
    this.edges = edges
  }
}

function createMap() {
  var mapdiv = document.createElement('div')
  mapdiv.id = 'map'
  mapdiv.class = 'map-div'
  // mapdiv.style.width = '400px'
  // mapdiv.style.height = '400px'
  let p1 = ol.proj.fromLonLat([0, 0])
  let p2 = ol.proj.fromLonLat([90, 90])
  let extents = [p1[0], p1[1], p2[0], p2[1]]
  document.body.appendChild(mapdiv)
  var map = new ol.Map({
        target: 'map',
        renderer:'canvas',
        layers: [
          // new ol.layer.Tile({
          //   source: new ol.source.OSM(),
          //   // resolution: 152.87405654296876,
          //   // tileSize: [1024,1024]
          // }),
          new ol.layer.Tile({
            // extent: extents,
            minResolution: 1,
            source: new ol.source.Stamen({
              layer: 'terrain'
            }),
            // maxResolution: 2000,
          })
        ],
        view: new ol.View({
          // center: ol.proj.fromLonLat([67.41, 8.82]),
          // projection: 'EPSG:9823',
          center: ol.proj.fromLonLat([0, 0]),
          zoom: 0,
          zoomFactor: 2,
          // resolution: 2,
          // maxResolution: 2
        })
  });
  // console.log(map.getView().getResolution())
  return map
}

function calculateCurvature() {
  console.log("calculate curvature")
  var data = {nodes: [], links: []}
  for (let id in vertices) {
    data.nodes.push({id: id})
  }
  let current_edges = {...edges}
  if (graphs.length > 0)
    current_edges = {...graphs[document.getElementById("threshold-slider").value].edges}
  console.log(current_edges)
  for (let id in current_edges) {
    let edge = current_edges[id]
    data.links.push({source: edge.start.id, target: edge.end.id})
  }
  // $.ajax({
  // type: "POST",
  // url: "./scripts/OllivierRicci.py",
  // data: { param: text}
  //   }).done(function( o ) {
  //      // do something
  //   })
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.onreadystatechange = function()
  {
      if(xmlHttp.readyState == 4 && xmlHttp.status == 200)
      {
          data = JSON.parse(xmlHttp.responseText)
          // console.log(data)
          console.log("data recv")
          let current_edges = {...edges}
          if (graphs.length > 0)
            current_edges = {...graphs[document.getElementById("threshold-slider").value].edges}
          for(let id in data.links) {
            let link = data.links[id]
            for (let id2 in current_edges) {
              let edge = current_edges[id2]
              if ((edge.start.id == link.source && edge.end.id == link.target) || (edge.start.id == link.target && edge.end.id == link.source)) {
                edge.weight = parseFloat(link.ricciCurvature)
                let edgeDiv = document.getElementById("edge" + id2)
                if (edgeDiv != null)
                  edgeDiv.querySelector(".weight").value = parseFloat(link.ricciCurvature)
                break
              }
            }
          }

          for (let line of linesDrawn) {
            scene.remove(line)
            line = null
          }
          linesDrawn = []
          linesCleared = true
      }
  }
  xmlHttp.open("post", "calc-curvature");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

  xmlHttp.send(JSON.stringify(data));
  console.log("data sent")


}

function calcSurface() {
  console.log("calculate surface")
  var data = {nodes: [], links: []}
  let current_edges = {...edges}
  if (graphs.length > 0)
    current_edges = {...graphs[document.getElementById("threshold-slider").value].edges}
  length = Object.keys(vertices).length
  for (let id in vertices) {
    let node = vertices[id]
    data.nodes.push({id: parseInt(id), city: String(node.name), lat: node.lat + 1E-10, long: node.long + 1E-10})
  }
  data.nodes.push({id: data.nodes.length, city: "border1", lat: 155.001, long: 180.001})
  data.nodes.push({id: data.nodes.length, city: "border2", lat: -155.001, long: -180.001})
  // data.nodes.push({id: length+3, city: "border3", lat: 155.001, long: 0.001}) // 138.5
  // data.nodes.push({id: length+4, city: "border4", lat: -155.001, long: 0.001})

  let splitCount = 0
  for (let id in current_edges) {
    let edge = current_edges[id]
    if (edge.split) {
      // continue
      console.log("split")
      data.nodes.push({id: data.nodes.length, city: "splitstart" + splitCount, lat: edge.startSplit[0], long: edge.startSplit[1]})
      data.nodes.push({id: data.nodes.length, city: "splitend" + splitCount, lat: edge.endSplit[0], long: edge.endSplit[1]})
      data.links.push({source: edge.start.id, target: data.nodes.length-2, ricciCurvature: edge.weight})
      data.links.push({source: edge.end.id, target: data.nodes.length-1, ricciCurvature: edge.weight})
      splitCount++
    } else {
      data.links.push({source: edge.start.id, target: edge.end.id, ricciCurvature: edge.weight})
    }
  }
  console.log()
  // console.log(vertices)
  // $.ajax({
  // type: "POST",
  // url: "./scripts/OllivierRicci.py",
  // data: { param: text}
  //   }).done(function( o ) {
  //      // do something
  //   })
  var xmlHttp = new XMLHttpRequest();
  // xmlHttp.responseType = "arraybuffer"
  xmlHttp.responseType = "text"
  var smooth_pen = document.getElementById("input-smooth").value
  var niter = document.getElementById("input-niter").value
  var send_data = {graph: data, smooth_pen: smooth_pen, niter: niter, map: heightMap}

  xmlHttp.onreadystatechange = function()
  {
      if(xmlHttp.readyState == 4 && xmlHttp.status == 200)

      {
          // document.getElementById("heatmap-img").setAttribute('src', 'data:image/png;base64,' + btoa(String.fromCharCode.apply(null, new Uint8Array(xmlHttp.response))))
          let scale = 2
          dataSent = false
          document.body.style.cursor = "auto"
          // data = JSON.parse(xmlHttp.responseText)
          data = xmlHttp.responseText
          data = data.substring(data.indexOf('['))
          data = JSON.parse(data)
          console.log(data)
          console.log("data recv")
          let hm = []
          if (graphs.length > 0)
            hm = graphs[document.getElementById("threshold-slider").value].heightmap
          else
            hm = calcHeightMap
          for (let i = 0 ; i < divisions ; i++) {
            for (let j = 0 ; j < divisions ; j++) {
              hm[j][divisions-1-i] = data[i*divisions + j]*scale
            }
          }

          updatePlaneHeights(hm);
      }
  }
  xmlHttp.open("post", "calc-surface");
  xmlHttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  console.log(send_data)
  xmlHttp.send(JSON.stringify(send_data));
  console.log("data sent")
  dataSent = true
  document.body.style.cursor = "progress"
}

function calcCurvMap(curvMap, vertices, edges) {

  for (let id in edges) {
    let startPt = [edges[id].start.mesh.position.x, edges[id].start.mesh.position.z]
    let endPt = [edges[id].end.mesh.position.x, edges[id].end.mesh.position.z]

    startPt = convert3JStoHM(startPt)
    endPt = convert3JStoHM(endPt)
    for (let i = 0 ; i < divisions ; i++) {
      for (let j = 0 ; j < divisions ; j++) {
        // if (distanceToLine(startPt, endPt, [i, j]) < 0.3)
        //   opacityMap[j][i] = 1
        if (Math.abs(dist(startPt, [i, j]) + dist([i, j], endPt) - dist(startPt, endPt)) < 0.2)
          curvMap[divisions-i][j] = edges[id].weight
      }
    }
  }
}

function updateShading(curvMap) {
  let ctx = document.getElementById('map').getElementsByTagName('canvas')[0].getContext('2d')
  // console.log(ctx)
  let scale = ctx.canvas.height/divisions
  ctx.lineWidth = 0.005 / scale
  ctx.strokeStyle = "#fff"
  let data = {
    width: 50,
    height: 50,
    values: curvMap.flat()
  }
  ctx.save()
  ctx.globalAlpha = 1.0
  ctx.scale(scale, scale)
  let color = d3.scaleLinear().domain([d3.min(data.values), 0, d3.max(data.values)]).range(['#FF0000', '#FFFFFF', '#00FF00']).nice()
  let contours = d3.contours().size([data.width, data.height])
  let thresholds = color.ticks(10)
  let path = d3.geoPath(null, ctx)
  // console.log(ctx.canvas.width)
  // console.log(thresholds)

  for (const d of thresholds) {
    // console.log(d)
    if (d == 0) {
      // console.log("skip")
      // continue
    }
    ctx.beginPath();
    path(contours.contour(data.values, d));
    ctx.fillStyle = color(d);
    ctx.fill();
    ctx.stroke();
  }

  ctx.restore()


}

function createAlphaMap(heightMap) {
  var alphaCanv = document.getElementById('alpha-canvas')
  if (alphaCanv == null) {
    console.log('doesnt exist')
    alphaCanv = document.createElement('canvas')
    alphaCanv.id = 'alpha-canvas'
    document.body.appendChild(alphaCanv)
  }
  let ctx = alphaCanv.getContext('2d')
  let height = heightMap.length
  let width = heightMap[0].length
  let imgData = ctx.createImageData(height, width)
  for (let i = 0, j = 0 ; i < height*width ; i += 1, j += 4) {
    let i_int = Math.floor(i/width)
    let r = Math.random()*255
    let s = 100
    imgData.data[j] = heightMap[i%width][width - i_int]*s
    imgData.data[j+1] = heightMap[i%width][width - i_int]*s
    imgData.data[j+2] = heightMap[i%width][width - i_int]*s
    imgData.data[j+3] = 255
  }
  ctx.putImageData(imgData, 0, 0)
  var aMap = new T.CanvasTexture(ctx.canvas)
  return aMap
}

function createAndUpdateAlphaMapD3(map) {
  var svg = d3.select('#alpha-svg')
  var width = 512
  var height = 512
  if (d3.select('#alpha-svg').empty()) {
    console.log('create svg')
    var svg = d3.select('body')
      .append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('id', 'alpha-svg')
        .attr('display', 'none')
  }
  svg.selectAll("*").remove()
  svg.append('rect')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('fill', '#333333')
  svg.append('g')

  // Add X axis
  var x = d3.scaleLinear()
    .domain([0, map.length])
    .range([ 0, width ])
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x))

  // Add Y axis
  var y = d3.scaleLinear()
    .domain([0, map[0].length])
    .range([ height, 0 ])
  svg.append("g")
    .call(d3.axisLeft(y))

  var color = d3.scaleLinear()
    .domain([0.0, 0.002]) // Points per square pixel. .002
    .range(["#333333", "white"])
  var data = []

  for (let i = 0 ; i < map.length ; i++) {
    for (let j = 0 ; j < map[i].length ; j++)
      if (map[i][j] == 1)
        data.push({x: i, y: j})
  }

  // Modify bandwidth / 4 // 100

  var densityData = d3.contourDensity()
      .x(function(d) { return x(d.x); })
      .y(function(d) { return y(d.y); })
      .size([width, height])
      .bandwidth(4)
      (data)


  svg.insert("g", "g")
    .selectAll("path")
    .data(densityData)
    .enter().append("path")
      .attr("d", d3.geoPath())
      .attr("fill", function(d) { return color(d.value); })

  var alphaCanv = document.getElementById('alpha-canvas')
  if (alphaCanv == null) {
    console.log('doesnt exist')
    alphaCanv = document.createElement('canvas')
    alphaCanv.id = 'alpha-canvas'
    alphaCanv.height = height
    alphaCanv.width = width
    alphaCanv.style.display = 'none'

    document.body.appendChild(alphaCanv)
  }
  var ctx = alphaCanv.getContext('2d')

  var svgEle = document.getElementById('alpha-svg')
  var img = document.createElement('img')
  img.setAttribute('src', "data:image/svg+xml;base64," + window.btoa(unescape(encodeURIComponent((new XMLSerializer()).serializeToString(svgEle)))))
  img.onload = function() {
    ctx.drawImage(img, 0, 0)
    // let mapCtx = document.getElementById('map').getElementsByTagName('canvas')[0].getContext('2d')
    // mapCtx.globalAlpha = 0.5
    // mapCtx.drawImage(img, (mapCtx.canvas.width-1000)/2, (mapCtx.canvas.height-1000)/2, 1000, 1000)
  }

  return new T.CanvasTexture(alphaCanv)

}

function fileSelectEdges(evt) {
    evt.stopPropagation()
    evt.preventDefault()

    var files = evt.dataTransfer.files; // FileList object.
    console.log(files)
    for (let file of files) {
      readEdgeFile(file)
    }
}

function readEdgeFile(file) {
  var reader = new FileReader()
  reader.onload = function() {
    let current_edges = {}
    let text = reader.result
    let lines = text.split('\n')
    let i = -1
    let inputNames = []
    let negative_edges = []
    if (file.name.substr(-3) != 'csv') {
    // if (file.name.substr(-3))
      let inputNameData = lines[0].split('\"')
      let k = -1
      for (let nameData of inputNameData) {
        console.log("here3")
        k++
        nameData = nameData.trim()
        if (nameData == '')
          continue
        if (k%2 == 1)
          inputNames.push(nameData)
        else {
          inputNames = inputNames.concat(nameData.split(' '))
        }
      }
    } else {
      inputNames = lines[0].split(',').slice(1)
    }

    if (file.name.substr(-3) != 'csv') {
      for (let i = 0 ; i < lines.length ; i++) {
        let line = lines[i]
        let data = line.split("\"")
        let currentNode = ''
        if (data.length > 1) { // Two word name - deal with double quotes
          currentNode = data[1]
          data = data[2].split(" ")
        } else {
          data = data[0].split(" ")
          currentNode = data[0]
          data = data.splice(1)
        }
        if (data[0] == '' || isNaN(data[0]))
          continue
        let currentId = names[currentNode]
        for (let j=0 ; j<i ; j++) {
          let weight = parseFloat(data[j])
          if (weight == 0)
            continue
          let endNode = inputNames[j]
          let endId = names[endNode]
          console.log(`${weight} edge from ${currentNode}(${currentId}) to ${endNode}(${endId})`)
          if (weight < 0) {
            let n = {start: currentId, end: endId, weight: weight}
            negative_edges.push(n)
            continue
          }
          addEdgeSec(null, currentId, endId, weight, vertices, current_edges)
        }
        // addVertex(null, (parseFloat(data[1])/90)*7, (parseFloat(data[2])/180)*10, true, data[0])
      }
    } else {
      for (let i = 0 ; i < lines.length ; i++) {
        let line = lines[i]
        let data = line.split(",")
        let currentNode = data[0]
        data = data.splice(1)

        // if (data[0] == '' || isNaN(data[0]))
        //   continue
        let currentId = names[currentNode]
        for (let j=0 ; j<i ; j++) {
          if (data[j] == '' || isNaN(data[0]))
            continue
          let weight = parseFloat(data[j])
          let endNode = inputNames[j]
          let endId = names[endNode]
          console.log(`${weight} edge from ${currentNode}(${currentId}) to ${endNode}(${endId})`)
          if (weight < 0) {
            let n = {start: currentId, end: endId, weight: weight}
            negative_edges.push(n)
            continue
          }
          addEdgeSec(null, currentId, endId, weight, vertices, current_edges)
        }
      }
    }
    negative_edges.sort((a, b) => -(a.weight - b.weight))
    for (let e of negative_edges)
      addEdgeSec(null, e.start, e.end, e.weight, vertices, current_edges)

    let newHeightMap = Array(divisions).fill().map(() => Array(divisions).fill(0.0));
    let newGraph = new GraphObj(vertices, current_edges, newHeightMap)
    graphs.push(newGraph)
    console.log(current_edges)


    // for (let id in current_edges) {
    //   let edge = current_edges[id]
    //   if (edge.weight < 0)
    //     negative_edges.push(edge)
    // }
    // graphs[document.getElementById("threshold-slider").value + 1].edges
    // edgeCollection.push(current_edges)
    // console.log(current_edges)
    document.getElementById("threshold-slider").max = graphs.length - 1
    document.getElementById("threshold-slider").value = graphs.length - 1
  }
  reader.readAsText(file)

}

function fileSelectNodes(evt) {
    evt.stopPropagation()
    evt.preventDefault()
    console.log("read node file")
    var files = evt.dataTransfer.files; // FileList object.
    // files is a FileList of File objects. List some properties.
    // var output = [];
    // for (var i = 0, f; f = files[i]; i++) {
    //   output.push('<li><strong>', escape(f.name), '</strong> (', f.type || 'n/a', ') - ',
    //               f.size, ' bytes, last modified: ',
    //               f.lastModifiedDate ? f.lastModifiedDate.toLocaleDateString() : 'n/a',
    //               '</li>');
    // }
    // document.getElementById('node-list').innerHTML = '<ul>' + output.join('') + '</ul>';

    var reader = new FileReader()

    reader.onload = function() {
      let text = reader.result
      let lines = text.split('\n')
      for (let line of lines) {
        let data = line.split(',')
        if (data[1] == '' || isNaN(data[1]))
          continue
        let projected = ol.proj.fromLonLat([data[2], data[1]])
        // p[1]/20048966.10*10, p[0]/20026376.39*10
        addVertex(null, projected[1]/20048966.10*10, projected[0]/20026376.39*10, true, data[0], parseFloat(data[1]), parseFloat(data[2]))

        // addVertex(null, (parseFloat(data[1])/167)*10, (parseFloat(data[2])/180)*10, true, data[0], parseFloat(data[1]), parseFloat(data[2]))
        // addVertex(null, (parseFloat(data[1])/180)*10, (parseFloat(data[2])/180)*10, true, data[0], parseFloat(data[1]), parseFloat(data[2]))
      }
    }
    reader.readAsText(files[0])

}

function dragOver(evt) {
  evt.stopPropagation()
  evt.preventDefault()
  evt.dataTransfer.dropEffect = 'copy' // Explicitly show this is a copy.
}

function getRandomIntInclusive(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min; //The maximum is inclusive and the minimum is inclusive
}

function getRandomArbitrary(min, max) {
  return Math.random() * (max - min) + min;
}

function convert3JStoVerts(x, y) {
  // Depends on plane geometry
  // Change from -10,10 to 0,49
  if (subPlanes.length != 0)
    return convert3JStoVertsSubgraph(x, y)
  x = parseFloat(x)
  y = parseFloat(y)
  x += 10
  x /= 20
  x *= 49
  y += 10
  y /= 20
  y *= 49
  let retval = Math.round(y)*50 + Math.round(x)
  return retval
}


/**
  Convert 3JS coords to vertex coords on plane with duplicate plane attached
  to account for wrap around
**/
function convert3JStoVertsExtended(x, y) {
  // Depends on plane geometry
  // Change from -10,10 to 0,49
  if (subPlanes.length != 0)
    return convert3JStoVertsSubgraph(x, y)
  x = parseFloat(x)
  y = parseFloat(y)
  x += 10
  x /= 20
  x *= 49
  y += 10
  y /= 20
  y *= 49
  let retval = Math.round(y)*50 + Math.round(x)
  let retval2 =  (Math.round(y)+50)*50 + Math.round(x)
  return [retval, retval2]
}

function convert3JStoVertsSubgraph(x, y) {
  let spObj = subPlanes[0]
  x = parseFloat(x)
  y = parseFloat(y)
  x -= spObj.xstart3JS
  x /= spObj.width
  x *= 49
  y -= spObj.ystart3JS
  y /= spObj.height
  y *= 49
  let retval = Math.round(y)*50 + Math.round(x)
  return retval
}

function convertVertsto3JS(x, y) {
  if (subPlanes.length != 0)
    return convertVertsto3JSSubgraph(x, y)
  x = parseFloat(x)
  y = parseFloat(y)
  x /= 49
  x *= 20
  x -= 10
  y /= 49
  y *= 20
  y -= 10
  return [y, x]
}

function convertVertsto3JSSubgraph(x, y) {
  let spObj = subPlanes[0]
  x = parseFloat(x)
  y = parseFloat(y)
  x /= 49
  x *= spObj.height
  x += spObj.ystart3JS
  y /= 49
  y *= spObj.width
  y += spObj.xstart3JS
  return [y, x]
}

function convert3JStoHM(point) {
  point[0] = (point[0] - planeXMin) // Change from (min,max) to (0, newmax)
  point[1] = (point[1] - planeYMin) // Change from (min,max) to (0, newmax)

  point[0] = Math.round((point[0] / planeW) * (divisions-1)) // Change from (0, planeWidth) to (0, divisions)
  point[1] = Math.round((point[1] / planeH) * (divisions-1)) // Change from (0, planeHeight) to (0, divisions)

  return point;
}

function convert3JStoHMgeneric(point, planeXMin, planeYMin, planeWidth, planeHeight) {
  point[0] = (point[0] - planeXMin) // Change from (min,max) to (0, newmax)
  point[1] = (point[1] - planeYMin) // Change from (min,max) to (0, newmax)
  point[0] = Math.round((point[0] / planeWidth) * (divisions-1)) // Change from (0, planeWidth) to (0, divisions)
  point[1] = Math.round((point[1] / planeHeight) * (divisions-1)) // Change from (0, planeHeight) to (0, divisions)
  return point;
}

function convert3JStoOM(point, divisions) {
  point[0] = (point[0] - planeXMin) // Change from (min,max) to (0, newmax)
  point[1] = (point[1] - planeYMin) // Change from (min,max) to (0, newmax)

  point[0] = Math.round((point[0] / planeW) * (divisions-1)) // Change from (0, planeWidth) to (0, divisions)
  point[1] = Math.round((point[1] / planeH) * (divisions-1)) // Change from (0, planeHeight) to (0, divisions)

  return point;
}

function convert3JStoLatLong(x, y) {
  return [x*155/10, y*180/10]
}

function distanceToLine(startPt, endPt, pt) {
  let t1 = (endPt[1] - startPt[1])*pt[0]
  let t2 = (endPt[0] - startPt[0])*pt[1]
  let t3 = (endPt[0] * startPt[1])
  let t4 = (endPt[1] * startPt[0])
  let t5 = Math.abs(t1 - t2 + t3 - t4)
  t5 /= Math.sqrt((endPt[1] - startPt[1])**2 + (endPt[0] - startPt[0])**2)
  return t5
}

function dist(startPt, endPt) {
  return Math.sqrt((startPt[0] - endPt[0])**2 + (startPt[1] - endPt[1])**2)
}
