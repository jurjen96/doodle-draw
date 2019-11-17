var canvas = document.getElementById('canvas')
var context = canvas.getContext('2d')
context.fillStyle = '#FFF'
context.lineCap = 'round'
context.fillRect(0, 0, canvas.width, canvas.height);
context.strokeStyle = 'black'
context.lineWidth = 8

let x = 0, y = 0
let isMouseDown = false

const stopDrawing = () => isMouseDown = false
const startDrawing = event => {
    isMouseDown = true;
   [x, y] = [event.offsetX, event.offsetY]
}
const drawLine = event => {
    if ( isMouseDown ) {
        const newX = event.offsetX
        const newY = event.offsetY
        context.beginPath()
        context.moveTo( x, y )
        context.lineTo( newX, newY )
        context.stroke();
        [x, y] = [newX, newY]
    }
}

canvas.addEventListener('mousedown', startDrawing)
canvas.addEventListener('mousemove', drawLine)
canvas.addEventListener('mouseup', stopDrawing)
canvas.addEventListener('mouseout', stopDrawing)

run = async () => {
  // Get the CanvasPixelArray from the given coordinates and dimensions.
  let imgData = context.getImageData(0,0,canvas.width,canvas.height) //.data
  let norm = tf.scalar(255.0)
  let tensor = tf.browser.fromPixels(imgData, 1)
                  .resizeNearestNeighbor([28,28])
                  .toFloat()
                  .div(norm)
                  .expandDims()

  const model = await tf.loadLayersModel('https://jurjen96.github.io/doodle-draw/tfjs_model/model.json')
  console.log(model)
  model.predict(tensor).data().then(predictions => {
    console.log(predictions)
  })
}

run()
