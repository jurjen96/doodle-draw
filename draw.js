const labels = [
  'car', 'chair', 'bird', 'broom', 'butterfly', 'candle', 'clock',
  'flashlight', 'flower', 'airplane', 'house', 'violin', 'sock' ]

var canvas = document.getElementById('doodle-canvas')
var context = canvas.getContext('2d')
context.fillStyle = '#FFF'
context.lineCap = 'round'
context.fillRect(0, 0, canvas.width, canvas.height);
context.strokeStyle = 'black'
context.lineWidth = 12

let x = 0, y = 0
let isMouseDown = false

const stopDrawing = () => {
  isMouseDown = false
  run()
}
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
// canvas.addEventListener('mouseout', stopDrawing)

run = async () => {
  // Get the CanvasPixelArray from the given coordinates and dimensions.
  let imgData = context.getImageData(0,0,canvas.width,canvas.height)

  let norm = tf.scalar(255.0)
  let invert = tf.scalar(1.0)
  let tensor = tf.browser.fromPixels(imgData, 1)
                  .resizeBilinear([28,28])
                  // .resizeNearestNeighbor([28,28])
                  .toFloat()
                  .div(norm) // Normalize the data from 255 max to 1.0
                  // .invert()
  tensor = invert.sub(tensor) // Invert the colors
  tensor.print()
  console.log(tensor.dataSync())
  // tf.browser.toPixels(tensor, document.getElementById("canvas2"))
  tensor = tensor.toFloat()
                 .expandDims()


  const model = await tf.loadLayersModel('https://jurjen96.github.io/doodle-draw/object_model/model.json')
  let predictions = await model.predict(tensor).data()
  console.log(predictions)
  let predicted = await tf.argMax(predictions).data()
  console.log(predicted)
  console.log(labels[predicted])
  $("#prediction").html(labels[predicted])
}

$("#clear").click(() => {
  console.log("Clearing")
  context.fillRect(0, 0, canvas.width, canvas.height);
})

$("#next").click(() => {
  console.log("Guessing")
  run()
})
