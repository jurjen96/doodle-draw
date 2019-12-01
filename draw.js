const labels = [
  'car', 'sailboat', 'bird', 'broom', 'butterfly', 'candle', 'clock',
  'flashlight', 'flower', 'airplane', 'house', 'violin', 'sock', 'saw' ]

const ranges = [0, 0.6, 0.9] // Depending on certainty replace the text how certain it is
const certainty_text = ["I'm not sure, but is it $A_AN:", "It looks like $A_AN:", "I see $A_AN:"]
const vowels = ['a', 'e', 'i', 'o', 'u'] // Used to check how we should replace $A_AN in certainty_text above

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
  if (isMouseDown)
    predict()
  isMouseDown = false
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
    context.moveTo(x, y)
    context.lineTo(newX, newY)
    context.stroke();
    [x, y] = [newX, newY]
  }
}

canvas.addEventListener('mousedown', startDrawing)
canvas.addEventListener('mousemove', drawLine)
canvas.addEventListener('mouseup', stopDrawing)
canvas.addEventListener('mouseout', stopDrawing)

predict = async () => {
  // Get the CanvasPixelArray from the given coordinates and dimensions.
  let imgData = context.getImageData(0,0,canvas.width,canvas.height)

  let norm = tf.scalar(255.0)
  let invert = tf.scalar(1.0)
  let tensor = tf.browser.fromPixels(imgData, 1)
                  .resizeBilinear([28,28])
                  // .resizeNearestNeighbor([28,28])
                  .toFloat()
                  .div(norm) // Normalize the data from 255 max to 1.0
  tensor = invert.sub(tensor) // Invert the colors (Black becomes white)
  tensor = tensor.toFloat()
                 .expandDims()

  // Uncomment line below for local development
  // const model = await tf.loadLayersModel('http://127.0.0.1:8887/object_model/model.json')
  const model = await tf.loadLayersModel('https://jurjen96.github.io/doodle-draw/object_model/model.json')
  let predictions = await model.predict(tensor).data()
  let predicted = await tf.argMax(predictions).data()

  sorted = [...predictions] // Create a copy of predictions

  sorted.sort((a, b) => b - a) // Sort based on prediction value from high -> low
  $('#prediction').html(labels[predicted])

  // Set the text how certain the network is that it is able to recognize a certain doodle
  for (range of ranges.slice().reverse()) {
    if (predictions[predicted] > range) {
      let text = certainty_text[ranges.indexOf(range)]
      // Replace the $A_AN depending if the recognized object word start with an vowel
      if (vowels.includes(labels[predicted][0]))
        text = text.replace('$A_AN', 'an')
      else
        text = text.replace('$A_AN', 'a')
      $('#certainty-text').html(text)
      break
    }
  }

  // Show the result for the best 5 predictions
  $('#advanced .content').empty()
  sorted.slice(0, 5).forEach((prediction) => {
    let div = $('<div>')
    div.html(labels[predictions.indexOf(prediction)] + ' ' + (prediction * 100).toFixed(1) + '%')
    div.appendTo($('#advanced .content'))
  })
}

reset = () => {
  // Resets everything to default value -> prediction is set to void
  context.fillRect(0, 0, canvas.width, canvas.height)
  $('#certainty-text').html('I think I saw a:')
  $('#prediction').html('Void')

  $('#advanced .content').empty()
  for (let _ of Array(5).keys()) {
    let div = $('<div>')
    div.html('Void 100.0%')
    div.appendTo($('#advanced .content'))
  }
}

$('#clear').click(reset)

$('#advanced .header').click(() => {
    $('#advanced .content').slideToggle(500, () => {
         // $('#advanced .content').is(':visible') ? 'Collapse' : 'Expand';
    })
})

reset()
