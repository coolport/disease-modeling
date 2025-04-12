# Messy notes on some prerequisites

Main: <https://www.youtube.com/watch?v=aircAruvnKk>\
On neutral network parameters (weights, biases, activation function)

- <https://www.youtube.com/watch?v=sM2Mm6aT_HI>
- <https://www.youtube.com/watch?v=SJ-hWwBF3zU>

## What is a neural network?

Example/ontext of notes: mnist dataset (handwritten numbers)

### Neurons

thing that holds a number ideally from 0 to 1

**in THIS case (mnist, image recognition)**\
take a picture of a digit with res of 28x28,
imagine all thse 784 pixels as neurons

a neuron can hold a value 0-1 correspedning to its grayscale value. (darkness, lightness)

- we call this the activation
- the neuron or pixel is lit up when it is 1, black or darkness with 0

Let's assign these 784 pixels as the first layer of our neural network

layers get smaller and smaller, with the last layer represending the output
only having 10 neurons or nodes, still holding the same activation value of 0-1
which represents how much the system or network thinks that this particular number
represents the picture the best

between these 2, are "Hidden layers"

for this example we have 2 hidden layers, each with 16 neurons each
activation of the last layer influence activaiton in the next layer

Q: how does one layer influences the next?
how does training work?

## Why the layered structure?

component ro pattern recognition, take 9, loop up top, line below it etc.
we hope that the second to the last layer (layer before output), has neurons that corresponed to these patterns or components, and we just need to know the ocmbination of pieces

this example's 2nd layer hope: smaller edges matchinhg that build the actual shapes in 3rd layer

generally this concept can be applied to everything in image recognition,speech recognition ,etce tec
refinement, from understanding smaller patterns and patching them up togfethr type shit; abstraction

## What parameters should the network have?

- (in our example) fo the 2nd layer to pick up on smaller edges?, for the 3rd.. etcetc
- what knobs should we tweak so that it's exprsesive enough for layers to pick up on patterns etc

1. what you do is assign a _WEIGHT_ (just numbers) to each one of the connections betweene our neurons (2nd layer ucrrently speakng) and the neurons from the first layer

2. take the activations from the first layer and compute their weighted sum according to these weights

### Weights

### Biases

### Activation Function

outputs from neurons (computed using features and their weights, added up and adjusted for bias)
are not useful YET.

we pass them through an activation function which **DECIDES ACTIVITY**

taken from sigmund in the video: small / negative values become near 0 - **INACTIVE**
bigger values become nearer to 1, **active**

like a gate ganun

to summarize, these are what activation functions do:

- make the original values useful, by acting as decision makers / filters
  and converters, that shape how much a specific neuron's output matters moving forward hte layers..

#### ReLU

ReLU(x) = max(0, x)
much more simple than sigmoid

- if input is positive, keep
- if input is negative, becomes 0
  neg = stay out, positive = unchanged, and see difference in higher values, not limited by 1

#### (From Video) Sigmoid Function

"squish weighted sums and make it fit into 0 to 1"

- squashes values to make everything fit into 0 and 1

  **NOT THE STANDARD ANYMORE (especially in hidden layers)**
  **REPLACED BY**:

- ReLU (rectified linear unit) way more common now -- maybe THE standard?
- and ReLU's variants
- still applied in some cases like sometimes in output layers etc

## Notationally compact way of presenting these connectoins : Practical arrangement of things (in general)

or basically how all of this is handled in practice

- organize all activations of one layer into a collumn as a vector
- organize all weights as a matrix, where each row is a connection from one layer to a particular neuron
- meaning taking the weighted sum of the activatoins in the first layer corresponsds t o one of the terms in the matrix vector product in the weights matrix
- organize biases as a vector and add the entire vector to the previous matrix+vector product
- wrap a sigmoid (or other activation function) throughout the whole expression (apply sigmoid function to each specific component of
  of the resulting vector inside)
- insert pic here

## Redefining neurons

- each neuron is more of a function that a simple number holder
- when it takes the outputs of all the functions of the previous layer and outputs a 0-1 value
