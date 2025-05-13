# Messy notes on some prerequisites

**Main:**

- <https://www.youtube.com/watch?v=aircAruvnKk> (Part 1)
- <https://www.youtube.com/watch?v=IHZwWFHWa-w> (Part 2)
- <https://www.youtube.com/watch?v=Ilg3gGewQ5U> (Part 3)

  **On neutral network parameters (weights, biases, activation function):**

- <https://www.youtube.com/watch?v=sM2Mm6aT_HI>
- <https://www.youtube.com/watch?v=SJ-hWwBF3zU>
- <https://www.youtube.com/watch?v=pg3hJpSopHQ>

Example/context of notes: mnist dataset (handwritten numbers)

## PART 1: What is a neural network?

(from pytorch.org)
Neural networks (NNs) are a collection of nested functions that are executed on some input data. These functions are defined by parameters (consisting of weights and biases), which in PyTorch are stored in tensors.

Training a NN happens in two steps:

Forward Propagation: In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.

Backward Propagation: In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent. For a more detailed walkthrough of backprop, check out this video from 3Blue1Brown.

---

- Input layer
- Hidden layer(s)
- Output layer

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

- (in our example) for the 2nd layer to pick up on smaller edges?, for the 3rd.. etcetc
- what "knobs" should we tweak so that it's exprsesive enough for layers to pick up on patterns etc

1. what you do is assign a _WEIGHT_ (just numbers) to each one of the connections betweene our neurons from one layer to another

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
  ***

## PART 2: How do neural networks **_learn?_**

context is still mnist

### Recap

28x28 = 784 pixels
each pixel grayscale value from 0 to 1
these determine the activation of the 784 neurons in the input layer of the
neural network
the activation of each neuron in the next layers is based on the weihted some on all the activations of the
previous layer + some specific bias then apply activation function

learning = tweaking weights and biases to alter correctness of model

motivation behind the layer structure is that 1st layer picksu p on edges, 2nd on bigger
patterns like loops and lines and last pieces patterns to recog digits

### Learning

weights and biases initially set randomly (example)

define cost function
it is done by

low if results are correct, large if netwrok is wrongish
get avg of cost of the thousands of training examples, this will bbe the measure
to how lousy the network is and to let it know how much its wrong

so
the network itself is basically a function, taking 784 numbers (pixels) and
outputs 10 numbers, parameterized by the 13,002 weights/biases

the cost function wraps the whole function, it takes the 13002 weights and biases
and outputs 1 number (the cost), it is defined by the networks
bhavior on the thousansds of training data.
so the cost function represents the specific configuration or parameters of
the model to produce a certain output or accuracy

so meaning.. the goal is to make the cost or the output of the cost function lower.

mathematically, u are to find the input of this function that will minimize
the value of this function

- not easy especially for complicated functions like the cost function of this example which
  contains like 13k weights
  (this example has like 2 hidden layers, practical projs have way more lol)
- instead of figuring out the perfect input (aka EXACT configuration of the thousands or millions or weights)
  that producin the perfect predictions on the model, the btter approahc is..

is to start at any input, then figure out which direction to go, to make the cost ouput lowe
figure out slope of where u are, shift to left is slope is positive, and vice versa..
you will find a **LOCAL** minimum..
analogy to this is a ball rolling down a hill(curvy graph), eventually settling down on a low point / valley

the local minimum is **doable**...
global minimum is very hard to find

l.

### Gradient Descent

moving to more complicated functions .. take a funcdtion with 2 inputs and 1 output

x y plane input space, the cost function being a graph above it
which direction decreases C(x,y) most quickly? (going donw hill)

in multivariable calculus the gradient of a function, gives u the direction of the steepest ascent
or which direction increases the function most quicklyu
taking the negative of this gives u the steepest slope

- the length of this gradient vector is an indication to how steep the steepest slope is

what matters rn is that a way to compute this vector exists. u dnt have to know actual multivar calc

#### Backpropagation

this is the algorithm that computes the gradient efficiently (pretty much how a neural network learns)

a neural network learning = minimizing the cost function

repeatedly nudging an input of af unction by some mulitple of the neagtive gradient
is called gradient descent

- to converge towards a local minimum of a cost function

#### Gradeint descent recap so far

neural network function

- input 784 numbers (pixels)
- output 10 numbers
- parameters (13,002 weights/biases)

cost function

- input: 13,002 weihgts/biases
- output 1 number(the cost) -- a single measure of lousiness
- parameters: the many training examples

the gradient of the cost function

- what changes to all these weights and biases cause the fastest change to the cost function
- "which changes to the parameters matter the most"

#### analysis of the network

..in the example with 2 hidden layers, we can get a 96% success on new imgs it hasnt seen

cant "draw" .. for ex inptting gibberish img will make it confidently output a digit
extremely constrained worldview

the setup of the network is actuallya rchaic in the vid. shit from the 80s . wala pa sa newwer technology like CNNs (convolutional neural netowrks), lstm, etc.
meaning foundational knowledge lang talaga and things here may or may not be the standard still
....though w further search fundamental (non negotiable)parts naman mostly in this vid and notes, like graient descent, cost function (mas nababasa ko "loss function" recently in place of this but same thing) all still necessary parts of implementing nad traininga neural network

## PART 3: Backpropagation

recap again: by learning, we mean what weights or biases minimize a cost function

the cost of a single training example:

- take output the network gives, along the output that you intended it to give (correct outuput)
- add up the squares of the differences of components (see pic)
- this will give the total cost of the network
- what we are looking for is the negative gradient of this cost function, which will tell
  how we need to change all the parameters to minimize the cost

backpropagation is an algorithm for computing that gradient.

### Walkthrough

## CONVOLUTIONAL NEURAL NETWORKS

<https://www.youtube.com/watch?v=8NbXFoVk1mo>
<https://www.youtube.com/watch?v=pj9-rr1wDhM>

<https://www.ibm.com/think/topics/convolutional-neural-networks>
<https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks>\

type of neural network that is distinguished from other types or subsets by its superior
performanmce with img, audio, speech inputs.

3 main layers:

- convolutional layer
- pooling layer
- fully connected (fc) layer

flatten layer - take multidimensional inp0ut lke img and make it into 1d (vector)

diff from normal neural networks (vanilla nn / fully conneted neural networks)

- no spacial awareness in vanilla nn, since filters/kernels arent used (grid that slides along an img to learn patterns)
- the whole grid or res say 28x28, like in the example earlier it just gets turned into a vector with 784 value
  and that gets passed to the next layers as usual, with no filter or pooling steps like in CNNs
- useable parin normal nn for basic computer vision like mnist

- typically utilized for classicifcation and computer vision tasks
- under the hood its just a shitton of matrix multiplication (imgs are 2d grid represent by matrix, filters are matrices as well and other things, and everythig in between is just matrix operations)

### Filters

aka kernels aka ?
performs the pattern recognition

goes through matrix of input img (say 28x28), applies convoltuional opearation that will produce
a new output based on the feature it s set to "highlight".
goes line by line and magnitude of this is called the filter _stride_

padding -

### Pooling

### Types of CNNs

- resnet, alexnet, vggnet, googlenet
