# Perceptron Library

This library provides a single ES Module file that contains the class `NeuralNetwork` which can be used to create a multilayer perceptron.

```js
// creates a neural network with 
//     128 input neurons
//     32 hidden neurons
//     2 ouput neurons
const nn = new NeuralNetwork(128,32,2);

// you can train the network using the 'learn' method
nn.learn(data, answers);
// e.g.
nn.learn([[0, 1]], [[1, 0]]);
// this will train the network that
// given the input [0, 1] it should
// output [1, 0]

// after training you can use the network
// by calling the 'createGuess' method
const guess = nn.createGuess([0, 1]);
// guess will contain an array describing
// the neural network's guess e.g.
[
    0.1283737,
    0.8822837
]
// the above would indicate the network
// guesses the answer is 1 (as index 1
// has the higher value)
```