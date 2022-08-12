// Example usage:

// Train the neural network to count the
// total number of 1s in a passed array.
// e.g:
// Input [0,1,0,1]
// Ouput [0,0,1,0,0]
// Explaination (index 2 is a 1 because 
//              there are 2 zeros in the
//              input array)

// import the neural network class
import { NeuralNetwork } from "./library/neuralNetwork.js";

// utility for generating data
function random_max(x) {
    return Math.floor(Math.random() * (x+1));
}

// create an instance of the neural network
const nn = new NeuralNetwork(4, 4, 5);

// generated data
let data = [];
let answers = [];

// generate 1000 data points
for (let i = 0; i < 100; i++) {
    // array of length 4 with random numbers that are either 0 or 1
    const cur_data = [random_max(1), random_max(1), random_max(1), random_max(1)];
    data.push(cur_data);

    // the answer will be the total number of 1s in the data array
    let cur_answer = [0,0,0,0,0];
    // index 0 will be 1 if there are zero 1s in cur_data
    // index 1 will be 1 if there is one 1 in cur_data
    // etc..
    cur_answer[rand.reduce((x,y)=>{return x+y})] = 1;
    answers.push(cur_answer);
}

// train 10000 times
for (let i = 0; i < 10000; i++) {
    console.log("The average cost over 100 data points is:",nn.learn([...data],[...answers]));
}

// run 4 test cases
// the largest value in the outputed
// array is the neural networks guess
console.log(nn.createGuess([1,0,1,0]));
console.log(nn.createGuess([1,1,1,0]));
console.log(nn.createGuess([0,0,1,0]));
console.log(nn.createGuess([0,0,1,1]));