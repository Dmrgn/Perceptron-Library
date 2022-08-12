import { Node } from "./node.js";

export class NeuralNetwork {

    constructor(ins, his, ous) {

        this.num_ins = ins;
        this.num_hiddens = his;
        this.num_outs = ous;

        this.hiddens = [];
        this.outs = [];

        for (var i = 0; i < this.num_hiddens; i++) {
            this.hiddens[i] = new Node(this.num_ins);
        }
        for (var i = 0; i < this.num_outs; i++) {
            this.outs[i] = new Node(this.num_hiddens);
        }

        this.createGuess = function (data) {
            //console.log("%c Creating guesses", "text-transform:uppercase; padding: 10px; background: rgb(10,60,100); color:white;");
            var data1 = [];
            for (var i = 0; i < this.num_hiddens; i++) {
                //console.log(data);
                data1[i] = this.hiddens[i].guess(data);
            }
            var data2 = [];
            for (var i = 0; i < this.num_outs; i++) {
                var x = this.outs[i].guess(data1);
                //console.log("Guess for out num " + i + " is: " + x);
                data2[i] = x;
            }
            //console.log(data2);
            return data2;
        }

        this.getCost = function (data, ex) {
            //console.log(data);
            var guess = this.createGuess(data);
            var diff = [];
            for (var i = 0; i < guess.length; i++) {
                diff[diff.length] = (guess[i] - ex[i]) ** 2;
            }
            var cost = 0;
            for (var i = 0; i < diff.length; i++) {
                cost += diff[i];
            }
            return cost;
        }

        this.getChangesToLayer = function (layer, ex) {
            var changes = [];
            for (var i = 0; i < layer.length; i++) {
                changes[i] = ex[i] - layer[i];
            }
            return changes;
        }

        this.getWeightsGrad = function (data, ex) {
            //console.log("%c Getting weights gradient", "text-transform:uppercase; padding: 10px; background: rgb(10,60,100); color:white;");
            var prev_layer_outs = [];
            var weightsGrad = [];
            for (var i = 0; i < this.num_hiddens; i++) {
                ////console.log(this.hiddens[i].guess(data));
                prev_layer_outs[i] = this.hiddens[i].guess(data);
            }
            weightsGrad[0] = [];
            for (var i = 0; i < this.num_outs; i++) {
                weightsGrad[0][i] = [];
                for (var j = 0; j < this.outs[i].weights.length; j++) {
                    var ratios = [];
                    ratios[0] = 2 * (this.outs[i].guess(prev_layer_outs) - ex[i]); // IDEA: check if works later
                    var x = this.outs[i].guessNoSig(prev_layer_outs);;
                    ratios[1] = Node.sigmoid(x) * (1 - Node.sigmoid(x)); // IDEA: check if works later
                    ratios[2] = prev_layer_outs[j]; // IDEA: check if works later
                    var ratio = ratios[0] * ratios[1] * ratios[2];
                    weightsGrad[0][i][j] = ratio * -1;
                }
            }
            weightsGrad[1] = [];
            for (var i = 0; i < this.num_hiddens; i++) {
                weightsGrad[1][i] = [];
                for (var j = 0; j < this.hiddens[i].weights.length; j++) {
                    var ratios = [];
                    for (var k = 0; k < this.outs.length; k++) {
                        ratios[k] = [];
                        ratios[k][0] = 2 * (this.outs[k].guess(prev_layer_outs) - ex[k]);
                        //console.log(ratios);
                        var x = this.outs[k].guessNoSig(prev_layer_outs);
                        ratios[k][1] = Node.sigmoid(x) * (1 - Node.sigmoid(x));
                        //console.log(ratios);
                        ratios[k][2] = this.outs[k].weights[i];
                        //console.log(ratios);
                        var x = this.hiddens[i].guessNoSig(data);
                        ratios[k][3] = Node.sigmoid(x) * (1 - Node.sigmoid(x));
                        //console.log(ratios);
                        ratios[k][4] = data[j];
                        //console.log(ratios);
                    }
                    var ratio = ratios[0][0] * ratios[0][1] * ratios[0][2] * ratios[0][3] * ratios[0][4] + ratios[1][0] * ratios[1][1] * ratios[1][2] * ratios[1][3] * ratios[1][4];
                    weightsGrad[1][i][j] = ratio * -1;
                }
            }
            //console.log(weightsGrad);
            return weightsGrad;
        }

        this.getBiasGrad = function () {
            // IDEA: make this work
        }

        this.adjustWeights = function (av_outs, av_hiddens) {
            for (var i = 0; i < this.num_outs; i++) {
                for (var j = 0; j < this.outs[i].weights.length; j++) {
                    this.outs[i].weights[j] += av_outs[i][j];
                }
            }

            for (var i = 0; i < this.num_hiddens; i++) {
                for (var j = 0; j < this.hiddens[i].weights.length; j++) {
                    this.hiddens[i].weights[j] += av_hiddens[i][j];
                }
            }
        }

        this.learn = function (data, answers) {
            var full_data = data;
            var full_ans = answers;
            var formed_data = []
            var formed_ans = []
            for (var j = 0; j < full_data.length; j++) {
                formed_data[j] = full_data[j];
                formed_ans[j] = full_ans[j];
            }
            
            var costs = [];
            //console.log("%c Getting current cost", "text-transform:uppercase; padding: 10px; background: rgb(10,60,100); color:white;");
            while (formed_data.length > 0) {
                var rand_index = Math.floor(Math.random() * formed_data.length);
                var cur_data = formed_data[rand_index];
                var cur_ans = formed_ans[rand_index];
                formed_data.splice(rand_index, 1);
                formed_ans.splice(rand_index, 1);
                costs[costs.length] = this.getCost(cur_data, cur_ans);
            }

            var cost = 0;
            for (var j = 0; j < costs.length; j++) {
                cost += costs[j];
            }
            cost = cost / costs.length;

            // console.log("Found average cost of " + full_data.length + " examples: " + cost);

            var formed_data = full_data;
            var formed_ans = full_ans;

            var outs_weight_grad = [];

            var hiddens_weight_grad = [];

            while (formed_data.length > 0) {
                var rand_index = Math.floor(Math.random() * full_data.length);
                var cur_data = formed_data[rand_index];
                var cur_ans = formed_ans[rand_index];
                formed_data.splice(rand_index, 1);
                formed_ans.splice(rand_index, 1);
                var weights_grad_curr = this.getWeightsGrad(cur_data, cur_ans);
                outs_weight_grad[outs_weight_grad.length] = weights_grad_curr[0];
                hiddens_weight_grad[hiddens_weight_grad.length] = weights_grad_curr[1];
            }

            var av_outs_grad = [];
            var av_hiddens_grad = [];
            var av_outs_grad = [];

            //console.log("%c Getting average negetive gradient per output", "text-transform:uppercase; padding: 10px; background: rgb(10,60,100); color:white;");

            for (var j = 0; j < outs_weight_grad[0].length; j++) {
                av_outs_grad[j] = [];
                for (var k = 0; k < outs_weight_grad[0][j].length; k++) {
                    av_outs_grad[j][k] = 0;
                }
            }

            for (var j = 0; j < outs_weight_grad.length; j++) {
                for (var k = 0; k < outs_weight_grad[j].length; k++) {
                    for (var u = 0; u < outs_weight_grad[j][k].length; u++) {
                        av_outs_grad[k][u] += outs_weight_grad[j][k][u];
                    }
                }
            }

            for (var j = 0; j < av_outs_grad.length; j++) {
                for (var k = 0; k < av_outs_grad[j].length; k++) {
                    av_outs_grad[j][k] = av_outs_grad[j][k] / outs_weight_grad.length;
                }
            }

            //console.log("Got average output grad:");
            //console.log(av_outs_grad);
            //console.log("%c Getting average negetive gradient per hidden", "text-transform:uppercase; padding: 10px; background: rgb(10,60,100); color:white;");

            for (var j = 0; j < hiddens_weight_grad[0].length; j++) {
                av_hiddens_grad[j] = [];
                for (var k = 0; k < hiddens_weight_grad[0][j].length; k++) {
                    av_hiddens_grad[j][k] = 0;
                }
            }

            for (var j = 0; j < hiddens_weight_grad.length; j++) {
                for (var k = 0; k < hiddens_weight_grad[j].length; k++) {
                    for (var u = 0; u < hiddens_weight_grad[j][k].length; u++) {
                        av_hiddens_grad[k][u] += hiddens_weight_grad[j][k][u];
                    }
                }
            }

            for (var j = 0; j < av_hiddens_grad.length; j++) {
                for (var k = 0; k < av_hiddens_grad[j].length; k++) {
                    av_hiddens_grad[j][k] = av_hiddens_grad[j][k] / hiddens_weight_grad.length;
                }
            }

            //console.log("Got average hidden grad:");
            //console.log(av_hiddens_grad);
            //console.log("%c Adjusting weights", "text-transform:uppercase; padding: 10px; background: rgb(10,60,100); color:white;");

            this.adjustWeights(av_outs_grad, av_hiddens_grad);

            return cost;
        }
    }
}