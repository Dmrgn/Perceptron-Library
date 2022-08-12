export class Node {

    // return the sigmoid of the passed number
    static sigmoid(num) {
        return 1/(1+(2.7182818284**(num*-1)));
    }

    constructor(previous) {
        this.weights = [];
        this.bias = 0; //Math.random() * 2 - 1;
        for (var i = 0; i < previous; i++) {
            this.weights[i] = Math.random() * 2 - 1;
        }

        this.guess = function (pinputs) {
            var sum = 0;
            for (var i = 0; i < pinputs.length; i++) {
                sum += pinputs[i] * this.weights[i];
            }

            sum += this.bias;
            sum = Node.sigmoid(sum);

            return sum;

        }

        this.guessNoSig = function (pinputs) {
            var sum = 0;
            for (var i = 0; i < pinputs.length; i++) {
                sum += pinputs[i] * this.weights[i];
            }
            sum += 1 * this.bias
            return sum;
        }

    }

}