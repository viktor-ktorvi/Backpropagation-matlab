# Backpropagation-matlab
Implementing backprop in MATLAB because I needed it for online learning in Simulink

## TODO
* I forgot to add biases to hidden layers. It should be that in the forward pass every mid result(input, first hidden ... all but the last one) gets ones as long as the batch size concatenated. Also, in the constructor the layer sizes should get a +1 before the weights are initialized.
