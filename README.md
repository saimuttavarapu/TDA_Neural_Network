# Topological Data Analysis of Neural Networks (TDANN)
This package includes a pipepline with the following steps:
* get an input of ONNX serialised pre-trained neural network
* get an additional input of training dataset using which the network was trained
* construct a weighted graph based on NN weights, activations, and different measures of divergence between them
* compute topological properties of the graph using Gudhi
