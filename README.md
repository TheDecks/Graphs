# Graphs

This code was used in my master thesis. It provides a way of communicating with KEGG PATHWAY database via its API to 
parse metabolic pathways into graph structure. The latter has its own lightweight implementation that creates 
substructures not by copying them into new objects but rather by storing appropriate references.

The files include implementation for graph structure, naive Neural Network with feedforward and backpropagation 
(adjustable only with number of layers, neurons in them, activation and cost functions) and methods for node and graph
classification.