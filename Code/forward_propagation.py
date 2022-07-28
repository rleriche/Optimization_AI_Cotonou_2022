"""
    A collection of methods for forward propagation in a neural network
    
    @author: Rodolphe Le Riche, Kevin Kpakpo, Brian Dedji Whannou
"""
import numpy as np
from typing import Union, Callable
from copy import deepcopy

# TODO: make sure that the list of weights is coherent
def forward_propagation(inputs: np.ndarray,
               weights: list[np.ndarray],
               activation_functions: Union[Callable,list[Callable],list[list[Callable]]]) -> np.ndarray:
    """
    Returns the output of the network.

            Parameters:
                    inputs: is an array of shape n,p 
                    (with p being the number of input features and n the number of observations)
                    weights: list of matrices. 
                             Each matrix represents a layer. 
                             The number of rows is the number of nodes. 
                             The number of columns is the number inputs of the layer (with the bias)
                    activation_functions: the user can provide either
                        * one activation function for the whole network (one callable)
                        * one activation function for each layer (list of callable)
                        * one activation function per nodes (list of list of callable)
            Outputs:
                    layer_output: is an array of shape n,q
                    (with q being the number of nodes in the last layer and n the number of observations)

    """
    
    # number of nodes per layers
    network_structure = []
    for layer_weights in weights:
        network_structure.append(layer_weights.shape[0])

    # all activation functions
    activation_functions_parsed = parse_activation_function(activation_functions, network_structure)
    
    bias = np.repeat(np.array([[1]]), inputs.shape[0], axis=0)

    layer_input = np.append(inputs,bias,axis = 1)

    for func,layer_weights in zip(activation_functions_parsed,weights):
        layer_combinaison = layer_weights.dot(layer_input.T)
        layer_output = apply_activation_functions(func,layer_combinaison.T)
        layer_input = np.append(layer_output.T,bias,axis = 1)

    return layer_output.T

def parse_activation_function(activation_functions, network_structure) -> list[list[Callable]]:
    activation_functions_parsed = deepcopy(activation_functions)
    if isinstance(activation_functions, Callable):
        activation_functions_parsed = []
        for number_nodes in network_structure:
            layer_funcs = [activation_functions for n in range(0,number_nodes)]
            activation_functions_parsed.append(layer_funcs)
    
    if isinstance(activation_functions,list):
        if isinstance(activation_functions[0], Callable):
            activation_functions_parsed = []
            for func, number_nodes in zip(activation_functions, network_structure):
                activation_functions_parsed.append([func for n in range(0,number_nodes)])
    return activation_functions_parsed

def apply_activation_functions(functions: list[object], values: np.ndarray) -> np.ndarray:
    all_activated_values = np.zeros(values.shape)
    for i in range(0,values.shape[0]):
        row_values = values[i,:]
        all_activated_values[i,:] = [node_func(value) for node_func,value in zip(functions,row_values)]
    return np.array(all_activated_values).T


def create_layer_weights(inputs_size: int,outputs_size: int)-> np.ndarray:
    avg = 0
    std = np.sqrt(1/inputs_size)
    layer_weights = np.random.normal(avg,std,(outputs_size,inputs_size))
    
    return layer_weights

def create_weights(network_structure: list[int]) -> list[np.ndarray]:
    np.random.seed(42)
    weights = []
    for layer in range(0,len(network_structure)-1):
        weights.append(create_layer_weights(
            inputs_size = network_structure[layer] + 1,
            outputs_size = network_structure[layer + 1]
        ))
    return weights

def vector_to_weights(vector: np.ndarray,network_structure: list[int]) -> list[np.ndarray]:
    weights = []
    for layer in range(0,len(network_structure)-1):
        input_size = (network_structure[layer] + 1)
        output_size = network_structure[layer + 1]
        selected_parameters = np.array(vector[:(input_size * output_size)])
        weights.append(selected_parameters.reshape((output_size,input_size)))
        vector = vector[(input_size * output_size):]
    
    return weights