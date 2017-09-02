import random
import numpy as np
import math
import pickle

# Activation funtions
# Sigmoid 
def sig(x):
	# inqeualities help prevent float overflow errors
	if x > 100000:
		return 1
	elif x < -500:
		return 0
	else:
		return 1.0/(1.0+math.e**(-x))

# Relu
def relu(x):
	return max(0,x)

def action_potential(x):
	return int(x > 0)

# Leaky relu
def leaky_relu(x):
	if x > 0:
		return x
	else:
		return 0.01*x


# Allows activations to be run on numpy matrices
sig = np.vectorize(sig)
relu = np.vectorize(relu)
leaky_relu = np.vectorize(leaky_relu)
action_potential = np.vectorize(action_potential)



class NN:
	# Creates a new neural network with random weights
	# def __init__(self):
	# 	self.bottom_layer = None
	# 	self.hidden_layer = None
	# 	self.output_layer = None
	# 	self.num_hidden_neurons = 10
	# 	self.num_output_neurons = 7

	def __init__(self,layer_vector,input_length):
		# Each layer is fully connected

		# List of matrices
		self.layers = []
		self.biases = []

		for layer_idx in range(len(layer_vector)):
			if layer_idx == 0:
				new_layer = np.random.uniform(low = -0.1, high = 0.1, size= (layer_vector[0],input_length))
				self.layers.append(new_layer)
				self.biases.append(np.random.uniform(low = -1, high = 0.1, size= (1,layer_vector[0])))
			else:
				num_neurons = layer_vector[layer_idx]
				num_weights_per_neuron = len(self.layers[-1])
				new_layer = np.random.rand(num_neurons,num_weights_per_neuron)
				new_layer = np.random.uniform(low = -0.1, high = 0.1, size= (num_neurons,num_weights_per_neuron))
				self.layers.append(new_layer)
				self.biases.append(np.random.uniform(low = -0.1, high = 0.1, size= (1,num_neurons)))
	
	# Set weights of the neural net manually, really just used in spawning
	def set_weights(self,layers,biases):
		self.layers = layers
		self.biases = biases
	
	def set_label(self,label):
		self.label = label


	def feed_forward(self, data):
		activation = sig
		for i in range(len(self.layers)):
			# Use sigmoid for final layer
			if i == len(self.layers)-1:
				activation = sig
			layer = self.layers[i]
			data = np.dot(data, layer.transpose())
			data = activation(data+self.biases[i])
		return data


	# Returns a slightly modified version of this NN
	def mutate(self, mutation_probability = 0.5, mutation_range=0.01):
		# Create copy
		layer_copy = []
		biases_copy = []
		for layer in self.layers:
			layer_copy.append(np.copy(layer))
		biases_copy = list(self.biases)

		# For each weight, mutate it with probability mutation_probability, 
		for l_idx in range(len(layer_copy)):
			for n_idx in range(len(layer_copy[l_idx])):
				for w_idx in range(len(layer_copy[l_idx][n_idx])):
					biases_copy[l_idx][0][n_idx] += random.uniform(-mutation_range,mutation_range)
					if np.random.binomial(1,mutation_probability):
						layer_copy[l_idx][n_idx][w_idx] += random.uniform(-mutation_range,mutation_range)
		new_nn = NN([],[])
		new_nn.set_weights(layer_copy,biases_copy)
		return new_nn

	# Save the network to disk
	def save(self, filename):
		with open(filename, 'wb') as fp:
			pickle.dump(self.layers, fp)

    # Loads a network from a file on the disk
	def load(self, filename):
		with open (filename, 'rb') as fp:
			self.layers = pickle.load(fp)