from NeuralNet import NN
import math
import random
import numpy as np
# Input : integer x
# Output: 1 is x >5, 0 otherwise


def mate_networks(nn1,nn2):
	new_layer = []
	for layer in nn1.layers:
		new_layer.append(np.copy(layer))
	new_biases = list(nn1.biases)
	for l_idx in range(len(nn1.layers)):
		for n_idx in range(len(nn1.layers[l_idx])):
			for w_idx in range(len(nn1.layers[l_idx][n_idx])):
				# Pick either mom's or dad's with 50% 
				if np.random.binomial(1, 0.5):
					new_layer[l_idx][n_idx][w_idx] = nn1.layers[l_idx][n_idx][w_idx]
					new_biases[l_idx][0][n_idx] = nn1.biases[l_idx][0][n_idx]
				else:
					new_layer[l_idx][n_idx][w_idx] = nn2.layers[l_idx][n_idx][w_idx]
					new_biases[l_idx][0][n_idx] = nn2.biases[l_idx][0][n_idx]

	new_nn = NN([],[])
	new_nn.set_weights(new_layer,new_biases)
	return new_nn


def get_error(prediction, actual):
	return abs(prediction-actual)

# inputs a list, normalizes it
def normalize(l):
	return [float(x)/sum(l) for x in l]

def evolve():
	pop_size = 20
	num_generations = 1000
	topology = [3,10,1]

	population = []
	for i in range(pop_size):
		nn = NN(topology,1)
		nn.set_label("rando")
		population.append(nn)

	for gen in range(num_generations):
		X = [-5,-4,-3,-2,-1,0, 1,2,3,4,5]
		#X = normalize(X)
		Y = [1,  1, 1, 0, 0,0, 0,0,1,1,1]

		scores = [0]*len(population)
		for i in range(len(population)):
			for j in range(len(X)):
				nn = population[i]
				
				prediction = nn.feed_forward(X[j]).tolist()[0][0]
		
				scores[i] += get_error(prediction, Y[j])
		#print scores

		pop_scores = zip(population,scores)
		pop_scores.sort(key=lambda tup: tup[1],reverse=False)
		population = [x[0] for x in pop_scores] # Population sorted by scores

		population = population[:10]
		# add offspring
		# for i in range(1):
		# 	nn = mate_networks(population[0],population[1])
		# 	nn.set_label("mated")
		# 	population.append(nn)
		for i in range(10):
			nn = population[0].mutate()
			nn.set_label("mutated")
			population.append(nn)

		# for i in range(4):
		# 	nn = NN(topology,1)
		# 	nn.set_label("rando")

		# 	population.append(nn)

	return population



best_population = evolve()
print best_population[0].label
print best_population[0].layers
print best_population[0].biases
X = range(-10,10)

#X = normalize(X)
for x in X:
	print "x:, ", x
	print best_population[0].feed_forward(x).tolist()[0][0]

"""
mutated
[array([[-5.66207496],
       [ 2.45387338],
       [-2.99396226]]), array([[ -8.53633555,   3.04907534,   8.41395783],
       [-11.41909576,  10.66136192,  17.27101726],
       [ -1.39784027,   0.15280367,   0.35014985],
       [  2.33585397, -12.48569325, -16.2805977 ],
       [  1.38256497,   1.10566531,   1.37017495],
       [ -0.37339448,  -0.15164508,   1.98587603],
       [ 13.37083537,  -0.63794105, -13.61564045],
       [  0.14136495,   1.62473779,  -0.46648139],
       [  5.79623082, -10.14179966, -15.96324309],
       [-11.1529537 ,   3.25644786,  17.79226596]]), array([[  1.01876919e+01,   3.30994112e+01,   1.60734578e+00,
         -3.42095235e+01,  -2.10632030e+01,  -5.94132955e+00,
         -2.71721411e+01,  -1.02891330e-02,  -3.72545254e+01,
          2.05158065e+01]])]
[array([[ 13.95962024,  -4.83441126,  -6.94529954]]), array([[  2.91116782,  -0.57785947, -17.21024102,   9.21742625,
          9.84804838,  15.04826913,  -2.91660207,  -4.18990023,
          7.24000631,  -3.0342725 ]]), array([[ 23.90028595]])]
x:,  -10
1.0
x:,  -9
1.0
x:,  -8
1.0
x:,  -7
1.0
x:,  -6
1.0
x:,  -5
1.0
x:,  -4
1.0
x:,  -3
1.0
x:,  -2
1.03716659198e-44
x:,  -1
6.82665335457e-45
x:,  0
6.78887228293e-45
x:,  1
6.86129263491e-45
x:,  2
1.29872316288e-44
x:,  3
1.0
x:,  4
1.0
x:,  5
1.0
x:,  6
1.0
x:,  7
1.0
x:,  8
1.0
x:,  9
1.0
[Finished in 2218.1s]"""
