# An AI that can play connect 4, made with genetic deep learning
import random
import numpy as np
import math
import Queue 
import matplotlib.pyplot as plt
import pickle
from NeuralNet import NN
from board import Board

NUM_ROWS = 4
NUM_COLS = 4
empty_token = 0.001
player_1_token = 1
player_2_token = -1
connect_x = 3 # Number of pieces that must be connected



# Mates two networks together, averages the weights, returns a new offspring NN,
# must have the same topologies
def mate_networks(nn1,nn2):
	new_layer = []
	for layer in nn1.layers:
		new_layer.append(np.copy(layer))
	for l_idx in range(len(nn1.layers)):
		for n_idx in range(len(nn1.layers[l_idx])):
			for w_idx in range(len(nn1.layers[l_idx][n_idx])):
				# Pick either mom's or dad's with 50% 
				if np.random.binomial(1, 0.5):
					new_layer[l_idx][n_idx][w_idx] = nn1.layers[l_idx][n_idx][w_idx]
				else:
					new_layer[l_idx][n_idx][w_idx] = nn2.layers[l_idx][n_idx][w_idx]

	new_nn = NN([],[])
	new_nn.set_weights(new_layer)
	return new_nn



def get_random_topology(max_layers = 5, max_nodes_in_layer=20):
	layers = [0]*random.randint(1,max_layers)
	for i in range(len(layers)):
		layers[i] = random.randint(1,max_nodes_in_layer)
	print layers


def list_to_queue(l):
	q = Queue.Queue()
	for x in l:
		q.put(x)
	return q


# Returns the next player given the current player
def switch_player(player):
	if player == 1:
		return 2
	else:
		return 1




# Given scores for different moves, ranks the moves by putting them in a queue
def get_ranked_moves(scores):
	if type(scores) != list:
		scores = scores.tolist()[0]
	q = Queue.Queue()
	d = zip(scores,range(len(scores)))
	d.sort(key=lambda tup: tup[0],reverse=True)	
	for x in d:
		q.put(x[1])
	return q


# Inputs two neural networks, they do battle at Connect 4, the winning network is returned
def battle(nn1,nn2, verbose=False):
	# Create new board
	nns = [nn1, nn2]
	board = Board(NUM_ROWS, NUM_COLS, player_1_token, player_2_token, empty_token, connect_x)
	player = 1	# Player 1 goes first

	if verbose:
		board.display()
		print ""

	while True:
		nn = nns[player-1] # gets the network player

		probs = []
		if player == 1:
			probs = nn.feed_forward(board.flatten())
		if player == 2:
			probs = nn.feed_forward(board.flatten()) #Player 2 sees the opposite board

 		q = get_ranked_moves(probs)
		while q.empty() == False:
			move = q.get()
			if not board.check_full_column(move):
				board.play_move(move,player)
				if verbose:
					board.display()
					print ""
				break
		
		# Check win
		if board.did_win() != "No Winner Yet":
			return nn

		#Check tie. If there is a tie, just return NN1 as the winner
		if board.is_full():
			return nn1

		# Change turns
		player = switch_player(player)

# Returns the fitness of the population (high variance -> high )
def get_population_fitness(scores):
	return np.var(scores)


# Every nn competes against the other nns, nn with most wins is returned
# This helps us get more diversity in our gene pool
def round_robin(list_of_nns):
	score_card = [0]*len(list_of_nns) # Each index corresponds to an nn
	for i in range(len(list_of_nns)):
		for j in range(len(list_of_nns)):
			if i != j:
				nn1 = list_of_nns[i]
				nn2 = list_of_nns[j]
				winner = battle(nn1,nn2)
				if nn1 == winner:
					#print "player ", i, " wins"
					score_card[i] += 1
				else:
					score_card[j] += 1
					#print "player ", j, " winsp"

	return score_card


# NNs compete in connect 4, winners survive, losers are removed from the population
# Think of the NN list as being wrapped around in a circle. if you beat the 
# NNs to the left and right of you, you "survive"
def survival_contest(population):
	survivors = []
	for i in range(len(population)):
		nn_L = population[(i-1)%len(population)]
		nn_M = population[i]
		winner1 = battle(nn_L,nn_M)
		winner2 = battle(nn_M, nn_L)
		if winner1 == winner2 and winner1 not in survivors:
			survivors.append(winner2)
	if len(survivors) < 2:
		survivors.append(population[0])
		survivors.append(population[1])
	return survivors






debug = False




if __name__ == "__main__" and debug == False:
	input_size = NUM_ROWS * NUM_COLS 

	topology_type1 = [10,10, NUM_COLS]
	
	# Create initial population
	pop_size = 20
	expected_wins = pop_size-1
	population = []
	for i in range(pop_size):
		nn = NN(topology_type1,input_size)
		population.append(nn)

	n_generations = 1000
	# for loop goes here
	for i in range(n_generations):
		print str(i) + "/" + str(n_generations)
		# Compute 'fitness' via a round robin tournament
		# scores = round_robin(population)
		# print "scores :" , scores
		# print "num below: ", len([x for x in scores if x < expected_wins])
		
		# pop_scores = zip(population,scores)
		# pop_scores.sort(key=lambda tup: tup[1],reverse=True)
		# population = [x[0] for x in pop_scores if x[1] >= expected_wins] # Remove networks with bad scores
		population = survival_contest(population)
		n_survivors = len(population)
		print "Survival percentage: ", (n_survivors+0.0)/pop_size
		
		# Mutate some survivors
		for i in range((pop_size-n_survivors)/3):
			rand = random.randint(0,n_survivors-1)
			mutation = population[rand].mutate()
			population.append(mutation)

		# Make some randos
		for i in range((pop_size-n_survivors)/3):
			population.append(NN(topology_type1,input_size))

		# Mate some survivors
		while len(population) < pop_size:
			rand1 = random.randint(0,n_survivors-1)
			rand2 = random.randint(0,n_survivors-1)
			offspring = mate_networks(population[rand1],population[rand2])
			population.append(offspring)

	# Round robin tournament
	scores = round_robin(population)
	# Get number of wins for each network, sort by scores
	pop_scores = zip(population,scores)
	pop_scores.sort(key=lambda tup: tup[1],reverse=True)
	population = [x[0] for x in pop_scores] # Population sorted by scores
	nn1 = population[0]
	nn2 = population[1]
	print nn1.layers
	print nn2.layers

	battle(nn2,nn1,verbose=True) # Does the best network still win?
	nn1.save("nn1.weights")
	nn2.save("nn2.weights")
	print "#######################"
	battle(nn1,nn2,verbose=True) # Does the best network still win?


