import random
import numpy as np
import math
import Queue 


class Board:
	def __init__(self, NUM_ROWS, NUM_COLS, player_1_token, player_2_token, empty_token,connect_x):
		self.board = []
		self.NUM_ROWS = NUM_ROWS
		self.NUM_COLS = NUM_COLS
		self.player_1_token = player_1_token
		self. player_2_token = player_2_token
		self.empty_token = empty_token
		self.connect_x = connect_x

		row = []
		for i in range(self.NUM_COLS):
			row.append(self.empty_token)
		for i in range(self.NUM_ROWS):
			self.board.append(list(row))

	# Display the board in ASCII
	def display(self):
		for row in range(self.NUM_ROWS):
			line = ""
			for col in range(self.NUM_COLS):
				if self.board[row][col] == self.player_1_token:
					line += "|  " + str(self.player_1_token) + " "
				if self.board[row][col] == self.player_2_token:
					line += "| " + str(self.player_2_token) + " "
				elif self.board[row][col] == self.empty_token:
					line += "|  " + str(0) + " "
				
			print line
			print "-----------------------------------"

	# Place the given player's token in a given column
	def play_move(self,column,player):
		# Column: which column to play on
		# Player: which player is playing, 1 or 2
		token = ""
		if player == 1:
			token = self.player_1_token
		else:
			token = self.player_2_token

		# Find the row the piece would fall in, find last 0
		column_values = [c[column] for c in self.board]
		row = self.rindex(column_values,self.empty_token)
		self.board[row][column] = token




	# returns true if there is still room in the given column for a piece
	def check_full_column(self, column):
		column_values = [c[column] for c in self.board]
		if len([c for c in column_values if c != self.empty_token])== self.NUM_ROWS:
			return True

	# Check to see if a player has won the game
	def did_win(self):
		# Check to see if these sequences are subsequences of rows, columns, or diags in the board
		win_1 = [self.player_1_token]*self.connect_x
		win_2 = [self.player_2_token]*self.connect_x

		# Check horizontal wins
		for row in self.board:
			if self.is_subsequence(win_1, row):
				return "Player 1 Wins row"
			elif self.is_subsequence(win_2, row):
				return "Player 2 Wins row"
		# Check column wins
		for col in range(self.NUM_COLS):
			column = [x[col] for x in self.board]
			if self.is_subsequence(win_1, column):
				return "Player 1 Wins col"
			elif self.is_subsequence(win_2, column):
				return "Player 2 Wins col"

		# Check for diagonal wins
		a = np.array(self.board)
		diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
		diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
		for diag in diags:
			if self.is_subsequence(win_1, diag):
				return "Player 1 Wins diag"
			elif self.is_subsequence(win_2, diag):
				return "Player 2 Wins diag" 

		return "No Winner Yet"


	# Checks the board to see if someone one. Returns that player if so, or else returns None
	def is_subsequence(self, x,y):
		if type(y) != list:
			y = y.tolist()
		for i in range(len(y)-len(x)+1):
			if x == y[i:i+len(x)]:
				return True
		return False

	# Finds the last index of a item in a list
	def rindex(self,mylist, myvalue):
	    return len(mylist) - mylist[::-1].index(myvalue) - 1


    # Returns true if the board is full, false otherwise
	def is_full(self):
		for row in self.board:
			for col in row:
				if col == self.empty_token:
					return False
		return True

	# Returns a list representation of the board
	def flatten(self):
		return np.matrix([item for sublist in self.board for item in sublist])

	# Used for the 2nd player
	def invert_and_to_list(self):
		inv_board = []
		for row in self.board:
			new_row = []
			for x in row:
				if x != self.empty_token:
					new_row.append(self.player_2_token)
				else:
					new_row.append(self.empty_token)
			inv_board.append(new_row)

		return np.matrix([item for sublist in inv_board for item in sublist])


