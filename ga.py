import numpy as np
import timeit
import os

def selection(pop, scores, k=3):
	# first random selection
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if np.random.rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = np.random.randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if np.random.rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, decode, n_bits, bounds, n_iter, n_pop, r_cross, r_mut, debug=False):
	# init time
	init_time = timeit.default_timer()
	# best scores
	best_scores = []
	# avgs
	avg = []
	# initial population of random bitstring
	pop = [np.random.randint(0, 2, n_bits * np.random.randint(1, bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(decode(pop[0]))
	# enumerate generations
	for gen in range(n_iter):
		print(f'> gen: {gen}, time: {timeit.default_timer() - init_time:.2f}s')
		# decode population
		### decoded = [decode(bounds, n_bits, p) for p in pop]
		decoded = [decode(p) for p in pop]
		# evaluate all candidates in the population
		scores = [objective(d) for d in decoded]

		avg.append(sum(scores)/len(scores))

		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				if debug:
					print(f'> gen: {gen}, new best {decoded[i]} = {-1 * scores[i]}')
				else:
					print(f'> gen: {gen}, new best = {-1 * scores[i]}')

		best_scores.append(best_eval)

		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)

				# implement underflow check here

				# store for next generation
				children.append(c)
		# replace population
		pop = children

		print(f'> gen: {gen}, average fitness = {avg[gen]}, std = {np.std(scores)}')

		f = open(f'{os.getcwd()}\\best.txt', 'w+')
		f.write(''.join(str(i) for i in best))
		f.close()

	return [best, best_eval, best_scores, avg]