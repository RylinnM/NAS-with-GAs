# new version
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, time
from absl import app
from nasbench import api
import numpy as np
import nas_ioh

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

# Parameters setting
pop_size = 4
pop_mu = 4
pop_lam = 20
learning_rate = 1.0 / np.sqrt(26)
np.random.seed(0)

def evolution_strategy(bench, population=[], pop_sigmas=[]):
    # def a function to convert a 3 bit binary number to a integer
    def bin2int(bitstring):
        return int(''.join(str(i) for i in bitstring), 2)

    # def a function to convert a 5 bit ternary number to a integer
    def tern2int(bitstring):
        return int(''.join(str(i) for i in bitstring), 3)

    # def a function to convert an integer to a 3 bit binary number, fill the left with 0
    def int2bin(real):
        return [int(i) for i in list('{:03b}'.format(real))]

    # def a function to convert an integer to a 5 bit ternary number, fill the left with 0
    def int2tern(real):
        return [int(x) for x in np.base_repr(real, 3).zfill(5)]

    # def a round function to round a float number to its nearest integer
    def rounding(x):
        return int(x + 0.5)

    # def a function to round every element in a list
    def rounding_list(x):
        return [rounding(i) for i in x]

    # def a encode function to encode a module to a bitstring
    def decode(real_list):
        real_list = rounding_list(real_list)
        bitstring = []
        for i in range(7):
            bitstring.extend(int2bin(real_list[i]))
        bitstring.extend(int2tern(real_list[-1]))
        return bitstring

    def encode(bitstring):
        real_list = []
        for i in range(7):
            real_list.append(bin2int(bitstring[i * 3:(i + 1) * 3]))
        real_list.append(tern2int(bitstring[-5:]))
        return real_list

    def check_valid(bs):
        matrix = np.zeros((7, 7))
        matrix[np.triu_indices(7, 1)] = bs[:21]
        ops = [0]
        ops.extend(bs[21:])
        ops.append(0)
        for bit in range(len(ops)):
            if ops[bit] == 1:
                ops[bit] = CONV1X1
            elif ops[bit] == 2:
                ops[bit] = CONV3X3
            elif ops[bit] == 0:
                ops[bit] = MAXPOOL3X3
        ops[0] = INPUT
        ops[6] = OUTPUT
        model_spec = api.ModelSpec(matrix=matrix, ops=list(ops))
        if bench.is_valid(model_spec):
            return True
        else:
            return False

    # def a init function to init a population of 7 real numbers of 0-7 and1 real number of 0-242
    def init_pop():
        real_string = np.random.uniform(0, 7, 7).tolist() + [np.random.uniform(0, 242)]
        rounded = rounding_list(real_string)
        if check_valid(decode(rounded)):
            return real_string

    # def a function to initialize the population sigma
    def init_sigma():
        sigma = np.random.uniform(0, 7)
        return sigma

    # def a recombination function to recombine two solutions
    def recombination(solution1, solution2, sigma1, sigma2):
        solution = []
        sigma = (sigma1 + sigma2) / 2
        for i in range(len(solution1)):
            solution.append((solution1[i] + solution2[i]) / 2)
        return solution, sigma

    # def a mutation function to mutate a solution
    def mutation(solution, sigma, tau0=learning_rate):
        sigma = sigma * np.exp(tau0 * np.random.normal(0, 1))
        for i in range(7):
            solution[i] = solution[i] + sigma * np.random.normal(0, 1)
            solution[i] = solution[i] if solution[i] <= 7 else 7
            solution[i] = solution[i] if solution[i] >= 0 else 0
        solution[-1] = solution[-1] + sigma * np.random.normal(0, 1)
        solution[-1] = solution[-1] if solution[-1] <= 242 else 242
        solution[-1] = solution[-1] if solution[-1] >= 0 else 0
        # check if the solution is valid, if not, redo the mutation
        if not check_valid(decode(rounding_list(solution))):
            return mutation(solution, sigma, tau0)

        return solution, sigma

    # def a function to select the best solution from a population
    def select_best(pop, scores):
        best = pop[0]
        for i in range(len(pop)):
            if scores[i] > scores[pop.index(best)]:
                best = pop[i]
        return best

    def group_selection(pop, sigmas, scores, tournament_k=5):
        # tournament selection
        for _ in range(len(pop)):
            pre_select = np.random.choice(len(scores), tournament_k, replace=False)
            best = pop[pre_select[0]]
            sigma = sigmas[pre_select[0]]
            max_f = scores[pre_select[0]]
            for p in pre_select:
                if scores[p] > max_f:
                    max_f = scores[p]
                    best = pop[p]
                    sigma = sigmas[p]
        return best, sigma

    f_opt = sys.float_info.min
    pop = []
    pop_sigmas = []
    parents = []
    parent_sigmas = []

    # initialize the population and the population sigma
    while len(population) < pop_size:
        solution = init_pop()
        if solution is not None:
            population.append(solution)
    while len(pop_sigmas) < pop_size:
        pop_sigmas.append(init_sigma())

    pop = population

    # select mu parents from the population
    while len(parents) < pop_mu:
        idx = np.random.choice(len(pop), pop_mu, replace=False)
        for i in idx:
            parents.append(pop[i])
            parent_sigmas.append(pop_sigmas[i])

    # recombine the parents to generate lam children
    children = []
    children_sigmas = []
    while len(children) < pop_lam:
        idx_chosen = np.random.choice(len(parents), 2, replace=False)
        solution1, solution2 = parents[idx_chosen[0]], parents[idx_chosen[1]]
        sigma1, sigma2 = parent_sigmas[idx_chosen[0]], parent_sigmas[idx_chosen[1]]
        child, child_sigma = recombination(solution1, solution2, sigma1, sigma2)
        children.append(child)
        children_sigmas.append(child_sigma)

    # mutate the children
    for i in range(len(children)):
        children[i], children_sigmas[i] = mutation(children[i], children_sigmas[i])

    # evaluate the children
    evaluation_pop = children
    evaluation_pop_sigmas = children_sigmas
    evaluation_fitness = []
    for i in range(len(evaluation_pop)):
        evaluation_fitness.append(nas_ioh.f(decode(evaluation_pop[i])))

    x_opt = select_best(evaluation_pop, evaluation_fitness)
    f_opt = evaluation_fitness[evaluation_pop.index(x_opt)]
    x_opt_decoded = decode(x_opt)
    next_pop = []
    next_pop_sigmas = []
    while len(next_pop) < pop_size:
        best, sigma = group_selection(evaluation_pop, evaluation_pop_sigmas, evaluation_fitness)
        next_pop.append(best)
        next_pop_sigmas.append(sigma)
    return x_opt, f_opt, next_pop, next_pop_sigmas, x_opt_decoded


def main(argv):
    del argv  # Unused
    for r in range(nas_ioh.runs):  # we execute the algorithm with 20 independent runs.
        f_best = sys.float_info.min
        population = []
        population_sigmas = []
        for _ in range(int(nas_ioh.budget / (pop_lam))):
            x, y, population, population_sigmas, decoded_x = evolution_strategy(nas_ioh.nasbench, population,
                                                                                population_sigmas)  # budget as 5000
            if y > f_best:
                f_best = y
                x_best = x
                x_best_decoded = decoded_x
        print("run", r, ", best x:", x_best_decoded, ", f :", f_best)
        nas_ioh.f.reset()  # Note that you must run the code after each independent run.


# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
    start = time.time()
    app.run(main)
    end = time.time()
    print("The program takes %s seconds" % (end - start))

# reset the recording status of the function after exceeding the budget of 5000
