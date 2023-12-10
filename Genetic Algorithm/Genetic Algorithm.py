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
pop_size = 10
parent_mu = 10
offspring_lambda = 10
tournament_k = 5
pb_m = 0.25
pb_c = 0.6
np.random.seed(0)

def genetic_algorithm(bench, population=[]):
    def init_pop():
        bitstring = np.random.randint(0, 2, 21).tolist() + np.random.randint(0, 3, 5).tolist()
        if check_valid(bitstring):
            return bitstring

    def crossover(pa1, pa2):
        if np.random.uniform(0, 1) < pb_c:
            for i in range(len(pa1)):
                if np.random.uniform(0, 1) < 0.5:
                    temp = pa1[i]
                    pa1[i] = pa2[i]
                    pa2[i] = temp
        return pa1, pa2

    def mutation(bitstring):
        for i in range(len(bitstring)):
            if np.random.uniform(0, 1) < pb_m:
                if i < 21:
                    bitstring[i] = 1 - bitstring[i]
                else:
                    if bitstring[i] == 0:
                        bitstring[i] = 1 if np.random.uniform(0, 1) < 0.5 else 2
                    if bitstring[i] == 1:
                        bitstring[i] = 0 if np.random.uniform(0, 1) < 0.5 else 2
                    if bitstring[i] == 2:
                        bitstring[i] = 1 if np.random.uniform(0, 1) < 0.5 else 0
                if check_valid(bitstring):
                    return bitstring
                else:
                    continue

    def tournament_selection(pop, scores):
        # tournament selection
        for _ in range(len(pop)):
            pre_select = np.random.choice(len(scores), tournament_k, replace=False)
            best = pop[pre_select[0]]
            max_f = scores[pre_select[0]]
            for p in pre_select:
                if scores[p] > max_f:
                    max_f = scores[p]
                    best = pop[p]
        return best

    def rank_selection(pop, scores):
        # rank selection
        rank = np.argsort(scores)
        rank = rank[::-1]
        return pop[rank[0]]

    def roulette_selection(pop, scores):
        # roulette wheel selection
        scores = np.array(scores)
        scores = scores - np.min(scores)
        scores = scores / np.sum(scores)
        return np.random.choice(pop, 1, p=scores)[0]


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
        return True if bench.is_valid(model_spec) else False

    while len(population) < pop_size:
        x = init_pop()
        if x:
            population.append(x)
        else:
            continue

    pop = population
    scores = []
    f_opt = sys.float_info.min
    parents = []
    offsprings = []
    # randomly choose items from pop as parents
    idx = np.random.choice(len(pop), parent_mu, replace=False)
    for i in idx:
        parents.append(pop[i])

    while len(offsprings) < offspring_lambda:
        rdc = np.random.choice(len(parents), 2, replace=False)
        off1, off2 = crossover(parents[rdc[0]], parents[rdc[1]])
        offsprings.append(off1)
        offsprings.append(off2)

    for i in range(offspring_lambda):
        mutation(offsprings[i])
        # check if the offspring is valid, if not, mutate it again
        while not check_valid(offsprings[i]):
            mutation(offsprings[i])


    evaluation_pop = offsprings
    for i in range(len(evaluation_pop)):
        scores.append(nas_ioh.f(evaluation_pop[i]))
    x_opt = rank_selection(evaluation_pop, scores)
    f_opt = scores[evaluation_pop.index(x_opt)]
    next_pop = []
    # select using rank selection to form next_pop
    idx = np.argsort(scores)
    idx = idx[::-1]
    #for i in range(int(pop_size / 2)):
        #next_pop.append(evaluation_pop[idx[i]])
    # use tournament selection to fill next_pop
    while len(next_pop) < int(pop_size / 2):
        next_pop.append(tournament_selection(evaluation_pop, scores))
    # use roulette wheel selection to fill next_pop
    #while len(next_pop) < int(pop_size / 2):
        #next_pop.append(roulette_selection(evaluation_pop, scores))
    return x_opt, next_pop, f_opt

def main(argv):
    del argv  # Unused
    for r in range(nas_ioh.runs):  # we execute the algorithm with 20 independent runs.
        auc = 0
        f_best = sys.float_info.min
        population = []
        for _ in range(int(nas_ioh.budget / (offspring_lambda))):
            x, population, y = genetic_algorithm(nas_ioh.nasbench, population)
            if y > f_best:
                f_best = y
                x_best = x
        print("run", r, ", best x:", x_best, ", f :", f_best)
        nas_ioh.f.reset()  # Note that you must run the code after each independent run.




# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
    start = time.time()
    app.run(main)
    end = time.time()
    print("The program takes %s seconds" % (end - start))
