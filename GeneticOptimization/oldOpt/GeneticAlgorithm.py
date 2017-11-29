'''

COMMENTED TUTORIAL on DEAP python package for evolutionary programming

'''

import random

from deap import base
from deap import creator
from deap import tools

# Create Fitness class for Minimization of 1 objective (weight = -1) NOTE: weights must be tuples
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Create individual with fitness attribute
creator.create("Individual", list, fitness=creator.FitnessMin)


# Initialize
IND_SIZE=10

toolbox = base.Toolbox()
# register: creates aliases (e.g. attr_float stands for random.random)
toolbox.register("attr_float", random.random)
# initRepeat -> 3 args: Individual, function, size
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

# Now, calling toolbox.individual() will call initRepeat() with the fixed arguments and return a complete
# creator.Individual composed of IND_SIZE floating point numbers with a maximizing single objective fitness attribute.

# Create population of Individuals (BAG)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(100)


'''GRID population could be used for the MEA  - neighbors are important, accessible with 2 indices'''
# toolbox.register("row", tools.initRepeat, list, toolbox.individual, n=N_COL)
# toolbox.register("population", tools.initRepeat, list, toolbox.row, n=N_ROW)
''' SEEDING: a first guess population can be used to initialize an evolutionary algorithm (e.g. from file) '''


# Evaluation

ind1 = toolbox.individual()


def eval(individual):
    # Do some hard computing on the individual
    a = sum(individual)
    b = len(individual)
    return a/b,  # must return a tuple!

ind1.fitness.values = eval(ind1)
print ind1.fitness.valid
#print ind1.fitness


# Mutation
mutant = toolbox.clone(ind1)
ind2, = tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
del mutant.fitness.values
ind2.fitness.values = eval(ind2)

print ind1.fitness
print ind2.fitness

# Crossover
child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
tools.cxBlend(child1, child2, 0.9)
del child1.fitness.values
del child2.fitness.values
child1.fitness.values = eval(child1)
child2.fitness.values = eval(child2)


# Selection

# selBest: selects N best individuals (2nd arg) among list provided
selected = tools.selBest([child1, child2], 1)
# the offspring must be cloned!
selected = tools.selBest(pop, 5)
offspring = [toolbox.clone(ind) for ind in selected]


'''USING TOOLBOX'''

# register functions

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval)


'''EXAMPLE'''

NGEN = 50
CXPB = 0.8
MUTPB = 0.05

for g in range(NGEN):

    print "Generation number ", g

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    # The population is entirely replaced by the offspring
    pop[:] = offspring

solution = tools.selBest(pop, 1)


'''TOOL DECORATION (very important)'''
MIN = -0.2
MAX = 1.4

# Example
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in xrange(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator

toolbox.register("mate", tools.cxBlend, alpha=0.2)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2)

toolbox.decorate("mate", checkBounds(MIN, MAX))
toolbox.decorate("mutate", checkBounds(MIN, MAX))
