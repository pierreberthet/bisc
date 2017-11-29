'''

Optimization of MEA spatial pattern to maximize driving function
for neuron with different alignment in the proximity of the MEA

Convention:
position in: um
current in:  nA

x: vertical (cortical depth)
y: horizontal (MEA plane)
z: depth wrt MEA

(MEA plane: x-y)

'''

"""TO DO:
-decorate solutions to have uA discretization (and maybe a max set of available)
-write own mutation function
-try different size and think about neuron population assumption
-try closely separated neurons with different alignments
"""

import random

from deap import base
from deap import creator
from deap import tools

from MEAutility import *
import matplotlib.pyplot as plt

# Create MEA and geometric neuron
N_side = 10
pitch = 15
mea = SquareMEA(dim=N_side, pitch=pitch)

vm_threshold_target = 100
vm_threshold_surround = -50

# vm_threshold_target = 15
# vm_threshold_surround = -5

target_x = 15
target_y = 0
target_z = 0
neuron_separation = 20

axon_dir = [0, 0, -1]
axon_length = 15.0
discrete = 5.0

# Target neuron
soma = [target_x, target_y, target_z]
neuron = GeometricNeuron(soma, align_dir=axon_dir, length=axon_length, discrete_points=discrete)
x_target = neuron.get_axon_points()

# Surrounding neurons
surround_neurons = []
x_surround = []
surround_somas = [[target_x, target_y+neuron_separation, target_z],
                  [target_x, target_y-neuron_separation, target_z]]
# [target_x, target_y + 2 * neuron_separation, target_z],
# [target_x, target_y - 2 * neuron_separation, target_z],
# [target_x, target_y + 3 * neuron_separation, target_z],
# [target_x, target_y - 3 * neuron_separation, target_z]
surround_dir = [[0, 0, -1],
                [0, 0, -1]]
# ,
# [0, 0, -1],
# [0, 0, -1],
# [0, 0, -1],
# [0, 0, -1]

for neur in range(len(surround_somas)):
    surround_neurons.append(GeometricNeuron(surround_somas[neur], align_dir=surround_dir[neur], length=axon_length, discrete_points=discrete))
    x_surround.append(surround_neurons[neur].get_axon_points())



def my_random(max):
    randint = random.randint(-max, max)
    return randint*1000


# def eval(individual):
#     mea.set_currents(individual)
#
#     energy = sum(np.abs(individual))
#     electro_neutrality = 1/(abs(sum(individual))+0.1)
#
#     #number of axon points above threshold
#     v_axon = mea.compute_field(x_target)
#     dv_axon = np.gradient(v_axon)
#     dv2_axon = np.gradient(dv_axon)
#
#     n_above = sum(1 for e in range(len(dv2_axon)) if dv2_axon[e]>vm_threshold)
#
#     return n_above, electro_neutrality, energy  # must return a tuple!

# # With multiple surrounding neurons
# def eval(individual):
#     mea.set_currents(individual)
#
#     energy = sum(np.abs(individual))
#
#     '''Maximize not only the number above threshold'''
#     #number of axon points above threshold for target
#     v_axon = mea.compute_field(x_target)
#     dv_axon = np.gradient(v_axon)
#     dv2_axon = np.gradient(dv_axon)
#     n_above_fitness = sum(1 for e in range(len(dv2_axon)) if dv2_axon[e]>vm_threshold_target)
#     # target_above_fitness = sum(dv2_axon-vm_threshold_target)
#
#     '''Mainimize not only the number above threshold'''
#     # number of axon points above threshold for surrounding neurons
#     surround_above_fitness = 0
#     n_above_surround = 0
#     for neur in range(len(surround_neurons)):
#         v_axon = mea.compute_field(x_surround[neur])
#         dv_axon = np.gradient(v_axon)
#         dv2_axon = np.gradient(dv_axon)
#         n_above_surround += sum(1 for e in range(len(dv2_axon)) if dv2_axon[e] > vm_threshold_surround)
#         # surround_above_fitness += sum(dv2_axon-vm_threshold_surround)
#
#     return n_above_fitness, n_above_surround , energy # must return a tuple!

# With multiple surrounding neurons + threshold distance normalization
def eval(individual):
    mea.set_currents(individual)

    energy = sum(np.abs(individual))

    tract = neuron.length / discrete

    '''Maximize not only the number above threshold, but also by how far'''
    #number of axon points above threshold for target
    v_axon = mea.compute_field(x_target)
    dv_axon = np.gradient(v_axon)#/tract
    dv2_axon = np.gradient(dv_axon)#/tract
    n_above_fitness = sum(1 for e in range(len(dv2_axon)) if dv2_axon[e]>=vm_threshold_target)
    n_above_fitness -= sum(abs(dv2_axon[e]-vm_threshold_target)/(vm_threshold_target-vm_threshold_surround)
                           for e in range(len(dv2_axon)) if dv2_axon[e]<vm_threshold_target)

    '''Mainimize not only the number above threshold, but also by how far'''
    # number of axon points above threshold for surrounding neurons
    surround_above_fitness = 0
    n_below_surround = 0
    for neur in range(len(surround_neurons)):
        v_axon = mea.compute_field(x_surround[neur])
        dv_axon = np.gradient(v_axon)#/tract
        dv2_axon = np.gradient(dv_axon)#/tract
        n_below_surround += sum(1 for e in range(len(dv2_axon)) if dv2_axon[e] <= vm_threshold_surround)
        n_below_surround -= sum(abs(dv2_axon[e] - vm_threshold_surround) / (vm_threshold_target - vm_threshold_surround)
                               for e in range(len(dv2_axon)) if dv2_axon[e] > vm_threshold_surround)
        # # n_above_surround += sum(1 for e in range(len(dv2_axon)) if dv2_axon[e] > vm_threshold_surround)
        # n_above_surround += sum(abs(dv2_axon[e] - vm_threshold_surround) / (vm_threshold_target - vm_threshold_surround)
        #                        for e in range(len(dv2_axon)) if dv2_axon[e] > vm_threshold_surround)
        # surround_above_fitness += sum(dv2_axon-vm_threshold_surround)

    return n_above_fitness, n_below_surround , energy # must return a tuple!

# # Create Fitness class for Minimization of 3 objectives: total energy, electroneutrality, neuron excitation
# creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0,))

# For increased selectivity -> minimize points above threshold for neurons placed close to target (second objective)
creator.create("FitnessMulti", base.Fitness, weights=(0.4, 0.4, -0.2,))

# Create individual with fitness attribute
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize
IND_SIZE=100

toolbox = base.Toolbox()
# register: creates aliases (e.g. attr_float stands for random.random)
toolbox.register("attr_float", my_random, 50)
# initRepeat -> 3 args: Individual, function, size
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register functions
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1000, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval)

# # Decorate functions
# MIN = 5e5 # 50 uA
# MAX = 1e4 # 1 uA
#
# def checkBounds(min, max):
#     def decorator(func):
#         def wrapper(*args, **kargs):
#             offspring = func(*args, **kargs)
#             for child in offspring:
#                 for i in xrange(len(child)):
#                     if abs(child[i]) > max:
#                         if child[i] > 0:
#                             child[i] = max
#                         else:
#                             child[i] = -max
#                     elif abs(child[i]) < min:
#                         child[i] = 0
#             return offspring
#         return wrapper
#     return decorator
#
# toolbox.decorate("mate", checkBounds(MIN, MAX))
# toolbox.decorate("mutate", checkBounds(MIN, MAX))


NGEN = 50
CXPB = 0.8
MUTPB = 0.1

pop = toolbox.population(100)

besties = []

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

    best_solution = tools.selBest(pop, 1)
    besties.append(best_solution[0])
    # print "Above threshold: ", toolbox.evaluate(best_solution[0])[0], " over ", neuron.points , \
    #       "Electroneutrality: ", toolbox.evaluate(best_solution[0])[1], " over 10 maximum", \
    #       "Total current: ", toolbox.evaluate(best_solution[0])[2]/1000, " (around ", \
    #       toolbox.evaluate(best_solution[0])[2]/(1000*mea.number_electrode) , " uA for each electrode)"

    # print "Target above threshold: ", toolbox.evaluate(best_solution[0])[0], " over ", neuron.points, \
    #     "  Surround above threshold: ", toolbox.evaluate(best_solution[0])[1], " over ", len(x_surround)*len(x_surround[0])

    print "Target above threshold fitness: ", toolbox.evaluate(best_solution[0])[0], \
          "  Surround below threshold: ", toolbox.evaluate(best_solution[0])[1], " over ", len(x_surround)*len(x_surround[0]) , \
          "  Total current: ", toolbox.evaluate(best_solution[0])[2]/1000, " (around ", \
          toolbox.evaluate(best_solution[0])[2]/(1000*mea.number_electrode) , " uA for each electrode)"


# Plot results

mea.set_currents(best_solution[0])

samples = 30.0
steps = np.linspace(0, neuron.length, num=samples)
x_target_upsampled = np.array([neuron.soma_pos + st * neuron.align_dir for st in steps])

tract = neuron.length/samples

axon = np.linspace(0, neuron.length, samples)
v_axon = mea.compute_field(x_target_upsampled)
dv_axon = np.gradient(v_axon)/tract
dv2_axon = np.gradient(dv_axon)/tract

fig1 = plt.figure()
plt.subplot(3, 1, 1)
plt.plot(axon, v_axon, 'b.-')
plt.title('V along axon')
plt.ylabel('mV')

plt.subplot(3, 1, 2)
plt.plot(axon, dv_axon, 'g.-')
plt.title('dV along axon')
plt.ylabel('mV/um')

plt.subplot(3, 1, 3)
plt.plot(axon, dv2_axon, 'r.-')
plt.title('dV2 along axon')
plt.ylabel('mV2/um2')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.savefig('./images/mea_target_4',
            dpi=300)

fig2 = plt.figure()
for neur in range(len(surround_neurons)):
    steps = np.linspace(0, surround_neurons[neur].length, num=samples)
    x_target_upsampled = np.array([surround_neurons[neur].soma_pos + st * surround_neurons[neur].align_dir for st in steps])

    v_axon = mea.compute_field(x_target_upsampled)
    dv_axon = np.gradient(v_axon)/tract
    dv2_axon = np.gradient(dv_axon)/tract

    plt.subplot(len(surround_neurons), 1, neur+1)
    plt.plot(axon, dv2_axon, 'b.-')
    plt.title('dV2 along axon')
    plt.ylabel('mV2/um2')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.savefig('./images/mea_surround_4',
            dpi=300)


unit = 5
bound = 80
x_vec = np.arange(1, bound, unit)
y_vec = np.arange(-bound, bound, unit)
z_vec = np.arange(-bound, bound, unit)

x, y, z = np.meshgrid(x_vec, y_vec, z_vec)
y_plane, z_plane = np.meshgrid(y_vec, z_vec)


v_grid = np.zeros((len(y_vec), len(z_vec)))

z_vec = z_vec[::-1]

for ii in range(len(z_vec)):
    for jj in range(len(y_vec)):
        v_grid[ii, jj] = mea.compute_field(np.array([10, y_vec[jj], z_vec[ii]]))
        # print np.array([10, y_vec[jj], z_vec[ii]])


v_grid_dv2_z = np.zeros((len(y_vec), len(z_vec)))
v_grid_dv2_y = np.zeros((len(y_vec), len(z_vec)))
for ii in range(len(y_vec)):
    v_grid_dv2_z[:, ii] = np.gradient(np.gradient(v_grid[:, ii]))
    v_grid_dv2_y[ii, :] = np.gradient(np.gradient(v_grid[ii, :]))

fig3 = plt.figure()
CS = plt.contour(y_vec, z_vec, v_grid/1000, linewidths=1, colors='k')
CS = plt.contourf(y_vec, z_vec, v_grid/1000, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('Potential in proximity of electrodes: x = 10 um')

# overlay neuron soma and axon (projection on (x=10,y,z))
axon_terminal = neuron.get_axon_end()
neuron_proj_ext = plt.plot([soma[1], axon_terminal[1]], [soma[2], axon_terminal[2]])
plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 5.0)
neuron_proj_int = plt.plot([soma[1], axon_terminal[1]], [soma[2], axon_terminal[2]])
plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 4.0)
plt.scatter(soma[1], soma[2], marker='^', c='k', s=600)
plt.scatter(soma[1], soma[2], marker='^', c='r', s=500)


for neur in range(len(surround_neurons)):
    axon_terminal = surround_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='k', s=400)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='#888888', s=400)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.savefig('./images/mea_potential_neurons_4',
            dpi=300)

# Plot dv2 field
fig4 = plt.figure()
CS1 = plt.contour(y_vec, z_vec, v_grid_dv2_z, linewidths=1, colors='k')
CS1 = plt.contourf(y_vec, z_vec, v_grid_dv2_z, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('Potential in proximity of electrodes: x = 10 um')

# overlay neuron soma and axon (projection on (x=10,y,z))
axon_terminal = neuron.get_axon_end()
neuron_proj_ext = plt.plot([soma[1], axon_terminal[1]], [soma[2], axon_terminal[2]])
plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 5.0)
neuron_proj_int = plt.plot([soma[1], axon_terminal[1]], [soma[2], axon_terminal[2]])
plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 4.0)
plt.scatter(soma[1], soma[2], marker='^', c='k', s=600)
plt.scatter(soma[1], soma[2], marker='^', c='r', s=500)


for neur in range(len(surround_neurons)):
    axon_terminal = surround_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='k', s=400)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='#888888', s=400)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.savefig('./images/mea_dv2potential_neurons_4',
            dpi=300)


plt.matshow(mea.get_current_matrix())
plt.colorbar()

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.savefig('./images/mea_currents_nA_4',
            dpi=300)
plt.show()


