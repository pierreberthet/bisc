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
-Initial Guess: Sparse Solution --> 25% of active electrodes?
"""

import random, math
import numpy as np
from numpy import linalg as la

from deap import base
from deap import creator
from deap import tools

from MEAutility import *
import matplotlib.pyplot as plt

save_images = False
image_name = "2neur_z_dir_20u"

der2filter = [1, -2, 1]

uniform_neurons = True
N_uni = 5
x_lim = [5, 25]

# If True -> surround neurons are lightly hyperpolarized
# If False -> surround neurons are as neutral as possible (dv2 = 0)
surround_hyper = True

# If True -> second derivative is evaluated through parabolic fitting
# If False -> "" through linear [2 -1 2] filter
parabolic = False

"""DEFINE FUNCTIONS"""


def save_figure(name):
    figmanager = plt.get_current_fig_manager()
    figmanager.window.showMaximized()
    file_im = './images/' + name + '_' + image_name
    plt.savefig(file_im, dpi=300)


# def my_random(max_val):
#     randint = round(random.randint(-max_val, max_val) / float(current_step)) * current_step
#     return randint*1000
def my_random(max_val):
    if random.random() < 0.25:
        randint = round(random.randint(-max_val, max_val) / float(current_step)) * current_step
        return randint * 1000
    else:
        return 0



# With multiple surrounding neurons + threshold distance normalization
def compute_fitness(individual):
    mea.set_currents(individual)

    x_energy = 1-(sum(np.abs(individual))/(mea.number_electrode*max_current*1000))
    x_sparsity = len(np.where(np.array(individual) == 0)[0])/float(mea.number_electrode)

    tract = float(axon_length)/(discrete-1)

    if not parabolic:
        '''Maximize not only the number above threshold, but also by how far'''
        n_above_target = 0

        for neur in range(len(target_neurons)):
            v_axon_ = mea.compute_field(x_target[neur])
            dv2_axon_ = np.convolve(v_axon_, der2filter, 'valid') / tract**2

            n_above_target += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] >= vm_threshold_target)
            n_above_target -= sum(abs(dv2_axon_[e] - vm_threshold_target) / (vm_threshold_target - vm_threshold_surround)
                              for e in range(len(dv2_axon_)) if dv2_axon_[e] < vm_threshold_target)

        # Normalize over number of targeting neurons
        x_targ = n_above_target / float(len(target_neurons) * len(dv2_axon_))

        if surround_hyper:
            '''Maximize number of surround neurons below surround threshold'''
            n_below_surround = 0

            for neur in range(len(surround_neurons)):
                v_axon_ = mea.compute_field(x_surround[neur])
                dv2_axon_ = np.convolve(v_axon_, der2filter, 'valid') / tract**2

                n_below_surround += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] <= vm_threshold_surround)
                n_below_surround -= sum(abs(dv2_axon_[e] - vm_threshold_surround) / (vm_threshold_target - vm_threshold_surround)
                                    for e in range(len(dv2_axon_)) if dv2_axon_[e] > vm_threshold_surround)

            # Normalize over number of surrounding neurons
            x_nontarg = n_below_surround / float(len(surround_neurons)*len(dv2_axon_))

        else:
            '''Maximize neutrality'''
            sum_neutral = 0

            for neur in range(len(surround_neurons)):
                v_axon_ = mea.compute_field(x_surround[neur])
                dv2_axon_ = np.convolve(v_axon_, der2filter, 'valid') / tract**2

                sum_neutral += sum(np.abs(dv2_axon_))/(len(dv2_axon_))

            # Normalize between 1 (max) and 0 (min) [1/(1+x)]
            x_nontarg = 1 / (1 + sum_neutral**0.2)
    else:
        '''Compute parabolic coefficient'''
        n_above_target = 0

        for neur in range(len(target_neurons)):
            v_axon_ = mea.compute_field(x_target[neur])
            n_axon_ = np.linspace(0, axon_length, num=len(v_axon_))
            p = np.polyfit(n_axon_, v_axon_, 2)

            if p[0] >= parabolic_threshold_target:
                n_above_target += 1
            else:
                n_above_target -= abs(parabolic_threshold_target - p[0]) / \
                                  (parabolic_threshold_target-parabolic_threshold_surround)

        # Normalize over number of targeting neurons
        x_targ = n_above_target / len(target_neurons)

        n_below_surround = 0

        for neur in range(len(surround_neurons)):
            v_axon_ = mea.compute_field(x_surround[neur])
            n_axon_ = np.linspace(0, axon_length, num=len(v_axon_))
            p = np.polyfit(n_axon_, v_axon_, 2)

            if p[0] <= parabolic_threshold_surround:
                n_below_surround += 1
            else:
                n_below_surround -= abs(parabolic_threshold_surround - p[0]) / \
                                    (parabolic_threshold_target - parabolic_threshold_surround)

        x_nontarg = n_below_surround / len(surround_neurons)

    fit_activation = alpha_target_surround*x_targ + (1-alpha_target_surround)*x_nontarg
    fit_energy = alpha_energy_sparsity*x_energy + (1-alpha_energy_sparsity)*x_sparsity

    return fit_activation, fit_energy,


def my_mutate(individual, mu, sigma, pgauss, pzero):

    size = len(individual)

    for i in range(size):
        # Apply gaussian mutation
        if random.random() < pgauss:
            individual[i] += random.gauss(mu, sigma)

            # round to current_step (nA -> uA -> nA conversion)
            individual[i] = round((individual[i]/1000) / float(current_step)) * current_step * 1000

            # Check limits
            if individual[i] > max_current * 1000:
                individual[i] = max_current * 1000
            if individual[i] < -max_current * 1000:
                individual[i] = -max_current * 1000

        # Apply zero mutation --> facilitates sparse solutions
        if random.random() < pzero:
            individual[i] = 0

    return individual,

"""START OPTIMIZATION"""

'''MEA and NEURONS initialization'''
# Create MEA and geometric neuron
N_side = 4
pitch = 15
current_step = 2
max_current = 20
mea = SquareMEA(dim=N_side, pitch=pitch)
bound = abs(mea[0][0].position[1]) + pitch
bound_uni = int(abs(mea[0][0].position[1]))

neuron_separation = 25
close_separation = 15

axon_length = 15
discrete = 15

# Target neuron(s)

target_x = 15
target_y = 0
target_z = 0

target_somas = [[target_x, target_y, target_z]]
target_dir = [[0, -0.5, -1]]
# target_somas = [[target_x, target_y, target_z],
#                 [target_x, target_y-15, target_z+15]]
# target_dir = [[0, np.random.randn(), np.random.randn()],
#               [0, np.random.randn(), np.random.randn()]]

target_neurons = []
x_target = []


for neur in range(len(target_somas)):
    target_neurons.append(GeometricNeuron(target_somas[neur], align_dir=target_dir[neur], length=axon_length,
                                          discrete_points=discrete))
    x_target.append(target_neurons[neur].get_axon_points())

# Surrounding neurons

surround_neurons = []
x_surround = []

surround_somas = []
surround_dir = []

all_somas = target_somas

if uniform_neurons:
    # Generate N random neurons (Check that they are not closer than close_separation
    for nn in range(N_uni):

        new_soma_not_found = True

        while new_soma_not_found:

            x_rand = random.randint(x_lim[0], x_lim[1])
            y_rand = random.randint(-bound_uni, bound_uni)
            z_rand = random.randint(-bound_uni, bound_uni)

            soma_surr = [x_rand, y_rand, z_rand]

            # Check proximity with other somas
            if all([la.norm(np.array(soma_surr)-np.array(all_somas[e])) > close_separation
                    for e in range(len(all_somas))]):
                # append new neuron and break while
                all_somas.append(soma_surr)
                axon_dir_surr = [0, np.random.randn(), np.random.randn()]
                surround_somas.append(soma_surr)
                surround_dir.append(axon_dir_surr)
                new_soma_not_found = False
            else:
                print 'Neuron too close: looking for another soma position'

else:

    surround_somas = [[target_x, target_y + close_separation, target_z],
                     [target_x, target_y, target_z + close_separation]]

    surround_dir = [[0, 0, -1],
                    [0, -1, -0.5]]

for neur in range(len(surround_somas)):
    surround_neurons.append(GeometricNeuron(surround_somas[neur], align_dir=surround_dir[neur], length=axon_length,
                                            discrete_points=discrete))
    x_surround.append(surround_neurons[neur].get_axon_points())

'''GA initialization'''

# FITNESS

# Physioligical Estimate: rm=30000, Ra=150, Vm_thresh = 30mV
space_const = np.sqrt(float(30000)/150)

vm_threshold_target = 0.8
vm_threshold_surround = 0.4

# parabolic_threshold_target = 2
# parabolic_threshold_surround = -0.5
parabolic_threshold_target = 0.3
parabolic_threshold_surround = -0.2

alpha_target_surround = 0.4
alpha_energy_sparsity = 0.5

# GA parameters
NGEN = 300
CXPB = 0.8
MUTPB = 0.1
IND_SIZE = mea.number_electrode
PBEST = 0.04
NSTALL = 75

# mutation and select
muGauss = 0
sigGauss = 2000
pGauss = 0.1
pZero = 0.2
tournSize = 3

# For increased selectivity -> minimize points above threshold for neurons placed close to target (second objective)
creator.create("FitnessMulti", base.Fitness, weights=(1, 1))

# Create individual with fitness attribute
creator.create("Individual", list, fitness=creator.FitnessMulti)


toolbox = base.Toolbox()
# register: creates aliases (e.g. attr_float stands for random.random)
toolbox.register("attr_float", my_random, max_current)
# initRepeat -> 3 args: Individual, function, size
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register functions
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", my_mutate, mu=muGauss, sigma=sigGauss, pgauss=pGauss, pzero=pZero)
toolbox.register("select", tools.selTournament, tournsize=tournSize)
toolbox.register("evaluate", compute_fitness)

pop = toolbox.population(mea.number_electrode)
n_best = int(math.ceil(IND_SIZE*PBEST))

besties = []
f1 = []
f2 = []
nstall = 0

for g in range(NGEN):

    print "Generation number ", g

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)

    # Apply crossover on the offspring, except for best solutions
    for child1, child2 in zip(offspring[n_best::2], offspring[n_best+1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring, except for best solutions
    for mutant in offspring[n_best::]:
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
    f1.append(toolbox.evaluate(best_solution[0])[0])
    f2.append(toolbox.evaluate(best_solution[0])[1])

    if len(x_surround) != 0:
        print "AF fitness: ", toolbox.evaluate(best_solution[0])[0], " over ", 1, \
              "AF neutrality: ", toolbox.evaluate(best_solution[0])[1], " over ", 1

    else:
        print "Target above threshold fitness: ", toolbox.evaluate(best_solution[0])[0], \
            "  No surrounding neurons.  Zero-current electrodes = ", toolbox.evaluate(best_solution[0])[2]

    if besties[g] == besties[g-1]:
        nstall += 1
        print 'Stall for ', nstall, ' iterations...'
    else:
        nstall = 0

    '''STOP CRITERION'''
    if nstall == NSTALL:
        print 'Solution in stall. Exiting loop'
        break


"""PLOT RESULTS"""

mea.set_currents(best_solution[0])

samples = 25.0
tract_up = float(axon_length+10)/(samples-1)
steps = np.linspace(-5, axon_length + 5, num=samples)
axon = np.linspace(-5, axon_length + 5, samples)
axon2 = np.linspace(-5, axon_length + 5, samples-2)

fig1 = plt.figure()
print 'TARGET NEURON(S)'
for neur in range(len(target_neurons)):

    x_target_upsampled = np.array([target_neurons[neur].soma_pos + st * target_neurons[neur].align_dir for st in steps])

    v_axon_up = mea.compute_field(x_target_upsampled)
    v_axon = mea.compute_field(x_target[neur])
    n_axon = np.linspace(0, axon_length, num=len(v_axon))
    dv2_axon = np.convolve(v_axon_up, der2filter, 'valid') / tract_up**2

    p = np.polyfit(n_axon, v_axon, 2)

    plt.subplot(len(target_neurons), 2, 2 * neur + 1)
    plt.plot(axon, v_axon_up, 'b.-')
    if neur == 0:
        plt.title('TARGET: V along axon')
    plt.ylabel('mV')
    plt.axvline(x=0, linewidth=2, color='k')
    plt.axvline(x=15, linewidth=2, color='k')

    plt.subplot(len(target_neurons), 2, 2*neur+2)
    plt.plot(axon2, dv2_axon, 'r.-')
    if neur == 0:
        plt.title('TARGET: dV2 along axon')
    plt.ylabel('mV2/um2')
    plt.axvline(x=0, linewidth=2, color='k')
    plt.axvline(x=15, linewidth=2, color='k')

    print 'target n: ', neur+1, ' p = ', p[0]


if save_images:
    save_figure('mea_target')

fig2 = plt.figure()
print 'SURROUNDING NEURON(S)'
for neur in range(len(surround_neurons)):

    x_surround_upsampled = np.array([surround_neurons[neur].soma_pos + st * surround_neurons[neur].align_dir for st in steps])

    v_axon_up = mea.compute_field(x_surround_upsampled)
    v_axon = mea.compute_field(x_surround[neur])
    n_axon = np.linspace(0, axon_length, num=len(v_axon))
    dv2_axon = np.convolve(v_axon_up, der2filter, 'valid') / tract_up**2

    p = np.polyfit(n_axon, v_axon, 2)

    plt.subplot(len(surround_neurons), 2, 2 * neur + 1)
    plt.plot(axon, v_axon_up, 'b.-')
    if neur == 0:
        plt.title('SURROUND: V along axon')
    plt.ylabel('mV')
    plt.axvline(x=0, linewidth=2, color='k')
    plt.axvline(x=15, linewidth=2, color='k')

    ax4 = plt.subplot(len(surround_neurons), 2, 2*neur+2)
    plt.plot(axon2, dv2_axon, 'r.-')
    if neur == 0:
        plt.title('SURROUND: dV2 along axon')
    plt.ylabel('mV2/um2')
    plt.axvline(x=0, linewidth=2, color='k')
    plt.axvline(x=15, linewidth=2, color='k')

    print 'surround n: ', neur+1, ' p = ', p[0]

if save_images:
    save_figure('mea_surround')

unit = 5
bound = abs(mea[0][0].position[1]) + pitch
x_vec = np.arange(1, bound, unit)
y_vec = np.arange(-bound, bound, unit)
z_vec = np.arange(-bound, bound, unit)

x, y, z = np.meshgrid(x_vec, y_vec, z_vec)
y_plane, z_plane = np.meshgrid(y_vec, z_vec)

v_grid = np.zeros((len(y_vec), len(z_vec)))

# maintain matrix orientation (row - z, column - y, [0,0] - top left corner)
z_vec = z_vec[::-1]

for ii in range(len(z_vec)):
    for jj in range(len(y_vec)):
        v_grid[ii, jj] = mea.compute_field(np.array([15, y_vec[jj], z_vec[ii]]))
        # print np.array([10, y_vec[jj], z_vec[ii]])

v_grid_dv2_z = np.zeros((len(y_vec), len(z_vec)))
v_grid_dv2_y = np.zeros((len(y_vec), len(z_vec)))

tract_image = (max(y_vec)-min(y_vec)) / (len(y_vec) - 1)

for ii in range(len(y_vec)):
    v_grid_dv2_z[:, ii] = np.gradient(np.gradient(v_grid[:, ii])) / tract_image**2
    v_grid_dv2_y[ii, :] = np.gradient(np.gradient(v_grid[ii, :])) / tract_image**2


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
for neur in range(len(target_neurons)):
    axon_terminal = target_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 2.0)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='k', s=600)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='r', s=500)


for neur in range(len(surround_neurons)):
    axon_terminal = surround_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='k', s=400)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='#888888', s=400)

if save_images:
    save_figure('mea_field')


# Plot dv2Z field
fig4 = plt.figure()
plt.subplot(221)
CSz = plt.contour(y_vec, z_vec, v_grid_dv2_z, linewidths=1, colors='k')
CSz = plt.contourf(y_vec, z_vec, v_grid_dv2_z, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('dv2 in z direction')

# overlay neuron soma and axon (projection on (x=10,y,z))
for neur in range(len(target_neurons)):
    axon_terminal = target_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 2.0)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='k', s=600)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='r', s=500)


for neur in range(len(surround_neurons)):
    axon_terminal = surround_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='k', s=400)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='#888888', s=400)

#Plot dv2Y
plt.subplot(222)
CSy = plt.contour(y_vec, z_vec, v_grid_dv2_y, linewidths=1, colors='k')
CSy = plt.contourf(y_vec, z_vec, v_grid_dv2_y, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('dv2 in y direction')

# overlay neuron soma and axon (projection on (x=10,y,z))
for neur in range(len(target_neurons)):
    axon_terminal = target_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 2.0)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='k', s=600)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='r', s=500)

for neur in range(len(surround_neurons)):
    axon_terminal = surround_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='k', s=400)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='#888888', s=400)

#plot dv2 45
plt.subplot(223)
v_grid_dv_neur1 = 0.5*v_grid_dv2_y + 0.5*v_grid_dv2_z
CS1 = plt.contour(y_vec, z_vec, v_grid_dv_neur1, linewidths=1, colors='k')
CS1 = plt.contourf(y_vec, z_vec, v_grid_dv_neur1, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('dv2 in +45 deg')

# overlay neuron soma and axon (projection on (x=10,y,z))
for neur in range(len(target_neurons)):
    axon_terminal = target_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 2.0)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='k', s=600)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='r', s=500)

for neur in range(len(surround_neurons)):
    axon_terminal = surround_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='k', s=400)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='#888888', s=400)

#plot dv2 -45
plt.subplot(224)
v_grid_dv_neur1 = 0.5*v_grid_dv2_y - 0.5*v_grid_dv2_z
CS1 = plt.contour(y_vec, z_vec, v_grid_dv_neur1, linewidths=1, colors='k')
CS1 = plt.contourf(y_vec, z_vec, v_grid_dv_neur1, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('dv2 in -45 deg')


# overlay neuron soma and axon (projection on (x=10,y,z))
for neur in range(len(target_neurons)):
    axon_terminal = target_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([target_somas[neur][1], axon_terminal[1]], [target_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 2.0)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='k', s=600)
    plt.scatter(target_somas[neur][1], target_somas[neur][2], marker='^', c='r', s=500)

for neur in range(len(surround_neurons)):
    axon_terminal = surround_neurons[neur].get_axon_end()
    neuron_proj_ext = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
    neuron_proj_int = plt.plot([surround_somas[neur][1], axon_terminal[1]], [surround_somas[neur][2], axon_terminal[2]])
    plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='k', s=400)
    plt.scatter(surround_somas[neur][1], surround_somas[neur][2], marker='^', c='#888888', s=400)

plt.tight_layout()

if save_images:
    save_figure('mea_dv2_z')


plt.matshow(mea.get_current_matrix()/1000)
plt.colorbar()

if save_images:
    save_figure('mea_currents')

fig5 = plt.figure()
ax = plt.subplot(111)
ax.plot(range(g+1), f1, 'b-', range(g+1), f2, 'r--')

if save_images:
    save_figure('mea_currents')

plt.show()











#
#     '''Minimize dv2y and dv2z on a uniform grid'''
#     vec_ = np.arange(-bound, bound, pitch/2)
#     v_grid_ = np.zeros((len(vec_), len(vec_)))
#
#     for ii in range(len(vec_)):
#         for jj in range(len(vec_)):
#             v_grid_[ii, jj] = mea.compute_field(np.array([15, vec_[ii], vec_[jj]]))
#             # print np.array([10, y_vec[jj], z_vec[ii]])
#
#     dv2_z = np.zeros((len(vec_), len(vec_)))
#     dv2_y = np.zeros((len(vec_), len(vec_)))
#     for kk in range(len(vec_)):
#         dv2_z[:, kk] = np.gradient(np.gradient(v_grid_[:, kk]))
#         dv2_y[kk, :] = np.gradient(np.gradient(v_grid_[kk, :]))
#
#     af_neutrality = sum(sum(np.abs(dv2_y))) + sum(sum(np.abs(dv2_z)))



# # number of axon points above threshold for target
# v_axon_ = mea.compute_field(x_target)
# dv2_axon_ = np.convolve(v_axon_, der2filter, 'valid') / tract
#
# n_above_fitness = sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e]>=vm_threshold_target)
# n_above_fitness -= sum(abs(dv2_axon_[e]-vm_threshold_target)/(vm_threshold_target-vm_threshold_surround)
#                        for e in range(len(dv2_axon_)) if dv2_axon_[e]<vm_threshold_target)

# Norm over number of points (0-1)
# x_targ = n_above_target / float(len(dv2_axon_))