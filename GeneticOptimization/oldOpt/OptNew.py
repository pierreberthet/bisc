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
"""

import random, math

from deap import base
from deap import creator
from deap import tools

from MEAutility import *
import matplotlib.pyplot as plt

save_images = True
image_name = "2neur_z_dir_20u"

"""DEFINE FUNCTIONS"""


def save_figure(name):
    figmanager = plt.get_current_fig_manager()
    figmanager.window.showMaximized()
    file_im = './images/' + name + '_' + image_name
    plt.savefig(file_im, dpi=300)


def my_random(max):
    randint = random.randint(-max, max)
    return randint*1000


# With multiple surrounding neurons + threshold distance normalization
def eval(individual):
    mea.set_currents(individual)

    energy = sum(np.abs(individual))

    sparseness = len(np.where(np.array(individual) == 0)[0])

    tract = neuron.length/discrete

    '''Maximize not only the number above threshold, but also by how far'''
    # number of axon points above threshold for target
    v_axon_ = mea.compute_field(x_target)
    dv_axon_ = np.gradient(v_axon_)/tract
    dv2_axon_ = np.gradient(dv_axon_)/tract
    n_above_fitness = sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e]>=vm_threshold_target)
    n_above_fitness -= sum(abs(dv2_axon_[e]-vm_threshold_target)/(vm_threshold_target-vm_threshold_surround)
                           for e in range(len(dv2_axon_)) if dv2_axon_[e]<vm_threshold_target)


    '''Maximize number of surround neurons below surround threshold'''
    # neutrality = 0
    n_below_surround = 0

    for neur in range(len(surround_neurons)):
        v_axon_ = mea.compute_field(x_surround[neur])
        dv_axon_ = np.gradient(v_axon_)/tract
        dv2_axon_ = np.gradient(dv_axon_)/tract
        n_below_surround += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] <= vm_threshold_surround)
        n_below_surround -= sum(abs(dv2_axon_[e] - vm_threshold_surround) / (vm_threshold_target - vm_threshold_surround)
                                for e in range(len(dv2_axon_)) if dv2_axon_[e] > vm_threshold_surround)
        # v_axon_ = mea.compute_field(x_surround[neur])
        # dv_axon_ = np.gradient(v_axon_)
        # dv2_axon_ = np.gradient(dv_axon_)
        # neutrality += sum((dv2_axon_[e]-vm_threshold_surround)**2 for e in range(len(dv2_axon_)))

    return  n_above_fitness, n_below_surround, sparseness,   #  energy, neutrality, must return a tuple!

# # With multiple surrounding neurons + threshold distance normalization
# def eval(individual):
#     '''TRY NOT TO COMPUTE DERIVATIVE ON 1ST AND LAST POINT'''
#     mea.set_currents(individual)
#
#     energy = sum(np.abs(individual))
#
#     sparseness = len(np.where(np.array(individual) == 0)[0])
#
#     tract = neuron.length/discrete
#
#     '''Maximize not only the number above threshold, but also by how far'''
#     # number of axon points above threshold for target
#
#     v_axon_ = mea.compute_field(x_target)
#     dv_axon_ = np.gradient(v_axon_)#/tract
#     dv2_axon_ = np.gradient(dv_axon_)#/tract
#     n_above_fitness = sum(1 for e in range(1, len(dv2_axon_)-1) if dv2_axon_[e]>=vm_threshold_target )
#     n_above_fitness -= sum(abs(dv2_axon_[e]-vm_threshold_target)/(vm_threshold_target-vm_threshold_surround)
#                            for e in range(1, len(dv2_axon_)-1) if dv2_axon_[e]<vm_threshold_target)
#
#
#     '''Maximize number of surround neurons below surround threshold'''
#     # neutrality = 0
#     n_below_surround = 0
#
#     for neur in range(len(surround_neurons)):
#         v_axon_ = mea.compute_field(x_surround[neur])
#         dv_axon_ = np.gradient(v_axon_)#/tract
#         dv2_axon_ = np.gradient(dv_axon_)#/tract
#         n_below_surround += sum(1 for e in range(1, len(dv2_axon_)-1) if dv2_axon_[e] <= vm_threshold_surround)
#         n_below_surround -= sum(abs(dv2_axon_[e] - vm_threshold_surround) / (vm_threshold_target - vm_threshold_surround)
#                                 for e in range(1, len(dv2_axon_)-1) if dv2_axon_[e] > vm_threshold_surround)
#         # v_axon_ = mea.compute_field(x_surround[neur])
#         # dv_axon_ = np.gradient(v_axon_)
#         # dv2_axon_ = np.gradient(dv_axon_)
#         # neutrality += sum((dv2_axon_[e]-vm_threshold_surround)**2 for e in range(len(dv2_axon_)))
#
#     return  n_above_fitness, n_below_surround, sparseness,   #  energy, neutrality, must return a tuple!


def my_mutate(individual, mu, sigma, pgauss, pzero):

    size = len(individual)

    for i in range(size):
        # Apply gaussian mutation
        if random.random() < pgauss:
            individual[i] += random.gauss(mu, sigma)

            # round to current_step (nA -> uA -> nA conversion)
            individual[i] = round((individual[i]/1000) // current_step) * 1000

        # Apply zero mutation --> facilitates sparse solutions
        if random.random() < pzero:
            individual[i] = 0

    return individual,

"""START OPTIMIZATION"""

# Create MEA and geometric neuron
N_side = 8
pitch = 15
current_step = 5
mea = SquareMEA(dim=N_side, pitch=pitch)

# vm_threshold_target = 15
# vm_threshold_surround = -5

# vm_threshold_target = 200
# vm_threshold_surround = -10
vm_threshold_target = 10
vm_threshold_surround = -2

target_x = 15
target_y = 0
target_z = 0
neuron_separation = 20
close_separation = 10

axon_dir = [0, -0.5, -1]
axon_length = 15.0
discrete = 3.0

# Target neuron
soma = [target_x, target_y, target_z]
neuron = GeometricNeuron(soma, align_dir=axon_dir, length=axon_length, discrete_points=discrete)
x_target = neuron.get_axon_points()

# Surrounding neurons
surround_neurons = []
x_surround = []
surround_somas = [[target_x, target_y + close_separation, target_z]]
# ,
# [target_x, target_y, target_z + close_separation]
# [target_x, target_y - close_separation, target_z],
# [target_x, target_y - close_separation, target_z - close_separation],
# [target_x, target_y + close_separation, target_z + close_separation]
    #               [target_x, target_y + 2 * neuron_separation, target_z],
    #               [target_x, target_y - 2 * neuron_separation, target_z],
    #               [target_x, target_y + 3 * neuron_separation, target_z],
    #               [target_x, target_y - 3 * neuron_separation, target_z]]

surround_dir = [[0, 0, -1]]
# ,
# [0, 0, -1]
# [0, -1, 1],
# [0, 0, 1],
# [0, 1, -1]
    # [0, 0, -1],
    #             [0, 0, -1]
    # [0, 0, -1],
    #             [0, 0, -1],
    #             [0, 0, -1],
    #             [0, 0, -1],
    #             [0, 0, -1],
    #             [0, 0, -1]]

for neur in range(len(surround_somas)):
    surround_neurons.append(GeometricNeuron(surround_somas[neur], align_dir=surround_dir[neur], length=axon_length, discrete_points=discrete))
    x_surround.append(surround_neurons[neur].get_axon_points())


# For increased selectivity -> minimize points above threshold for neurons placed close to target (second objective)
creator.create("FitnessMulti", base.Fitness, weights=(1, 1, 1))

# Create individual with fitness attribute
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize
IND_SIZE=mea.number_electrode

toolbox = base.Toolbox()
# register: creates aliases (e.g. attr_float stands for random.random)
toolbox.register("attr_float", my_random, 50)
# initRepeat -> 3 args: Individual, function, size
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register functions
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", my_mutate, mu=0, sigma=5000, pgauss=0.2, pzero=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval)

# toolbox.decorate("mate", checkBounds(MIN, MAX))
# toolbox.decorate("mutate", checkBounds(MIN, MAX))

# GA parameters
NGEN = 100
CXPB = 0.8
MUTPB = 0.1

pop = toolbox.population(mea.number_electrode)

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

    if len(x_surround) != 0:
        print "Target above threshold fitness: ", toolbox.evaluate(best_solution[0])[0], \
              "Surround below threshold: ", toolbox.evaluate(best_solution[0])[1], " over ", len(x_surround) * len(x_surround[0]),   \
              "Zero-current electrodes = ", toolbox.evaluate(best_solution[0])[2]


    else:
        print "Target above threshold fitness: ", toolbox.evaluate(best_solution[0])[0], \
            "  No surrounding neurons.  Zero-current electrodes = ", toolbox.evaluate(best_solution[0])[2]
        # "  Total current: ", toolbox.evaluate(best_solution[0])[2]/1000, " (around ", \
        # toolbox.evaluate(best_solution[0])[2]/(1000*mea.number_electrode) , " uA for each electrode)", \

"""PLOT RESULTS"""

mea.set_currents(best_solution[0])

samples = 50.0
tract = neuron.length/samples
steps = np.linspace(0, neuron.length, num=samples)
x_target_upsampled = np.array([neuron.soma_pos + st * neuron.align_dir for st in steps])

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

if save_images:
    save_figure('mea_target')

fig2 = plt.figure()
for neur in range(len(surround_neurons)):
    tract = surround_neurons[neur].length / samples
    steps = np.linspace(0, surround_neurons[neur].length, num=samples)
    x_target_upsampled = np.array([surround_neurons[neur].soma_pos + st * surround_neurons[neur].align_dir for st in steps])

    v_axon = mea.compute_field(x_target_upsampled)
    dv_axon = np.gradient(v_axon)/tract
    dv2_axon = np.gradient(dv_axon)/tract

    plt.subplot(len(surround_neurons), 1, neur+1)
    plt.plot(axon, dv2_axon, 'b.-')
    plt.title('dV2 along axon')
    plt.ylabel('mV2/um2')

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

if save_images:
    save_figure('mea_field')


# Plot dv2 field
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


plt.subplot(223)
CSy = plt.contour(y_vec, z_vec, v_grid_dv2_y, linewidths=1, colors='k')
CSy = plt.contourf(y_vec, z_vec, v_grid_dv2_y, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('dv2 in y direction')

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


plt.subplot(122)
v_grid_dv_neur = abs(neuron.align_dir[1])*v_grid_dv2_y + abs(neuron.align_dir[2])*v_grid_dv2_z
CSneur = plt.contour(y_vec, z_vec, v_grid_dv_neur, linewidths=1, colors='k')
CSneur = plt.contourf(y_vec, z_vec, v_grid_dv_neur, 50, cmap=plt.cm.jet)
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
            [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
            marker='o', c='b', s=20)
plt.title('dv2 in target direction')

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


plt.tight_layout()

if save_images:
    save_figure('mea_dv2_z')


plt.matshow(mea.get_current_matrix()/1000)
plt.colorbar()

if save_images:
    save_figure('mea_currents')

plt.show()


