'''

'Monte Carlo' evaluation of Genetic Optimization

-Simulate N scenarios with GA and Monopolar
-For every scenario evaluate
  1. target activation
  2. surround inactivation
  3. energy consumption
  4. 'complexity' of scenario
 both for GA, MONO, and BIPOLAR
-average results (mean, boxplots, etc.)
-tests GA - MONO?

Use function singleRun to perform GA, MONO, BIPOLAR and return values

'''

import SimulateMEA as sim
import matplotlib.pyplot as plt
import MEAutility as mea

import numpy as np
from scipy import stats
import pickle
import os
from numpy import linalg as la

verbose = False
plot = False
save = True

N_target = 1
N_surround = 4
axon_length = 15
discrete = 15
trg_above_thresh = 1
Ngen = 100
Nstall = 20
N_sim = 10
monopolar_current = -10000
vm_targ = 1.5
vm_surr = 0.5

# meaParam: N_side, pitch, current_step, max_current, monopolaramp
meaParam = [4, 15, 2, 20, monopolar_current]
# neuronParam: close_separation, xlim, n_target, n_surround, axon_length, discrete, trg_above_thresh
neuronParam = [15, [5, 15], [5, 30], N_target, N_surround, axon_length, discrete, trg_above_thresh]
# fitParam: vm_target, vm_surround, alpha_target_surround, alpha_energy_sparsity
fitParam = [vm_targ, vm_surr, 0.4, 0.5]
# gaParam: NGEN, CXPB, MUTPB, PBEST, NSTALL
gaParam = [Ngen, 0.8, 0.1, 0.04, 100]
# mutselParam: muGauss, sdGauss, pgauss, pzero, tourn_size
mutselParam = [0, 4000, 0.2, 0.2, 3]

s = sim.SimulateScenario()

# complexity = [proximity, divergence, overlap]

# performance: [null_electrodes, average_current
#              n_above_target, max_af_target, min_af_target, mean_af_target, median_af_targ, sd_af_target,
#              n_below_surround, max_af_surround, min_af_surround, mean_af_surround, median_af_surround, sd_af_surround]

# performances: [ga_performance, mono_performance, bi_performance]

# neurons: [target_neurons, surround_neurons]


currents = []
neurons = []

# Initialize output arrays
# Scenario complexity
axon_dist = np.zeros(N_sim)
div = np.zeros(N_sim)
overlap = np.zeros(N_sim)

# Energy
ga_null = np.zeros(N_sim)
ga_mean_curr = np.zeros(N_sim)
# AF
# MAX
ga_max_targ = np.zeros(N_sim)
ga_max_surr = np.zeros([N_sim, N_surround])
mono_max_targ = np.zeros(N_sim)
mono_max_surr = np.zeros([N_sim, N_surround])
bi_max_targ = np.zeros(N_sim)
bi_max_surr = np.zeros([N_sim, N_surround])
# MIN
ga_min_targ = np.zeros(N_sim)
ga_min_surr = np.zeros([N_sim, N_surround])
mono_min_targ = np.zeros(N_sim)
mono_min_surr = np.zeros([N_sim, N_surround])
bi_min_targ = np.zeros(N_sim)
bi_min_surr = np.zeros([N_sim, N_surround])
# MEAN
ga_mean_targ = np.zeros(N_sim)
ga_mean_surr = np.zeros([N_sim, N_surround])
mono_mean_targ = np.zeros(N_sim)
mono_mean_surr = np.zeros([N_sim, N_surround])
bi_mean_targ = np.zeros(N_sim)
bi_mean_surr = np.zeros([N_sim, N_surround])
# MEDIAN
ga_median_targ = np.zeros(N_sim)
ga_median_surr = np.zeros([N_sim, N_surround])
mono_median_targ = np.zeros(N_sim)
mono_median_surr = np.zeros([N_sim, N_surround])
bi_median_targ = np.zeros(N_sim)
bi_median_surr = np.zeros([N_sim, N_surround])
# SD
ga_sd_targ = np.zeros(N_sim)
ga_sd_surr = np.zeros([N_sim, N_surround])
mono_sd_targ = np.zeros(N_sim)
mono_sd_surr = np.zeros([N_sim, N_surround])
bi_sd_targ = np.zeros(N_sim)
bi_sd_surr = np.zeros([N_sim, N_surround])

for n in range(N_sim):
    print 'Scenario Number ', (n+1), ' out of ', N_sim
    [complexity, performances, currents, neurons] = s.simulate_scenario(meaParam, neuronParam,
                                                                        fitParam, gaParam, mutselParam, verbose=verbose)

    # Scenario complexity
    axon_dist[n] = complexity[0]
    div[n] = complexity[1]
    overlap[n] = complexity[2]
    # Energy
    ga_null[n] = performances[0][0]
    ga_mean_curr[n] = performances[0][1]
    # AF
    # MAX
    ga_max_targ[n] = performances[0][3]
    ga_max_surr[n, :] = np.array(performances[0][9])
    mono_max_targ[n] = performances[1][3]
    mono_max_surr[n, :] = np.array(performances[1][9])
    bi_max_targ[n] = performances[2][3]
    bi_max_surr[n, :] = np.array(performances[2][9])
    # MIN
    ga_min_targ[n] = performances[0][4]
    ga_min_surr[n, :] = np.array(performances[0][10])
    mono_min_targ[n] = performances[1][4]
    mono_min_surr[n, :] = np.array(performances[1][10])
    bi_min_targ[n] = performances[2][4]
    bi_min_surr[n, :] = np.array(performances[2][10])
    # MEAN
    ga_mean_targ[n] = performances[0][5]
    ga_mean_surr[n, :] = np.array(performances[0][11])
    mono_mean_targ[n] = performances[1][5]
    mono_mean_surr[n, :] = np.array(performances[1][11])
    bi_mean_targ[n] = performances[2][5]
    bi_mean_surr[n, :] = np.array(performances[2][11])
    # MEDIAN
    ga_median_targ[n] = performances[0][6]
    ga_median_surr[n, :] = np.array(performances[0][12])
    mono_median_targ[n] = performances[1][6]
    mono_median_surr[n, :] = np.array(performances[1][12])
    bi_median_targ[n] = performances[2][6]
    bi_median_surr[n, :] = np.array(performances[2][12])
    # SD
    ga_sd_targ[n] = performances[0][7]
    ga_sd_surr[n, :] = np.array(performances[0][13])
    mono_sd_targ[n] = performances[1][7]
    mono_sd_surr[n, :] = np.array(performances[1][13])
    bi_sd_targ[n] = performances[2][7]
    bi_sd_surr[n, :] = np.array(performances[2][13])

'''PLOTS'''

if plot:
    fig1 = plt.figure()
    ax0 = plt.subplot(121)
    ax0.scatter(overlap, ga_min_targ, c='r', marker='o', s=10)
    ax0.scatter(overlap, mono_min_targ, c='b', marker='*', s=10)
    ax0.scatter(overlap, bi_min_targ, c='k', marker='^', s=10)

    ax1 = plt.subplot(122)
    ax1.scatter(overlap, np.max(ga_max_surr, 1), c='r', marker='o', s=10)
    ax1.scatter(overlap, np.max(mono_max_surr, 1), c='b', marker='*', s=10)
    ax1.scatter(overlap, np.max(bi_max_surr, 1), c='k', marker='^', s=10)
    # for ss in range(N_surround):
    #     ax1.scatter(overlap, ga_max_surr[:,ss] , c='r', marker='o', s=50)
    #     ax1.scatter(overlap, mono_max_surr[:,ss], c='b', marker='*', s=50)
    #     ax1.scatter(overlap, bi_max_surr[:,ss], c='k', marker='^', s=50)

    fig2 = plt.figure()
    ax2 = plt.subplot(121)
    ax2.scatter(axon_dist, ga_min_targ, c='r', marker='o', s=50)
    ax2.scatter(axon_dist, mono_min_targ, c='b', marker='*', s=50)
    ax2.scatter(axon_dist, bi_min_targ, c='k', marker='^', s=50)

    ax3 = plt.subplot(122)
    ax3.scatter(axon_dist, np.max(ga_max_surr, 1), c='r', marker='o', s=50)
    ax3.scatter(axon_dist, np.max(mono_max_surr, 1), c='b', marker='*', s=50)
    ax3.scatter(axon_dist, np.max(bi_max_surr, 1), c='k', marker='^', s=50)
    # for ss in range(N_surround):
    #     ax3.scatter(axon_dist, ga_max_surr[:,ss], c='r', marker='o', s=50)
    #     ax3.scatter(axon_dist, mono_max_surr[:,ss], c='b', marker='*', s=50)
    #     ax3.scatter(axon_dist, bi_max_surr[:,ss], c='k', marker='^', s=50)2000 NOK

    fig3 = plt.figure()
    ax4 = plt.subplot(121)
    ax4.scatter(div, ga_min_targ, c='r', marker='o', s=50)
    ax4.scatter(div, mono_min_targ, c='b', marker='*', s=50)
    ax4.scatter(div, bi_min_targ, c='k', marker='^', s=50)

    ax5 = plt.subplot(122)
    ax5.scatter(div, np.max(ga_max_surr, 1), c='r', marker='o', s=50)
    ax5.scatter(div, np.max(mono_max_surr, 1), c='b', marker='*', s=50)
    ax5.scatter(div, np.max(bi_max_surr, 1), c='b', marker='*', s=50)
    #
    # for ss in range(4):
    #     ax5.scatter(div, ga_max_surr[:,ss], c='r', marker='o', s=50)
    #     ax5.scatter(div, mono_max_surr[:,ss], c='b', marker='*', s=50)
    #     ax5.scatter(div, bi_max_surr[:,ss], c='k', marker='^', s=50)


    fig4 = plt.figure()
    ax6 = plt.subplot(121)
    ax6.boxplot([ga_min_targ, mono_min_targ, bi_min_targ])
    ax7 = plt.subplot(122)
    ax7.boxplot([ga_max_surr, mono_max_surr, bi_max_surr])

    fig5 = plt.figure()
    ax7 = plt.subplot(111)
    ax7.scatter(overlap, ga_null, c='r', marker='o', s=50)
    ax7.scatter(overlap, ga_mean_curr, c='b', marker='*', s=50)

    fig6 = plt.figure()
    ax8 = plt.subplot(111)
    ax8.scatter(overlap, (16 - ga_null)*ga_mean_curr, c='r', marker='o', s=50)


    '''Linear Regression'''

    slope_ga_surr, intercept_ga_surr, r_value_ga_surr, p_value_ga_surr, std_err_ga_surr = \
        stats.linregress(overlap, np.max(ga_max_surr, 1))

    slope_mono_surr, intercept_mono_surr, r_value_mono_surr, p_value_mono_surr, std_err_mono_surr = \
        stats.linregress(overlap, np.max(mono_max_surr, 1))

    slope_bi_surr, intercept_bi_surr, r_value_bi_surr, p_value_bi_surr, std_err_bi_surr = \
        stats.linregress(overlap, np.max(bi_max_surr, 1))

    X_overlap = np.linspace(0, np.max(overlap), num=50)
    Y_ga = slope_ga_surr*X_overlap + intercept_ga_surr
    Y_mono = slope_mono_surr*X_overlap + intercept_mono_surr
    Y_bi = slope_bi_surr*X_overlap + intercept_bi_surr

    # Add lines to plot
    ax1.plot(X_overlap, Y_ga, c='r', lw=2, ls='--')
    ax1.plot(X_overlap, Y_mono, c='b', lw=2, ls='-.')
    ax1.plot(X_overlap, Y_bi, c='k', lw=2, ls=':')


    fig7 = plt.figure()
    ax10 = plt.subplot(111)

    plt.xlabel('Complexity [0, 2]', fontsize=20)
    plt.ylabel('dv2', fontsize=20)
    plt.title('Maximum surround AF value VS complexity', fontsize=20)
    overlap_2 = np.abs(np.cos(div)) + (1-axon_dist/np.max(axon_dist))

    bi = ax10.scatter(overlap_2, np.max(mono_max_surr, 1), c='b', marker='*', s=50)
    mono = ax10.scatter(overlap_2, np.max(bi_max_surr, 1), c='k', marker='^', s=50)
    ga = ax10.scatter(overlap_2, np.max(ga_max_surr, 1), c='r', marker='o', s=50)

    plt.legend((ga, mono, bi),
               ('GA', 'Monopolar', 'Bipolar'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=18)

    plt.savefig('Sim3.pdf' ,
                dpi=300)

    fig4 = plt.figure()
    ax6 = plt.subplot(121)
    plt.xlabel('GA - MONO - BI', fontsize=20)
    plt.ylabel('dv2', fontsize=20)
    plt.title('Minimum Target dv2', fontsize=20)
    ax6.boxplot([ga_min_targ, mono_min_targ, bi_min_targ])
    ax7 = plt.subplot(122)
    plt.xlabel('GA - MONO - BI', fontsize=20)
    plt.ylabel('dv2', fontsize=20)
    plt.title('Maximum Surround dv2', fontsize=20)
    ax7.boxplot([np.max(ga_max_surr, 1), np.max(mono_max_surr, 1), np.max(bi_max_surr, 1)])

    plt.savefig('Sim4.pdf',
                dpi=300)

if save:
    # Saving the objects:

    # file string

    file_name = os.path.abspath('Simulation_Output') + '/Simulation_Nsim_' + str(N_sim) + '_Ngen_' + str(Ngen) + \
                '_Nstall_' + str(Nstall) + '_Ntrg_' + str(N_target) + '_Nsrr_' + str(N_surround) + '_Mono_' +  \
                str(monopolar_current) + '_vtarg_' + str(vm_targ) + '_vsurr_' + str(vm_surr) + '_trgAboveThresh_' + \
                str(trg_above_thresh) + '_discrete_' + str(discrete) + '.pickle'

    obj_field_name = ['axon_dist', 'div', 'overlap', 'ga_null', 'ga_mean_curr',
                      'ga_max_targ', 'ga_max_surr', 'ga_min_targ', ' ga_min_surr',
                      'ga_mean_targ', ' ga_mean_surr', 'ga_median_targ', ' ga_median_surr',
                      'ga_sd_targ', ' ga_sd_surr',
                      'mono_max_targ', ' mono_max_surr', ' mono_min_targ', ' mono_min_surr',
                      'mono_mean_targ', ' mono_mean_surr', 'mono_median_targ', ' mono_median_surr',
                      'mono_sd_targ', ' mono_sd_surr',
                      'bi_max_targ', ' bi_max_surr', ' bi_min_targ', ' bi_min_surr',
                      'bi_mean_targ', ' bi_mean_surr', 'bi_median_targ', ' bi_median_surr',
                      'bi_sd_targ', ' bi_sd_surr']

    obj_field = [axon_dist, div, overlap, ga_null, ga_mean_curr,
                 ga_max_targ, ga_max_surr, ga_min_targ, ga_min_surr,
                 ga_mean_targ, ga_mean_surr, ga_median_targ, ga_median_surr, ga_sd_targ, ga_sd_surr,
                 mono_max_targ, mono_max_surr, mono_min_targ, mono_min_surr,
                 mono_mean_targ, mono_mean_surr, mono_median_targ, mono_median_surr, mono_sd_targ, mono_sd_surr,
                 bi_max_targ, bi_max_surr, bi_min_targ, bi_min_surr,
                 bi_mean_targ, bi_mean_surr, bi_median_targ, bi_median_surr, bi_sd_targ, bi_sd_surr]

    with open(file_name, 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([obj_field_name, obj_field], f)

# # # Getting back the objects:
#
# # file string
# file_name = os.path.abspath('Simulation_Output/Simulation_Nsim_5_Ngen_300_Nstall100_Ntrg_1_Nsrr_4.pickle')
#
# with open(file_name) as f:  # Python 3: open(..., 'rb')
#     obj_field_name, obj_field = pickle.load(f)
#
# axon_dist = obj_field[0]
# div = obj_field[1]
# overlap = obj_field[2]
# ga_null = obj_field[3]
# ga_mean_curr = obj_field[4]
#
# ga_max_targ = obj_field[5]
# ga_max_surr = obj_field[6]
# ga_min_targ = obj_field[7]
# ga_min_surr = obj_field[8]
#
# ga_mean_targ = obj_field[9]
# ga_mean_surr = obj_field[10]
# ga_sd_targ = obj_field[11]
# ga_sd_surr = obj_field[12]
#
# mono_max_targ = obj_field[13]
# mono_max_surr = obj_field[14]
# mono_min_targ = obj_field[15]
# mono_min_surr = obj_field[16]
#
# mono_mean_targ = obj_field[17]
# mono_mean_surr = obj_field[18]
# mono_sd_targ = obj_field[19]
# mono_sd_surr = obj_field[20]
#
# bi_max_targ = obj_field[21]
# bi_max_surr = obj_field[22]
# bi_min_targ = obj_field[23]
# bi_min_surr = obj_field[24]
#
# bi_mean_targ = obj_field[25]
# bi_mean_surr = obj_field[26]
# bi_sd_targ = obj_field[27]
# bi_sd_surr = obj_field[28]
#


































# target_neurons = neurons[0]
# surround_neurons = neurons[1]
#
# meaPhantom = mea.SquareMEA(dim = 4, pitch=15)
#
# fig1 = plt.figure()
# ax0 = plt.subplot(111)
# ax0.scatter([meaPhantom.electrodes[elec].position[1] for elec in range(0, meaPhantom.number_electrode)],
#             [meaPhantom.electrodes[elec].position[2] for elec in range(0, meaPhantom.number_electrode)],
#             marker='o', c='b', s=20)
#
# for neur in range(len(target_neurons)):
#     axon_terminal = target_neurons[neur].get_axon_end()
#     neuron_proj_ext = ax0.plot([target_neurons[neur].soma_pos[1], axon_terminal[1]], [target_neurons[neur].soma_pos[2], axon_terminal[2]])
#     plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
#     neuron_proj_int = ax0.plot([target_neurons[neur].soma_pos[1], axon_terminal[1]], [target_neurons[neur].soma_pos[2], axon_terminal[2]])
#     plt.setp(neuron_proj_int, 'color', 'r', 'linewidth', 2.0)
#     ax0.scatter(target_neurons[neur].soma_pos[1], target_neurons[neur].soma_pos[2], marker='^', c='k', s=600)
#     ax0.scatter(target_neurons[neur].soma_pos[1], target_neurons[neur].soma_pos[2], marker='^', c='r', s=500)
#
#
# for neur in range(len(surround_neurons)):
#     axon_terminal = surround_neurons[neur].get_axon_end()
#     neuron_proj_ext = ax0.plot([surround_neurons[neur].soma_pos[1], axon_terminal[1]], [surround_neurons[neur].soma_pos[2], axon_terminal[2]])
#     plt.setp(neuron_proj_ext, 'color', 'k', 'linewidth', 3.0)
#     neuron_proj_int = ax0.plot([surround_neurons[neur].soma_pos[1], axon_terminal[1]], [surround_neurons[neur].soma_pos[2], axon_terminal[2]])
#     plt.setp(neuron_proj_int, 'color', '0.5', 'linewidth', 2.0)
#     ax0.scatter(surround_neurons[neur].soma_pos[1], surround_neurons[neur].soma_pos[2], marker='^', c='k', s=400)
#     ax0.scatter(surround_neurons[neur].soma_pos[1], surround_neurons[neur].soma_pos[2], marker='^', c='#888888', s=400)
#
#
# fig = plt.figure()
# ax1 = plt.subplot(131)
# ax2 = plt.subplot(132)
# ax3 = plt.subplot(133)
#
# ax1.matshow(currents[0])
# ax2.matshow(currents[1])
# ax3.matshow(currents[2])
#
