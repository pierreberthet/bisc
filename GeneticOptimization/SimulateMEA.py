'''

Functions to generate a scenario and simulate it with GA optimization and MONO


'''

import random, math
import heapq
import numpy as np
from numpy import linalg as la

from deap import base
from deap import creator
from deap import tools

from MEAutility import *
import matplotlib.pyplot as plt


class SimulateScenario:

    def __init__(self):
        # Const and utilities
        self.der2filter = [1, -2, 1]
        self.monopolar_amp = -20000
        self.current_step = 2
        self.max_current = 20
        self.mea = []
        self.axon_length = 15
        self.discrete = 15
        self.trg_above_thresh = 15
        self.target_neurons = []
        self.x_target = []
        self.surround_neurons = []
        self.x_surround = []
        self.vm_threshold_target = 1
        self.vm_threshold_surround = 0
        self.alpha_target_surround = 0.5
        self.alpha_energy_sparsity = 0.5

    def simulate_scenario(self, meaParam, neuronParam, fitParam, gaParam, mutselParam, verbose=False):

        print 'Simulating:'

        '''MEA and NEURONS initialization'''

        # meaParam[0] = N_side
        N_side = meaParam[0]
        # meaParam[1] = pitch
        pitch = meaParam[1]
        # meaParam[2] = current_step
        self.current_step = meaParam[2]
        # meaParam[3] = max_current
        self.max_current = meaParam[3]
        # meaParam[4] = monopolar_amp
        self.monopolar_amp = meaParam[4]

        # create Mea
        self.mea = SquareMEA(dim=N_side, pitch=pitch)

        bound_uni = int(abs(self.mea[0][0].position[1])-float(pitch/2))

        # neuronParam[0] = close_separation
        close_separation = neuronParam[0]
        # neuronParam[1] = x_lim_trg
        x_lim_trg = neuronParam[1]
        # neuronParam[2] = x_lim_trg
        x_lim_surr = neuronParam[2]
        # neuronParam[3] = N_target
        N_target = neuronParam[3]
        # neuronParam[4] = N_surround
        N_surround = neuronParam[4]
        # neuronParam[5] = axon_length
        self.axon_length = neuronParam[5]
        # neuronParam[6] = discrete
        self.discrete = neuronParam[6]
        # neuronParam[7] = trg_above_thresh
        self.trg_above_thresh = neuronParam[7]

        self.target_neurons = []
        self.x_target = []
        self.surround_neurons = []
        self.x_surround = []
        target_somas = []
        target_dir = []
        surround_somas = []
        surround_dir = []
        all_somas = []

        # Generate N random neurons (Check that they are not closer than close_separation
        for nn in range(N_target):

            new_soma_not_found = True

            while new_soma_not_found:

                x_rand = random.randint(x_lim_trg[0], x_lim_trg[1])
                y_rand = random.randint(-bound_uni, bound_uni)
                z_rand = random.randint(-bound_uni, bound_uni)

                soma_targ = [x_rand, y_rand, z_rand]

                # Check proximity with other somas
                if all([la.norm(np.array(soma_targ) - np.array(all_somas[e])) > close_separation
                        for e in range(len(all_somas))]):
                    # append new neuron and break while
                    all_somas.append(soma_targ)
                    axon_dir_targ = [0, np.random.randn(), np.random.randn()]
                    target_somas.append(soma_targ)
                    target_dir.append(axon_dir_targ)
                    new_soma_not_found = False

        for neur in range(len(target_somas)):
            self.target_neurons.append(GeometricNeuron(target_somas[neur], align_dir=target_dir[neur],
                                                       length=self.axon_length, discrete_points=self.discrete))
            self.x_target.append(self.target_neurons[neur].get_axon_points())

        for nn in range(N_surround):

            new_soma_not_found = True

            while new_soma_not_found:

                x_rand = random.randint(x_lim_surr[0], x_lim_surr[1])
                y_rand = random.randint(-bound_uni, bound_uni)
                z_rand = random.randint(-bound_uni, bound_uni)

                soma_surr = [x_rand, y_rand, z_rand]

                # Check proximity with other somas
                if all([la.norm(np.array(soma_surr) - np.array(all_somas[e])) > close_separation
                        for e in range(len(all_somas))]):
                    # append new neuron and break while
                    all_somas.append(soma_surr)
                    axon_dir_surr = [0, np.random.randn(), np.random.randn()]
                    surround_somas.append(soma_surr)
                    surround_dir.append(axon_dir_surr)
                    new_soma_not_found = False

        for neur in range(len(surround_somas)):
            self.surround_neurons.append(GeometricNeuron(surround_somas[neur], align_dir=surround_dir[neur],
                                                         length=self.axon_length, discrete_points=self.discrete))
            self.x_surround.append(self.surround_neurons[neur].get_axon_points())

        '''GA initialization'''

        # FITNESS

        # fitParam[0] = vm_threshold_target
        self.vm_threshold_target = fitParam[0]
        # fitParam[1] = vm_threshold_surround
        self.vm_threshold_surround = fitParam[1]
        # fitParam[2] = alpha_target_surround
        self.alpha_target_surround = fitParam[2]
        # fitParam[3] = alpha_energy_sparsity
        self.alpha_energy_sparsity = fitParam[3]

        # gaParam[0] = NGEN
        NGEN = gaParam[0]
        # gaParam[1] = CXPB
        CXPB = gaParam[1]
        # gaParam[2] = MUTPB
        MUTPB = gaParam[2]
        # gaParam[3] = PBEST
        PBEST = gaParam[3]
        # gaParam[4] = NSTALL
        NSTALL = gaParam[4]
        IND_SIZE = self.mea.number_electrode

        # mutselParam[0] = muGauss
        muGauss = mutselParam[0]
        # mutselParam[1] = sigGauss
        sigGauss = mutselParam[1]
        # mutselParam[2] = pGauss
        pGauss = mutselParam[2]
        # mutselParam[3] = pZero
        pZero = mutselParam[3]
        # mutselParam[4] = tournsize
        tournSize = mutselParam[4]

        # Maximize neuron activation and energy efficiency
        creator.create("FitnessMulti", base.Fitness, weights=(1, 1))

        # Create individual with fitness attribute
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        # register: creates aliases (e.g. attr_float stands for random.random)
        toolbox.register("attr_float", self.my_random, self.max_current)
        # initRepeat -> 3 args: Individual, function, size
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register functions
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.my_mutate, mu=muGauss, sigma=sigGauss, pgauss=pGauss, pzero=pZero)
        toolbox.register("select", tools.selTournament, tournsize=tournSize)
        toolbox.register("evaluate", self.compute_fitness)

        pop = toolbox.population(self.mea.number_electrode)
        n_best = int(math.ceil(IND_SIZE * PBEST))

        besties = []
        f1 = []
        f2 = []
        nstall = 0

        for g in range(NGEN):
            if verbose:
                print "Generation number ", g

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = map(toolbox.clone, offspring)

            # Apply crossover on the offspring, except for best solutions
            for child1, child2 in zip(offspring[n_best::2], offspring[n_best + 1::2]):
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

            if verbose:

                if len(self.x_surround) != 0:
                    print "AF fitness: ", toolbox.evaluate(best_solution[0])[0], " over ", 1, \
                          "AF neutrality: ", toolbox.evaluate(best_solution[0])[1], " over ", 1

                else:
                    print "Target above threshold fitness: ", toolbox.evaluate(best_solution[0])[0], \
                          "No surrounding neurons.  Zero-current electrodes = ", toolbox.evaluate(best_solution[0])[2]

            if besties[g] == besties[g - 1]:
                nstall += 1
                if verbose:
                    print 'Stall for ', nstall, ' iterations...'
            else:
                nstall = 0

            '''STOP CRITERION'''
            if nstall == NSTALL:
                print 'Solution in stall. Exiting loop'
                break

        print 'DONE'

        # Compute figure of merits

        # Scenario Complexity

        # Figure of merit: dot product between axon directions normalized over the distance between mid axon points
        proximity = []
        divergence = []
        # Overlap = targ_dir*surr_dir + (1 - inter_axon_dist/max_d)
        overlap = []
        max_d = np.abs(bound_uni)

        for tt in range(len(self.target_neurons)):
            for ss in range(len(self.surround_neurons)):
                mid_axon_targ = np.array(target_somas[tt]) + np.array(target_dir[tt]) * 0.5 * self.axon_length
                mid_axon_surr = np.array(surround_somas[ss]) + np.array(surround_dir[ss]) * 0.5 * self.axon_length
                inter_axon_dist = la.norm(mid_axon_targ - mid_axon_surr)
                if tt == 0 and ss == 0:
                    proximity = inter_axon_dist
                    divergence = abs(np.dot(self.target_neurons[tt].align_dir, self.surround_neurons[ss].align_dir))
                    overlap = abs(np.dot(self.target_neurons[tt].align_dir, self.surround_neurons[ss].align_dir)) + \
                                 (1 - inter_axon_dist / max_d)

                else:
                    if inter_axon_dist < proximity:
                        proximity = inter_axon_dist
                        divergence = abs(np.dot(self.target_neurons[tt].align_dir, self.surround_neurons[ss].align_dir))

                    if abs(np.dot(self.target_neurons[tt].align_dir, self.surround_neurons[ss].align_dir)) + \
                          (1 - inter_axon_dist / max_d) > overlap:
                        overlap = abs(np.dot(self.target_neurons[tt].align_dir, self.surround_neurons[ss].align_dir)) + \
                              (1 - inter_axon_dist / max_d)

            # divergence in degrees
            divergence = np.arccos(divergence) * 180 / np.pi

        # GA Stimulation Performance
        null_electrodes_GA = len(np.where(np.array(self.mea.get_currents()) == 0)[0])
        average_current_GA = sum(np.abs(self.mea.get_currents())) / float((self.mea.number_electrode -
                                                                           null_electrodes_GA)*1000)

        # Average target above activation threshold
        n_above_target_GA = 0
        max_af_target_GA = np.zeros(len(self.target_neurons))
        min_af_target_GA = np.zeros(len(self.target_neurons))
        mean_af_target_GA = np.zeros(len(self.target_neurons))
        median_af_target_GA = np.zeros(len(self.target_neurons))
        sd_af_target_GA = np.zeros(len(self.target_neurons))

        tract = float(self.axon_length) / (self.discrete - 1)

        for neur in range(len(self.target_neurons)):
            v_axon_ = self.mea.compute_field(self.x_target[neur])
            dv2_axon_ = np.convolve(v_axon_, self.der2filter, 'valid') / tract ** 2

            n_above_target_GA += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] >= self.vm_threshold_target)
            max_af_target_GA[neur] = np.max(dv2_axon_)
            min_af_target_GA[neur] = np.min(dv2_axon_)
            mean_af_target_GA[neur] = np.mean(dv2_axon_)
            median_af_target_GA[neur] = np.median(dv2_axon_)
            sd_af_target_GA[neur] = np.std(dv2_axon_)
        # Normalize over number of targeting neurons
        n_above_target_GA /= float(len(self.target_neurons))

        n_below_surround_GA = 0
        max_af_surround_GA = np.zeros(len(self.surround_neurons))
        min_af_surround_GA = np.zeros(len(self.surround_neurons))
        mean_af_surround_GA = np.zeros(len(self.surround_neurons))
        median_af_surround_GA = np.zeros(len(self.surround_neurons))
        sd_af_surround_GA = np.zeros(len(self.surround_neurons))

        tract = float(self.axon_length) / (self.discrete - 1)

        for neur in range(len(self.surround_neurons)):
            v_axon_ = self.mea.compute_field(self.x_surround[neur])
            dv2_axon_ = np.convolve(v_axon_, self.der2filter, 'valid') / tract ** 2

            n_below_surround_GA += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] <= self.vm_threshold_surround)
            max_af_surround_GA[neur] = np.max(dv2_axon_)
            min_af_surround_GA[neur] = np.min(dv2_axon_)
            mean_af_surround_GA[neur] = np.mean(dv2_axon_)
            median_af_surround_GA[neur] = np.median(dv2_axon_)
            sd_af_surround_GA[neur] = np.std(dv2_axon_)
        # Normalize over number of surrounding neurons
        n_below_surround_GA /= float(len(self.surround_neurons))

        # Monopolar Stimulation Performance
        meaM = SquareMEA(dim=N_side, pitch=pitch)

        closest_ind = []
        dist = meaM.pitch*meaM.dim

        for neur in range(len(self.target_neurons)):
            # find closest electrode to mid point of axon hillock
            mid_axon = np.array(target_somas[neur]) + np.array(target_dir[neur]) * 0.5 * self.axon_length
            for ee in range(meaM.number_electrode):
                if la.norm(mid_axon-np.array(meaM.electrodes[ee].position)) < dist:
                    dist = la.norm(mid_axon-np.array(meaM.electrodes[ee].position))
                    closest_ind = ee

            # set current for electrode closest to target 'neur' to -10000
            meaM.get_electrode_array()[closest_ind].set_current(self.monopolar_amp)

        # Compute Performance:
        null_electrodes_MONO = len(np.where(np.array(meaM.get_currents()) == 0)[0])
        average_current_MONO = sum(np.abs(meaM.get_currents())) / float((meaM.number_electrode -
                                                                         null_electrodes_MONO) * 1000)

        # Average target above activation threshold
        n_above_target_MONO = 0
        max_af_target_MONO = np.zeros(len(self.target_neurons))
        min_af_target_MONO = np.zeros(len(self.target_neurons))
        mean_af_target_MONO = np.zeros(len(self.target_neurons))
        median_af_target_MONO = np.zeros(len(self.target_neurons))
        sd_af_target_MONO = np.zeros(len(self.target_neurons))

        for neur in range(len(self.target_neurons)):
            v_axon_ = meaM.compute_field(self.x_target[neur])
            dv2_axon_ = np.convolve(v_axon_, self.der2filter, 'valid') / tract ** 2

            n_above_target_MONO += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] >= self.vm_threshold_target)
            max_af_target_MONO[neur] = np.max(dv2_axon_)
            min_af_target_MONO[neur] = np.min(dv2_axon_)
            mean_af_target_MONO[neur] = np.mean(dv2_axon_)
            median_af_target_MONO[neur] = np.median(dv2_axon_)
            sd_af_target_MONO[neur] = np.std(dv2_axon_)
        # Normalize over number of targeting neurons
        n_above_target_MONO /= float(len(self.target_neurons))

        n_below_surround_MONO = 0
        max_af_surround_MONO = np.zeros(len(self.surround_neurons))
        min_af_surround_MONO = np.zeros(len(self.surround_neurons))
        mean_af_surround_MONO = np.zeros(len(self.surround_neurons))
        median_af_surround_MONO = np.zeros(len(self.surround_neurons))
        sd_af_surround_MONO = np.zeros(len(self.surround_neurons))

        for neur in range(len(self.surround_neurons)):
            v_axon_ = meaM.compute_field(self.x_surround[neur])
            dv2_axon_ = np.convolve(v_axon_, self.der2filter, 'valid') / tract ** 2

            n_below_surround_MONO += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] <= self.vm_threshold_surround)
            max_af_surround_MONO[neur] = np.max(dv2_axon_)
            min_af_surround_MONO[neur] = np.min(dv2_axon_)
            mean_af_surround_MONO[neur] = np.mean(dv2_axon_)
            median_af_surround_MONO[neur] = np.median(dv2_axon_)
            sd_af_surround_MONO[neur] = np.std(dv2_axon_)
        # Normalize over number of surrounding neurons
        n_below_surround_MONO /= float(len(self.surround_neurons))

        # Bipolar Stimulation Performance
        meaB = SquareMEA(dim=N_side, pitch=pitch)

        closest_ind_mid = []
        closest_ind_end = []
        dist_mid = meaB.pitch*meaM.dim
        dist_end = meaB.pitch*meaM.dim

        for neur in range(len(self.target_neurons)):
            # find closest electrode to mid point
            mid_axon = np.array(target_somas[neur]) + np.array(target_dir[neur]) * 0.5 * self.axon_length
            end_point_pos = np.array(self.target_neurons[neur].get_axon_end())
            for ee in range(meaB.number_electrode):
                if la.norm(mid_axon - np.array(meaB.electrodes[ee].position)) < dist_mid:
                    dist_mid = la.norm(mid_axon - np.array(meaB.electrodes[ee].position))
                    closest_ind_mid = ee
                    # find closest electrode to end point
            for ee in range(meaB.number_electrode):
                if la.norm(end_point_pos - np.array(meaB.electrodes[ee].position)) < dist_end and \
                                ee != closest_ind_mid:  # keep it bipolar (the 2 ind cannot coincide)
                    dist_end = la.norm(end_point_pos - np.array(meaB.electrodes[ee].position))
                    closest_ind_end = ee

            # set current for electrode closest to soma 'neur' to monopolar_amp (neg)
            meaB.get_electrode_array()[closest_ind_mid].set_current(self.monopolar_amp)
            # set current for electrode closest to soma 'neur' to -monopolar_amp (pos)
            meaB.get_electrode_array()[closest_ind_end].set_current(-self.monopolar_amp)

        # Compute Performance:
        null_electrodes_BI = len(np.where(np.array(meaB.get_currents()) == 0)[0])
        average_current_BI = sum(np.abs(meaB.get_currents())) / float((meaB.number_electrode -
                                                                       null_electrodes_BI) * 1000)

        # Average target above activation threshold
        n_above_target_BI = 0
        max_af_target_BI = np.zeros(len(self.target_neurons))
        min_af_target_BI = np.zeros(len(self.target_neurons))
        mean_af_target_BI = np.zeros(len(self.target_neurons))
        median_af_target_BI = np.zeros(len(self.target_neurons))
        sd_af_target_BI = np.zeros(len(self.target_neurons))

        for neur in range(len(self.target_neurons)):
            v_axon_ = meaB.compute_field(self.x_target[neur])
            dv2_axon_ = np.convolve(v_axon_, self.der2filter, 'valid') / tract ** 2

            n_above_target_BI += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] >= self.vm_threshold_target)
            max_af_target_BI[neur] = np.max(dv2_axon_)
            min_af_target_BI[neur] = np.min(dv2_axon_)
            mean_af_target_BI[neur] = np.mean(dv2_axon_)
            median_af_target_BI[neur] = np.median(dv2_axon_)
            sd_af_target_BI[neur] = np.std(dv2_axon_)
        # Normalize over number of targeting neurons
        n_above_target_BI /= float(len(self.target_neurons))

        n_below_surround_BI = 0
        max_af_surround_BI = np.zeros(len(self.surround_neurons))
        min_af_surround_BI = np.zeros(len(self.surround_neurons))
        mean_af_surround_BI = np.zeros(len(self.surround_neurons))
        median_af_surround_BI = np.zeros(len(self.surround_neurons))
        sd_af_surround_BI = np.zeros(len(self.surround_neurons))

        for neur in range(len(self.surround_neurons)):
            v_axon_ = meaB.compute_field(self.x_surround[neur])
            dv2_axon_ = np.convolve(v_axon_, self.der2filter, 'valid') / tract ** 2

            n_below_surround_BI += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] <= self.vm_threshold_surround)
            max_af_surround_BI[neur] = np.max(dv2_axon_)
            min_af_surround_BI[neur] = np.min(dv2_axon_)
            mean_af_surround_BI[neur] = np.mean(dv2_axon_)
            median_af_surround_BI[neur] = np.median(dv2_axon_)
            sd_af_surround_BI[neur] = np.std(dv2_axon_)
        # Normalize over number of surrounding neurons
        n_below_surround_BI /= float(len(self.surround_neurons))

        # RETURN

        complexity = [proximity, divergence, overlap]

        ga_performance = [null_electrodes_GA, average_current_GA, n_above_target_GA, max_af_target_GA, min_af_target_GA,
                          mean_af_target_GA, median_af_target_GA, sd_af_target_GA, n_below_surround_GA,
                          max_af_surround_GA, min_af_surround_GA, mean_af_surround_GA, median_af_surround_GA,
                          sd_af_surround_GA]

        mono_performance = [null_electrodes_MONO, average_current_MONO, n_above_target_MONO, max_af_target_MONO,
                            min_af_target_MONO, mean_af_target_MONO, median_af_target_MONO, sd_af_target_MONO,
                            n_below_surround_MONO, max_af_surround_MONO, min_af_surround_MONO, mean_af_surround_MONO,
                            median_af_surround_MONO, sd_af_surround_MONO]

        bi_performance = [null_electrodes_BI, average_current_BI, n_above_target_BI, max_af_target_BI,
                          min_af_target_BI, mean_af_target_BI, median_af_target_BI, sd_af_target_BI,
                          n_below_surround_BI, max_af_surround_BI, min_af_surround_BI, mean_af_surround_BI,
                          median_af_surround_BI, sd_af_surround_BI]

        performances = [ga_performance, mono_performance, bi_performance]

        neurons = [self.target_neurons, self.surround_neurons]

        ga_currents = self.mea.get_current_matrix()
        mono_currents = meaM.get_current_matrix()
        bi_currents = meaB.get_current_matrix()

        currents = [ga_currents, mono_currents, bi_currents]

        return complexity, performances, currents, neurons

    '''Auxiliary functions'''
    # GA functions
    def my_random(self, max_val):
        if random.random() < 0.25:
            randint = round(random.randint(-max_val, max_val) / float(self.current_step)) * self.current_step
            return randint * 1000
        else:
            return 0

    def compute_fitness(self, individual):
        self.mea.set_currents(individual)

        x_energy = 1 - (sum(np.abs(individual)) / (self.mea.number_electrode * self.max_current * 1000))
        x_sparsity = len(np.where(np.array(individual) == 0)[0]) / float(self.mea.number_electrode)

        tract = float(self.axon_length) / (self.discrete - 1)

        '''Maximize not only the number above threshold, but also by how far'''
        n_above_target = 0

        for neur in range(len(self.target_neurons)):
            v_axon_ = self.mea.compute_field(self.x_target[neur])
            dv2_axon_ = np.convolve(v_axon_, self.der2filter, 'valid') / tract ** 2

            # Check AF only on self.trg_above_thresh maximum points
            if self.trg_above_thresh > len(dv2_axon_):
                self.trg_above_thresh = len(dv2_axon_)
            dv2_axon_max = heapq.nlargest(self.trg_above_thresh, dv2_axon_)

            n_above_target += sum(1 for e in range(len(dv2_axon_max)) if dv2_axon_max[e] >= self.vm_threshold_target)
            n_above_target -= sum(abs(dv2_axon_max[e] - self.vm_threshold_target) /
                                  (self.vm_threshold_target - self.vm_threshold_surround) for e in range(len(dv2_axon_max))
                                  if dv2_axon_max[e] < self.vm_threshold_target)

        # Normalize over number of targeting neurons
        x_targ = n_above_target / float(len(self.target_neurons) * len(dv2_axon_max))

        '''Maximize number of surround neurons below surround threshold'''
        n_below_surround = 0
        tract = float(self.axon_length) / (self.discrete - 1)

        # Use all points for surrounding neurons
        for neur in range(len(self.surround_neurons)):
            v_axon_ = self.mea.compute_field(self.x_surround[neur])
            dv2_axon_ = np.convolve(v_axon_, self.der2filter, 'valid') / tract ** 2

            n_below_surround += sum(1 for e in range(len(dv2_axon_)) if dv2_axon_[e] <= self.vm_threshold_surround)
            n_below_surround -= sum(
                abs(dv2_axon_[e] - self.vm_threshold_surround) / (self.vm_threshold_target - self.vm_threshold_surround)
                for e in range(len(dv2_axon_)) if dv2_axon_[e] > self.vm_threshold_surround)

            # Normalize over number of surrounding neurons
        x_nontarg = n_below_surround / float(len(self.surround_neurons) * len(dv2_axon_))

        fit_activation = self.alpha_target_surround * x_targ + (1 - self.alpha_target_surround) * x_nontarg
        fit_energy = self.alpha_energy_sparsity * x_energy + (1 - self.alpha_energy_sparsity) * x_sparsity

        return fit_activation, fit_energy,

    def my_mutate(self, individual, mu, sigma, pgauss, pzero):
        size = len(individual)

        for i in range(size):
            # Apply gaussian mutation
            if random.random() < pgauss:
                individual[i] += random.gauss(mu, sigma)

                # round to current_step (nA -> uA -> nA conversion)
                individual[i] = round((individual[i] / 1000) / float(self.current_step)) * self.current_step * 1000

                # Check limits
                if individual[i] > self.max_current * 1000:
                    individual[i] = self.max_current * 1000
                if individual[i] < -self.max_current * 1000:
                    individual[i] = -self.max_current * 1000

            # Apply zero mutation --> facilitates sparse solutions
            if random.random() < pzero:
                individual[i] = 0

        return individual,
