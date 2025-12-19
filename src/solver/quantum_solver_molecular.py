import sys

from typing import Counter
from emu_mps import BitStrings, MPSBackend, MPSConfig
import networkx as nx
from itertools import islice
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
from pulser import Pulse, Sequence
import time
import logging #used to turn of logging in emu_mps
from dataclasses import replace

import pulser as pl
from pulser.devices import DigitalAnalogDevice, AnalogDevice
from pulser import Register
from pulser.waveforms import ConstantWaveform, InterpolatedWaveform
from pulser_simulation import QutipEmulator
from emu_sv import SVBackend, SVConfig,  BitStrings, StateResult

import optimiser as optimiser
import permutations as permutations
from quantum_opt_tools import reciprocal_dist_matrix, permute_sequence_registers



from pulser.channels.dmm import DMM


dmm = DMM(
    clock_period=4,
    min_duration=16,
    max_duration=2**26,
    mod_bandwidth=8,
    bottom_detuning=-2 * np.pi * 20,  # detuning between 0 and -20 MHz
    total_bottom_detuning=-2 * np.pi * 2000,  # total detuning
)

mock_device = replace(
    DigitalAnalogDevice.to_virtual(), #Other experiments we were using DigitalAnalog
    dmm_objects=(dmm, DMM()),
    reusable_channels=True,
)
#print(mock_device.dmm_channels)


def compute_min_max_u(register_coord, 
                      graph):
    """U_min corresponds to minimal energy of two nodes connected in the
    graph. U_max corresponds to maximal energy of two nodes NOT connected in
    the graph.

    Args:
        register_coord: Register coordinates of atoms representing the graph
        graph: graph of the problem

    Returns:
        u_min: Minimal energy of two nodes connectes
        u_max: Maximal energy of two nodes not connected"""

    interaction_coefficient = DigitalAnalogDevice.interaction_coeff
    distance_list_connected = []
    distance_list_no_connected = []

    for edge in graph.edges():
        v1 = register_coord[edge[0]]
        v2 = register_coord[edge[1]]
        distance_list_connected.append(dist.euclidean(v1, v2))
    u_min = interaction_coefficient / (np.max(distance_list_connected) ** 6)

    graph_complementary = nx.complement(graph)
    for edge in graph_complementary.edges():
        v1 = register_coord[edge[0]]
        v2 = register_coord[edge[1]]
        distance_list_no_connected.append(dist.euclidean(v1, v2))
    u_max = interaction_coefficient / (np.min(distance_list_no_connected) ** 6)

    return u_min, u_max

def simple_adiabatic_sequence(
    register,
    parameters
) -> Sequence:
    """Creates the adiabatic sequence

    Args:
        register: arrangement of atoms in the quantum processor
        parameters: Dictionary with the parameters for the sequence
    Returns:
        sequence
    """
    #print('Parameters used in the quantum evolution:', parameters)

    delta_0 = -parameters["detuning_maximum"]  
    delta_f = -delta_0
    
    channel_name = "rydberg_global"
    dmm_channel_name = "dmm_0"
    opt_params_fresnel={}
    duration=4000
    omega_list=[1e-9, 4.850055051963065, 12.566370614359172, 11.882582923954192,1e-9]
    delta_list=[-AnalogDevice.channels["rydberg_global"].max_abs_detuning,-23.735663070454464,2.429518510655143,13.330733571729787,31.06390344878028]

    
    # get the detuning map
    detuning_map = register.define_detuning_map(parameters["dmm_map"], dmm_channel_name)

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(parameters["duration"], [1e-9, parameters["amplitude_maximum"] , 1e-9]),
        InterpolatedWaveform(parameters["duration"], [delta_0, 0, delta_f]),
        0,
    )   

    amp = InterpolatedWaveform(duration,omega_list)
    det = InterpolatedWaveform(duration,delta_list)

 

    sequence = pl.Sequence(register, mock_device)
    sequence.declare_channel(channel_name, channel_name)
    sequence.config_detuning_map(detuning_map, dmm_channel_name)
    #sequence.add(adiabatic_pulse, channel_name)
    sequence.add(Pulse(amp, det, 0), "rydberg_global")

    dmm_wave = ConstantWaveform(duration, parameters["dmm_detuning"])
    sequence.add_dmm_detuning(dmm_wave, dmm_channel_name)

    #sequence = Sequence(register, device)
    #sequence.declare_channel("ising", "rydberg_global")

    #sequence.add(adiabatic_pulse, "ising")


    return sequence


def simple_quantum_loop(parameters, 
                        register,
                        list_perm, 
                        run_qutip = True,
                        run_emu_mps=False, 
                        run_sv = False):
    """Simulation of the sequence in the quantum device

    Args:
        parameters: Parameters used in the sequence
        register: arrangement of atoms in the quantum processor
        run_qutip: If true the simulation will be with the QuTip emulator
        run_emu_mps: If true the simulation will be with emu-mps

    Returns:
        counts:Distribution of the samples
    """

    #parameter_omega, parameter_detuning = np.reshape(params.astype(int), 2)
    seq = simple_adiabatic_sequence(
        register,
        parameters
    )
    #print(seq)

    if run_qutip:
        simul = QutipEmulator.from_sequence(seq)
        res = simul.run()
        counts = res.sample_final_state(N_samples=5000)  # Sample from the state vector

    if run_sv:

        
        dt = 10
        seq_duration = seq.get_duration() // dt * dt

        eval_times = [t/seq_duration for t in range(dt, seq_duration, dt)]
        final_time = eval_times[-1]
        #print("final time", final_time)
        state = StateResult(evaluation_times=[final_time])

        bitstrings = BitStrings(evaluation_times=[final_time], num_shots=5000)

        #state = StateResult(evaluation_times=[duration])
        config = SVConfig(
                 dt = dt,
                 observables = [bitstrings, state], 
                 log_level = logging.WARN,
                 )
        
        #seq.draw()
        sim = SVBackend(seq, config=config)
        results =sim.run()

        #final_state = results['state'][duration]

        #counts = results["bitstrings"][final_time]
        last_elem = -1
        final_state = results.state[last_elem]
        counts = final_state.sample(num_shots=5000)


    if run_emu_mps:
        start = time.time()
        
        dt = 10

        seq_duration = seq.get_duration() // dt * dt

        eval_times = [t/seq_duration for t in range(dt, seq_duration, dt)]
        final_time = eval_times[-1]
        #print("final time", final_time)
        bitstrings = BitStrings(evaluation_times=[final_time], num_shots=5000)
        config = MPSConfig(
                dt=dt, #precision = 1e-7,
                max_bond_dim=200,#400
                observables=[
                    bitstrings,
                ], log_level = logging.WARN #don't print stuff for the many runs
                )# 
        #seq.draw()
        sim = MPSBackend(seq, config=config)
        results = sim.run()
        #bitstrings = results[bitstrings.name()]
        #counts_mps = Counter(bitstrings[final_time])
        results.get_result_times(bitstrings)
        counts_mps = results.get_result(bitstrings, 1.0)
        counts = {}
        
        reverse_perm = permutations.invert_permutation(list_perm)
        for bitstring, count in counts_mps.items():
            list_bitstring = permutations.permute_list(list(bitstring),reverse_perm)
            new_bitstring = ''.join(list_bitstring)
            counts[new_bitstring] = count 
        end = time.time()
        print('simulation time', end - start)

    #print(counts)

    return counts



def process_quantum_solution(count_dist,register, num_solutions =4):
    """Process the counts distributions of the quantum solution
    
    Args:
        counts_dist: Counts distribution
        register: Atomic register of the graph
        num_solutions: The number of solutions to process, starting with the highest count
        
    Returns:
        solution: List with the quantum solutions"""
        
    #count_dist = dict(sorted(counts_d.items(), key = lambda item: item[1], reverse = True))
        
    solution = []
    #ADD SOMETHING TO DEAZL WITH NOT IS 
    for item in islice(count_dist,num_solutions): 
        #print(item)
        element = 0
        solutions_iterations = []
        for bit_solution in item:
            if int(bit_solution) == 1:
                solutions_iterations.append(register.qubit_ids[element])
                #print(reg.qubit_ids[element])
            element += 1
        solution.append(solutions_iterations)
            
    return solution


def q_solver(graph, 
            coordinates_layout, 
            num_sol,
            display_info = False):
    """MWIS quantum solver

    Args:
        graph: graph to solved
        coordinates_layout: Full Layout Coordinates
        num_sol: Number of MWIS solutions 

    Returns:
        solution: List of MWIS 
    """

    params = {"duration": 4000} # Time of for the quantum evolution 

    if display_info:

        print('..........................')
        print('Current graph:', graph)

    weights = nx.get_node_attributes(graph, 'weight') # Get the weights of the graph 
 
    array_weights_origin = np.array([weights[node] for node in weights.keys()]) # Create array with the weights of the graph
    atoms_id = [node for node in weights.keys()] # List of atoms id for the register of the graph 

    # Iterate through the coordinates layout and 
    # the atom id of the graph to obtain the register to used
    coord = {}
    for id_atom in atoms_id:
        for id_trap in coordinates_layout.keys():
            if id_atom==id_trap:
                coord[id_trap] = coordinates_layout[id_trap]

    reg = Register(coord) # Create register representing the graph 

    matrix_int  = reciprocal_dist_matrix(reg)
    list_perm = optimiser.minimize_bandwidth(matrix_int)
    opt_reg = permute_sequence_registers(reg,list_perm)
    permuted_weights = [array_weights_origin[i] for i in list_perm]


    interval_interaction = compute_min_max_u(coord, graph ) # Compute the min and max values for the amplitude and detuning


    params["detuning_maximum"] = interval_interaction[1]*2
    params["amplitude_maximum"] = interval_interaction[1]

    array_weights = permuted_weights / np.max(permuted_weights) # Normalize the weights 
    min_weight = np.min(array_weights)
    #spread_detuning = params["detuning_maximum"] * (1 - min_weight)# Compute the detuning to used in DMM
    spread_detuning =  31.06390344878028 * (1 - min_weight) 
    
    #array_weights = (array_weights - min_weight) / (1 - min_weight)

    weights_rev = 1 - array_weights # Reverse the weigths for the DMM

    params["dmm_map"] = dict(zip(opt_reg.qubit_ids, weights_rev)) # Assigned the DMM map to each register 
    params["dmm_detuning"] = -spread_detuning # Total negative detuning used in DMM
    
    # Quantum evolution of the register 
    if display_info:
        print('The quantum magic starts...')
        print('..........................')
    num_atoms = len(reg.qubits)
    
    if num_atoms < 22:
        params["dmm_map"] = dict(zip(reg.qubit_ids, weights_rev))
        params["dmm_detuning"] = -spread_detuning # Total negative detuning used in DMM
        reg.draw()
        count_dist = simple_quantum_loop(params, reg, list_perm, run_qutip= False ,run_emu_mps= False, run_sv= True)
    else:
        new_det_map_weights= optimiser.permute_list(list(weights_rev),list_perm)
        #detuning_map = opt_reg.define_detuning_map(dict(zip(opt_reg.qubit_ids, new_det_map_weights)) )
        params["dmm_map"] = dict(zip(opt_reg.qubit_ids, new_det_map_weights))
        params["dmm_detuning"] = -spread_detuning # Total negative detuning used in DMM
        opt_reg.draw()
        count_dist = simple_quantum_loop(params, opt_reg, list_perm, run_qutip= False ,run_emu_mps= True, run_sv= False)
    if display_info:
        print('..........................')
        print('The quantum magic has finish')
    count= dict(sorted(count_dist.items(), key = lambda item: item[1], reverse = True))
    
    if display_info:
        plt.figure(figsize=(3,2)) 
        nx.draw(graph, with_labels=True)
    
    #plot_distribution(count)
    
    solution = process_quantum_solution(count,reg, num_sol) #Given the counts obtain a list of MWIS



    if display_info:
        print('Quantum Solution for current graph:', solution)
        reg.draw()

    return solution



def plot_distribution(C):
    #C = simple_quantum_loop(optimal_parameters, register)
    #C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    print(C)
    plt.figure(figsize=(25, 15))
    plt.xlabel("bitstings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5)
    plt.xticks(rotation="vertical")
    plt.show()
