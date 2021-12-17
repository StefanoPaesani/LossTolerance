from CodesFunctions.LTCodeClass import LTCode, powerset
from CodesFunctions.graphs import *
import matplotlib.pyplot as plt
import time
import networkx as nx

from itertools import combinations
from heapq import nlargest

if __name__ == '__main__':
    ##################################
    #####     INITIALISATION     #####
    ##################################
    ##### Define graph dimensions & calculation parameters

    max_m = 4
    num_MC_trials = 100
    # loss_prob_list = np.linspace(0, 1, 10)
    loss_prob = 0.1

    code_qubits_num = 6

    ##### initialise lists

    graph_codes = {}
    labels = []

    #############################################################
    #####     GENERATE ALL POSSIBLE N QUBIT GRAPH CODES     #####
    #############################################################

    graph_nodes = list(range(code_qubits_num))
    all_possible_edges = combinations(graph_nodes, 2)
    all_graphs_by_edges = list(powerset(all_possible_edges))
    num_graphs = len(all_graphs_by_edges)

    all_input_choices = [graph_nodes[:i] for i in range(1, code_qubits_num+1)]
    num_input_choices = len(all_input_choices)
    tot_num_codes = num_graphs * num_input_choices

    for edges_conf_ix, graph_edges in enumerate(all_graphs_by_edges):
        for input_conf_ix, input_conf in enumerate(all_input_choices):
            graph = nx.Graph()
            graph.add_nodes_from(graph_nodes)
            graph.add_edges_from(graph_edges)

            graph_codes[(edges_conf_ix, input_conf_ix)] = LTCode(graph, input_conf, [])
            labels.append(str((edges_conf_ix, input_conf_ix)))

    ########################################################
    ### PERFORM ULTRA LOSS TOLERANCE TESTS - FIRST ROUND ###
    ########################################################
    graph_dict_keys = list(graph_codes.keys())

    ######## FAST CALCULATION
    print("Starting 1st round of calculation.", tot_num_codes, "codes and", num_MC_trials, "samples")
    start_time = time.time()
    codes_LT_probs_list = [graph_codes[code_key].ULTRA_teleport_prob_MC_estimation(loss_prob, max_m, num_MC_trials)
                           for code_key in graph_dict_keys
                           ]
    end_time = time.time()
    print("Completed! Time used:", end_time - start_time, "s")


    ######## MUCH SLOWER CALCULATION BECAUSE IT PRINTS A LOT OF STUFF
    # codes_LT_probs_list = []
    # initial_time = time.time()
    # for code_ix, code_key in enumerate(graph_dict_keys):
    #     print("Performing test on code", code_ix, "of", tot_num_codes, ": ", labels[code_ix])
    #     start_time = time.time()
    #
    #     tele_succ_probs = graph_codes[code_key].ULTRA_teleport_prob_MC_estimation(loss_prob, max_m, num_MC_trials)
    #     end_time = time.time()
    #     print("time used for this code:", end_time - start_time, "s")
    #     codes_LT_probs_list.append(tele_succ_probs)
    # final_time = time.time()
    # print("\nCompleted! Time used:", final_time - initial_time, "s")

    ####################################################
    ########     FIND BEST CODES FROM ROUND 1  #########
    ####################################################

    num_best_codes = int(np.sqrt(tot_num_codes))
    best_probs = nlargest(num_best_codes, codes_LT_probs_list)
    min_best_prob = min(best_probs)

    best_dict_keys_list = [code_key for code_key_ix, code_key in enumerate(graph_dict_keys)
                           if codes_LT_probs_list[code_key_ix] >= min_best_prob]

    graph_codes = {key: graph_codes[key] for key in best_dict_keys_list}


    ########################################################
    ### PERFORM ULTRA LOSS TOLERANCE TESTS - SECOND ROUND ###
    ########################################################
    graph_dict_keys = list(graph_codes.keys())
    tot_num_codes = len(graph_dict_keys)

    num_MC_trials = 10000

    ######## FAST CALCULATION
    print("Starting 2st round of calculation.", tot_num_codes, "codes and", num_MC_trials, "samples")
    start_time = time.time()
    codes_LT_probs_list = [graph_codes[code_key].ULTRA_teleport_prob_MC_estimation(loss_prob, max_m, num_MC_trials)
                           for code_key in graph_dict_keys
                           ]
    end_time = time.time()
    print("Completed! Time used:", end_time - start_time, "s")


    ####################################################
    ########     FIND BEST CODES FROM ROUND 2  #########
    ####################################################

    num_best_codes = 20
    best_probs = nlargest(num_best_codes, codes_LT_probs_list)
    print(best_probs)

    best_codes = [graph_codes[graph_dict_keys[codes_LT_probs_list.index(good_prob)]] for good_prob in best_probs]

    ####################################
    ########     PLOT DATA     #########
    ####################################

    n = num_best_codes
    i = 2
    while i * i < n:
        while n % i == 0:
            n = n / i
        i = i + 1

    n_plot_rows = num_best_codes/n
    n_plot_cols = n

    for code_ix in range(num_best_codes):
        plt.subplot(n_plot_rows, n_plot_cols, code_ix+1)
        best_codes[code_ix].image(with_labels=True)
    plt.show()
