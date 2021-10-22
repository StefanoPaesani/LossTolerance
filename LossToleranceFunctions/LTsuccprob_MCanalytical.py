from CodesFunctions.GraphStateClass import GraphState
from LossToleranceFunctions.LT_qubitencoding import get_possible_stabs_meas, MC_decoding, trasmission_scan_MCestimate

import qecc as q
import numpy as np
import networkx as nx

from collections import Counter
from scipy.optimize import minimize, Bounds

from CodesFunctions.graphs import *


def succprob_poly_withMC(graph, transm=0.9, MC_samples=1000, t_sampl_func=None, in_qubit=0, printing=False):
    """
    Returns an analytical estimate (lower bound) of the success probability for recovering a qubit encoded in the
    graph, in the form of a polinomial of the form

            a_1 t^(e_11) (1-t)^(e_12) + a_2 t^(e_21) (1-t)^(e_22) + ... + a_n t^(e_n1) (1-t)^(e_n2)

    returning it in the form of the dictionary {(e_11, e_12): a_1, (e_21, e_22): a_2, ... , (e_n1, e_n2): a_n}
    """
    gstate = GraphState(graph)
    poss_stabs_list = get_possible_stabs_meas(gstate, in_qubit)

    if t_sampl_func == None:
        trans_sampl_func = lambda t: transm
    else:
        trans_sampl_func = lambda t: t_sampl_func(t)

    success_measurements = []
    for test_ix in range(MC_samples):
        # print(trans_sampl_func(transm))
        decoding_succ, decoding_meas = MC_decoding(poss_stabs_list, trans_sampl_func(transm),
                                                   in_qubit, printing=False, provide_measures=True)
        if decoding_succ:
            success_measurements.append(decoding_meas)

    ## filter out duplicates
    # print('success_measurements', success_measurements)
    success_meas_filter = list(set(success_measurements))
    polynom_all_exponents = [(len(this_meas[0]), len(this_meas[1])) for this_meas in success_meas_filter]

    if printing:
        print("success_meas_filter :", success_meas_filter)
        print("polynom_all_exponents :", polynom_all_exponents)

    return dict(Counter(polynom_all_exponents))


# Function that converts the polynomial expression into the success probability as a function of transmission t
def probsucc_poly_fromexpress(t, poly_express):
    return sum([poly_express[exps] * (t ** exps[0]) * ((1 - t) ** exps[1]) for exps in poly_express])


# Function that estimates the LT threshold from a given polynomial expression
def LTthresold_from_polyexpress(poly_express):
    # New approach based on inverting the polynomial (given that it is always monotone) - Ste 24/09/21
    temppoly = np.poly1d([0])

    for exp in poly_express:
        temppoly += poly_express[exp] * (np.poly1d([1, 0]) ** exp[0]) * (np.poly1d([-1, 1]) ** exp[1])

    all_roots = (temppoly - np.poly1d([1, 0])).roots
    real_roots = all_roots[np.isreal(all_roots)]
    real_roots = all_roots[(0.501 < all_roots) & (all_roots < 0.99)]
    if len(real_roots) == 0:
        return 1.
    else:
        return np.real(real_roots[0])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    in_qubit = 0

    ### random graph
    # graph = gen_random_connected_graph(6)
    # gstate = GraphState(graph)

    ### ring graph
    graph = gen_ring_graph(5)
    gstate = GraphState(graph)

    ### 5-qubit Shor-code graph
    # graph = nx.Graph()
    # nodes = list(range(6))
    # graph.add_nodes_from(nodes)
    # graph.add_edges_from(product([0], nodes[1:]))
    # graph.add_edges_from([(nodes[i], nodes[i + 1]) for i in range(1, 5)] + [(5, 1)])
    # gstate = GraphState(graph)

    ### Cascaded 3-rings
    # graph = nx.Graph()
    # nodes = list(range(7))
    # graph.add_nodes_from(nodes)
    # graph.add_edges_from([(0, 2), (0, 1), (1, 2), (1, 6), (1, 5), (5, 6), (2, 3), (2, 4), (3, 4)])
    # gstate = GraphState(graph)


    ## get list of possible measurements to encode & decode the state
    poss_stabs_list = get_possible_stabs_meas(gstate, in_qubit)

    ##############################################################################
    ################################### SINGLE TEST ##############################
    ##############################################################################

    # define channel transmission
    # transmission = 0.8
    # decoding_succ, decoding_meas = MC_decoding(poss_stabs_list, transmission, in_qubit, printing=False, provide_measures=True)
    #
    # ## see if we succeded or failed
    # if decoding_succ:
    #     print("Succeded :)")
    #     print("Meas. qubits :", decoding_meas[0])
    #     print("Lost qubits :", decoding_meas[1])
    # else:
    #     print("Failed :(")

    ##############################################################################
    ######   ESTIMATE ANALYTICAL SUCCESS PROBABILITY FOR GRAPH  ##################
    ##############################################################################

    MC_samples_meas = 1000

    transm = 0.82

    ### define channel transmission
    # sampl_func = lambda t: transm   # for constant transmission value
    sampl_func = lambda t: np.random.normal(transm, 0.06)  # for Gaussianly distribured values

    polynomial_expression = succprob_poly_withMC(graph, transm, MC_samples_meas, sampl_func, in_qubit, printing=False)
    print('polynomial_expression:', polynomial_expression)

    ## calculate threshold according to the analytical success probability estimate
    LT_threshold = LTthresold_from_polyexpress(polynomial_expression)
    print('LT_threshold:', LT_threshold)

    ### estimate succ probability from Monte-Carlo
    # eff_list_num = 11
    # MC_sims = 1000
    #
    # MCsucc_prob_list = trasmission_scan_MCestimate(poss_stabs_list, eff_list_num, MC_sims, in_qubit)
    # eff_list = np.linspace(0, 1, eff_list_num)

    ### plot all
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    t_list = np.linspace(0, 1, 100)
    # plt.plot(eff_list, MCsucc_prob_list, 'r.', label='MC estim.')
    plt.plot(t_list, probsucc_poly_fromexpress(t_list, polynomial_expression), c='blue', label='Analyt.')

    plt.plot(t_list, t_list, 'k:', label='Direct')
    plt.legend()
    plt.show()
