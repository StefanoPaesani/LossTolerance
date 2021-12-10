import qecc as q
import numpy as np
from itertools import product


# keeps only the stabilizers in which the input_qubit has a target pauli op. or 'I'
def filter_stabs_input_op_compatible(stabs_list, input_pauli_op, input_qubit):
    return [this_stabs for this_stabs in stabs_list if
            (this_stabs[input_qubit] in [input_pauli_op, 'I'])]


# keeps only the stabilizers in which the input_qubit has a target pauli operator
def filter_stabs_input_op_only(stabs_list, input_pauli_op, input_qubit):
    return [this_stabs for this_stabs in stabs_list if
            (this_stabs[input_qubit] == input_pauli_op)]


# keeps only the stabilizers all qubits operators are compatible with a measurement
def filter_stabs_measurement_compatible(stabs_list, measurement):
    temp_stabs_list = stabs_list
    for ix, this_op in enumerate(measurement):
        temp_stabs_list = filter_stabs_input_op_compatible(temp_stabs_list, this_op, ix)
    return temp_stabs_list


######### Finding EC families

# find possible measurements that could lead to some inderect measurement
def find_all_meas_strats(stabs_list, input_pauli_op, input_qubit):
    poss_ops_list = [[] if ix != input_qubit else [input_pauli_op] for ix in range(len(stabs_list[0]))]
    for this_stab in stabs_list:
        for ix, this_op in enumerate(this_stab):
            if ix != input_qubit:
                if this_op not in poss_ops_list[ix]:
                    poss_ops_list[ix].append(this_op)
    # print(poss_ops_list)
    return [''.join(ops) for ops in product(*poss_ops_list)]


def find_ind_meas_EC_families(stabs_list, input_pauli_op, input_qubit):
    ind_meas_stabs = filter_stabs_input_op_only(stabs_list, input_pauli_op, input_qubit)
    syndrome_stabs = [this_stab for this_stab in stabs_list if this_stab[input_qubit] != input_pauli_op]
    all_meas = find_all_meas_strats(stabs_list, input_pauli_op, input_qubit)
    temp_EC_families = []
    valid_meas = []
    for this_meas in all_meas:
        comp_ind_meas_stabs = filter_stabs_measurement_compatible(ind_meas_stabs, this_meas)
        if comp_ind_meas_stabs:
            comp_syndr_stabs = filter_stabs_measurement_compatible(syndrome_stabs, this_meas)
            valid_meas.append(this_meas)
            temp_EC_families.append(comp_ind_meas_stabs+comp_syndr_stabs)
    return dict(zip(valid_meas, temp_EC_families))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx

    from itertools import chain

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    # three graph
    # branching = [2, 2]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    ### fully connected graph
    # graph = gen_fullyconnected_graph(4)
    # gstate = GraphState(graph)

    ## star graph
    graph = gen_star_graph(4)
    gstate = GraphState(graph)

    ### ring graph
    # graph = gen_ring_graph(4)
    # gstate = GraphState(graph)

    ### line graph L543021 with no loss-tolerance
    # graph_nodes = list(range(6))
    # graph_edges = [(5, 4), (4, 3), (3, 0), (0, 2), (2, 1)]
    # gstate = graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    ### Graph equivalent to L543021 with loss-tolerance
    # graph_nodes = list(range(6))
    # graph_edges = [(5, 4), (4, 3), (4, 0), (4, 2), (0, 3), (3, 2), (2, 1)]
    # gstate = graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    #########################################################################################
    ################################### SINGLE TEST - DECODING ##############################
    #########################################################################################

    ind_meas_op = 'Z'

    ### gets all stabs for a graph
    all_stabs = [this_op.op for this_op in q.from_generators(gstate.stab_gens)][2:]
    print(all_stabs)

    ###
    filtered_stabs_compatible = filter_stabs_input_op_compatible(all_stabs, ind_meas_op, in_qubit)
    print(filtered_stabs_compatible)

    ###
    asd = find_ind_meas_EC_families(filtered_stabs_compatible, ind_meas_op, in_qubit)
    print(asd)

