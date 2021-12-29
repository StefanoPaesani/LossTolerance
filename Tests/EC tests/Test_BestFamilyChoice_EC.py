import qecc as q
import numpy as np
from itertools import product, combinations, chain
from copy import copy


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
    # print('\n\n')
    # print('Initial :', temp_stabs_list)
    for ix, this_op in enumerate(measurement):
        if this_op == 'I':
            temp_stabs_list = filter_stabs_input_op_only(temp_stabs_list, this_op, ix)
        else:
            temp_stabs_list = filter_stabs_input_op_compatible(temp_stabs_list, this_op, ix)
        # print('At step ', ix, ':', temp_stabs_list)
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
    all_meas = find_all_meas_strats(stabs_list, input_pauli_op, input_qubit)
    # print('all_meas', all_meas)
    temp_EC_families = []
    valid_meas = []
    for this_meas in all_meas:
        # comp_ind_meas_stabs = filter_stabs_measurement_compatible(stabs_list, this_meas)
        meas_comp_stabs = filter_stabs_measurement_compatible(stabs_list, this_meas)
        comp_ind_meas_stabs = filter_stabs_input_op_only(meas_comp_stabs, input_pauli_op, input_qubit)
        if comp_ind_meas_stabs:
            ### select as logical operator the indirect measurement stabilizer with minimal weight
            log_op_stab = min(comp_ind_meas_stabs, key=lambda x: sum(map(lambda y: y != 'I', x)))
            # log_op_stab = max(comp_ind_meas_stabs, key=lambda x: sum(map(lambda y: y != 'I', x)))
            valid_meas.append(this_meas)
            comp_syndr_stabs = filter_stabs_input_op_only(meas_comp_stabs, 'I', input_qubit)
            temp_EC_families.append((log_op_stab, comp_syndr_stabs))
    return dict(zip(valid_meas, temp_EC_families))


######### Generating possible error sets

def positions_k_errors(qbt_indices, k):
    return combinations(qbt_indices, k)


def positions_max_errors(qbt_indices, max_k):
    return list(chain.from_iterable([positions_k_errors(qbt_indices, k) for k in range(max_k + 1)]))


######### Indirect measure with error

# calculates if a stabilizer measurement outcome is flipped for a given configuration of physical errors
def get_stab_flip_from_error_conf(stab, error_conf):
    temp_val = 0
    for e_ix in error_conf:
        if stab[e_ix] != 'I':
            temp_val += 1
    return temp_val % 2


# calculates the syndroms for a list of stabilizers for a given configuration of physical errors
def get_syndroms_from_error_conf(stab_list, error_conf):
    return tuple([get_stab_flip_from_error_conf(stab, error_conf) for stab in stab_list])


# calculate the probability structure for a given error configuration
# the probability strucure is a tuple of the form:
# (num_correct_pauliX, num_wrong_pauliX, num_correct_pauliY, num_wrong_pauliY, num_correct_pauliZ, num_wrong_pauliZ)
# the probability associated to this structure is sum_P (1-p_P)^(num_wrong_pauliP) * p_P^(num_wrong_pauliP)
# with P in {X,Y,Z} and p_P the error probability for measuring the Pauli operator P.
def get_prob_structure_for_error_conf(measurement, error_conf, input_qubit):
    temp_conf = [0, 0, 0, 0, 0, 0]
    qbts_without_err = [i for i, _ in enumerate(measurement) if ((i not in error_conf) and (i != input_qubit))]
    for ix in qbts_without_err:
        if measurement[ix] == 'X':
            temp_conf[0] += 1
        elif measurement[ix] == 'Y':
            temp_conf[2] += 1
        elif measurement[ix] == 'Z':
            temp_conf[4] += 1
    for ix in error_conf:
        if measurement[ix] == 'X':
            temp_conf[1] += 1
        elif measurement[ix] == 'Y':
            temp_conf[3] += 1
        elif measurement[ix] == 'Z':
            temp_conf[5] += 1
    return tuple(temp_conf)


# function to calculate the probability associated to a dictionary with the coefficients associate to different
# probability structures
def calculate_prob_from_struct_coeff_dict(err_prob_structs_coeffs_dict, err_prob_X, err_prob_Y=None,
                                          err_prob_Z=None):
    if err_prob_Y is None:
        err_prob_Y = err_prob_X
    if err_prob_Z is None:
        err_prob_Z = err_prob_X
    temp_prob = 0
    for err_prob_struct in err_prob_structs_coeffs_dict:
        struct_coeff = err_prob_structs_coeffs_dict[err_prob_struct]
        if struct_coeff > 0:
            temp_prob += struct_coeff * ((1 - err_prob_X) ** err_prob_struct[0]) * (err_prob_X ** err_prob_struct[1]) * \
                         ((1 - err_prob_Y) ** err_prob_struct[2]) * (err_prob_Y ** err_prob_struct[3]) * \
                         ((1 - err_prob_Z) ** err_prob_struct[4]) * (err_prob_Z ** err_prob_struct[5])
    return temp_prob


# function to calculate that gets the dictionary with the coefficients associate to different probability structures
# when summing two of such probabilities
def sum_prob_struct_coeffs_dicts(prob_dict1, prob_dict2):
    temp_struct_coeffs_dict = copy(prob_dict1)
    for prob_struct in prob_dict2:
        if prob_struct in temp_struct_coeffs_dict:
            temp_struct_coeffs_dict[prob_struct] += prob_dict2[prob_struct]
        else:
            temp_struct_coeffs_dict[prob_struct] = prob_dict2[prob_struct]
    return temp_struct_coeffs_dict


######### EC family decoder
# calculates the lookup dictionary
# gets: syndroms, whether the logical operator gets flipped, and probability structure
# for all possible errors and a given measurement
# returns a dictionary of the form:
# {syndrome: {0: prob_struct_coeffs_dict_for_not_flipped, 1: prob_struct_coeffs_dict_for_flipped}, ...}
def calculate_syndromes_dictionary(measurement, log_op, syndr_stabs_list, input_qubit, max_error_num=None):
    qbts_indices = [i for i, _ in enumerate(measurement) if i != in_qubit]
    if max_error_num is None:
        max_error_num = len(measurement) - 1
    error_confs_list = positions_max_errors(qbts_indices, max_error_num)

    temp_syndr_dict = dict()
    for error_conf in error_confs_list:
        this_syndrom = get_syndroms_from_error_conf(syndr_stabs_list, error_conf)
        this_logop_flip = get_stab_flip_from_error_conf(log_op, error_conf)
        this_prob_struct = get_prob_structure_for_error_conf(measurement, error_conf, input_qubit)
        # print('\nNEW STEP')
        # print(temp_syndr_dict)
        # print('this_syndrom',this_syndrom,'this_logop_flip',this_logop_flip, 'this_prob_struct',this_prob_struct)
        if this_syndrom not in temp_syndr_dict:
            # print('The syndrom is new')
            temp_syndr_dict[this_syndrom] = {this_logop_flip: {this_prob_struct: 1.},
                                             1 - this_logop_flip: dict()}
        else:
            # print('Syndrom already encountered')
            old_prob_struct_coeffs = temp_syndr_dict[this_syndrom][this_logop_flip]
            # print('prob struct coeffs this syndr and logop values:', old_prob_struct_coeffs)
            temp_syndr_dict[this_syndrom][this_logop_flip] = sum_prob_struct_coeffs_dicts(
                old_prob_struct_coeffs, {this_prob_struct: 1.})
        # print(temp_syndr_dict)
    return temp_syndr_dict


def log_op_error_prob_from_lookup_dict(lookup_dict, err_prob_X, err_prob_Y=None, err_prob_Z=None,
                                       num_prob_thresh=1e-12):
    # print(lookup_dict)
    succ_probs_syndromes_raw = dict()
    for syndr in lookup_dict:
        # probability to get a flipped indirect measurement outcome for a given stabilizer
        succ_probs_syndromes_raw[syndr] = calculate_prob_from_struct_coeff_dict(lookup_dict[syndr][1],
                                                                                err_prob_X, err_prob_Y, err_prob_Z)
    # print(succ_probs_syndromes_raw)
    ## build all syndrome probabilities:
    syndr_probs = dict()
    for syndr in lookup_dict:
        syndr_probs[syndr] = calculate_prob_from_struct_coeff_dict(
            sum_prob_struct_coeffs_dicts(lookup_dict[syndr][0], lookup_dict[syndr][1]),
            err_prob_X, err_prob_Y, err_prob_Z)
    # print(syndr_probs)
    ### Normalize meas. success prob for given syndrome, and perform error correction
    succ_probs_syndromes = dict()
    for syndr in succ_probs_syndromes_raw:
        if syndr_probs[syndr] > num_prob_thresh:
            temp_prob = succ_probs_syndromes_raw[syndr] / syndr_probs[syndr]
        else:
            temp_prob = 0.
        # for error correction, majority voting is assumed here
        succ_probs_syndromes[syndr] = min(temp_prob, 1 - temp_prob)
    ### Get final probability
    return sum([succ_probs_syndromes[syndr] * syndr_probs[syndr] for syndr in succ_probs_syndromes])


################################################
###################  Tests  ####################
################################################

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

    # ### star graph
    # graph = gen_star_graph(4)
    # gstate = GraphState(graph)

    # ## ring graph
    graph = gen_ring_graph(5)
    gstate = GraphState(graph)

    ### line graph L543021 with no loss-tolerance
    # graph_nodes = list(range(6))
    # graph_edges = [(5, 4), (4, 3), (3, 0), (0, 2), (2, 1)]
    # gstate = graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    ### Graph equivalent to L543021 with loss-tolerance
    # graph_nodes = list(range(6))
    # graph_edges = [(5, 4), (4, 3), (4, 0), (4, 2), (0, 3), (3, 2), (2, 1)]
    # gstate = graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    ### Generate random graph
    # graph = gen_random_connected_graph(6)
    # gstate = GraphState(graph)

    #########################################################################################
    ################################### SINGLE TEST - DECODING ##############################
    #########################################################################################
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    ind_meas_op = 'Y'

    ### gets all stabs for a graph
    all_stabs = [this_op.op for this_op in q.from_generators(gstate.stab_gens)]
    all_stabs.remove('I' * len(all_stabs[0]))
    print(all_stabs)

    ### filter stabilizers to get only those for indirect measurement
    # filtered_stabs_ind_meas = filter_stabs_input_op_only(all_stabs, ind_meas_op, in_qubit)
    filtered_stabs_ind_meas = filter_stabs_input_op_compatible(all_stabs, ind_meas_op, in_qubit)
    print(filtered_stabs_ind_meas)

    ### identify all possible indirect measurement families
    meas_families = find_ind_meas_EC_families(filtered_stabs_ind_meas, ind_meas_op, in_qubit)
    print(meas_families)

    ## select best family
    best_meas = max(meas_families, key=lambda x: len(meas_families[x][1]))
    best_fam = meas_families[best_meas]
    print()
    print(best_meas, best_fam)

    ## run decoder:  build syndromes lookup table
    syndromes_probs_dict = calculate_syndromes_dictionary(best_meas, best_fam[0], best_fam[1], in_qubit,
                                                          max_error_num=None)
    print(syndromes_probs_dict)

    ### Get final probability for best graph
    print()
    error_prob = 0.1
    final_prob_best = log_op_error_prob_from_lookup_dict(syndromes_probs_dict, error_prob)
    print('Final probability best family:', final_prob_best)

    ### Compare with all oterh families
    log_error_rate_list = []
    for this_meas in meas_families:
        this_fam = meas_families[this_meas]
        syndromes_probs_dict = calculate_syndromes_dictionary(this_meas, this_fam[0], this_fam[1], in_qubit,
                                                              max_error_num=None)
        log_error_rate = log_op_error_prob_from_lookup_dict(syndromes_probs_dict, error_prob)
        log_error_rate_list.append(log_error_rate)
    print(log_error_rate_list)
    is_best_really_best_array = np.array(log_error_rate_list) >= (final_prob_best - 1e-8)
    print(is_best_really_best_array)
    print(np.all(is_best_really_best_array))



    # ##################################### PLots
    #
    # error_vals = np.linspace(0, 0.4, 20)
    # log_err_list = [log_op_error_prob_from_lookup_dict(syndromes_probs_dict, x) for x in error_vals]
    # plt.plot(error_vals, error_vals, label="direct")
    # plt.plot(error_vals, log_err_list, label='decoded')
    # plt.legend()
    # plt.show()
