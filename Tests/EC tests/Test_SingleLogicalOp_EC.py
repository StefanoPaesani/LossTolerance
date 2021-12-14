import qecc as q
import numpy as np
from itertools import product, combinations, chain


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
            valid_meas.append(this_meas)
            comp_syndr_stabs = filter_stabs_input_op_only(meas_comp_stabs, 'I', input_qubit)
            # temp_EC_families.append([ comp_ind_meas_stabs, comp_syndr_stabs])
            temp_EC_families.append(comp_ind_meas_stabs + comp_syndr_stabs)
            # temp_EC_families.append(comp_ind_meas_stabs)
    return dict(zip(valid_meas, temp_EC_families))


######### Generating possible error sets

def positions_k_errors(qbt_indices, k):
    return combinations(qbt_indices, k)


def positions_max_errors(qbt_indices, max_k):
    return list(chain.from_iterable([positions_k_errors(qbt_indices, k) for k in range(max_k + 1)]))


######### Indirect measure with error

# # calculates if the indirect measurement with a given stabilizer is faulty for a given configuration of physical errors
# def get_ind_outcome_stab_error(ind_meas_stab, error_conf):
#     temp_val = 0
#     for e_ix in error_conf:
#         if ind_meas_stab[e_ix] != 'I':
#             temp_val += 1
#     return temp_val % 2


# # calculates the probability of a faulty indirect measurement for a given configuration of physical errors
# # this assumes majority voting between the stabilizer to determine the ind. meas. outcome.
# # If the stabilizers are perfectly split, the outcome is decided randomly with 50% chance of succeeding
# def get_total_ind_outcome_error_prob(ind_meas_stab_list, error_conf):
#     stab_error_list = [get_ind_outcome_stab_error(ind_meas_stab, error_conf) for ind_meas_stab in ind_meas_stab_list]
#     maj_vot_threshold = len(stab_error_list) / 2.
#     num_wrong_stabs = sum(stab_error_list)
#     if num_wrong_stabs < maj_vot_threshold:
#         return 0
#     elif num_wrong_stabs > maj_vot_threshold:
#         return 1
#     else:
#         return 0.5


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


# def calculate_ind_meas_error_prob(err_prob_structs_coeffs_dict, err_prob_X, err_prob_Y=None, err_prob_Z=None):
#     if err_prob_Y is None:
#         err_prob_Y = err_prob_X
#     if err_prob_Z is None:
#         err_prob_Z = err_prob_X
#     temp_prob = 0
#     for err_prob_struct in err_prob_structs_coeffs_dict:
#         struct_coeff = err_prob_structs_coeffs_dict[err_prob_struct]
#         if struct_coeff>0:
#             temp_prob += struct_coeff * ((1-err_prob_X)**err_prob_struct[0]) * (err_prob_X**err_prob_struct[1]) * \
#                          ((1-err_prob_Y)**err_prob_struct[2]) * (err_prob_Y**err_prob_struct[3]) * \
#                          ((1-err_prob_Z)**err_prob_struct[4]) * (err_prob_Z**err_prob_struct[5])
#     return temp_prob


######### EC family decoder
# def decode_single_family(measurement, ind_meas_stabs, in_qubit, max_num_errors=None):
#     qbts_indices = [i for i, _ in enumerate(measurement) if i != in_qubit]
#     if max_num_errors is None:
#         max_num_errors = len(measurement) - 1
#     all_error_pos = positions_max_errors(qbts_indices, max_num_errors)
#     print(all_error_pos)
#     all_error_prob_struct_list = [get_prob_structure_for_error_conf(measurement, error_conf, in_qubit) for error_conf in
#                                   all_error_pos]
#     print(all_error_prob_struct_list)
#     incorrect_ind_meas_prob_list = [get_total_ind_outcome_error_prob(ind_meas_stabs, error_conf) for error_conf in
#                                     all_error_pos]
#     print(incorrect_ind_meas_prob_list)
#     probs_structs_keys = list(set(all_error_prob_struct_list))
#     print(probs_structs_keys)
#     err_prob_structs_coeffs_dict = dict(zip(probs_structs_keys, [0. for _ in probs_structs_keys]))
#     for ix, err_struct in enumerate(all_error_prob_struct_list):
#         err_prob_structs_coeffs_dict[err_struct] += incorrect_ind_meas_prob_list[ix]
#     return err_prob_structs_coeffs_dict


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
    # graph = gen_ring_graph(6)
    # gstate = GraphState(graph)

    ### line graph L543021 with no loss-tolerance
    # graph_nodes = list(range(6))
    # graph_edges = [(5, 4), (4, 3), (3, 0), (0, 2), (2, 1)]
    # gstate = graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    ### Graph equivalent to L543021 with loss-tolerance
    # graph_nodes = list(range(6))
    # graph_edges = [(5, 4), (4, 3), (4, 0), (4, 2), (0, 3), (3, 2), (2, 1)]
    # gstate = graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    ### Generate random graph
    graph = gen_random_connected_graph(7)
    gstate = GraphState(graph)

    #########################################################################################
    ################################### SINGLE TEST - DECODING ##############################
    #########################################################################################

    ind_meas_op = 'X'

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
    print()
    print(meas_families)

    ###### Tests to check if a single logical operator is present
    ###### (and all other ind. measurements are generated via multiplication with other stabiliers)
    single_log_op_meas_families = dict()
    single_log_op_meas_families_gen = dict()
    test_true_gen = dict()
    for this_meas in meas_families:
        stabs_fam = meas_families[this_meas]
        ind_meas_stabs = filter_stabs_input_op_only(stabs_fam, ind_meas_op, in_qubit)
        syndrome_stabs = [this_stab for this_stab in stabs_fam if this_stab not in ind_meas_stabs]
        single_log_op_fam = [ind_meas_stabs[0]] + syndrome_stabs
        single_log_op_meas_families[this_meas] = single_log_op_fam

        generated_fam = q.PauliList(single_log_op_fam).generated_group()
        generated_fam = list(set([stab.op for stab in generated_fam]))
        generated_fam.remove('I' * len(this_meas))
        single_log_op_meas_families_gen[this_meas] = generated_fam
        test_true_gen[this_meas] = (set(stabs_fam) == set(generated_fam))

    print(single_log_op_meas_families_gen)
    print(single_log_op_meas_families)
    print(test_true_gen)
    print(np.all(list(test_true_gen.values())))

    # these_stabs = meas_families['ZXZZX']
    # print(these_stabs)
    # these_pauli_coll = q.PauliList(these_stabs)
    # print(these_pauli_coll)
    # asd = these_pauli_coll.centralizer_gens(group_gens=these_pauli_coll)
    # asd = these_pauli_coll.centralizer_gens()
    # print(asd)

    ### select best family
    # best_meas = max(meas_families, key=lambda x: len(meas_families[x]))
    # best_fam = meas_families[best_meas]
    # print()
    # print(best_meas, best_fam)
    #
    # # get decoder running
    # decoded_prob_struct = decode_single_family(best_meas, best_fam, in_qubit, max_num_errors=None)
    # print()
    # print(decoded_prob_struct)
    #
    # ##################################### PLots
    #
    # error_vals = np.linspace(0, 1, 20)
    # plt.plot(error_vals, error_vals, label="direct")
    # plt.plot(error_vals, [calculate_ind_meas_error_prob(decoded_prob_struct, x) for x in error_vals], label='decoded')
    # plt.legend()
    # plt.show()
