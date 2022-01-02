import qecc as q
import numpy as np
from itertools import product, combinations, chain
from copy import copy
from MiscFunctions.PauliOpsFunctions import single_qubit_commute, count_target_pauli_in_stabs
from MiscFunctions.StabilizersFilteringFunctions import filter_stabs_input_op_compatible, filter_stabs_input_op_only, \
    filter_stabs_measurement_compatible, filter_stabs_compatible_qubits_ops, filter_stabs_given_qubits_ops


######### Finding EC families

#### Arbitrary indirect measurements

# find possible measurement strategies that could lead to some measurement in arbitrary indirect basis
# (pathfinding-condition-style)
def find_all_ind_meas_strats(stabs_list, input_qubit, pref_pauli='Z'):
    num_qubits = len(stabs_list[0])
    all_meas = {}
    for stab_ix1 in range((2 ** num_qubits) - 1):
        for stab_ix2 in range(stab_ix1 + 1, (2 ** num_qubits) - 1):
            stab1 = stabs_list[stab_ix1]
            stab2 = stabs_list[stab_ix2]
            ## checks which qubits have anticommuting Paulis
            anticomm_qbts = [qbt for qbt in range(num_qubits) if single_qubit_commute(stab1, stab2, qbt)]
            # print()
            # print(stab_ix1, stab1, stab_ix2, stab2, anticomm_qbts)
            ## checks that there are exactly two qubits with anticommuting Paulis: the input and an output
            if len(anticomm_qbts) == 2 and input_qubit in anticomm_qbts:
                compatible_stabs = filter_stabs_given_qubits_ops(stabs_list,
                                                                 {anticomm_qbts[0]: 'I', anticomm_qbts[1]: 'I'})
                non_inout_qubits = [ix for ix in range(num_qubits) if ix not in anticomm_qbts]
                non_inout_paulis = [stab1[ix] if stab1[ix] != 'I' else stab2[ix] for ix in non_inout_qubits]
                compatible_stabs = filter_stabs_compatible_qubits_ops(compatible_stabs,
                                                                      dict(zip(non_inout_qubits, non_inout_paulis)))
                # print('good inout, compatible_stabs:', compatible_stabs)
                poss_ops_list = [['I'] if ix in anticomm_qbts else [] for ix in range(num_qubits)]
                free_qubits = []
                for ix in range(num_qubits):
                    if ix not in anticomm_qbts:
                        if (stab1[ix] != 'I' or stab2[ix] != 'I'):
                            poss_ops_list[ix] = [stab1[ix] if stab1[ix] != 'I' else stab2[ix]]
                        else:
                            free_qubits.append(ix)
                # print('free_qubits', free_qubits)
                for this_stab in compatible_stabs:
                    for ix, this_op in enumerate(this_stab):
                        if ix in free_qubits:
                            if this_op not in poss_ops_list[ix]:
                                poss_ops_list[ix].append(this_op)
                # print('poss_ops_list', poss_ops_list)
                this_strat_all_meas = (''.join(ops) for ops in product(*poss_ops_list))
                # this_strat_all_meas = list(this_strat_all_meas)
                # print(this_strat_all_meas)
                for this_meas in this_strat_all_meas:
                    meas_comp_stabs = filter_stabs_measurement_compatible(compatible_stabs, this_meas)
                    ## Uses these stabilizers for this measurement if this_meas is not already included
                    if this_meas not in all_meas:
                        all_meas[this_meas] = ((anticomm_qbts, (stab1, stab2)), meas_comp_stabs)
                    ## If this_meas was already included, updated its strategy to this stabs if they are more than the
                    ## previous case, or if they are the same number but contain more of the prefered Pauli operator.
                    else:
                        previous_stabs = all_meas[this_meas][1]
                        if (len(meas_comp_stabs) > len(previous_stabs)) or (
                                len(meas_comp_stabs) == len(previous_stabs) and count_target_pauli_in_stabs(
                            meas_comp_stabs, pref_pauli) > count_target_pauli_in_stabs(previous_stabs, pref_pauli)):
                            all_meas[this_meas] = ((anticomm_qbts, (stab1, stab2)), meas_comp_stabs)
    return all_meas


#### SIngle-operator Indirect Measurement


# find possible measurements that could lead to some inderect measurement
def find_single_op_ind_meas_strats(stabs_list, input_pauli_op, input_qubit):
    poss_ops_list = [[] if ix != input_qubit else [input_pauli_op] for ix in range(len(stabs_list[0]))]
    for this_stab in stabs_list:
        for ix, this_op in enumerate(this_stab):
            if ix != input_qubit:
                if this_op not in poss_ops_list[ix]:
                    poss_ops_list[ix].append(this_op)
    # print(poss_ops_list)
    return (''.join(ops) for ops in product(*poss_ops_list))


def find_single_op_ind_meas_EC_families(stabs_list, input_pauli_op, input_qubit):
    all_meas = find_single_op_ind_meas_strats(stabs_list, input_pauli_op, input_qubit)
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


# # function to calculate that gets the dictionary with the coefficients associate to different probability structures
# # when summing two of such probabilities
# def sum_prob_struct_coeffs_dicts(prob_dict1, prob_dict2):
#     temp_struct_coeffs_dict = copy(prob_dict1)
#     for prob_struct in prob_dict2:
#         if prob_struct in temp_struct_coeffs_dict:
#             temp_struct_coeffs_dict[prob_struct] += prob_dict2[prob_struct]
#         else:
#             temp_struct_coeffs_dict[prob_struct] = prob_dict2[prob_struct]
#     return temp_struct_coeffs_dict


# function to calculate that gets the dictionary with the coefficients associate to different probability structures
# when summing multiple of such probabilities
def sum_prob_struct_coeffs_dicts(prob_dict_list):
    temp_struct_coeffs_dict = copy(prob_dict_list[0])
    for temp_prob_dict in prob_dict_list[1:]:
        for prob_struct in temp_prob_dict:
            if prob_struct in temp_struct_coeffs_dict:
                temp_struct_coeffs_dict[prob_struct] += temp_prob_dict[prob_struct]
            else:
                temp_struct_coeffs_dict[prob_struct] = temp_prob_dict[prob_struct]
    return temp_struct_coeffs_dict


######### EC family decoder

# calculates the lookup dictionary
# gets: syndroms, whether the logical operator gets flipped, and probability structure
# for all possible errors and a given measurement.
# Returns a dictionary of the form:
# {syndrome: {0: prob_struct_coeffs_dict_for_not_flipped, 1: prob_struct_coeffs_dict_for_flipped}, ...}
# If include_direct_meas is True, direct measurement of the input qubit to determine its correct value is also assumed.
def calculate_syndromes_dictionary_single_ind_meas(measurement, log_op, syndr_stabs_list, input_qubit,
                                                   include_direct_meas=False, max_error_num=None):
    if include_direct_meas:
        qbts_indices = [i for i, _ in enumerate(measurement)]
        stabs_for_decoder = copy(syndr_stabs_list)
        stabs_for_decoder.append(log_op)
        # print(stabs_for_decoder)
    else:
        qbts_indices = [i for i, _ in enumerate(measurement) if i != input_qubit]
        stabs_for_decoder = syndr_stabs_list
    if max_error_num is None:
        max_error_num = len(measurement) - 1
    error_confs_list = positions_max_errors(qbts_indices, max_error_num)
    temp_syndr_dict = dict()
    for error_conf in error_confs_list:
        this_syndrom = get_syndroms_from_error_conf(stabs_for_decoder, error_conf)
        if include_direct_meas:
            this_logop_flip = 1 if input_qubit in error_conf else 0
        else:
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
                [old_prob_struct_coeffs, {this_prob_struct: 1.}])
        # print(temp_syndr_dict)
    return temp_syndr_dict


# calculates the lookup dictionary for a Teleportation measurement
# gets: syndroms, whether the logical operator gets flipped, and probability structure
# for all possible errors and a given measurement.
# Returns a dictionary of the form:
# {syndrome: {0: prob_struct_coeffs_dict_for_not_flipped, 1: prob_struct_coeffs_dict_for_flipped}, ...}
def calculate_syndromes_dictionary_teleport(measurement, log_op1, log_op2, syndr_stabs_list, inout_qubits,
                                            max_error_num=None):
    input_qubit = inout_qubits[0]
    qbts_indices = [i for i, _ in enumerate(measurement) if i not in inout_qubits]
    # print('qbts_indices', qbts_indices)
    stabs_for_decoder = syndr_stabs_list
    if max_error_num is None:
        max_error_num = len(measurement) - 1
    error_confs_list = positions_max_errors(qbts_indices, max_error_num)
    temp_syndr_dict = dict()
    for error_conf in error_confs_list:
        this_syndrom = get_syndroms_from_error_conf(stabs_for_decoder, error_conf)
        this_logop1_flip = get_stab_flip_from_error_conf(log_op1, error_conf)
        this_logop2_flip = get_stab_flip_from_error_conf(log_op2, error_conf)
        this_tele_flips = (this_logop1_flip, this_logop2_flip)
        this_prob_struct = get_prob_structure_for_error_conf(measurement, error_conf, input_qubit)
        # print('\nNEW STEP')
        # print('error_conf', error_conf)
        # print(temp_syndr_dict)
        # print('this_syndrom', this_syndrom, 'this_tele_flips', this_tele_flips, 'this_prob_struct', this_prob_struct)
        if this_syndrom not in temp_syndr_dict:
            # print('The syndrom is new')
            temp_syndr_dict[this_syndrom] = {(0, 0): dict(), (0, 1): dict(), (1, 0): dict(), (1, 1): dict()}
            temp_syndr_dict[this_syndrom][this_tele_flips] = {this_prob_struct: 1.}
        else:
            # print('Syndrom already encountered')
            old_prob_struct_coeffs = temp_syndr_dict[this_syndrom][this_tele_flips]
            # print('prob struct coeffs this syndr and logop values:', old_prob_struct_coeffs)
            temp_syndr_dict[this_syndrom][this_tele_flips] = sum_prob_struct_coeffs_dicts(
                [old_prob_struct_coeffs, {this_prob_struct: 1.}])
        # print(temp_syndr_dict)
    return temp_syndr_dict


#### Full Decoder


#### Indirect Measurement Decoder

### Full decoder for indirect measurement that calculates the lookup table for a given graph.
# If include_direct_meas is True, direct measurement of the input qubit to determine its correct value is also assumed.
def ind_meas_EC_decoder(graph_state, ind_meas_op, in_qubit, include_direct_meas=False, max_error_num=None):
    if ind_meas_op not in ('X', 'Y', 'Z'):
        raise ValueError('The indirect measument Pauli operator needs to be one of (X, Y, Z)')
    ### gets all stabs for a graph
    all_stabs = [this_op.op for this_op in q.from_generators(graph_state.stab_gens)]
    all_stabs.remove('I' * len(all_stabs[0]))
    ### filter stabilizers to get only those for indirect measurement
    filtered_stabs_ind_meas = filter_stabs_input_op_compatible(all_stabs, ind_meas_op, in_qubit)
    ### identify all possible indirect measurement families
    meas_families = find_single_op_ind_meas_EC_families(filtered_stabs_ind_meas, ind_meas_op, in_qubit)
    # print(meas_families)
    ## select best family
    best_meas = max(meas_families, key=lambda x: len(meas_families[x][1]))
    best_fam = meas_families[best_meas]
    ## run decoder:  build syndromes lookup table and return it
    return calculate_syndromes_dictionary_single_ind_meas(best_meas, best_fam[0], best_fam[1], in_qubit,
                                                          include_direct_meas=include_direct_meas,
                                                          max_error_num=max_error_num)


#### Teleportation Decoder

### Full decoder for teleportation that calculates the lookup table for a given graph.
def teleportation_EC_decoder(graph_state, in_qubit, pref_pauli='Z', max_error_num=None):
    num_qbts = len(graph_state)
    ### gets all stabs for a graph
    all_stabs = [this_op.op for this_op in q.from_generators(graph_state.stab_gens)]
    all_stabs.remove('I' * len(all_stabs[0]))
    ### identify all possible teleportation measurement families
    meas_families = find_all_ind_meas_strats(all_stabs, in_qubit, pref_pauli=pref_pauli)
    # print('All families:')
    # print(meas_families, len(meas_families))
    ## select best family
    if meas_families:
        best_meas = max(meas_families, key=lambda x: len(meas_families[x][1]) + (
            0 if len(meas_families[x][1]) == 0 else count_target_pauli_in_stabs(meas_families[x][1], pref_pauli) / (
                    num_qbts * len(meas_families[x][1]))))
        best_fam = meas_families[best_meas]
        # print('best_meas', best_meas)
        # print('best_fam', best_fam)
        return calculate_syndromes_dictionary_teleport(best_meas, best_fam[0][1][0], best_fam[0][1][1], best_fam[1],
                                                       best_fam[0][0], max_error_num=max_error_num)
    else:
        return False


### Calculates the error probability given the lookup dictionary output from the single ind. measurement decoder
def log_op_error_prob_from_lookup_dict(lookup_dict, err_prob_X, err_prob_Y=None, err_prob_Z=None,
                                       num_prob_thresh=1e-12):
    if lookup_dict:
        if err_prob_Y is None:
            err_prob_Y = err_prob_X
        if err_prob_Z is None:
            err_prob_Z = err_prob_X
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
                sum_prob_struct_coeffs_dicts([lookup_dict[syndr][0], lookup_dict[syndr][1]]),
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
    else:
        return 1


### Calculates the error probability given the lookup dictionary output from the teleportation decoder
def teleport_error_prob_from_lookup_dict(lookup_dict, err_prob_X, err_prob_Y=None, err_prob_Z=None,
                                         num_prob_thresh=1e-12):
    if lookup_dict:
        if err_prob_Y is None:
            err_prob_Y = err_prob_X
        if err_prob_Z is None:
            err_prob_Z = err_prob_X

        # print(succ_probs_syndromes_raw)
        ## build all syndrome probabilities:
        syndr_probs = dict()
        for syndr in lookup_dict:
            syndr_probs[syndr] = calculate_prob_from_struct_coeff_dict(
                sum_prob_struct_coeffs_dicts(list(lookup_dict[syndr].values())),
                err_prob_X, err_prob_Y, err_prob_Z)
        # print(syndr_probs)
        ### Normalize meas. success prob for given syndrome, and perform error correction
        succ_probs_syndromes = dict()
        for syndr in lookup_dict:
            if syndr_probs[syndr] > num_prob_thresh:
                # for error correction, majority voting is assumed here
                succ_probs_syndromes[syndr] = 1 - max(
                    [calculate_prob_from_struct_coeff_dict(lookup_dict[syndr][log_ops_errors],
                                                           err_prob_X, err_prob_Y,
                                                           err_prob_Z) / syndr_probs[syndr]
                     for log_ops_errors in lookup_dict[syndr]])
            else:
                succ_probs_syndromes[syndr] = 0.
        ### Get final probability
        return sum([succ_probs_syndromes[syndr] * syndr_probs[syndr] for syndr in lookup_dict])
    else:
        return 1


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
    # graph = gen_star_graph(3)
    # gstate = GraphState(graph)

    # ## ring graph
    # graph = gen_ring_graph(5)
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
    graph = gen_random_connected_graph(6)
    gstate = GraphState(graph)

    ###############################################################################################
    ################################### PAULI MEASUREMENT - DECODING ##############################
    ###############################################################################################
    # gstate.image(input_qubits=[in_qubit])
    # plt.show()
    #
    # syndromes_probs_dict_X = ind_meas_EC_decoder(gstate, 'X', in_qubit, include_direct_meas=False, max_error_num=None)
    # syndromes_probs_dict_Y = ind_meas_EC_decoder(gstate, 'Y', in_qubit, include_direct_meas=False, max_error_num=None)
    # syndromes_probs_dict_Z = ind_meas_EC_decoder(gstate, 'Z', in_qubit, include_direct_meas=False, max_error_num=None)
    # syndromes_probs_dict_Z_withinput = ind_meas_EC_decoder(gstate, 'Z', in_qubit, include_direct_meas=True,
    #                                                        max_error_num=None)
    # print(syndromes_probs_dict_Z_withinput)
    #
    # ##################################### Plots
    #
    # error_vals = np.linspace(0, 1, 21)
    # log_err_list_X = [log_op_error_prob_from_lookup_dict(syndromes_probs_dict_X, x) for x in error_vals]
    # log_err_list_Y = [log_op_error_prob_from_lookup_dict(syndromes_probs_dict_Y, x) for x in error_vals]
    # log_err_list_Z = [log_op_error_prob_from_lookup_dict(syndromes_probs_dict_Z, x) for x in error_vals]
    # log_err_list_Z_withinput = [log_op_error_prob_from_lookup_dict(syndromes_probs_dict_Z_withinput, x) for x in
    #                             error_vals]
    #
    # plt.plot(error_vals, error_vals, 'k:', label='', )
    # plt.plot(error_vals, log_err_list_X, 'r', label='X', linewidth=3)
    # plt.plot(error_vals, log_err_list_Y, 'k', label='Y', linewidth=2, alpha=1)
    # plt.plot(error_vals, log_err_list_Z, 'b', label='Z', alpha=1)
    # plt.plot(error_vals, log_err_list_Z_withinput, 'b--', label='Z-with dir.', alpha=1)
    # plt.xlabel('Physical error probability')
    # plt.ylabel('Logical error probability')
    # plt.legend()
    # plt.show()

    #################################################################################################
    ################################### STATE TELEPORTATION - DECODING ##############################
    #################################################################################################
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    syndromes_probs_dict_full = teleportation_EC_decoder(gstate, in_qubit, pref_pauli='Z', max_error_num=None)
    print(syndromes_probs_dict_full)

    ##################################### Plots

    error_vals = np.linspace(0, 1, 21)
    log_err_list_tele = [teleport_error_prob_from_lookup_dict(syndromes_probs_dict_full, x) for x in error_vals]

    plt.plot(error_vals, error_vals, 'k:', label='', )
    plt.plot(error_vals, log_err_list_tele, 'r', label='Full decoder', linewidth=2)
    plt.xlabel('Physical error probability')
    plt.ylabel('Teleported state error probability')
    plt.legend()
    plt.show()
