from copy import copy
import qecc as q

CliffordGateList = ['I', 'H', 'S']

############################
##### LOCAL CLIFFORDS ######
############################


def single_local_clifford(cgate, pauli, qb_idx):
    """
    Rules for operating an individual Clifford gate on a single Pauli matrix
    """
    if cgate not in CliffordGateList:
        raise ValueError("Clifford gate needs to be in:", CliffordGateList)
    if not isinstance(pauli, q.Pauli):
        raise ValueError("Pauli operator needs to be a qecc.Pauli object.")

    pauli_op = pauli.op[qb_idx]
    new_op = list(pauli.op)
    new_phase = pauli.ph

    if cgate == 'H':
        if pauli_op == 'X':
            new_op[qb_idx] = 'Z'
        if pauli_op == 'Z':
            new_op[qb_idx] = 'X'
        if pauli_op == 'Y':
            new_phase = (new_phase + 2) % 4

    if cgate == "S":
        if pauli_op == 'X':
            new_op[qb_idx] = 'Y'
        if pauli_op == 'Y':
            new_op[qb_idx] = 'X'
            new_phase = (new_phase + 2) % 4
    new_pauli = q.Pauli("".join(new_op), phase=new_phase)
    return new_pauli


def local_cliffords(cgates, pauli):
    """
    Operates multiple Clifford gates on a multi-qubit Pauli group element
    """

    if not isinstance(pauli, q.Pauli):
        raise ValueError("Pauli operator needs to be a qecc.Pauli object.")

    new_pauli = copy(pauli)
    for qb_idx in range(len(new_pauli)):
        local_cgates = cgates[qb_idx]
        local_cgates = local_cgates[::-1]  # rearrange them so that last operations in the list are applied first
        for this_cgate in local_cgates:
            new_pauli = single_local_clifford(this_cgate, new_pauli, qb_idx)
    return new_pauli


def local_cliffords_on_stab_list(local_cgates, stab_list):
    """
    Operates Clifford gates on a list of multi-qubit Pauli group elements
    """
    return [local_cliffords(local_cgates, this_stab) for this_stab in stab_list]


###############################
##### PAULI MEASUREMENTS ######
###############################

def pauli_measurement_on_stab_list(pauli_measurement, stab_list, outcome=0):
    """
    Updates a list of stabilizers after a measurement is performed
    """
    if not (outcome == 0 or outcome == 1):
        raise ValueError("Measurement outcome value can be only 0 or 1.")

    if not isinstance(pauli_measurement, q.Pauli):
        pauli_meas = q.Pauli(pauli_measurement)
    else:
        pauli_meas = pauli_measurement

    for check_stab in stab_list:
        if not isinstance(check_stab, q.Pauli):
            raise ValueError("All input stabilizers need to be qecc.Pauli objects.")

    new_pauli_list = copy(stab_list)
    for stab_ix, this_stab in enumerate(stab_list):
        if q.com(pauli_meas, this_stab) == 1:
            anticomm_pauli = this_stab
            anticomm_pauli.ph = (anticomm_pauli.ph + 2*outcome) % 4
            new_pauli_list[stab_ix].op = pauli_meas.op
            new_pauli_list[stab_ix].ph = (pauli_meas.ph + 2*outcome) % 4
            break

    for other_stabs_ix in range(stab_ix, len(stab_list)):
        this_stab = stab_list[other_stabs_ix]
        if q.com(pauli_meas, this_stab) == 1:
            new_pauli_list[other_stabs_ix] = anticomm_pauli * this_stab

    return new_pauli_list


def pauli_measurement_on_code(pauli_measurement, stab_gens_list, logical_ops_list, outcome=0):
    """
    Updates stabilizers and logical operators after a measurement is performed
    """
    if not (outcome == 0 or outcome == 1):
        print(outcome)
        raise ValueError("Measurement outcome value can be only 0 or 1.")

    if not isinstance(pauli_measurement, q.Pauli):
        pauli_meas = q.Pauli(pauli_measurement)
    else:
        pauli_meas = pauli_measurement

    for check_stab in stab_gens_list:
        if not isinstance(check_stab, q.Pauli):
            raise ValueError("All input stabilizers need to be qecc.Pauli objects.")

    for log_op_zs in logical_ops_list[0]:
        if not isinstance(log_op_zs, q.Pauli):
            raise ValueError("All logical Z operators need to be qecc.Pauli objects.")

    for log_op_xs in logical_ops_list[1]:
        if not isinstance(log_op_xs, q.Pauli):
            raise ValueError("All logical X operators need to be qecc.Pauli objects.")



    check_all_comm = True
    new_stab_gens_list = copy(stab_gens_list)
    for stab_ix, this_stab in enumerate(stab_gens_list):
        if q.com(pauli_meas, this_stab) == 1:
            check_all_comm = False
            anticomm_pauli = this_stab
            anticomm_pauli.ph = (anticomm_pauli.ph + 2*outcome) % 4
            new_stab_gens_list[stab_ix].op = pauli_meas.op
            new_stab_gens_list[stab_ix].ph = (pauli_meas.ph + 2*outcome) % 4
            break

    for other_stabs_ix in range(stab_ix, len(stab_gens_list)):
        this_stab = stab_gens_list[other_stabs_ix]
        if q.com(pauli_meas, this_stab) == 1:
            new_stab_gens_list[other_stabs_ix] = anticomm_pauli * this_stab

    new_Z_log_ops_list = copy(logical_ops_list[0])
    new_X_log_ops_list = copy(logical_ops_list[1])
    # if measurement anticommutes with some stabilizer generator, update logical operators
    if not check_all_comm:
        for logop_ix, this_logop in enumerate(logical_ops_list[0]):
            if q.com(pauli_meas, this_logop) == 1:
                new_Z_log_ops_list[logop_ix] = anticomm_pauli * this_logop
        for logop_ix, this_logop in enumerate(logical_ops_list[1]):
            if q.com(pauli_meas, this_logop) == 1:
                new_X_log_ops_list[logop_ix] = anticomm_pauli * this_logop

        return new_stab_gens_list, [new_Z_log_ops_list, new_X_log_ops_list]

    # if measurement commutes with all stabilizer generators, check that it commutes with all logical operators (i.e.
    # the measurement is a stabilizer), and if so return the same logical operators. If instead it anticommutes with
    # some logical operator, it means we measured a logical operator, and thus we collapsed the whole logical state to
    # an eigenstate of a logical operator. In this case both te stabilizer generators and logical operators would remain
    # untouched, but here we also raise an error because teleportation can no longer be performed.
    else:
        for logop_ix, this_logop in enumerate(logical_ops_list[0]):
            if (q.com(pauli_meas, logical_ops_list[0][logop_ix]) == 1) or \
                    (q.com(pauli_meas, logical_ops_list[1][logop_ix]) == 1):
                raise ValueError("Performed measurement in a logical operator: logical state is collapsed and "
                                 "teleportation can no longer be performed.")
        return new_stab_gens_list, [new_Z_log_ops_list, new_X_log_ops_list]



