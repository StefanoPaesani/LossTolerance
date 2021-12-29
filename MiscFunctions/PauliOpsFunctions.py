def single_qubit_commute(pauli1, pauli2, qbt):
    """
    Returns 0 if the operators on the qbt-th qubit of the two operators in the Pauli group commute,
    and 1 if they anticommute.
    """
    if pauli1[qbt] == 'I' or pauli2[qbt] == 'I' or pauli1[qbt] == pauli2[qbt]:
        return 0
    else:
        return 1


def count_target_pauli_in_stab(stab, target_pauli):
    return stab.count(target_pauli)


def count_target_pauli_in_stabs(stabs_list, target_pauli):
    return count_target_pauli_in_stab(''.join(stabs_list), target_pauli)
