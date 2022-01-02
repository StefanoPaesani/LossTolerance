# keeps only the stabilizers in which the input_qubit has a target pauli op. or 'I'
def filter_stabs_input_op_compatible(stabs_list, input_pauli_op, qbt_ix):
    return [this_stabs for this_stabs in stabs_list if (input_pauli_op == 'I' or
            (this_stabs[qbt_ix] in [input_pauli_op, 'I']))]


# keeps only the stabilizers in which the input_qubit has a target pauli operator
def filter_stabs_input_op_only(stabs_list, input_pauli_op, qbt_ix):
    return [this_stabs for this_stabs in stabs_list if
            (this_stabs[qbt_ix] == input_pauli_op)]


# keeps only the stabilizers all qubits operators are compatible with a measurement
def filter_stabs_measurement_compatible(stabs_list, measurement):
    temp_stabs_list = stabs_list
    for ix, this_op in enumerate(measurement):
        if this_op == 'I':
            temp_stabs_list = filter_stabs_input_op_only(temp_stabs_list, this_op, ix)
        else:
            temp_stabs_list = filter_stabs_input_op_compatible(temp_stabs_list, this_op, ix)
    return temp_stabs_list


# keeps only the stabilizers all qubits operators are compatible with a stablizer
def filter_stabs_indmeas_compatible(stabs_list, log_op, input_qubit):
    temp_stabs_list = stabs_list
    for ix, this_op in enumerate(log_op):
        if ix == input_qubit:
            temp_stabs_list = filter_stabs_input_op_only(temp_stabs_list, 'I', ix)
        else:
            temp_stabs_list = filter_stabs_input_op_compatible(temp_stabs_list, this_op, ix)
    return temp_stabs_list


# keeps only the stabilizers all qubits with compatible operators on given qubits, inputed as a dict {qbt_ix:
# operator, ..}
def filter_stabs_compatible_qubits_ops(stabs_list, qbt_ops_dict):
    temp_stabs_list = stabs_list
    for ix in qbt_ops_dict:
        this_op = qbt_ops_dict[ix]
        temp_stabs_list = filter_stabs_input_op_compatible(temp_stabs_list, this_op, ix)
    return temp_stabs_list


# keeps only the stabilizers all qubits with exact operators on given qubits, inputed as a dict {qbt_ix: operator, ..}
def filter_stabs_given_qubits_ops(stabs_list, qbt_ops_dict):
    temp_stabs_list = stabs_list
    for ix in qbt_ops_dict:
        this_op = qbt_ops_dict[ix]
        if this_op == 'I':
            temp_stabs_list = filter_stabs_input_op_only(temp_stabs_list, this_op, ix)
        else:
            temp_stabs_list = filter_stabs_input_op_compatible(temp_stabs_list, this_op, ix)
    return temp_stabs_list
