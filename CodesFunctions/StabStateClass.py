import numpy as np

import qecc as q

from CodesFunctions.stabs_graph_equiv import stab_to_graph
from CodesFunctions.local_transformations import local_cliffords_on_stab_list, pauli_measurement_on_stab_list


class StabState(object):
    r"""
    Class representing a Graph state on :math:`n` qubits.

    :param stab_generators: iterator of stabilizer generators defining the state
    """

    def __init__(self, stab_generators):
        # stab_gens represent the n stabilizer generators defining the n-qubit stabilizer state

        # TODO: various checks, including that if the set in stab_gens are stabilizer but not necessarily
        #  the set of minimal stabilizer generators, it extracts the generators (QECC has such function).

        # Check that stab_gens is the correct object, and ceonvert it to qecc.Pauli otherwise
        self.stab_gens = []
        for check_stab in stab_generators:
            if not isinstance(check_stab, q.Pauli):
                self.stab_gens.append(q.Pauli(check_stab))
            else:
                self.stab_gens.append(check_stab)

        self.nqbits = len(stab_generators[0])

    def __hash__(self):
        # We need a hash function to store GraphStates as dict keys or in sets.
        return hash(self.stab_gens)

    def __len__(self):
        """
        Yields the number of qubits in the graph.
        """
        return self.nqbits

    def __repr__(self):
        """
        Representation for StabState when printing.
        """
        return self.stab_gens.__repr__()

    #####################
    ## REPRESENTATIONS ##
    #####################

    def as_binary(self):
        """
        Obtains the binary representation of the stabilizer state
        """
        nq = self.nqbits
        bin_matr = np.zeros((2 * nq, len(self.stab_gens)), dtype=np.int)
        for idx_stab, this_stab in enumerate(self.stab_gens):
            for idx_q in range(nq):
                if this_stab.op[idx_q] == 'X':
                    bin_matr[idx_q + nq, idx_stab] = 1
                elif this_stab.op[idx_q] == 'Y':
                    bin_matr[idx_q, idx_stab] = 1
                    bin_matr[idx_q + nq, idx_stab] = 1
                elif this_stab.op[idx_q] == 'Z':
                    bin_matr[idx_q, idx_stab] = 1
        return bin_matr

    def as_graph(self):
        """
        Obtains a Graph State locally equivalent to the stabilizer state
        """
        return stab_to_graph(self)

    # def stab_vecstate(self):
    # TODO The stabilizer_subspace function from QECC is really to slow to do this...takes already 10 seconds for
    #  7 qubits. Need to find a better way to do this, possible using clifford circuits and Aaronson style stuff
    #  (see comments in the analogue function in the GraphState class). IDEA: find local Cliffords to pass to graph
    #  state, then find state for graph state, and finally apply local cliffords to pass back to the original state.
    # return self.stab_gens.stabilizer_subspace()

    ######################
    ## LOCAL OPERATIONS ##
    ######################

    def local_cliffords(self, local_Cliffords):
        self.stab_gens = local_cliffords_on_stab_list(local_Cliffords, self.stab_gens)

    def local_pauli_measure(self, local_Pauli_meas, outcome=0):
        self.stab_gens = pauli_measurement_on_stab_list(local_Pauli_meas, self.stab_gens, outcome=outcome)

    #######################
    ## LOCAL EQUIVALENCE ##
    #######################

    # TODO: add function to check local equivalence of two stabilizer states, which is done passing into their
    #  local-equivalent graph states and then checking local equivalence of graphs.


if __name__ == '__main__':
    stab_gens = ["XZII", "ZXZI", "IZXZ", "IIZX"]
    stab_state = StabState(stab_gens)
    print('Initial stabilizers:')
    print(stab_state)

    loc_Cliff = ['H', 'I', 'H', 'I']
    print('\nOperating local Cliffords:', loc_Cliff)
    stab_state.local_cliffords(loc_Cliff)
    print('Updated stabilizers:')
    print(stab_state)

    pauli_meas = "ZIZI"
    print('\nOperating local Pauli measurement:', pauli_meas)
    stab_state.local_pauli_measure(pauli_meas)
    print('Updated stabilizers:')
    print(stab_state)
