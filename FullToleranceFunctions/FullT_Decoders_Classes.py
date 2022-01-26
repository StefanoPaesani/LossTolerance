import qecc as q
import numpy as np
from itertools import product, combinations, chain

from MiscFunctions.PauliOpsFunctions import single_qubit_commute, count_target_pauli_in_stabs
from MiscFunctions.StabilizersFilteringFunctions import filter_stabs_input_op_compatible, filter_stabs_input_op_only, \
    filter_stabs_measurement_compatible, filter_stabs_indmeas_compatible, filter_stabs_given_qubits_ops, \
    filter_stabs_compatible_qubits_ops


class FullT_IndMeasDecoder(object):
    r"""
    Class representing the decoder to indirectly measure a Pauli operator in a graph.

    The state of the decoder is identified by the variables:
    'poss_strat_list': list of remaining strategies (dicts of measures, compatible stabilizers, ...) available to the decoder.
    'meas_config': current strategy of the decoder.

    'measured_qubits': List of qubits that have already been measured.
    'qubits_to_measure': List of qubits that still have to be measured.
    'lost_qubits': List of qubits that have already been measured to be lost.
    'mOUT_OUT': List of qubits in which an arbitrary measure has been tried, successfully.
    'mOUT_Z': List of qubits in which an arbitrary measure has been tried, unsuccessfully but an indirect Z outcome was obtained
    'mOUT_na': List of qubits in which an arbitrary measure has been tried, and no outcome was obtained
    'mX_X': List of qubits in which a X measure has been tried, and a X outcome has been obtained.
    'mX_Z': List of qubits in which a X measure has been tried, and an indirect Z outcome was obtained.
    'mX_na': List of qubits in which a X measure has been tried, and no outcome was obtained.
    'mY_Y': List of qubits in which a Y measure has been tried, and a Y outcome has been obtained.
    'mY_Z': List of qubits in which a Y measure has been tried, and an indirect Z outcome was obtained.
    'mY_na': List of qubits in which a Y measure has been tried, and no outcome was obtained.
    'mZ_Z': List of qubits in which a Z measure has been tried, and a Z outcome was obtained
    'mZ_na': List of qubits in which a Z measure has been tried, and neither direct or indirect outcomes were obtained.

    'new_strategy': True if a new strategy needs to be initialized (eg. because the previous one has failed due to loss).
    'finished': False if the decoder hasn't finished, True if there are no strategies left (decoder failed) or if it succeds.
    """

    def __init__(self, gstate, indmeas_pauli, in_qubit=0, pref_pauli='Z', printing=False):
        # Initialize the decoder
        self.gstate = gstate
        self.n_qbts = len(self.gstate)
        self.indmeas_pauli = indmeas_pauli
        self.in_qubit = in_qubit
        self.pref_pauli = pref_pauli
        self.meas_qubit_ix = 0  # Index of next qubit to be measured
        self.meas_type = 'I'  # Pauli operator to be measured next
        self.mOUT_OUT = []  # tracks qubits that we tried to measure in arbitrary bases, and succeded
        self.mOUT_Z = []  # tracks qubits that we tried to measure in arbitrary bases, failed but obtained an outcome in (indirect) Z measurement
        self.mOUT_na = []  # tracks qubits that we tried to measure in arbitrary bases, and obtained no outcome
        self.mX_X = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in XY
        self.mX_Z = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in (indirect) Z measurement
        self.mX_na = []  # tracks qubits that we tried to measure in XY, and obtained no outcome
        self.mY_Y = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in XY
        self.mY_Z = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in (indirect) Z measurement
        self.mY_na = []  # tracks qubits that we tried to measure in XY, and obtained no outcome
        self.mZ_Z = []  # tracks qubits that we tried to measure in Z, and obtained a direct outcome in Z
        self.mZ_na = []  # tracks qubits that we tried to measure in Z, and obtained no outcome either direct or indirect
        self.measured_qubits = []  # tracks which qubits have been already measured
        self.lost_qubits = []  # tracks which qubits have been already lost
        self.qubits_to_measure = []  # tracks which qubits are still to be measured
        self.on_track = True  # tracks if we're still on route to succeding or if we have failed
        self.finished = False  # tracks if the decoding process has finished or not
        self.new_strategy = True  # tracks if the current measurement strategy needs to be changed
        self.meas_config = []

        self.printing = printing

        # Find all valid strategies for the decoder, and initialize its state to be capable to access all of them
        self.all_strats = self.get_indirect_meas_strats()
        self.poss_strat_list = self.all_strats

    def reset_decoder_state(self):
        self.meas_qubit_ix = 0
        self.meas_type = 'I'
        self.mOUT_OUT = []
        self.mOUT_Z = []
        self.mOUT_na = []
        self.mX_X = []
        self.mX_Z = []
        self.mX_na = []
        self.mY_Y = []
        self.mY_Z = []
        self.mY_na = []
        self.mZ_Z = []
        self.mZ_na = []
        self.measured_qubits = []
        self.lost_qubits = []
        self.qubits_to_measure = []
        self.on_track = True
        self.finished = False
        self.new_strategy = True
        self.meas_config = ()
        self.poss_strat_list = self.all_strats

    def get_indirect_meas_strats(self):
        if self.indmeas_pauli not in ('X', 'Y', 'Z'):
            raise ValueError('The indirect measument Pauli operator needs to be one of (X, Y, Z)')
        ### gets all stabs for a graph
        all_stabs = [this_op.op for this_op in q.from_generators(self.gstate.stab_gens)]
        all_stabs.remove('I' * self.n_qbts)
        ### filter stabilizers to get those compatible with the indirect measurement
        filtered_stabs_ind_meas_comp = filter_stabs_input_op_compatible(all_stabs, self.indmeas_pauli, self.in_qubit)
        # print(filtered_stabs_ind_meas_comp)
        ### further filter all stabs to get only those forming possible valid logical operators
        filtered_stabs_ind_meas_only = filter_stabs_input_op_only(filtered_stabs_ind_meas_comp, self.indmeas_pauli,
                                                                  self.in_qubit)
        ### further filter all stabs get only those forming possible valid syndromes
        filtered_stabs_ind_meas_syndr = filter_stabs_input_op_only(filtered_stabs_ind_meas_comp, 'I', self.in_qubit)
        # print(filtered_stabs_ind_meas_only)
        # print(filtered_stabs_ind_meas_syndr)

        all_meas = {}
        for log_op_stab in filtered_stabs_ind_meas_only:
            # print('\nlog_op_stab:', log_op_stab)
            log_op_weight = self.n_qbts - log_op_stab.count('I')
            log_op_prefpauli_weight = log_op_stab.count(self.pref_pauli) / (log_op_weight + 1)
            log_op_val = log_op_weight - log_op_prefpauli_weight
            ### Get only the stabilizers that are also compatible with this logical operator
            compatible_syndromes = filter_stabs_indmeas_compatible(filtered_stabs_ind_meas_syndr, log_op_stab,
                                                                   self.in_qubit)
            # print('compatible_stabs', compatible_syndromes)
            poss_ops_list = [[self.indmeas_pauli] if ix == self.in_qubit else [] for ix in range(self.n_qbts)]
            free_qubits = []
            ### Populate the measurement bases that are fixed and given by logical operator stabilizer
            for ix in range(self.n_qbts):
                # print('qbt_ix:', ix)
                if ix != self.in_qubit:
                    if log_op_stab[ix] != 'I':
                        # print('occupied qbt')
                        poss_ops_list[ix] = [log_op_stab[ix]]
                    else:
                        # print('free qubt')
                        free_qubits.append(ix)
            ### For all free qubits, include 'I' (loss) as a possible compatible measurement
            for ix in free_qubits:
                poss_ops_list[ix].append('I')
            # print('free_qubits:', free_qubits)
            ### Find all possible measurement bases for qubits with bases not fixed by the logical operator considered
            for this_comp_stab in compatible_syndromes:
                for ix, this_op in enumerate(this_comp_stab):
                    if ix in free_qubits:
                        if this_op not in poss_ops_list[ix]:
                            poss_ops_list[ix].append(this_op)
            ### Find all possible measurements
            this_strat_all_meas = (''.join(ops) for ops in product(*poss_ops_list))

            this_strat_all_meas = list(this_strat_all_meas)
            # print('this_strat_all_meas:', this_strat_all_meas)
            ### Update the all_meas dictionary using the measurements compatible with this logical operator.
            for this_meas in this_strat_all_meas:
                meas_comp_stabs = filter_stabs_measurement_compatible(compatible_syndromes, this_meas)
                ## Uses these stabilizers for this measurement if this_meas is not already included
                if this_meas not in all_meas:
                    all_meas[this_meas] = (log_op_stab, meas_comp_stabs, log_op_val)
                # If this_meas was already included, updated its strategy to this measurement if there are equal or
                # more compatible stabilizers than the existing case (from EC decoders), and if the weight of the
                # logical operator is smaller (from LT decoders).
                else:
                    previous_stabs = all_meas[this_meas][1]
                    if len(meas_comp_stabs) >= len(previous_stabs):
                        if log_op_val < all_meas[this_meas][2]:
                            all_meas[this_meas] = (log_op_stab, meas_comp_stabs, log_op_val)
        return all_meas

    ###### Strategy decisions functions

    def decide_new_strat(self):
        # get strategies with minimal logical operator weight (from LT decoders)
        min_weight = min([self.poss_strat_list[x][2] for x in self.poss_strat_list])
        temp_strats = dict((k, self.poss_strat_list[k]) for k in self.poss_strat_list
                           if self.poss_strat_list[k][2] == min_weight)
        # between the selected strategies, get the one with maximal number of syndrome stabilizers, considering also
        # a preferred Pauli basis (from EC decoders).
        best_meas = max(temp_strats, key=lambda x: len(self.poss_strat_list[x][1]) + (
            0 if len(self.poss_strat_list[x][1]) == 0 else count_target_pauli_in_stabs(self.poss_strat_list[x][1],
                                                                                       self.pref_pauli) / (
                                                                   self.n_qbts * len(self.poss_strat_list[x][1]))))
        self.meas_config = (best_meas, temp_strats[best_meas])
        self.new_strategy = False

    def decide_next_meas(self):
        if self.new_strategy:
            self.decide_new_strat()
            self.new_strategy = False
        if self.printing:
            print("testing measurement meas_config", self.meas_config)

        # try to measure a qubit
        # first try to see if there are qubits to measure in the logical operator.
        self.qubits_to_measure = [x for x, pauli_op in enumerate(self.meas_config[1][0])
                                  if ((x != self.in_qubit) and (x not in self.measured_qubits) and (pauli_op != 'I'))]
        # If no qubits in the logical operators are left to measure, try all including the others
        if not self.qubits_to_measure:
            self.qubits_to_measure = [x for x, pauli_op in enumerate(self.meas_config[0])
                                      if
                                      ((x != self.in_qubit) and (x not in self.measured_qubits) and (pauli_op != 'I'))]
        # If there are still no qubits left, we have measured all of them!
        if not self.qubits_to_measure:
            if self.printing:
                print("SUCCEDING: no more qubits to measure")
            # If there are no more qubits to measure, we've succeded and we stop.
            self.finished = True
            self.meas_qubit_ix = 0
            self.meas_type = 'I'
        else:
            qubits_to_measure_in_XYs = [x for x in self.qubits_to_measure if self.meas_config[0][x] in ['X', 'Y']]
            # Pick one of the qubits (starting from the largest one, inspired by trees)
            # Qubits to be measured in XYs are measured first.
            if len(qubits_to_measure_in_XYs) > 0:
                self.meas_qubit_ix = qubits_to_measure_in_XYs[-1]
            else:
                self.meas_qubit_ix = self.qubits_to_measure[-1]
            self.meas_type = self.meas_config[0][self.meas_qubit_ix]
            if self.printing:
                print("measuring qubit", self.meas_qubit_ix, "in basis", self.meas_type)

    ##### Filtering functions to update indirect measurements stategies after trying to measure a qubit

    # keeps only stabilizers in which the lost_qbt_ix is allowed to be lost
    def filter_strats_lost_qubit(self, lost_qbt_ix):
        return dict((k, self.poss_strat_list[k]) for k in self.poss_strat_list if
                    k[lost_qbt_ix] == 'I')

    # keeps only the stabilizers in which the meas_qbt_ix is the fixed basis measured or in 'I'
    # (for qubits measured in fixed basis that cannot anymore be measured in other bases)
    def filter_strats_measured_qubit_fixed_basis(self, fixed_basis, meas_qbt_ix):
        return dict((k, self.poss_strat_list[k]) for k in self.poss_strat_list if
                    k[meas_qbt_ix] in [fixed_basis, 'I'])

    ##### Decoder updating function upon measurement trials

    def update_decoder_after_measure(self, outcome_basis):
        # Update for X or Y measurements
        self.measured_qubits.append(self.meas_qubit_ix)
        if self.meas_type in ['X', 'Y']:
            if outcome_basis == self.meas_type:
                if self.printing:
                    print("qubit is X or Y FULLY MEASURED")
                if self.meas_type == 'X':
                    self.mX_X.append(self.meas_qubit_ix)
                elif self.meas_type == 'Y':
                    self.mY_Y.append(self.meas_qubit_ix)
                self.poss_strat_list = self.filter_strats_measured_qubit_fixed_basis(self.meas_type, self.meas_qubit_ix)
            elif outcome_basis == 'Zind':
                # if direct X or Y meas failed, we can still indirectly measure it in Z from the layer below
                # before starting a new strategy.
                self.new_strategy = True
                if self.printing:
                    print('Direct' + self.meas_type + 'meas failed, but qubit was Z inDIRECTLY MEASURED')
                if self.meas_type == 'X':
                    self.mX_Z.append(self.meas_qubit_ix)
                elif self.meas_type == 'Y':
                    self.mY_Z.append(self.meas_qubit_ix)
                self.poss_strat_list = self.filter_strats_measured_qubit_fixed_basis('Z', self.meas_qubit_ix)
            elif outcome_basis == 'na':
                self.new_strategy = True
                if self.printing:
                    print('Measurement in ' + self.meas_type + 'failed to provide any outcome, qubit is LOST')
                self.lost_qubits.append(self.meas_qubit_ix)
                if self.meas_type == 'X':
                    self.mX_na.append(self.meas_qubit_ix)
                elif self.meas_type == 'Y':
                    self.mY_na.append(self.meas_qubit_ix)
                self.poss_strat_list = self.filter_strats_lost_qubit(self.meas_qubit_ix)
            else:
                raise ValueError('Outcome basis ' + outcome_basis + ' for ' + self.meas_type +
                                 ' measurement not recognized, use one in (X, Y, Zind, na)')
        # Update for Z measurements
        elif self.meas_type == 'Z':
            if outcome_basis == 'Z':
                if self.printing:
                    print("qubit is Z ONLY DIRECTLY measured")
                self.mZ_Z.append(self.meas_qubit_ix)
                self.poss_strat_list = self.filter_strats_measured_qubit_fixed_basis(self.meas_type, self.meas_qubit_ix)
            elif outcome_basis == 'na':
                self.new_strategy = True
                if self.printing:
                    print('Measurement in ' + self.meas_type + 'failed to provide any outcome, qubit is LOST')
                self.lost_qubits.append(self.meas_qubit_ix)
                self.mZ_na.append(self.meas_qubit_ix)
                self.poss_strat_list = self.filter_strats_lost_qubit(self.meas_qubit_ix)
            else:
                raise ValueError('Outcome basis ' + outcome_basis + ' for ' + self.meas_type +
                                 ' measurement not recognized, use one in (Z, na)')
        else:
            raise ValueError("Measurement basis not recognized, use one in (X, Y, Z)")


class FullT_TeleportationDecoder(object):
    r"""
    Class representing the decoder to teleport the logical qubit into an output one in a graph.

    The state of the decoder is identified by the variables:
    'poss_strat_list': list of remaining strategies (dicts of measures, compatible stabilizers, ...) available to the decoder.
    'meas_config': current strategy of the decoder.

    'measured_qubits': List of qubits that have already been measured.
    'qubits_to_measure': List of qubits that still have to be measured.
    'lost_qubits': List of qubits that have already been measured to be lost.
    'mOUT_OUT': List of qubits in which an arbitrary measure has been tried, successfully.
    'mOUT_Z': List of qubits in which an arbitrary measure has been tried, unsuccessfully but an indirect Z outcome was obtained
    'mOUT_na': List of qubits in which an arbitrary measure has been tried, and no outcome was obtained
    'mX_X': List of qubits in which a X measure has been tried, and a X outcome has been obtained.
    'mX_Z': List of qubits in which a X measure has been tried, and an indirect Z outcome was obtained.
    'mX_na': List of qubits in which a X measure has been tried, and no outcome was obtained.
    'mY_Y': List of qubits in which a Y measure has been tried, and a Y outcome has been obtained.
    'mY_Z': List of qubits in which a Y measure has been tried, and an indirect Z outcome was obtained.
    'mY_na': List of qubits in which a Y measure has been tried, and no outcome was obtained.
    'mZ_Z': List of qubits in which a Z measure has been tried, and a Z outcome was obtained
    'mZ_na': List of qubits in which a Z measure has been tried, and neither direct or indirect outcomes were obtained.

    'new_strategy': True if a new strategy needs to be initialized (eg. because the previous one has failed due to loss).
    'finished': False if the decoder hasn't finished, True if there are no strategies left (decoder failed) or if it succeds.
    """

    def __init__(self, gstate, in_qubit=0, pref_pauli='Z', printing=False):
        # Initialize the decoder
        self.gstate = gstate
        self.n_qbts = len(self.gstate)
        self.in_qubit = in_qubit
        self.pref_pauli = pref_pauli
        self.meas_qubit_ix = 0  # Index of next qubit to be measured
        self.meas_type = 'I'  # Pauli operator to be measured next
        self.mOUT_OUT = []  # tracks qubits that we tried to measure in arbitrary bases, and succeded
        self.mOUT_Z = []  # tracks qubits that we tried to measure in arbitrary bases, failed but obtained an outcome in (indirect) Z measurement
        self.mOUT_na = []  # tracks qubits that we tried to measure in arbitrary bases, and obtained no outcome
        self.mX_X = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in XY
        self.mX_Z = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in (indirect) Z measurement
        self.mX_na = []  # tracks qubits that we tried to measure in XY, and obtained no outcome
        self.mY_Y = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in XY
        self.mY_Z = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in (indirect) Z measurement
        self.mY_na = []  # tracks qubits that we tried to measure in XY, and obtained no outcome
        self.mZ_Z = []  # tracks qubits that we tried to measure in Z, and obtained a direct outcome in Z
        self.mZ_na = []  # tracks qubits that we tried to measure in Z, and obtained no outcome either direct or indirect
        self.measured_qubits = []  # tracks which qubits have been already measured
        self.lost_qubits = []  # tracks which qubits have been already lost
        self.qubits_to_measure = []  # tracks which qubits are still to be measured
        self.on_track = True  # tracks if we're still on route to succeding or if we have failed
        self.finished = False  # tracks if the decoding process has finished or not
        self.new_strategy = True  # tracks if the current measurement strategy needs to be changed
        self.meas_config = []

        self.printing = printing

        # Find all valid strategies for the decoder, and initialize its state to be capable to access all of them
        self.all_strats = self.get_teleportation_strats()
        self.poss_strat_list = self.all_strats

    def reset_decoder_state(self):
        self.meas_qubit_ix = 0
        self.meas_type = 'I'
        self.mOUT_OUT = []
        self.mOUT_Z = []
        self.mOUT_na = []
        self.mX_X = []
        self.mX_Z = []
        self.mX_na = []
        self.mY_Y = []
        self.mY_Z = []
        self.mY_na = []
        self.mZ_Z = []
        self.mZ_na = []
        self.measured_qubits = []
        self.lost_qubits = []
        self.qubits_to_measure = []
        self.on_track = True
        self.finished = False
        self.new_strategy = True
        self.meas_config = ()
        self.poss_strat_list = self.all_strats

    def get_teleportation_strats(self):
        ### gets all stabs for a graph
        all_stabs = [this_op.op for this_op in q.from_generators(self.gstate.stab_gens)]
        all_stabs.remove('I' * self.n_qbts)

        all_meas = {}
        for stab_ix1 in range((2 ** self.n_qbts) - 1):
            for stab_ix2 in range(stab_ix1 + 1, (2 ** self.n_qbts) - 1):
                stab1 = all_stabs[stab_ix1]
                stab2 = all_stabs[stab_ix2]
                ## checks which qubits have anticommuting Paulis
                anticomm_qbts = [qbt for qbt in range(self.n_qbts) if single_qubit_commute(stab1, stab2, qbt)]
                # print()
                # print(stab_ix1, stab1, stab_ix2, stab2, anticomm_qbts)
                ## checks that there are exactly two qubits with anticommuting Paulis: the input and an output
                if len(anticomm_qbts) == 2 and self.in_qubit in anticomm_qbts:
                    compatible_stabs = filter_stabs_given_qubits_ops(all_stabs,
                                                                     {anticomm_qbts[0]: 'I', anticomm_qbts[1]: 'I'})
                    non_inout_qubits = [ix for ix in range(self.n_qbts) if ix not in anticomm_qbts]
                    non_inout_paulis = [stab1[ix] if stab1[ix] != 'I' else stab2[ix] for ix in non_inout_qubits]
                    compatible_stabs = filter_stabs_compatible_qubits_ops(compatible_stabs,
                                                                          dict(zip(non_inout_qubits, non_inout_paulis)))
                    # print('good inout, compatible_stabs:', compatible_stabs)
                    poss_ops_list = [['I'] if ix in anticomm_qbts else [] for ix in range(self.n_qbts)]
                    free_qubits = []
                    for ix in range(self.n_qbts):
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
                                    len(meas_comp_stabs) == len(previous_stabs) and count_target_pauli_in_stabs(meas_comp_stabs, self.pref_pauli) > count_target_pauli_in_stabs(previous_stabs, self.pref_pauli)): all_meas[this_meas] = ((anticomm_qbts, (stab1, stab2)), meas_comp_stabs)
        return all_meas

        # all_meas = {}
        # for log_op_stab in filtered_stabs_ind_meas_only:
        #     # print('\nlog_op_stab:', log_op_stab)
        #     log_op_weight = self.n_qbts - log_op_stab.count('I')
        #     log_op_prefpauli_weight = log_op_stab.count(self.pref_pauli) / (log_op_weight + 1)
        #     log_op_val = log_op_weight - log_op_prefpauli_weight
        #     ### Get only the stabilizers that are also compatible with this logical operator
        #     compatible_syndromes = filter_stabs_indmeas_compatible(filtered_stabs_ind_meas_syndr, log_op_stab,
        #                                                            self.in_qubit)
        #     # print('compatible_stabs', compatible_syndromes)
        #     poss_ops_list = [[self.indmeas_pauli] if ix == self.in_qubit else [] for ix in range(self.n_qbts)]
        #     free_qubits = []
        #     ### Populate the measurement bases that are fixed and given by logical operator stabilizer
        #     for ix in range(self.n_qbts):
        #         # print('qbt_ix:', ix)
        #         if ix != self.in_qubit:
        #             if log_op_stab[ix] != 'I':
        #                 # print('occupied qbt')
        #                 poss_ops_list[ix] = [log_op_stab[ix]]
        #             else:
        #                 # print('free qubt')
        #                 free_qubits.append(ix)
        #     ### For all free qubits, include 'I' (loss) as a possible compatible measurement
        #     for ix in free_qubits:
        #         poss_ops_list[ix].append('I')
        #     # print('free_qubits:', free_qubits)
        #     ### Find all possible measurement bases for qubits with bases not fixed by the logical operator considered
        #     for this_comp_stab in compatible_syndromes:
        #         for ix, this_op in enumerate(this_comp_stab):
        #             if ix in free_qubits:
        #                 if this_op not in poss_ops_list[ix]:
        #                     poss_ops_list[ix].append(this_op)
        #     ### Find all possible measurements
        #     this_strat_all_meas = (''.join(ops) for ops in product(*poss_ops_list))
        #
        #     this_strat_all_meas = list(this_strat_all_meas)
        #     # print('this_strat_all_meas:', this_strat_all_meas)
        #     ### Update the all_meas dictionary using the measurements compatible with this logical operator.
        #     for this_meas in this_strat_all_meas:
        #         meas_comp_stabs = filter_stabs_measurement_compatible(compatible_syndromes, this_meas)
        #         ## Uses these stabilizers for this measurement if this_meas is not already included
        #         if this_meas not in all_meas:
        #             all_meas[this_meas] = (log_op_stab, meas_comp_stabs, log_op_val)
        #         # If this_meas was already included, updated its strategy to this measurement if there are equal or
        #         # more compatible stabilizers than the existing case (from EC decoders), and if the weight of the
        #         # logical operator is smaller (from LT decoders).
        #         else:
        #             previous_stabs = all_meas[this_meas][1]
        #             if len(meas_comp_stabs) >= len(previous_stabs):
        #                 if log_op_val < all_meas[this_meas][2]:
        #                     all_meas[this_meas] = (log_op_stab, meas_comp_stabs, log_op_val)
        # return all_meas


########################################################################################################################
##############################
###          MAIN          ###
##############################


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from CodesFunctions.graphs import *
    import networkx as nx

    from itertools import chain


    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


    branching = None

    ## index of the input qubit (output qubit is free)
    in_qubit = 0

    ## define graph state

    # ### three graph
    # branching = [2, 2]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    ## fully connected graph
    graph = gen_fullyconnected_graph(4)
    gstate = GraphState(graph)

    # ### ring graph
    # graph = gen_ring_graph(5)
    # gstate = GraphState(graph)

    # ## Graph equivalent to L543021 with loss-tolerance
    # graph_nodes = list(range(6))
    # graph_edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (1,5), (3, 6)]
    # gstate = graphstate_from_nodes_and_edges(graph_nodes, graph_edges)

    # ### Generate random graph
    # graph = gen_random_connected_graph(6)
    # gstate = GraphState(graph)

    #########################################################################################
    ################################### SINGLE TEST - DECODING ##############################
    #########################################################################################
    gstate.image(input_qubits=[in_qubit])
    plt.show()

    meas_pauli = 'Y'

    ## initialize the decoder
    IndMeas_decoder = FullT_IndMeasDecoder(gstate, meas_pauli, in_qubit, printing=True)

    print('\nFinal families:')
    print(IndMeas_decoder.all_strats)

    # print('\nCurrent strat:')
    # print(IndMeas_decoder.meas_config, IndMeas_decoder.new_strategy)
    # print('Calculating new strat')
    # IndMeas_decoder.decide_new_strat()
    # print('Current strat:')
    # print(IndMeas_decoder.meas_config, IndMeas_decoder.new_strategy)

    IndMeas_decoder.decide_next_meas()
    IndMeas_decoder.decide_next_meas()
