import qecc as q
import numpy as np


###################################
####### FULL DECODER CLASS ########
###################################

class LT_FullDecoder(object):
    r"""
    Class representing the decoder to retrieve the encoded state in a loss-tolerant graph.

    The state of a full-decoder is identified by the variables:
    'poss_strat_list': list of remaining strategies available to the decoder.
    'meas_config': current strategy of the decoder.
    'out_qubit_ix': index of qubit currently identified as output.
    'meas_out': identifies the current state of the output. True if it has already been measured in an arbitrary basis.

    'measured_qubits': List of qubits that have already been measured.
    'lost_qubits': List of qubits that have already been measured to be lost.
    'mOUT_OUT_qbts': List of qubits in which an arbitrary measure has been tried, successfully.
    'mOUT_Z_qbts': List of qubits in which an arbitrary measure has been tried, unsuccessfully but an indirect Z outcome was obtained
    'mOUT_na_qbts': List of qubits in which an arbitrary measure has been tried, and no outcome was obtained
    'mX_X_qbts': List of qubits in which a X measure has been tried, and a X outcome has been obtained.
    'mX_Z_qbts': List of qubits in which a X measure has been tried, and an indirect Z outcome was obtained.
    'mX_na_qbts': List of qubits in which a X measure has been tried, and no outcome was obtained.
    'mY_Y_qbts': List of qubits in which a Y measure has been tried, and a Y outcome has been obtained.
    'mY_Z_qbts': List of qubits in which a Y measure has been tried, and an indirect Z outcome was obtained.
    'mY_na_qbts': List of qubits in which a Y measure has been tried, and no outcome was obtained.
    'mZ_Z_qbts': List of qubits in which a Z measure has been tried, and a Z outcome was obtained.
    'mZ_na_qbts': List of qubits in which a Z measure has been tried, and no outcome was obtained.

    'poss_outs_dict': dictionary that identifies which output qubits are better for current strategy.
    'new_strategy': True if a new strategy needs to be initialized (eg. because the previous one has failed due to loss).
    'finished': False if the decoder hasn't finished, True if there are no strategies left (decoder failed) or if it succeds.
    """

    def __init__(self, gstate, in_qubit=0, printing=False):
        # Initialize the decoder
        self.gstate = gstate
        self.in_qubit = in_qubit
        self.meas_out = False  # tracks if the output qubit for current strategy has already been measured or not
        self.out_qubit_ix = in_qubit  # tracks the index of the current output qubit
        self.meas_qubit_ix = 0  # Index of next qubit to be measured
        self.meas_type = 'I'  # Pauli operator to be measured next
        self.mOUT_OUT_qbts = []  # tracks qubits that we tried to measure in arbitrary bases, and succeded
        self.mOUT_Z_qbts = []  # tracks qubits that we tried to measure in arbitrary bases, failed but obtained an outcome in (indirect) Z measurement
        self.mOUT_na_qbts = []  # tracks qubits that we tried to measure in arbitrary bases, and obtained no outcome
        self.mX_X_qbts = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in XY
        self.mX_Z_qbts = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in (indirect) Z measurement
        self.mX_na_qbts = []  # tracks qubits that we tried to measure in XY, and obtained no outcome
        self.mY_Y_qbts = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in XY
        self.mY_Z_qbts = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in (indirect) Z measurement
        self.mY_na_qbts = []  # tracks qubits that we tried to measure in XY, and obtained no outcome
        self.mZ_Z_qbts = []  # tracks qubits that we tried to measure in Z, and obtained an outcome in Z (direct or indirect)
        self.mZ_na_qbts = []  # tracks qubits that we tried to measure in Z, and obtained no outcome
        self.measured_qubits = []  # tracks which qubits have been already measured
        self.lost_qubits = []  # tracks which qubits have been already lost
        self.on_track = True  # tracks if we're still on route to succeding or if we have failed
        self.finished = False  # tracks if the decoding process has finished or not
        self.new_strategy = True  # tracks if the current measurement strategy needs to be changed
        self.meas_config = []
        self.poss_outs_dict = []

        self.printing = printing

        # Find all valid strategies for the decoder, and initialize its state to be capable to access all of them
        self.all_strats = self.get_possible_decoding_strats()
        self.poss_strat_list = self.all_strats

    def reset_decoder_state(self):
        self.gstate = gstate
        self.in_qubit = in_qubit
        self.meas_out = False
        self.out_qubit_ix = in_qubit
        self.meas_qubit_ix = 0
        self.meas_type = 'I'
        self.mOUT_OUT_qbts = []
        self.mOUT_Z_qbts = []
        self.mOUT_na_qbts = []
        self.mX_X_qbts = []
        self.mX_Z_qbts = []
        self.mX_na_qbts = []
        self.mY_Y_qbts = []
        self.mY_Z_qbts = []
        self.mY_na_qbts = []
        self.mZ_Z_qbts = []
        self.mZ_na_qbts = []
        self.measured_qubits = []
        self.lost_qubits = []
        self.on_track = True
        self.finished = False
        self.new_strategy = True
        self.meas_config = []
        self.poss_outs_dict = []
        self.poss_strat_list = self.all_strats

    def get_possible_decoding_strats(self):
        num_qubits = len(self.gstate)
        # print("num_qubits", num_qubits)
        # get all possible 2^N stabilizers.
        # TODO: to be improved checking only "smart" stabilizers.
        all_stabs = list(q.from_generators(self.gstate.stab_gens))
        poss_stabs_list = []
        for stab_ix1 in range(1, (2 ** num_qubits) - 1):
            for stab_ix2 in range(stab_ix1, 2 ** num_qubits):
                stab1 = all_stabs[stab_ix1].op
                stab2 = all_stabs[stab_ix2].op
                ## checks which qubits have anticommuting Paulis
                anticomm_qbts = [qbt for qbt in range(num_qubits) if single_qubit_commute(stab1, stab2, qbt)]

                ## checks that there are exactly two qubits with anticommuting Paulis: the input and an output
                if len(anticomm_qbts) == 2 and self.in_qubit in anticomm_qbts:
                    measurement = [stab1[qbt] if stab1[qbt] != 'I' else stab2[qbt] for qbt in range(num_qubits)]
                    other_meas_qubits = [qbt for qbt in range(num_qubits)
                                         if measurement[qbt] != 'I' and qbt not in anticomm_qbts]
                    meas_weight = num_qubits - measurement.count('I')
                    Z_weight = measurement.count('Z') / (meas_weight + 1)
                    poss_stabs_list.append(
                        [anticomm_qbts, other_meas_qubits, [stab1, stab2], measurement, meas_weight, Z_weight])
                    # print(stab1, stab2, anticomm_qbts, other_meas_qubits, measurement, meas_weight)
        ### order them such that we always prefer mstrategies with smaller weight, and with more Zs in the non-trivial paulis.
        poss_stabs_list.sort(key=lambda x: x[4] - x[5])
        return poss_stabs_list

    ###### Strategy decisions functions

    def decide_newstrat(self):
        self.meas_config = self.poss_strat_list[0]
        temp_out_qb_ix = self.meas_config[0][1]
        if temp_out_qb_ix != self.out_qubit_ix:
            self.meas_out = False
            self.out_qubit_ix = temp_out_qb_ix

        ### Decide which nodes are better to keep as outputs
        self.poss_outs_dict = {}
        for strat in self.poss_strat_list:
            this_out = strat[0][1]
            weight = strat[4]
            add_to_sum = 1. / weight
            if this_out in self.poss_outs_dict:
                self.poss_outs_dict[this_out] += add_to_sum
            else:
                self.poss_outs_dict[this_out] = add_to_sum
        # hack to have larger indexes first, to facilitate their selection for first in the strategy
        self.poss_outs_dict = dict(reversed(list(self.poss_outs_dict.items())))
        # sort the possible outputs from the one least likely to get good strategies to the best one
        self.poss_outs_dict = dict(sorted(self.poss_outs_dict.items(), key=lambda item: item[1]))

    def decide_next_meas(self):
        if self.new_strategy:
            self.decide_newstrat()
            self.new_strategy = False
        if self.printing:
            print("testing measurement meas_config", self.meas_config)
        if not self.meas_out:
            # start strategy by trying to measure the output qubit
            self.meas_qubit_ix = self.meas_config[0][1]
            self.meas_type = 'XYout'
            if self.printing:
                print("measuring out qubit", self.meas_qubit_ix)
        else:
            # try to measure a non-output qubit
            qubits_to_measure = [x for x in self.meas_config[1] if x not in self.measured_qubits]
            if len(qubits_to_measure) == 0:
                if self.printing:
                    print("SUCCEDING: no more qubits to measure")
                # If there are no more qubits to measure, we've succeded and we stop.
                self.finished = True
                self.meas_qubit_ix = 0
                self.meas_type = 'I'
            else:
                # check if there are some 'safe' options that are not in the possible outputs list
                qubits_to_meas_nooutputs = [x for x in qubits_to_measure if x not in self.poss_outs_dict]
                if len(qubits_to_meas_nooutputs) > 0:
                    # if there are safe options, pick one of these (starting from the largest one, inspired by
                    # trees)
                    qubits_to_measure_in_XYs = [x for x in qubits_to_meas_nooutputs if
                                                self.meas_config[3][x] in ['X', 'Y']]
                    # Qubits to be measured in XYs are measured first, similarly as for outputs.
                    if len(qubits_to_measure_in_XYs) > 0:
                        self.meas_qubit_ix = qubits_to_measure_in_XYs[-1]
                    else:
                        self.meas_qubit_ix = qubits_to_measure[-1]
                    self.meas_type = self.meas_config[3][self.meas_qubit_ix]
                else:
                    # if not, Pick one of the qubits (starting from the largest one, inspired by trees)
                    qubits_to_meas = [x for x in self.poss_outs_dict if x in qubits_to_measure]
                    qubits_to_measure_in_XYs = [x for x in qubits_to_meas if self.meas_config[3][x] in ['X', 'Y']]
                    # Qubits to be measured in XYs are measured first, similarly as for outputs.
                    if len(qubits_to_measure_in_XYs) > 0:
                        self.meas_qubit_ix = qubits_to_measure_in_XYs[-1]
                    else:
                        self.meas_qubit_ix = qubits_to_measure[-1]
                    self.meas_type = self.meas_config[3][self.meas_qubit_ix]
                if self.printing:
                    print("measuring qubit", self.meas_qubit_ix, "in basis", self.meas_type)

    ##### filtering functions for measuring in full decoder

    # keeps only the possible strategies in which the lost_qbt_ix is allowed to be lost
    def filter_strat_lost_qubit(self, lost_qbt_ix):
        return [these_stabs for these_stabs in self.poss_strat_list if these_stabs[3][lost_qbt_ix] == 'I']

    # keeps only the possible strategies where meas_qbt_ix is not an output and have it in the fixed basis measured
    # (for qubits measured in fixed basis that cannot anymore be used as outputs or measured in other bases)
    def filter_strat_measured_qubit_fixed_basis(self, fixed_basis, meas_qbt_ix):
        return [these_stabs for these_stabs in self.poss_strat_list if
                (these_stabs[0][1] != meas_qbt_ix and (these_stabs[3][meas_qbt_ix] in [fixed_basis, 'I']))]

    # keeps only the possible strategies in which the meas_qbt_ix is an output or it is not measured
    # (for qubits measured in an arbitrary output basis that cannot anymore be measured in fixed bases)
    def filter_strat_measured_qubit_output_basis(self, meas_qbt_ix):
        return [these_stabs for these_stabs in self.poss_strat_list if
                (these_stabs[0][1] == meas_qbt_ix or these_stabs[3][meas_qbt_ix] == 'I')]

    ##### Measurement Functions

    def get_probs_XY_meas(self, transm, t_xi=1, t_yi=1, t_zi=0):
        if self.meas_type == 'XYout':
            t_xyi = max(t_xi, t_yi)
        elif self.meas_type == 'X':
            t_xyi = t_xi
        elif self.meas_type == 'Y':
            t_xyi = t_yi
        p_XY_XY = transm * t_xyi  # Prob. to measure in X (Y) and obtain  X (Y) outcome.
        p_XY_Z = (1 - transm) * t_zi  # Prob. to measure in X (Y), fail direct meas, but obtain indir. Z outcome
        p_XY_na = 1 - p_XY_XY - p_XY_Z  # transm * (1 - t_xyi) + (1 - transm) * (1 - t_xyi)
        # Prob. to measure in X (Y) and obtain no outcome (lost)
        return [self.meas_type, 'Z', 'na'], [p_XY_XY, p_XY_Z, p_XY_na]

    def get_probs_Z_meas(self, transm, t_zi=0):
        p_Z_Z = transm + (
                1 - transm) * t_zi  # Prob. to measure in Z and obtain Z outcome, either directly or indirectly
        p_Z_na = 1 - p_Z_Z  # Prob. to measure in Z and obtain no outcome (qubit is lost)
        return ['Z', 'na'], [p_Z_Z, p_Z_na]

    def update_decoder_XYmeas(self, outcome_basis):
        if outcome_basis in ['X', 'Y', 'XYout']:
            if self.printing:
                print("qubit is X or Y FULLY MEASURED")
            self.measured_qubits.append(self.meas_qubit_ix)
            if self.meas_type == 'X':
                self.mX_X_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'Y':
                self.mY_Y_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'XYout':
                self.mOUT_OUT_qbts.append(self.meas_qubit_ix)
            if self.meas_out:
                self.poss_strat_list = self.filter_strat_measured_qubit_fixed_basis(self.meas_type, self.meas_qubit_ix)
            else:
                self.meas_out = True
                self.poss_strat_list = self.filter_strat_measured_qubit_output_basis(self.meas_qubit_ix)
        elif outcome_basis == 'Z':
            # if direct X or Y meas failed, we can still indirectly measure it in Z from the layer below
            # before starting a new strategy.
            self.new_strategy = True
            if self.printing:
                print("Direct XY meas failed, but qubit was Z inDIRECTLY MEASURED")
            self.measured_qubits.append(self.meas_qubit_ix)
            if self.meas_type == 'X':
                self.mX_Z_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'Y':
                self.mY_Z_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'XYout':
                self.mOUT_Z_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_strat_measured_qubit_fixed_basis('Z', self.meas_qubit_ix)
        elif outcome_basis == 'na':
            self.new_strategy = True
            if self.printing:
                print("Measurement XY failed to provide any outcome, qubit is LOST")
            self.lost_qubits.append(self.meas_qubit_ix)
            if self.meas_type == 'X':
                self.mX_na_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'Y':
                self.mY_na_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'XYout':
                self.mOUT_na_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_strat_lost_qubit(self.meas_qubit_ix)
        else:
            raise ValueError("Outcome basis not recognized, use one in (X, Y, XY, Z, na)")

    def update_decoder_Zmeas(self, outcome_basis):
        if outcome_basis == 'Z':
            if self.printing:
                print("qubit Z MEASURED SUCCEDED")
            self.measured_qubits.append(self.meas_qubit_ix)
            self.mZ_Z_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_strat_measured_qubit_fixed_basis(self.meas_type, self.meas_qubit_ix)
        elif outcome_basis == 'na':
            if self.printing:
                print("qubit is LOST")
            self.lost_qubits.append(self.meas_qubit_ix)
            self.mZ_na_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_strat_lost_qubit(self.meas_qubit_ix)
            self.new_strategy = True
        else:
            raise ValueError("Outcome basis not recognized, use one in (Z, na)")

    ##### Monte-Carlo decoder simulation

    def MC_decoding(self, transm, t_xi=1, t_yi=1, t_zi=0, provide_measures=False):
        # reset the decoder state
        self.reset_decoder_state()
        while not self.finished:
            if self.printing:
                print()
                print("starting new measurement, meas_out", self.meas_out, ", measured_qubits", self.measured_qubits)
                print("Current stabs:", self.poss_strat_list)
                print("Strategy status",
                      [self.mOUT_OUT_qbts, self.mOUT_Z_qbts, self.mOUT_na_qbts,
                       self.mX_X_qbts, self.mX_Z_qbts, self.mX_na_qbts,
                       self.mY_Y_qbts, self.mY_Z_qbts, self.mY_na_qbts,
                       self.mZ_Z_qbts, self.mZ_na_qbts])

            # if there are no possible measurement to do, we have failed and we stop
            if len(self.poss_strat_list) == 0:
                self.on_track = False
                self.finished = True
                # print("failing")
            else:
                # Decide the next step of the decoer: which qubit to measure and in what basis
                self.decide_next_meas()

                # Perform the step
                if not self.finished:
                    if self.meas_type == 'Z':
                        poss_outcomes, outcome_probs = self.get_probs_Z_meas(transm, t_zi=t_zi)
                        this_outcome = np.random.choice(poss_outcomes, p=outcome_probs)
                        self.update_decoder_Zmeas(this_outcome)
                    elif self.meas_type in ['X', 'Y', 'XYout']:
                        poss_outcomes, outcome_probs = self.get_probs_XY_meas(transm, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi)
                        this_outcome = np.random.choice(poss_outcomes, p=outcome_probs)
                        self.update_decoder_XYmeas(this_outcome)
                    else:
                        raise ValueError("meas_type not recognized, use one in (X, Y, Z, XYout)")

        # see if we succeded or failed
        deco_final_status = (
                        tuple(self.mOUT_OUT_qbts), tuple(self.mOUT_Z_qbts), tuple(self.mOUT_na_qbts),
                        tuple(self.mX_X_qbts), tuple(self.mX_Z_qbts), tuple(self.mX_na_qbts),
                        tuple(self.mY_Y_qbts), tuple(self.mY_Z_qbts), tuple(self.mY_na_qbts),
                        tuple(self.mZ_Z_qbts), tuple(self.mZ_na_qbts))
        if self.on_track:
            if provide_measures:
                if t_xi == 1 and t_yi == 1 and t_zi == 0:
                    return True, (tuple(self.measured_qubits), tuple(self.lost_qubits))
                else:
                    return True, deco_final_status
            else:
                return True
        else:
            if provide_measures:
                if t_xi == 1 and t_yi == 1 and t_zi == 0:
                    return False, (tuple(self.measured_qubits), tuple(self.lost_qubits))
                else:
                    return False, deco_final_status
            else:
                return False


########################################################
####### SINGLE-PAULI INDIRECT MEASUREMENT CLASS ########
########################################################


class LT_IndMeasDecoder(object):
    r"""
    Class representing the decoder to indirectly measure a Pauli operator in a graph.

    The state of the decoder is identified by the variables:
    'poss_strat_list': list of remaining strategies available to the decoder.
    'meas_config': current strategy of the decoder.

    'measured_qubits': List of qubits that have already been measured.
    'lost_qubits': List of qubits that have already been measured to be lost.
    'mOUT_OUT_qbts': List of qubits in which an arbitrary measure has been tried, successfully.
    'mOUT_Z_qbts': List of qubits in which an arbitrary measure has been tried, unsuccessfully but an indirect Z outcome was obtained
    'mOUT_na_qbts': List of qubits in which an arbitrary measure has been tried, and no outcome was obtained
    'mX_X_qbts': List of qubits in which a X measure has been tried, and a X outcome has been obtained.
    'mX_Z_qbts': List of qubits in which a X measure has been tried, and an indirect Z outcome was obtained.
    'mX_na_qbts': List of qubits in which a X measure has been tried, and no outcome was obtained.
    'mY_Y_qbts': List of qubits in which a Y measure has been tried, and a Y outcome has been obtained.
    'mY_Z_qbts': List of qubits in which a Y measure has been tried, and an indirect Z outcome was obtained.
    'mY_na_qbts': List of qubits in which a Y measure has been tried, and no outcome was obtained.
    'mZ_Z_qbts': List of qubits in which a Z measure has been tried, and a Z outcome was obtained.
    'mZ_na_qbts': List of qubits in which a Z measure has been tried, and no outcome was obtained.

    'new_strategy': True if a new strategy needs to be initialized (eg. because the previous one has failed due to loss).
    'finished': False if the decoder hasn't finished, True if there are no strategies left (decoder failed) or if it succeds.
    """

    def __init__(self, gstate, indmeas_pauli, in_qubit=0, printing=False):
        # Initialize the decoder
        self.gstate = gstate
        self.indmeas_pauli = indmeas_pauli
        self.in_qubit = in_qubit
        self.meas_qubit_ix = 0  # Index of next qubit to be measured
        self.meas_type = 'I'  # Pauli operator to be measured next
        self.mOUT_OUT_qbts = []  # tracks qubits that we tried to measure in arbitrary bases, and succeded
        self.mOUT_Z_qbts = []  # tracks qubits that we tried to measure in arbitrary bases, failed but obtained an outcome in (indirect) Z measurement
        self.mOUT_na_qbts = []  # tracks qubits that we tried to measure in arbitrary bases, and obtained no outcome
        self.mX_X_qbts = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in XY
        self.mX_Z_qbts = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in (indirect) Z measurement
        self.mX_na_qbts = []  # tracks qubits that we tried to measure in XY, and obtained no outcome
        self.mY_Y_qbts = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in XY
        self.mY_Z_qbts = []  # tracks qubits that we tried to measure in XY, and obtained an outcome in (indirect) Z measurement
        self.mY_na_qbts = []  # tracks qubits that we tried to measure in XY, and obtained no outcome
        self.mZ_Z_qbts = []  # tracks qubits that we tried to measure in Z, and obtained an outcome in Z (direct or indirect)
        self.mZ_na_qbts = []  # tracks qubits that we tried to measure in Z, and obtained no outcome
        self.measured_qubits = []  # tracks which qubits have been already measured
        self.lost_qubits = []  # tracks which qubits have been already lost
        self.on_track = True  # tracks if we're still on route to succeding or if we have failed
        self.finished = False  # tracks if the decoding process has finished or not
        self.new_strategy = True  # tracks if the current measurement strategy needs to be changed
        self.meas_config = []

        self.printing = printing

        # Find all valid strategies for the decoder, and initialize its state to be capable to access all of them
        self.all_stabs = self.get_indirect_meas_stabs()
        self.poss_strat_list = self.all_stabs

    def reset_decoder_state(self):
        self.meas_qubit_ix = 0
        self.meas_type = 'I'
        self.mOUT_OUT_qbts = []
        self.mOUT_Z_qbts = []
        self.mOUT_na_qbts = []
        self.mX_X_qbts = []
        self.mX_Z_qbts = []
        self.mX_na_qbts = []
        self.mY_Y_qbts = []
        self.mY_Z_qbts = []
        self.mY_na_qbts = []
        self.mZ_Z_qbts = []
        self.mZ_na_qbts = []
        self.measured_qubits = []
        self.lost_qubits = []
        self.on_track = True
        self.finished = False
        self.new_strategy = True
        self.meas_config = []
        self.poss_strat_list = self.all_stabs

    def get_indirect_meas_stabs(self):
        num_qubits = len(self.gstate)
        # print("num_qubits", num_qubits)
        # get all possible 2^N stabilizers.
        # TODO: to be improved checking only "smart" stabilizers.
        all_stabs = q.from_generators(self.gstate.stab_gens)
        poss_stabs_list = []
        for this_stab0 in all_stabs:
            this_stab = this_stab0.op
            if this_stab[self.in_qubit] == self.indmeas_pauli:
                meas_weight = num_qubits - this_stab.count('I')
                Z_weight = this_stab.count('Z') / (meas_weight + 1)
                meas_qubits = [qbt for qbt in range(num_qubits)
                               if ((qbt != self.in_qubit) and (this_stab[qbt] != 'I'))]
                ### order them such that we always prefer mstrategies with smaller weight, and with more Zs in the non-trivial paulis.
                poss_stabs_list.append([this_stab, meas_qubits, meas_weight, Z_weight])
        poss_stabs_list.sort(key=lambda x: x[2] - x[3])
        return poss_stabs_list

    ###### Strategy decisions functions

    def decide_newstrat(self):
        # decide new strategy
        self.meas_config = self.poss_strat_list[0]
        self.new_strategy = False

    def decide_next_meas(self):
        if self.new_strategy:
            self.decide_newstrat()
            self.new_strategy = False
        if self.printing:
            print("testing measurement meas_config", self.meas_config)

        # try to measure a  qubit
        qubits_to_measure = [x for x in self.meas_config[1] if x not in self.measured_qubits]
        if len(qubits_to_measure) == 0:
            if self.printing:
                print("SUCCEDING: no more qubits to measure")
            # If there are no more qubits to measure, we've succeded and we stop.
            self.finished = True
            self.meas_qubit_ix = 0
            self.meas_type = 'I'
        else:
            qubits_to_measure_in_XYs = [x for x in qubits_to_measure if self.meas_config[0][x] in ['X', 'Y']]
            # Pick one of the qubits (starting from the largest one, inspired by trees)
            # Qubits to be measured in XYs are measured first, similarly as for outputs.
            if len(qubits_to_measure_in_XYs) > 0:
                self.meas_qubit_ix = qubits_to_measure_in_XYs[-1]
            else:
                self.meas_qubit_ix = qubits_to_measure[-1]
            self.meas_type = self.meas_config[0][self.meas_qubit_ix]
            if self.printing:
                print("measuring qubit", self.meas_qubit_ix, "in basis", self.meas_type)

    ##### filtering functions for measuring in stabilizers for indirect measurements

    # keeps only stabilizers in which the lost_qbt_ix is allowed to be lost
    def filter_stabs_lost_qubit(self, lost_qbt_ix):
        return [these_stabs for these_stabs in self.poss_strat_list if these_stabs[0][lost_qbt_ix] == 'I']

    # keeps only the stabilizers in which the meas_qbt_ix is the fixed basis measured or in 'I'
    # (for qubits measured in fixed basis that cannot anymore be measured in other bases)
    def filter_stabs_measured_qubit_fixed_basis(self, fixed_basis, meas_qbt_ix):
        return [these_stabs for these_stabs in self.poss_strat_list if
                (these_stabs[0][meas_qbt_ix] in [fixed_basis, 'I'])]

    ##### Measurement Functions

    def get_probs_XY_meas(self, transm, t_xi=1, t_yi=1, t_zi=0):
        if self.meas_type == 'XYout':
            t_xyi = max(t_xi, t_yi)
        elif self.meas_type == 'X':
            t_xyi = t_xi
        elif self.meas_type == 'Y':
            t_xyi = t_yi
        p_XY_XY = transm * t_xyi  # Prob. to measure in X (Y) and obtain  X (Y) outcome.
        p_XY_Z = (1 - transm) * t_zi  # Prob. to measure in X (Y), fail direct meas, but obtain indir. Z outcome
        p_XY_na = 1 - p_XY_XY - p_XY_Z  # transm * (1 - t_xyi) + (1 - transm) * (1 - t_xyi)
        # Prob. to measure in X (Y) and obtain no outcome (lost)
        return [self.meas_type, 'Z', 'na'], [p_XY_XY, p_XY_Z, p_XY_na]

    def get_probs_Z_meas(self, transm, t_zi=0):
        p_Z_Z = transm + (
                1 - transm) * t_zi  # Prob. to measure in Z and obtain Z outcome, either directly or indirectly
        p_Z_na = 1 - p_Z_Z  # Prob. to measure in Z and obtain no outcome (qubit is lost)
        return ['Z', 'na'], [p_Z_Z, p_Z_na]

    def update_decoder_XYmeas(self, outcome_basis):
        if outcome_basis in ['X', 'Y']:
            if self.printing:
                print("qubit is X or Y FULLY MEASURED")
            self.measured_qubits.append(self.meas_qubit_ix)
            if self.meas_type == 'X':
                self.mX_X_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'Y':
                self.mY_Y_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_stabs_measured_qubit_fixed_basis(self.meas_type, self.meas_qubit_ix)
        elif outcome_basis == 'Z':
            # if direct X or Y meas failed, we can still indirectly measure it in Z from the layer below
            # before starting a new strategy.
            self.new_strategy = True
            if self.printing:
                print("Direct XY meas failed, but qubit was Z inDIRECTLY MEASURED")
            self.measured_qubits.append(self.meas_qubit_ix)
            if self.meas_type == 'X':
                self.mX_Z_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'Y':
                self.mY_Z_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_stabs_measured_qubit_fixed_basis('Z', self.meas_qubit_ix)
        elif outcome_basis == 'na':
            self.new_strategy = True
            if self.printing:
                print("Measurement XY failed to provide any outcome, qubit is LOST")
            self.lost_qubits.append(self.meas_qubit_ix)
            if self.meas_type == 'X':
                self.mX_na_qbts.append(self.meas_qubit_ix)
            elif self.meas_type == 'Y':
                self.mY_na_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_stabs_lost_qubit(self.meas_qubit_ix)
        else:
            raise ValueError("Outcome basis not recognized, use one in (X, Y, XYout, Z, na)")

    def update_decoder_Zmeas(self, outcome_basis):
        if outcome_basis == 'Z':
            if self.printing:
                print("qubit Z MEASURED SUCCEDED")
            self.measured_qubits.append(self.meas_qubit_ix)
            self.mZ_Z_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_stabs_measured_qubit_fixed_basis(self.meas_type, self.meas_qubit_ix)
        elif outcome_basis == 'na':
            if self.printing:
                print("qubit is LOST")
            self.lost_qubits.append(self.meas_qubit_ix)
            self.mZ_na_qbts.append(self.meas_qubit_ix)
            self.poss_strat_list = self.filter_stabs_lost_qubit(self.meas_qubit_ix)
            self.new_strategy = True
        else:
            raise ValueError("Outcome basis not recognized, use one in (Z, na)")

    ##### Monte-Carlo decoder simulation

    def MC_decoding(self, transm, t_xi=1, t_yi=1, t_zi=0, provide_measures=False):
        # reset the decoder state
        self.reset_decoder_state()
        while not self.finished:
            if self.printing:
                print()
                print("starting new measurement, measured_qubits", self.measured_qubits)
                print("Current stabs:", self.poss_strat_list)
                print("Strategy status",
                      [self.mOUT_OUT_qbts, self.mOUT_Z_qbts, self.mOUT_na_qbts,
                       self.mX_X_qbts, self.mX_Z_qbts, self.mX_na_qbts,
                       self.mY_Y_qbts, self.mY_Z_qbts, self.mY_na_qbts,
                       self.mZ_Z_qbts, self.mZ_na_qbts])

            # if there are no possible measurement to do, we have failed and we stop
            if len(self.poss_strat_list) == 0:
                self.on_track = False
                self.finished = True
                # print("failing")
            else:
                # Decide the next step of the decoer: which qubit to measure and in what basis
                self.decide_next_meas()

                # Perform the step
                if not self.finished:
                    if self.meas_type == 'Z':
                        poss_outcomes, outcome_probs = self.get_probs_Z_meas(transm, t_zi=t_zi)
                        this_outcome = np.random.choice(poss_outcomes, p=outcome_probs)
                        self.update_decoder_Zmeas(this_outcome)
                    elif self.meas_type in ['X', 'Y']:
                        poss_outcomes, outcome_probs = self.get_probs_XY_meas(transm, t_xi=t_xi, t_yi=t_yi, t_zi=t_zi)
                        this_outcome = np.random.choice(poss_outcomes, p=outcome_probs)
                        self.update_decoder_XYmeas(this_outcome)
                    else:
                        raise ValueError("meas_type not recognized, use one in (X, Y, Z)")

        # see if we succeded or failed
        deco_final_status = (
                        tuple(self.mOUT_OUT_qbts), tuple(self.mOUT_Z_qbts), tuple(self.mOUT_na_qbts),
                        tuple(self.mX_X_qbts), tuple(self.mX_Z_qbts), tuple(self.mX_na_qbts),
                        tuple(self.mY_Y_qbts), tuple(self.mY_Z_qbts), tuple(self.mY_na_qbts),
                        tuple(self.mZ_Z_qbts), tuple(self.mZ_na_qbts))
        if self.on_track:
            if provide_measures:
                if t_xi == 1 and t_yi == 1 and t_zi == 0:
                    return True, (tuple(self.measured_qubits), tuple(self.lost_qubits))
                else:
                    return True, deco_final_status
            else:
                return True
        else:
            if provide_measures:
                if t_xi == 1 and t_yi == 1 and t_zi == 0:
                    return False, (tuple(self.measured_qubits), tuple(self.lost_qubits))
                else:
                    return False, deco_final_status
            else:
                return False


##############################
### OTHER USEFUL FUNCTIONS ###
##############################

def single_qubit_commute(pauli1, pauli2, qbt):
    """
    Returns 0 if the operators on the qbt-th qubit of the two operators in the Pauli group commute,
    and 1 if they anticommute.
    """
    if pauli1[qbt] == 'I' or pauli2[qbt] == 'I' or pauli1[qbt] == pauli2[qbt]:
        return 0
    else:
        return 1


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

    # three graph
    # branching = [2, 2]
    # graph = gen_tree_graph(branching)
    # gstate = GraphState(graph)

    ### fully connected graph
    # graph = gen_fullyconnected_graph(7)
    # gstate = GraphState(graph)

    ### ring graph
    graph = gen_ring_graph(3)
    gstate = GraphState(graph)

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

    ## initialize the decoder
    full_decoder = LT_FullDecoder(gstate, in_qubit, printing=True)

    # define channel transmission
    transmission = 0.8
    decoding_succ, meas = full_decoder.MC_decoding(transmission, provide_measures=True)

    ## see if we succeded or failed
    if decoding_succ:
        print("Succeded :)")
    else:
        print("Failed :(")
    print(meas)

    #########################################################################################
    ######################## SINGLE TEST - INDIRECT PAULI MEASUREMENT #######################
    #########################################################################################

    # ## initialize the decoder
    # indmeas_decoder = LT_IndMeasDecoder(gstate, 'X', in_qubit, printing=True)
    #
    # # define channel transmission
    # transmission = 0.6
    # decoding_succ, meas = indmeas_decoder.MC_decoding(transmission, provide_measures=True)
    #
    # ## see if we succeded or failed
    # if decoding_succ:
    #     print("Succeded :)")
    # else:
    #     print("Failed :(")
    # print(meas)
