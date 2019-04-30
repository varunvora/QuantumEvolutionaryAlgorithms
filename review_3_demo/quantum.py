import numpy as np
import unittest
import random
import itertools
from functools import reduce
from math import sqrt, pi, e, log
import time


####
## Gates
####
class Gate(object):
    i_ = np.complex(0, 1)
    ## One qubit gates
    # Hadamard gate
    H = 1. / sqrt(2) * np.matrix('1 1; 1 -1')
    # Pauli gates
    X = np.matrix('0 1; 1 0')
    Y = np.matrix([[0, -i_], [i_, 0]])
    Z = np.matrix([[1, 0], [0, -1]])

    # Defined as part of the Bell state experiment
    W = 1 / sqrt(2) * (X + Z)
    V = 1 / sqrt(2) * (-X + Z)

    # Other useful gates
    eye = np.eye(2, 2)

    S = np.matrix([[1, 0], [0, i_]])
    Sdagger = np.matrix([[1, 0], [0, -i_]])  # convenience Sdagger = S.conjugate().transpose()

    T = np.matrix([[1, 0], [0, e ** (i_ * pi / 4.)]])
    Tdagger = np.matrix([[1, 0], [0, e ** (-i_ * pi / 4.)]])  # convenience Tdagger= T.conjugate().transpose()

    # CNOT Gate (control is qubit 0, target qubit 1), this is the default CNOT gate
    CNOT2_01 = np.matrix('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0')
    # control is qubit 1 target is qubit 0
    CNOT2_10 = np.kron(H, H) * CNOT2_01 * np.kron(H, H)  # =np.matrix('1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0')

    # operates on 2 out of 3 entangled qubits, control is first subscript, target second
    CNOT3_01 = np.kron(CNOT2_01, eye)
    CNOT3_10 = np.kron(CNOT2_10, eye)
    CNOT3_12 = np.kron(eye, CNOT2_01)
    CNOT3_21 = np.kron(eye, CNOT2_10)
    CNOT3_02 = np.matrix(
        '1 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0; 0 0 0 1 0 0 0 0; 0 0 0 0 0 1 0 0; 0 0 0 0 1 0 0 0; 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 1 0')
    CNOT3_20 = np.matrix(
        '1 0 0 0 0 0 0 0; 0 0 0 0 0 1 0 0; 0 0 1 0 0 0 0 0; 0 0 0 0 0 0 0 1; 0 0 0 0 1 0 0 0; 0 1 0 0 0 0 0 0; 0 0 0 0 0 0 1 0; 0 0 0 1 0 0 0 0')

    # operates on 2 out of 4 entangled qubits, control is first subscript, target second
    CNOT4_01 = np.kron(CNOT3_01, eye)
    CNOT4_10 = np.kron(CNOT3_10, eye)
    CNOT4_12 = np.kron(CNOT3_12, eye)
    CNOT4_21 = np.kron(CNOT3_21, eye)
    CNOT4_13 = np.kron(eye, CNOT3_02)
    CNOT4_31 = np.kron(eye, CNOT3_20)
    CNOT4_02 = np.kron(CNOT3_02, eye)
    CNOT4_20 = np.kron(CNOT3_20, eye)
    CNOT4_23 = np.kron(eye, CNOT3_12)
    CNOT4_32 = np.kron(eye, CNOT3_21)
    CNOT4_03 = np.eye(16, 16)
    CNOT4_03[np.array([8, 9])] = CNOT4_03[np.array([9, 8])]
    CNOT4_03[np.array([10, 11])] = CNOT4_03[np.array([11, 10])]
    CNOT4_03[np.array([12, 13])] = CNOT4_03[np.array([13, 12])]
    CNOT4_03[np.array([14, 15])] = CNOT4_03[np.array([15, 14])]
    CNOT4_30 = np.eye(16, 16)
    CNOT4_30[np.array([1, 9])] = CNOT4_30[np.array([9, 1])]
    CNOT4_30[np.array([3, 11])] = CNOT4_30[np.array([11, 3])]
    CNOT4_30[np.array([5, 13])] = CNOT4_30[np.array([13, 5])]
    CNOT4_30[np.array([7, 15])] = CNOT4_30[np.array([15, 7])]

    # operates on 2 out of 5 entangled qubits, control is first subscript, target second
    CNOT5_01 = np.kron(CNOT4_01, eye)
    CNOT5_10 = np.kron(CNOT4_10, eye)
    CNOT5_02 = np.kron(CNOT4_02, eye)
    CNOT5_20 = np.kron(CNOT4_20, eye)
    CNOT5_03 = np.kron(CNOT4_03, eye)
    CNOT5_30 = np.kron(CNOT4_30, eye)
    CNOT5_12 = np.kron(CNOT4_12, eye)
    CNOT5_21 = np.kron(CNOT4_21, eye)
    CNOT5_13 = np.kron(CNOT4_13, eye)
    CNOT5_31 = np.kron(CNOT4_31, eye)
    CNOT5_14 = np.kron(eye, CNOT4_03)
    CNOT5_41 = np.kron(eye, CNOT4_30)
    CNOT5_23 = np.kron(CNOT4_23, eye)
    CNOT5_32 = np.kron(CNOT4_32, eye)
    CNOT5_24 = np.kron(eye, CNOT4_13)
    CNOT5_42 = np.kron(eye, CNOT4_31)
    CNOT5_34 = np.kron(eye, CNOT4_23)
    CNOT5_43 = np.kron(eye, CNOT4_32)
    CNOT5_04 = np.eye(32, 32)
    CNOT5_04[np.array([16, 17])] = CNOT5_04[np.array([17, 16])]
    CNOT5_04[np.array([18, 19])] = CNOT5_04[np.array([19, 18])]
    CNOT5_04[np.array([20, 21])] = CNOT5_04[np.array([21, 20])]
    CNOT5_04[np.array([22, 23])] = CNOT5_04[np.array([23, 22])]
    CNOT5_04[np.array([24, 25])] = CNOT5_04[np.array([25, 24])]
    CNOT5_04[np.array([26, 27])] = CNOT5_04[np.array([27, 26])]
    CNOT5_04[np.array([28, 29])] = CNOT5_04[np.array([29, 28])]
    CNOT5_04[np.array([30, 31])] = CNOT5_04[np.array([31, 30])]
    CNOT5_40 = np.eye(32, 32)
    CNOT5_40[np.array([1, 17])] = CNOT5_40[np.array([17, 1])]
    CNOT5_40[np.array([3, 19])] = CNOT5_40[np.array([19, 3])]
    CNOT5_40[np.array([5, 21])] = CNOT5_40[np.array([21, 5])]
    CNOT5_40[np.array([7, 23])] = CNOT5_40[np.array([23, 7])]
    CNOT5_40[np.array([9, 25])] = CNOT5_40[np.array([25, 9])]
    CNOT5_40[np.array([11, 27])] = CNOT5_40[np.array([27, 11])]
    CNOT5_40[np.array([13, 29])] = CNOT5_40[np.array([29, 13])]
    CNOT5_40[np.array([15, 31])] = CNOT5_40[np.array([31, 15])]


####
## States
####
class State(object):
    i_ = np.complex(0, 1)
    ## One qubit states (basis)
    # standard basis (z)
    zero_state = np.matrix('1; 0')
    one_state = np.matrix('0; 1')
    # diagonal basis (x)
    plus_state = 1 / sqrt(2) * np.matrix('1; 1')
    minus_state = 1 / sqrt(2) * np.matrix('1; -1')
    # circular basis (y)
    plusi_state = 1 / sqrt(2) * np.matrix([[1], [i_]])  # also known as clockwise arrow state
    minusi_state = 1 / sqrt(2) * np.matrix([[1], [-i_]])  # also known as counterclockwise arrow state

    # 2-qubit states
    bell_state = 1 / sqrt(2) * np.matrix('1; 0; 0; 1')

    @staticmethod
    def change_to_x_basis(state):
        return Gate.H * state

    @staticmethod
    def change_to_y_basis(state):
        return Gate.H * Gate.Sdagger * state

    @staticmethod
    def change_to_w_basis(state):
        # W=1/sqrt(2)*(X+Z)
        return Gate.H * Gate.T * Gate.H * Gate.S * state

    @staticmethod
    def change_to_v_basis(state):
        # V=1/sqrt(2)*(-X+Z)
        return Gate.H * Gate.Tdagger * Gate.H * Gate.S * state

    @staticmethod
    def is_fully_separable(qubit_state):
        try:
            separated_state = State.separate_state(qubit_state)
            for state in separated_state:
                State.string_from_state(state)
            return True
        except StateNotSeparableException as e:
            return False

    @staticmethod
    def get_first_qubit(qubit_state):
        return State.separate_state(qubit_state)[0]

    @staticmethod
    def get_second_qubit(qubit_state):
        return State.separate_state(qubit_state)[1]

    @staticmethod
    def get_third_qubit(qubit_state):
        return State.separate_state(qubit_state)[2]

    @staticmethod
    def get_fourth_qubit(qubit_state):
        return State.separate_state(qubit_state)[3]

    @staticmethod
    def get_fifth_qubit(qubit_state):
        return State.separate_state(qubit_state)[4]

    @staticmethod
    def all_state_strings(n_qubits):
        return [''.join(map(str, state_desc)) for state_desc in itertools.product([0, 1], repeat=n_qubits)]

    @staticmethod
    def state_from_string(qubit_state_string):
        if not all(x in '01' for x in qubit_state_string):
            raise Exception("Description must be a string in binary")
        state = None
        for qubit in qubit_state_string:
            if qubit == '0':
                new_contrib = State.zero_state
            elif qubit == '1':
                new_contrib = State.one_state
            if state is None:
                state = new_contrib
            else:
                state = np.kron(state, new_contrib)
        return state

    @staticmethod
    def string_from_state(qubit_state):
        separated = State.separate_state(qubit_state)
        desc = ''
        for state in separated:
            if np.allclose(state, State.zero_state):
                desc += '0'
            elif np.allclose(state, State.one_state):
                desc += '1'
            else:
                raise StateNotSeparableException("State is not separable")
        return desc

    @staticmethod
    def separate_state(qubit_state):
        """This only works if the state is fully separable at present

        Throws exception if not a separable state"""
        n_entangled = QuantumRegister.num_qubits(qubit_state)
        if list(qubit_state.flat).count(1) == 1:
            separated_state = []
            idx_state = list(qubit_state.flat).index(1)
            add_factor = 0
            size = qubit_state.shape[0]
            while (len(separated_state) < n_entangled):
                size = size / 2
                if idx_state < (add_factor + size):
                    separated_state += [State.zero_state]
                    add_factor += 0
                else:
                    separated_state += [State.one_state]
                    add_factor += size
            return separated_state
        else:
            # Try a few naive separations before giving up
            cardinal_states = [State.zero_state, State.one_state, State.plus_state, State.minus_state,
                               State.plusi_state, State.minusi_state]
            for separated_state in itertools.product(cardinal_states, repeat=n_entangled):
                candidate_state = reduce(lambda x, y: np.kron(x, y), separated_state)
                if np.allclose(candidate_state, qubit_state):
                    return separated_state
            # TODO: more general separation methods
            raise StateNotSeparableException(
                "TODO: Entangled qubits not represented yet in quantum computer implementation. Can currently do manual calculations; see TestBellState for Examples")

    @staticmethod
    def measure(state):
        """finally some probabilities, whee. To properly use, set the qubit you measure to the result of this function
            to collapse it. state=measure(state). Currently supports only up to three entangled qubits """
        state_z = state
        n_qubits = QuantumRegister.num_qubits(state)
        probs = Probability.get_probabilities(state_z)
        rand = random.random()
        for idx, state_desc in enumerate(State.all_state_strings(n_qubits)):
            if rand < sum(probs[0:(idx + 1)]):
                return State.state_from_string(state_desc)

    @staticmethod
    def get_bloch(state):
        return np.array(
            (Probability.expectation_x(state), Probability.expectation_y(state), Probability.expectation_z(state)))

    @staticmethod
    def pretty_print_gate_action(gate, n_qubits):
        for s in list(itertools.product([0, 1], repeat=n_qubits)):
            sname = ('%d' * n_qubits) % s
            state = State.state_from_string(sname)
            print(sname, '->', State.string_from_state(gate * state))


class StateNotSeparableException(Exception):
    def __init__(self, args=None):
        self.args = args


class Probability(object):
    @staticmethod
    def get_probability(coeff):
        return (coeff * coeff.conjugate()).real

    @staticmethod
    def get_probabilities(state):
        return [Probability.get_probability(x) for x in state.flat]

    @staticmethod
    def get_correlated_expectation(state):
        probs = Probability.get_probabilities(state)
        return probs[0] + probs[3] - probs[1] - probs[2]

    @staticmethod
    def pretty_print_probabilities(state):
        probs = Probability.get_probabilities(state)
        am_desc = '|psi>='
        pr_desc = ''
        for am, pr, state_desc in zip(state.flat, probs, State.all_state_strings(QuantumRegister.num_qubits(state))):
            if am != 0:
                if am != 1:
                    am_desc += '%r|%s>+' % (am, state_desc)
                else:
                    am_desc += '|%s>+' % (state_desc)
            if pr != 0:
                pr_desc += 'Pr(|%s>)=%f; ' % (state_desc, pr)
        print(am_desc[0:-1])
        print(pr_desc)
        if state.shape == (4, 1):
            print("<state>=%f" % float(probs[0] + probs[3] - probs[1] - probs[2]))

    @staticmethod
    def expectation_x(state):
        state_x = State.change_to_x_basis(state)
        prob_zero_state = (state_x.item(0) * state_x.item(0).conjugate()).real
        prob_one_state = (state_x.item(1) * state_x.item(1).conjugate()).real
        return prob_zero_state - prob_one_state

    @staticmethod
    def expectation_y(state):
        state_y = State.change_to_y_basis(state)
        prob_zero_state = (state_y.item(0) * state_y.item(0).conjugate()).real
        prob_one_state = (state_y.item(1) * state_y.item(1).conjugate()).real
        return prob_zero_state - prob_one_state

    @staticmethod
    def expectation_z(state):
        state_z = state
        prob_zero_state = (state_z.item(0) * state_z.item(0).conjugate()).real
        prob_one_state = (state_z.item(1) * state_z.item(1).conjugate()).real
        return prob_zero_state - prob_one_state


class QuantumRegister(object):
    def __init__(self, name, state=State.zero_state, entangled=None):
        self._entangled = [self]
        self._state = state
        self.name = name
        self.idx = None
        self._noop = []  # after a measurement set this so that we can allow no further operations. Set to Bloch coords if bloch operation performed

    @staticmethod
    def num_qubits(state):
        num_qubits = log(state.shape[0], 2)
        if state.shape[1] != 1 or num_qubits not in [1, 2, 3, 4, 5]:
            raise Exception("unrecognized state shape")
        else:
            return int(num_qubits)

    def get_entangled(self):
        return self._entangled

    def set_entangled(self, entangled):
        self._entangled = entangled
        for qb in self._entangled:
            qb._state = self._state
            qb._entangled = self._entangled

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        for qb in self._entangled:
            qb._state = state
            qb._entangled = self._entangled
            qb._noop = self._noop

    def get_noop(self):
        return self._noop

    def set_noop(self, noop):
        self._noop = noop
        for qb in self._entangled:
            qb._noop = noop

    def is_entangled(self):
        return len(self._entangled) > 1

    def is_entangled_with(self, qubit):
        return qubit in self._entangled

    def get_indices(self, target_qubit):
        if not self.is_entangled_with(target_qubit):
            search = self._entangled + target_qubit.get_entangled()
        else:
            search = self._entangled
        return search.index(self), search.index(target_qubit)

    def get_num_qubits(self):
        return QuantumRegister.num_qubits(self._state)

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.name == other.name and np.array(self._noop).shape == np.array(other._noop).shape and np.allclose(
            self._noop, other._noop) and np.array(self.get_state()).shape == np.array(
            other.get_state()).shape and np.allclose(self.get_state(),
                                                     other.get_state()) and QuantumRegisterCollection.orderings_equal(
            self._entangled, other._entangled)


class QuantumRegisterSet(object):
    """Created this so I could have some set like features for use, even though QuantumRegisters are mutable"""
    registers = []

    def __init__(self, registers):
        for r in registers:
            if r not in self.registers:
                self.registers += [r]

    def intersection(self, quantumregisterset):
        intersection = []

        if self.size() >= quantumregisterset:
            qrs1 = self
            qrs2 = quantumregisterset
        else:
            qrs1 = quantumregisterset
            qrs2 = self
        # now qrs2 is the smaller set
        intersection = [qr for qr in qrs1 if qr in qrs2]
        return QuantumRegisterSet(intersection)

    def size(self):
        return len(self.registers)


class QuantumRegisterCollection(object):
    def __init__(self, qubits):
        self._qubits = qubits
        for idx, qb in enumerate(self._qubits):
            qb.idx = idx
        self.num_qubits = len(qubits)

    def get_quantum_register_containing(self, name):
        for qb in self._qubits:
            if qb.name == name:
                return qb
            else:
                for entqb in qb.get_entangled():
                    if entqb.name == name:
                        return entqb
        raise Exception("qubit %s not found in %s" % (name, repr(self._qubits)))

    def get_quantum_registers(self):
        return self._qubits

    def entangle_quantum_registers(self, first_qubit, second_qubit):
        new_entangle = first_qubit.get_entangled() + second_qubit.get_entangled()
        if len(first_qubit.get_entangled()) >= len(second_qubit.get_entangled()):
            self._remove_quantum_register_named(second_qubit.name)
            first_qubit.set_entangled(new_entangle)
        else:
            self._remove_quantum_register_named(first_qubit.name)
            second_qubit.set_entangled(new_entangle)

    def _remove_quantum_register_named(self, name):
        self._qubits = [qb for qb in self._qubits if qb.name != name]

    def is_in_canonical_ordering(self):
        return self.get_qubit_order() == list(range(self.num_qubits))

    @staticmethod
    def is_in_increasing_order(qb_list):
        for a, b in zip(qb_list, qb_list[1:]):
            if not a.idx < b.idx:
                return False
        return True

    def get_entangled_qubit_order(self):
        ordering = []
        for qb in self._qubits:
            ent_order = []
            for ent in qb.get_entangled():
                ent_order += [ent]
            ordering += [ent_order]
        return ordering

    def get_qubit_order(self):
        ordering = []
        for qb in self._qubits:
            for ent in qb.get_entangled():
                ordering += [ent.idx]
        return ordering

    def add_quantum_register(self, qubit):
        qubit.idx = self.num_qubits
        self._qubits += [qubit]
        self.num_qubits += 1

    @staticmethod
    def orderings_equal(order_one, order_two):
        return [qb.idx for qb in order_one] == [qb.idx for qb in order_two]



class TestQuantumRegister(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()
        self.q0 = QuantumRegister("q0")
        self.q1 = QuantumRegister("q1")

    def tearDown(self):
        print(self._testMethodName, "%.3f" % (time.time() - self.startTime))
        self.q0 = None
        self.q1 = None

    def test_get_num_qubits(self):
        self.assertTrue(self.q0.get_num_qubits() == self.q0.get_num_qubits() == 1)

    def test_equality(self):
        self.assertEqual(self.q0, self.q0)
        self.assertNotEqual(self.q0, self.q1)


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print(self._testMethodName, "%.3f" % (time.time() - self.startTime))

    def test_measure_probs_plus(self):
        measurements = []
        for i in range(100000):
            measurements += [State.measure(State.plus_state)]
        result = (1. * sum(measurements)) / len(measurements)
        self.assertTrue(np.allclose(list(result.flat), np.array((0.5, 0.5)), rtol=1e-2))

    def test_measure_probs_minus(self):
        measurements = []
        for i in range(100000):
            measurements += [State.measure(State.minus_state)]
        result = (1. * sum(measurements)) / len(measurements)
        self.assertTrue(np.allclose(list(result.flat), np.array((0.5, 0.5)), rtol=1e-2))

    def test_collapse(self):
        result = State.measure(State.minus_state)
        for i in range(100):
            new_measure = State.measure(result)
            self.assertTrue(np.allclose(result, new_measure))
            result = new_measure

    def test_measure_bell(self):
        """ Tests the measurement of a 2 qubit entangled system"""
        measurements = []
        for i in range(100000):
            measurements += [State.measure(State.bell_state)]
        result = (1. * sum(measurements)) / len(measurements)
        self.assertTrue(np.allclose(list(result.flat), np.array((0.5, 0.0, 0.0, 0.5)), rtol=1e-2))


class TestGetBloch(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print(self._testMethodName, "%.3f" % (time.time() - self.startTime))

    def test_get_bloch(self):
        self.assertTrue(np.allclose(State.get_bloch(State.zero_state), np.array((0, 0, 1))))
        self.assertTrue(np.allclose(State.get_bloch(State.one_state), np.array((0, 0, -1))))
        self.assertTrue(np.allclose(State.get_bloch(State.plusi_state), np.array((0, 1, 0))))
        self.assertTrue(np.allclose(State.get_bloch(State.minusi_state), np.array((0, -1, 0))))
        self.assertTrue(np.allclose(State.get_bloch(Gate.Z * State.plus_state), np.array((-1, 0, 0))))
        self.assertTrue(np.allclose(State.get_bloch(Gate.Z * State.minus_state), np.array((1, 0, 0))))

        # assert the norms are 1 for cardinal points (obviously) but also for a few other points at higher T depth on the Bloch Sphere
        for state in [State.zero_state, State.one_state, State.plusi_state, State.minusi_state,
                      Gate.Z * State.plus_state, Gate.H * Gate.T * Gate.Z * State.plus_state,
                      Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * Gate.T * Gate.Z * State.plus_state]:
            self.assertAlmostEqual(np.linalg.norm(state), 1.0)


class TestGetBloch2(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print(self._testMethodName, "%.3f" % (time.time() - self.startTime))

    def get_bloch_2(self, state):
        """ equal to get_bloch just a different way of calculating things. Used for testing get_bloch. """
        return np.array((((state * state.conjugate().transpose() * Gate.X).trace()).item(0),
                         ((state * state.conjugate().transpose() * Gate.Y).trace()).item(0),
                         ((state * state.conjugate().transpose() * Gate.Z).trace()).item(0)))

    def test_get_bloch_2(self):
        self.assertTrue(np.allclose(self.get_bloch_2(State.zero_state), State.get_bloch(State.zero_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(State.one_state), State.get_bloch(State.one_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(State.plusi_state), State.get_bloch(State.plusi_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(State.minusi_state), State.get_bloch(State.minusi_state)))
        self.assertTrue(
            np.allclose(self.get_bloch_2(Gate.Z * State.plus_state), State.get_bloch(Gate.Z * State.plus_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(Gate.H * Gate.T * Gate.Z * State.plus_state), State.get_bloch(
            Gate.H * Gate.T * Gate.Z * State.plus_state)))  # test for arbitrary gates


class TestCNOTGate(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print(self._testMethodName, "%.3f" % (time.time() - self.startTime))

    def test_CNOT(self):
        self.assertTrue(np.allclose(Gate.CNOT2_01 * State.state_from_string('00'), State.state_from_string('00')))
        self.assertTrue(np.allclose(Gate.CNOT2_01 * State.state_from_string('01'), State.state_from_string('01')))
        self.assertTrue(np.allclose(Gate.CNOT2_01 * State.state_from_string('10'), State.state_from_string('11')))
        self.assertTrue(np.allclose(Gate.CNOT2_01 * State.state_from_string('11'), State.state_from_string('10')))


class TestTGate(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print(self._testMethodName, "%.3f" % (time.time() - self.startTime))

    def test_T(self):
        # This is useful to check some of the exercises on IBM's quantum experience.
        # "Ground truth" answers from IBM's calculations which unfortunately are not reported to high precision.
        red_state = Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * State.zero_state
        green_state = Gate.S * Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * State.zero_state
        blue_state = Gate.H * Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * State.zero_state
        self.assertTrue(np.allclose(State.get_bloch(red_state), np.array((0.5, 0.5, 0.707)), rtol=1e-3))
        self.assertTrue(np.allclose(State.get_bloch(green_state), np.array((0.427, 0.457, 0.780)), rtol=1e-3))
        self.assertTrue(np.allclose(State.get_bloch(blue_state), np.array((0.457, 0.427, 0.780)), rtol=1e-3))
        # Checking norms
        for state in [red_state, green_state, blue_state]:
            self.assertAlmostEqual(np.linalg.norm(state), 1.0)



if __name__ == '__main__':
    unittest.main()
