from math import sqrt, pi
from random import random

Gate, Probability, QuantumRegister = None, None, None
import itertools
from pprint import pprint
from logging import error
import numpy as np


class Qubit:
    def __init__(self):
        self.measured_value = None
        self.a = random()
        self.b = sqrt(1 - pow(self.a, 2))

    def __str__(self):
        return f'{self.a}|0> + {self.b}|1>'

    def get_documentation(self, method_name: str) -> str:
        return getattr(self, method_name).__doc__

    def is_measured(self) -> bool:
        return self.measured_value is not None

    def measure(self):
        """
        The qubit collapses into a single state- 0 or 1
        based on the amplitude of its components.
        """
        if not self.is_measured():
            self.measured_value = 0 if random() < pow(self.a, 2) else 1
            self.a, self.b = int(not self.measured_value), self.measured_value
        return self.measured_value

    # defining single qubit logic gates
    def hadamard_gate(self):
        """
        Hadamard gate forms a uniformly random input
        """
        if self.measured_value is None:
            self.a, self.b = (self.a + self.b) / sqrt(2), (self.a - self.b) / sqrt(2)

    def pauli_x_gate(self):
        """
        equivalent to logical NOT gate
        It equates to a rotation around the X-axis of the Bloch sphere by pi radians.
        """
        if self.measured_value is None:
            self.a, self.b = self.b, self.a

    def pauli_y_gate(self):
        """
        It equates to a rotation around the Y-axis of the Bloch sphere by pi radians.
        """
        if self.measured_value is None:
            self.a, self.b = self.b, self.a

    def pauli_z_gate(self):
        """
        It equates to a rotation around the Z-axis of the Bloch sphere by pi radians.
        """
        if self.measured_value is None:
            self.b = -self.b

    def rotation(self, bi, isGreater):
        if self.measured_value is not None:
            return
        dt = 0
        sign = 0
        ri = self.measured_value
        positive = self.a * self.b > 0
        aNOT = not self.a
        bNOT = not self.b
        if (isGreater):
            if not ri and bi:
                dt = np.pi * .05  # angle of rotation
                if aNOT:
                    sign = 1  # sign of the rotation
                elif bNOT:
                    sign = 0
                elif positive:
                    sign = -1
                else:
                    sign = 1
            elif ri and not bi:
                dt = np.pi * .025
                if aNOT:
                    sign = 0
                elif bNOT:
                    sign = 1
                elif positive:
                    sign = 1
                else:
                    sign = -1
            elif ri and bi:
                dt = np.pi * .025
                if aNOT:
                    sign = 0
                elif bNOT:
                    sign = 1
                elif positive:
                    sign = 1
                else:
                    sign = -1
        else:
            if ri and not bi:
                dt = np.pi * .01
                if aNOT:
                    sign = 1
                elif bNOT:
                    sign = 0
                elif positive:
                    sign = -1
                else:
                    sign = 1
            elif ri and bi:
                dt = np.pi * .005
                if aNOT:
                    sign = 0
                elif bNOT:
                    sign = 1
                elif positive:
                    sign = 1
                else:
                    sign = -1

        t = sign * dt  # product of angle with the sign
        # applying the rotation gate as a dot product
        self.a, self.b = np.dot(
            np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]), np.array([self.a, self.b])
        )

        # keys: ri, bi, isGreater, alpha*beta -> [">" is greater than 0, "<" is less than 0, "0"]
        rotation_angle_lookup = {
            (0, 0, False): 0,
            (0, 0, True): 0,
            (0, 1, False): 0,
            (0, 1, True): 0.05 * pi,
            (1, 0, False): 0.01 * pi,
            (1, 0, True): 0.025 * pi,
            (1, 1, False): 0.005 * pi,
            (1, 1, True): 0.025 * pi}


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


if __name__ == "__main__":
    q = Qubit()
    print(q)
    q.pauli_x_gate()
    print(q)
    q.measure()
    print(q)
    q.pauli_x_gate()
    print(q)
