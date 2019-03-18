from math import sqrt
from random import random
from pprint import pprint
from logging import error

class Qubit:
    def __init__(self):
        self.measured_value = None
        self.a = random()
        self.b = sqrt(1 - pow(self.a, 2))
        pprint(vars(self))

    def __str__(self):
        return f'{self.a}|0> + {self.b}|1>'

    def __setattr__(self, key, value):
        try:
            pprint(vars(self))
            if self.is_measured():
                error(f'Can not set {value} for {key} after qubit is measured')
                return
            super(Qubit, self).__setattr__(key, value)
        except AttributeError:
            super(Qubit, self).__setattr__(key, value)

    def is_measured(self) -> bool:
        return self.measured_value is not None

    def measure(self):
        if not self.is_measured():
            self.measured_value = 0 if random() < pow(self.a, 2) else 1
            self.a, self.b = int(not self.measured_value), self.measured_value
        return self.measured_value

    # defining single qubit logic gates

    def hadamard_gate(self):
        self.a, self.b = (self.a + self.b) / sqrt(2), (self.a - self.b) / sqrt(2)

    def pauli_x_gate(self):
        """
        equivalent to logical NOT gate
        It equates to a rotation around the X-axis of the Bloch sphere by pi radians.
        """
        self.a, self.b = self.b, self.a

    def pauli_y_gate(self):
        """
        It equates to a rotation around the Y-axis of the Bloch sphere by pi radians.
        """
        # todo
        pass

    def pauli_z_gate(self):
        """
        It equates to a rotation around the Z-axis of the Bloch sphere by pi radians.
        """
        self.b = -self.b


if __name__ == "__main__":
    q = Qubit()
    print(q)
    q.pauli_x_gate()
    print(q)
    q.measure()
    print(q)
    q.pauli_x_gate()
    print(q)
