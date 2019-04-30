from math import sqrt, pi
from random import random
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
        if not self.is_measured():
            self.measured_value = 0 if random() < pow(self.a, 2) else 1
            self.a, self.b = int(not self.measured_value), self.measured_value

    # defining single qubit logic gates
    def hadamard_gate(self):
        """
        Creates a superposition with equal probabilities of alpha and beta.
        """
        if not self.is_measured():
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


if __name__ == "__main__":
    q = Qubit()
    print(q)
    q.pauli_x_gate()
    print(q)
    q.measure()
    print(q)
    q.pauli_x_gate()
    print(q)
