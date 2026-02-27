import cirq
import numpy as np


class BloombergMarketsInflection:

    def __init__(self):
        # Q0 = S&P 500 (hub)
        # Q1 = NASDAQ
        # Q2 = Nikkei
        # Q3 = Oil
        # Q4 = Gold
        # Q5 = Silver
        # Q6 = Tech
        # Q7 = Risk
        self.n_qubits = 8
        self.qubits = cirq.LineQubit.range(self.n_qubits)

    # --------------------------------------------------
    # 1️⃣ Build Nonlinear Market Circuit
    # --------------------------------------------------
    def build_market_circuit(self, oil_shift=0.0):

        circuit = cirq.Circuit()

        percent_moves = [
            -0.58,   # SPX
            -0.87,   # NASDAQ
            0.16,    # Nikkei
            0.30 + oil_shift,  # Oil (parametric)
            0.94,    # Gold
            7.48,    # Silver
            -0.40,   # Tech proxy
            -0.50    # Risk sentiment
        ]

        # Encode rotations
        for i, move in enumerate(percent_moves):
            angle = move * np.pi / 8  # scaled for stability
            circuit.append(cirq.ry(angle).on(self.qubits[i]))

        # --------------------------------------------------
        # 2️⃣ Put SPX in Superposition (critical)
        # --------------------------------------------------
        circuit.append(cirq.H(self.qubits[0]))

        # --------------------------------------------------
        # 3️⃣ Nonlinear Cross-Asset Phase Coupling (ZZ)
        # --------------------------------------------------
        interaction_strength = 0.7

        circuit.append(
            cirq.ZZ(self.qubits[0], self.qubits[3]) ** interaction_strength
        )  # SPX ↔ Oil

        circuit.append(
            cirq.ZZ(self.qubits[0], self.qubits[4]) ** interaction_strength
        )  # SPX ↔ Gold

        circuit.append(
            cirq.ZZ(self.qubits[0], self.qubits[5]) ** interaction_strength
        )  # SPX ↔ Silver

        return circuit

    # --------------------------------------------------
    # 4️⃣ Compute SPX Expectation
    # --------------------------------------------------
    def compute_spx_expectation(self, oil_shift=0.0):

        circuit = self.build_market_circuit(oil_shift)

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        state_vector = result.final_state_vector
        qubit_map = {q: i for i, q in enumerate(self.qubits)}

        observable = cirq.Z(self.qubits[0])

        exp = observable.expectation_from_state_vector(
            state_vector=state_vector,
            qubit_map=qubit_map
        ).real

        return exp

    # --------------------------------------------------
    # 5️⃣ Inflection Detector
    # --------------------------------------------------
    def detect_inflection(self):

        shifts = np.linspace(-1.0, 1.0, 41)
        values = []

        for s in shifts:
            exp = self.compute_spx_expectation(oil_shift=s)
            values.append(exp)

        values = np.array(values)

        # Numerical second derivative
        second_derivative = np.gradient(np.gradient(values))

        inflection_points = []

        for i in range(1, len(second_derivative)):
            if second_derivative[i-1] * second_derivative[i] < 0:
                inflection_points.append(float(shifts[i]))

        return shifts, values, inflection_points


if __name__ == "__main__":

    engine = BloombergMarketsInflection()

    shifts, values, inflections = engine.detect_inflection()

    print("\n=== SP500 Nonlinear Expectation Curve ===")
    for s, v in zip(shifts, values):
        print(f"Oil Shift {round(s,2)} → SPX Expectation {round(v,6)}")

    print("\n=== Detected Inflection Points ===")
    print(inflections if inflections else "No inflection detected")
