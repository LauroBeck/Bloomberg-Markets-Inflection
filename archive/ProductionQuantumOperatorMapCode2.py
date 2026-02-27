import cirq
import numpy as np


class QuantumMarketOperatorMap:

    def __init__(self):
        self.qubits = cirq.LineQubit.range(10)
        self.circuit = cirq.Circuit()

    # ----------------------------------
    # 1️⃣ Build Operator Layer
    # ----------------------------------
    def build_operator_layer(self):

        values = [
            2677.88 / 5000 * np.pi,
            (0.0055 + 0.05) / 0.10 * np.pi,
            4500 / 5000 * np.pi,
            14000 / 20000 * np.pi,
            4000 / 5000 * np.pi
        ]

        for i, angle in enumerate(values):
            self.circuit.append(cirq.ry(angle).on(self.qubits[i]))

        flags = [1, 0, 1, 1, 1]
        for i, flag in enumerate(flags):
            self.circuit.append(cirq.ry(flag * np.pi).on(self.qubits[5 + i]))

        self.circuit.append(cirq.CNOT(self.qubits[0], self.qubits[2]))
        self.circuit.append(cirq.CNOT(self.qubits[2], self.qubits[3]))

        return self.circuit

    # ----------------------------------
    # 2️⃣ Compute Expectation Values
    # ----------------------------------
    def compute_expectations(self):

        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)

        state_vector = result.final_state_vector
        expectations = {}

        qubit_map = {q: i for i, q in enumerate(self.qubits)}

        for i, q in enumerate(self.qubits):
            observable = cirq.Z(q)

            exp = observable.expectation_from_state_vector(
                state_vector=state_vector,
                qubit_map=qubit_map
            ).real

            expectations[f"Qubit_{i}"] = exp

        return expectations

    # ----------------------------------
    # 3️⃣ Compute Correlations
    # ----------------------------------
    def compute_correlations(self):

        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)

        state_vector = result.final_state_vector
        qubit_map = {q: i for i, q in enumerate(self.qubits)}

        correlations = {}
        pairs = [(0, 2), (2, 3)]

        for i, j in pairs:
            observable = cirq.Z(self.qubits[i]) * cirq.Z(self.qubits[j])

            exp = observable.expectation_from_state_vector(
                state_vector=state_vector,
                qubit_map=qubit_map
            ).real

            correlations[f"Z{i}Z{j}"] = exp

        return correlations


if __name__ == "__main__":

    engine = QuantumMarketOperatorMap()

    engine.build_operator_layer()

    expectations = engine.compute_expectations()
    correlations = engine.compute_correlations()

    print("\nSingle Qubit Expectations:")
    for k, v in expectations.items():
        print(f"{k}: {round(v, 6)}")

    print("\nCorrelation Operators:")
    for k, v in correlations.items():
        print(f"{k}: {round(v, 6)}")
