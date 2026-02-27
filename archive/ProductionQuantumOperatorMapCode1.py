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
            2677.88 / 5000 * np.pi,                # Russell level
            (0.0055 + 0.05) / 0.10 * np.pi,        # Russell % change
            4500 / 5000 * np.pi,                   # S&P 500
            14000 / 20000 * np.pi,                 # Nasdaq
            4000 / 5000 * np.pi                    # Euro Stoxx
        ]

        for i, angle in enumerate(values):
            self.circuit.append(cirq.ry(angle).on(self.qubits[i]))

        flags = [1, 0, 1, 1, 1]  # gas, oil, fraud, economic, guest
        for i, flag in enumerate(flags):
            self.circuit.append(cirq.ry(flag * np.pi).on(self.qubits[5 + i]))

        # Market correlation entanglement
        self.circuit.append(cirq.CNOT(self.qubits[0], self.qubits[2]))
        self.circuit.append(cirq.CNOT(self.qubits[2], self.qubits[3]))

        return self.circuit

    # ----------------------------------
    # 2️⃣ Compute Expectation Values (UPDATED)
    # ----------------------------------
    def compute_expectations(self):

        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)

        state_vector = result.final_state_vector

        expectations = {}
        qubit_map = {q: i for i, q in enumerate(self.qubits)}

        for i, q in enumerate(self.qubits):
            z_op = cirq.Z(q)

            exp = cirq.expectation_from_state_vector(
                state_vector,
                z_op,
                qubit_map=qubit_map
            ).real

            expectations[f"Qubit_{i}"] = exp

        return expectations

    # ----------------------------------
    # 3️⃣ Compute Correlation Operators ⟨Zi Zj⟩
    # ----------------------------------
    def compute_correlations(self):

        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)

        state_vector = result.final_state_vector
        qubit_map = {q: i for i, q in enumerate(self.qubits)}

        correlations = {}

        # Example: core market correlation block
        pairs = [(0, 2), (2, 3)]  # Russell-S&P, S&P-Nasdaq

        for i, j in pairs:
            zz_op = cirq.Z(self.qubits[i]) * cirq.Z(self.qubits[j])

            exp = cirq.expectation_from_state_vector(
                state_vector,
                zz_op,
                qubit_map=qubit_map
            ).real

            correlations[f"Z{i}Z{j}"] = exp

        return correlations

    # ----------------------------------
    # 4️⃣ Financial Signal Mapping
    # ----------------------------------
    def map_financial_signals(self, expectations):

        mapping = {
            0: "Russell 2000 Level",
            1: "Russell Momentum",
            2: "S&P 500",
            3: "Nasdaq",
            4: "Euro Stoxx",
            5: "Natural Gas",
            6: "Heating Oil",
            7: "Fraud Risk",
            8: "Economic Stress",
            9: "Geopolitical Signal"
        }

        interpreted = {}

        for key, value in expectations.items():
            idx = int(key.split("_")[1])
            interpreted[mapping[idx]] = {
                "Z_expectation": round(value, 6),
                "Signal_strength": round((1 - value) / 2, 6)
            }

        return interpreted


# ----------------------------------
# Execute
# ----------------------------------
if __name__ == "__main__":

    engine = QuantumMarketOperatorMap()

    engine.build_operator_layer()


   def compute_expectations(self):

    simulator = cirq.Simulator()
    result = simulator.simulate(self.circuit)

    state_vector = result.final_state_vector
    expectations = {}

    # Full qubit map
    qubit_map = {q: i for i, q in enumerate(self.qubits)}

    for i, q in enumerate(self.qubits):
        observable = cirq.Z(q)

        exp = observable.expectation_from_state_vector(
            state_vector=state_vector,
            qubit_map=qubit_map
        ).real

        expectations[f"Qubit_{i}"] = exp

    return expectations

    print("\nQuantum Operator Map Results:\n")
    for k, v in result_map.items():
        print(f"{k}: {v}")

    print("\nMarket Correlation Operators ⟨ZiZj⟩:\n")
    for k, v in correlations.items():
        print(f"{k}: {round(v, 6)}")
