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

        # Continuous market signals
        values = [
            2677.88 / 5000 * np.pi,   # Russell level
            (0.0055 + 0.05) / 0.10 * np.pi,  # Russell % change
            4500 / 5000 * np.pi,      # S&P 500
            14000 / 20000 * np.pi,    # Nasdaq
            4000 / 5000 * np.pi       # Euro Stoxx
        ]

        for i, angle in enumerate(values):
            self.circuit.append(cirq.ry(angle).on(self.qubits[i]))

        # Binary macro flags
        flags = [1, 0, 1, 1, 1]
        for i, flag in enumerate(flags):
            self.circuit.append(cirq.ry(flag * np.pi).on(self.qubits[5+i]))

        # Add correlation entanglement
        self.circuit.append(cirq.CNOT(self.qubits[0], self.qubits[2]))  # Russell → S&P
        self.circuit.append(cirq.CNOT(self.qubits[2], self.qubits[3]))  # S&P → Nasdaq

        return self.circuit

    # ----------------------------------
    # 2️⃣ Compute Expectation Values
    # ----------------------------------
    def compute_expectations(self):

        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)

        expectations = {}

        for i, q in enumerate(self.qubits):
            z_op = cirq.Z(q)
            exp = result.expectation_from_state_vector(
                result.final_state_vector,
                observables=z_op,
                qubit_map={q: i}
            ).real
            expectations[f"Qubit_{i}"] = exp

        return expectations

    # ----------------------------------
    # 3️⃣ Financial Signal Mapping
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

        for i, value in expectations.items():
            idx = int(i.split("_")[1])
            interpreted[mapping[idx]] = {
                "Z_expectation": round(value, 4),
                "Signal_strength": round((1 - value)/2, 4)
            }

        return interpreted


# -----------------------------
# Execute
# -----------------------------
if __name__ == "__main__":

    engine = QuantumMarketOperatorMap()

    engine.build_operator_layer()

    expectations = engine.compute_expectations()

    result_map = engine.map_financial_signals(expectations)

    print("\nQuantum Operator Map Results:\n")
    for k, v in result_map.items():
        print(f"{k}: {v}")
