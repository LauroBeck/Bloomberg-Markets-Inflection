import cirq
import numpy as np


class BloombergQuantumDashboard:

    def __init__(self):
        self.n_qubits = 16
        self.qubits = cirq.LineQubit.range(self.n_qubits)
        self.circuit = cirq.Circuit()

    # --------------------------------------------------
    # 1️⃣ Encode Dashboard Tiles into Rotations
    # --------------------------------------------------
    def encode_dashboard(self):

        # Percent moves from screenshot (approx normalized)
        percent_moves = [
            -0.60,  # CSI 300
            -0.43,  # Shanghai
            -0.70,  # Shenzhen
            -0.05,  # China 10Y
            0.003,  # China 30Y
            -0.41,  # Copper
            -0.33,  # Steel
            -0.13,  # Iron Ore
            -1.05,  # Crude
            0.15,   # Moutai
            0.32,   # CATL
            0.16,   # Ping An
            0.10,   # Merchants Bank
            -0.32,  # Midea
            -2.03,  # Cambricon
            -0.50   # Risk sentiment placeholder
        ]

        # Normalize to rotation angles
        for i, move in enumerate(percent_moves):
            angle = move * np.pi  # scale to π
            self.circuit.append(cirq.ry(angle).on(self.qubits[i]))

        return self.circuit

    # --------------------------------------------------
    # 2️⃣ Sector Entanglement Layer
    # --------------------------------------------------
    def entangle_sectors(self):

        # Indices entangled
        self.circuit.append(cirq.CNOT(self.qubits[0], self.qubits[1]))
        self.circuit.append(cirq.CNOT(self.qubits[1], self.qubits[2]))

        # Commodities entangled
        self.circuit.append(cirq.CNOT(self.qubits[5], self.qubits[6]))
        self.circuit.append(cirq.CNOT(self.qubits[6], self.qubits[7]))
        self.circuit.append(cirq.CNOT(self.qubits[7], self.qubits[8]))

        # Equity cluster entanglement
        self.circuit.append(cirq.CNOT(self.qubits[9], self.qubits[10]))
        self.circuit.append(cirq.CNOT(self.qubits[10], self.qubits[11]))
        self.circuit.append(cirq.CNOT(self.qubits[11], self.qubits[12]))
        self.circuit.append(cirq.CNOT(self.qubits[12], self.qubits[13]))
        self.circuit.append(cirq.CNOT(self.qubits[13], self.qubits[14]))

        return self.circuit

    # --------------------------------------------------
    # 3️⃣ Expectation Map
    # --------------------------------------------------
    def compute_expectations(self):

        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit)

        state_vector = result.final_state_vector
        qubit_map = {q: i for i, q in enumerate(self.qubits)}

        expectations = {}

        for i, q in enumerate(self.qubits):
            observable = cirq.Z(q)

            exp = observable.expectation_from_state_vector(
                state_vector=state_vector,
                qubit_map=qubit_map
            ).real

            expectations[f"Tile_{i}"] = exp

        return expectations


if __name__ == "__main__":

    dashboard = BloombergQuantumDashboard()

    dashboard.encode_dashboard()
    dashboard.entangle_sectors()

    expectations = dashboard.compute_expectations()

    print("\n=== Bloomberg Quantum Dashboard State ===")
    for k, v in expectations.items():
        print(f"{k}: {round(v,6)}")
