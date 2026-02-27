# 10qubits1.py
import cirq
import numpy as np


class BloombergQuantumEncoder:

    def __init__(self):
        self.qubits = cirq.LineQubit.range(10)
        self.circuit = cirq.Circuit()

    def encode_market_data(self):
        # --- Russell 2000 ---
        russell_index_value = 2677.88
        normalized_russell = russell_index_value / 5000 * np.pi
        self.circuit.append(cirq.ry(normalized_russell).on(self.qubits[0]))

        russell_percent_change = 0.55 / 100
        normalized_change = (russell_percent_change + 0.05) / 0.10 * np.pi
        self.circuit.append(cirq.ry(normalized_change).on(self.qubits[1]))

        # --- S&P 500 ---
        sp500_value = 4500
        self.circuit.append(cirq.ry(sp500_value / 5000 * np.pi).on(self.qubits[2]))

        # --- Nasdaq ---
        nasdaq_value = 14000
        self.circuit.append(cirq.ry(nasdaq_value / 20000 * np.pi).on(self.qubits[3]))

        # --- Euro Stoxx ---
        euro_stoxx_value = 4000
        self.circuit.append(cirq.ry(euro_stoxx_value / 5000 * np.pi).on(self.qubits[4]))

        # --- Binary Flags ---
        flags = [1, 0, 1, 1, 1]  # gas, oil, fraud, economic, guest

        for i, flag in enumerate(flags):
            self.circuit.append(cirq.ry(flag * np.pi).on(self.qubits[5 + i]))

        self.circuit.append(cirq.measure(*self.qubits, key='m'))

        return self.circuit

    def simulate(self, repetitions=1000):
        simulator = cirq.Simulator()
        results = simulator.run(self.circuit, repetitions=repetitions)
        return results


def analyze_results(results):
    measurements = results.measurements['m']
    shots = len(measurements)

    print("\nMeasurement Proportions:")
    for i in range(measurements.shape[1]):
        proportion = np.sum(measurements[:, i]) / shots
        print(f"Qubit {i}: {proportion:.4f}")


if __name__ == "__main__":
    encoder = BloombergQuantumEncoder()
    circuit = encoder.encode_market_data()

    print("Quantum Circuit:")
    print(circuit)

    results = encoder.simulate()
    analyze_results(results)
