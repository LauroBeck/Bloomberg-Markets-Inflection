import cirq
import numpy as np
from scipy.linalg import eigvals


class BloombergMarketsQuantumSimulator:

    def __init__(self):
        # Q0 = SPX
        # Q1 = NASDAQ
        # Q2 = Nikkei
        # Q3 = Oil
        # Q4 = Gold
        self.n_qubits = 5
        self.qubits = cirq.LineQubit.range(self.n_qubits)

        # Coupling strengths (J terms)
        self.J_spx_oil = 0.8
        self.J_spx_gold = 0.7

        # Local bias (h terms)
        self.h = np.array([-0.58, -0.87, 0.16, 0.30, 0.94]) * np.pi / 8

    # --------------------------------------------------
    # 1️⃣ Build Effective Hamiltonian Matrix
    # --------------------------------------------------
    def build_hamiltonian_matrix(self, oil_shift=0.0, gold_shift=0.0):

        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            bitstring = format(i, f"0{self.n_qubits}b")
            z = np.array([1 if b == "0" else -1 for b in bitstring])

            # Local field contribution
            local = np.sum(self.h * z)

            # Interaction contribution
            interaction = (
                self.J_spx_oil * z[0] * z[3] +
                self.J_spx_gold * z[0] * z[4]
            )

            H[i, i] = local + interaction

        return H

    # --------------------------------------------------
    # 2️⃣ Multi-Step Trotter Time Evolution
    # --------------------------------------------------
    def trotter_evolution(self, oil_shift=0.0, gold_shift=0.0, steps=10, dt=0.2):

        circuit = cirq.Circuit()

        # Initial superposition on SPX
        circuit.append(cirq.H(self.qubits[0]))

        for _ in range(steps):

            # Local rotations
            for i, angle in enumerate(self.h):
                circuit.append(cirq.rz(angle * dt).on(self.qubits[i]))

            # ZZ couplings
            circuit.append(
                cirq.ZZ(self.qubits[0], self.qubits[3]) ** (self.J_spx_oil * dt)
            )
            circuit.append(
                cirq.ZZ(self.qubits[0], self.qubits[4]) ** (self.J_spx_gold * dt)
            )

        return circuit

    # --------------------------------------------------
    # 3️⃣ Noise Channel (Volatility)
    # --------------------------------------------------
    def add_decoherence(self, circuit, gamma=0.05):

        noisy_circuit = cirq.Circuit()

        for moment in circuit:
            noisy_circuit.append(moment)

            for q in self.qubits:
                noisy_circuit.append(cirq.depolarize(gamma).on(q))

        return noisy_circuit

    # --------------------------------------------------
    # 4️⃣ Compute SPX Expectation
    # --------------------------------------------------
    def compute_spx_expectation(self, oil_shift=0.0, gold_shift=0.0, noise=0.0):

        circuit = self.trotter_evolution(oil_shift, gold_shift)

        if noise > 0:
            circuit = self.add_decoherence(circuit, gamma=noise)

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        state = result.final_state_vector
        qubit_map = {q: i for i, q in enumerate(self.qubits)}

        observable = cirq.Z(self.qubits[0])

        return observable.expectation_from_state_vector(
            state_vector=state,
            qubit_map=qubit_map
        ).real

    # --------------------------------------------------
    # 5️⃣ 2D Oil × Gold Surface
    # --------------------------------------------------
    def compute_phase_surface(self):

        oil_vals = np.linspace(-1, 1, 15)
        gold_vals = np.linspace(-1, 1, 15)

        surface = np.zeros((len(oil_vals), len(gold_vals)))

        for i, o in enumerate(oil_vals):
            for j, g in enumerate(gold_vals):
                surface[i, j] = self.compute_spx_expectation(
                    oil_shift=o,
                    gold_shift=g
                )

        return oil_vals, gold_vals, surface

    # --------------------------------------------------
    # 6️⃣ Eigenvalue Spectrum
    # --------------------------------------------------
    def compute_eigen_spectrum(self):

        H = self.build_hamiltonian_matrix()
        eigenvalues = np.sort(np.real(eigvals(H)))

        return eigenvalues


if __name__ == "__main__":

    engine = BloombergMarketsQuantumSimulator()

    print("\n=== SPX Expectation (No Noise) ===")
    print(engine.compute_spx_expectation())

    print("\n=== SPX Expectation (With Volatility Noise) ===")
    print(engine.compute_spx_expectation(noise=0.1))

    print("\n=== Hamiltonian Eigenvalue Spectrum ===")
    print(engine.compute_eigen_spectrum())

    print("\nComputing 2D Phase Surface...")
    oil, gold, surface = engine.compute_phase_surface()
    print("Surface shape:", surface.shape)
