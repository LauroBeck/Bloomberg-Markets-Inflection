import cirq
import numpy as np
from scipy.linalg import eigvals


class BloombergMarketsQuantumSimulatorV3:

    def __init__(self):

        # Qubits
        # Q0 = SPX
        # Q1 = NASDAQ
        # Q2 = Nikkei
        # Q3 = Oil
        # Q4 = Gold
        self.n_qubits = 5
        self.qubits = cirq.LineQubit.range(self.n_qubits)

        # --- Base Couplings ---
        self.J_spx_oil = 0.8
        self.J_spx_gold = 0.7

        # NEW: SPX–NASDAQ entanglement
        self.J_spx_nasdaq = 1.2

        # Sector weights (tech heavy Nasdaq)
        self.sector_weights = {
            "spx": 1.0,
            "nasdaq": 1.4,   # Tech beta amplification
            "nikkei": 0.6,
            "oil": 0.5,
            "gold": 0.5
        }

        # Local bias fields
        self.h = np.array([-0.58, -0.87, 0.16, 0.30, 0.94]) * np.pi / 8


    # --------------------------------------------------
    # Volatility-Dependent Coupling
    # --------------------------------------------------
    def adjust_couplings_for_volatility(self, volatility):

        # Volatility amplifies systemic correlations
        scale = 1 + 2 * volatility

        self.J_spx_nasdaq_eff = self.J_spx_nasdaq * scale
        self.J_spx_oil_eff = self.J_spx_oil * (1 + volatility)
        self.J_spx_gold_eff = self.J_spx_gold * (1 + volatility)


    # --------------------------------------------------
    # Hamiltonian Matrix (Sector Weighted)
    # --------------------------------------------------
    def build_hamiltonian_matrix(
        self,
        oil_shift=0.0,
        gold_shift=0.0,
        nasdaq_shift=0.0,
        volatility=0.0
    ):

        self.adjust_couplings_for_volatility(volatility)

        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            bitstring = format(i, f"0{self.n_qubits}b")
            z = np.array([1 if b == "0" else -1 for b in bitstring])

            # --- Local fields with sector weights ---
            local = np.sum(self.h * z)

            local += oil_shift * self.sector_weights["oil"] * z[3]
            local += gold_shift * self.sector_weights["gold"] * z[4]
            local += nasdaq_shift * self.sector_weights["nasdaq"] * z[1]

            # --- Interactions ---
            interaction = (
                self.J_spx_oil_eff * z[0] * z[3] +
                self.J_spx_gold_eff * z[0] * z[4] +
                self.J_spx_nasdaq_eff * z[0] * z[1]   # NEW ENTANGLEMENT
            )

            H[i, i] = local + interaction

        return H


    # --------------------------------------------------
    # Eigen Spectrum + Gap
    # --------------------------------------------------
    def compute_spectrum(self, **kwargs):

        H = self.build_hamiltonian_matrix(**kwargs)
        eigenvalues = np.sort(np.real(eigvals(H)))

        gap = eigenvalues[1] - eigenvalues[0]

        return eigenvalues, gap


    # --------------------------------------------------
    # SPX Expectation via Ground State
    # --------------------------------------------------
    def compute_spx_ground_expectation(self, **kwargs):

        H = self.build_hamiltonian_matrix(**kwargs)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        ground_state = eigenvectors[:, 0]

        expectation = 0
        for i in range(len(ground_state)):
            bitstring = format(i, f"0{self.n_qubits}b")
            z_spx = 1 if bitstring[0] == "0" else -1
            prob = np.abs(ground_state[i]) ** 2
            expectation += z_spx * prob

        return expectation.real


    # --------------------------------------------------
    # Regime Classifier
    # --------------------------------------------------
    def classify_regime(self, oil_shift, gold_shift, nasdaq_shift, volatility):

        eigenvalues, gap = self.compute_spectrum(
            oil_shift=oil_shift,
            gold_shift=gold_shift,
            nasdaq_shift=nasdaq_shift,
            volatility=volatility
        )

        spx_bias = self.compute_spx_ground_expectation(
            oil_shift=oil_shift,
            gold_shift=gold_shift,
            nasdaq_shift=nasdaq_shift,
            volatility=volatility
        )

        # --- Regime Logic ---

        if gap < 0.05:
            regime = "Systemic Risk / Phase Transition"

        elif nasdaq_shift < -0.5 and volatility > 0.3:
            regime = "AI Bubble Compression"

        elif oil_shift < -0.5:
            regime = "Commodity Shock"

        else:
            regime = "Stable / Neutral"

        return {
            "regime": regime,
            "spectral_gap": gap,
            "spx_bias": spx_bias
        }


# --------------------------------------------------
# RUN EXAMPLE
# --------------------------------------------------

if __name__ == "__main__":

    engine = BloombergMarketsQuantumSimulatorV3()

    # Simulate AI selloff scenario
    result = engine.classify_regime(
        oil_shift=0.0,
        gold_shift=0.0,
        nasdaq_shift=-0.8,   # tech shock
        volatility=0.4      # elevated vol
    )

    print("\n=== Regime Classification ===")
    print(result)
