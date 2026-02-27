import numpy as np
from scipy.linalg import eigvals


class BloombergMarketsQuantumSimulatorV4:

    def __init__(self):

        # Q0 = SPX
        # Q1 = NASDAQ
        # Q2 = Nikkei
        # Q3 = Oil
        # Q4 = Gold
        # Q5 = VIX (NEW)
        self.n_qubits = 6

        # Base couplings
        self.J_spx_nasdaq = 1.2
        self.J_spx_oil = 0.8
        self.J_spx_gold = 0.7
        self.J_spx_vix = 1.5  # NEW: SPX–Volatility entanglement

        # Local bias
        self.h = np.array([-0.58, -0.87, 0.16, 0.30, 0.94, 0.5]) * np.pi / 8


    # --------------------------------------------------
    # Build Hamiltonian (Now Includes VIX Qubit)
    # --------------------------------------------------
    def build_hamiltonian(self,
                          oil_shift=0,
                          gold_shift=0,
                          nasdaq_shift=0,
                          vix_shift=0):

        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=float)

        for i in range(dim):
            bits = format(i, f"0{self.n_qubits}b")
            z = np.array([1 if b == "0" else -1 for b in bits])

            local = np.sum(self.h * z)

            # External Shifts
            local += oil_shift * z[3]
            local += gold_shift * z[4]
            local += nasdaq_shift * z[1]
            local += vix_shift * z[5]

            # Interactions
            interaction = (
                self.J_spx_nasdaq * z[0] * z[1] +
                self.J_spx_oil * z[0] * z[3] +
                self.J_spx_gold * z[0] * z[4] +
                self.J_spx_vix * z[0] * z[5]
            )

            H[i, i] = local + interaction

        return H


    # --------------------------------------------------
    # Spectrum + Ground State
    # --------------------------------------------------
    def compute_spectrum(self, **kwargs):

        H = self.build_hamiltonian(**kwargs)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        gap = eigenvalues[1] - eigenvalues[0]

        return eigenvalues, eigenvectors, gap


    # --------------------------------------------------
    # Spectral Gap vs Volatility Curve
    # --------------------------------------------------
    def spectral_gap_curve(self, nasdaq_shift=-0.8):

        vol_values = np.linspace(0, 1.5, 30)
        gaps = []

        for v in vol_values:
            _, _, gap = self.compute_spectrum(
                nasdaq_shift=nasdaq_shift,
                vix_shift=v
            )
            gaps.append(gap)

        return vol_values, np.array(gaps)


    # --------------------------------------------------
    # Automatic Crisis Detection
    # --------------------------------------------------
    def detect_crisis_threshold(self, nasdaq_shift=-0.8):

        vol_values, gaps = self.spectral_gap_curve(nasdaq_shift)

        threshold = 0.03  # Critical gap compression

        crisis_points = vol_values[gaps < threshold]

        if len(crisis_points) > 0:
            return float(crisis_points[0])
        else:
            return None


    # --------------------------------------------------
    # Eigenvector Overlap (Regime Rotation Speed)
    # --------------------------------------------------
    def regime_rotation_speed(self, shift_start=-0.2, shift_end=-0.8):

        _, vecs1, _ = self.compute_spectrum(nasdaq_shift=shift_start)
        _, vecs2, _ = self.compute_spectrum(nasdaq_shift=shift_end)

        ground1 = vecs1[:, 0]
        ground2 = vecs2[:, 0]

        overlap = np.abs(np.dot(np.conjugate(ground1), ground2))

        rotation_speed = 1 - overlap  # Larger = faster regime change

        return rotation_speed


    # --------------------------------------------------
    # Probability-Weighted Regime Output
    # --------------------------------------------------
    def probabilistic_regime(self,
                             oil_shift=0,
                             gold_shift=0,
                             nasdaq_shift=0,
                             vix_shift=0):

        eigenvalues, _, gap = self.compute_spectrum(
            oil_shift=oil_shift,
            gold_shift=gold_shift,
            nasdaq_shift=nasdaq_shift,
            vix_shift=vix_shift
        )

        # Softmax probability from energy levels
        beta = 5  # inverse temperature (confidence)
        probs = np.exp(-beta * eigenvalues)
        probs /= np.sum(probs)

        systemic_prob = float(np.sum(probs[:2]))  # weight of lowest 2 states

        if gap < 0.03:
            crisis_prob = min(1.0, 0.8 + systemic_prob)
        else:
            crisis_prob = systemic_prob

        ai_prob = float(abs(nasdaq_shift) / 2)
        commodity_prob = float(abs(oil_shift) / 2)

        neutral_prob = max(0.0, 1 - crisis_prob - ai_prob - commodity_prob)

        return {
            "Systemic Risk Probability": round(crisis_prob, 3),
            "AI Bubble Compression Probability": round(ai_prob, 3),
            "Commodity Shock Probability": round(commodity_prob, 3),
            "Neutral Probability": round(neutral_prob, 3),
            "Spectral Gap": round(float(gap), 6)
        }


# --------------------------------------------------
# Example Run
# --------------------------------------------------

if __name__ == "__main__":

    engine = BloombergMarketsQuantumSimulatorV4()

    print("\n=== Spectral Gap vs Volatility ===")
    vol, gaps = engine.spectral_gap_curve()
    print("Min Gap:", gaps.min())

    print("\n=== Crisis Threshold Detection ===")
    print("Critical Vol Level:", engine.detect_crisis_threshold())

    print("\n=== Regime Rotation Speed ===")
    print("Rotation Metric:", engine.regime_rotation_speed())

    print("\n=== Probabilistic Regime Output ===")
    print(engine.probabilistic_regime(
        nasdaq_shift=-0.8,
        vix_shift=0.6
    ))
