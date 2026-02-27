import numpy as np
from scipy.linalg import expm
import json


class BloombergMarketsQuantumSimulator:

    def __init__(self):

        # Qubits:
        # 0 = SPX
        # 1 = NASDAQ
        # 2 = Nikkei
        # 3 = Oil
        # 4 = Gold
        # 5 = VIX

        self.n_qubits = 6

        # Coupling strengths
        self.J_spx_nasdaq = 1.2
        self.J_spx_vix = 1.3
        self.J_spx_oil = 0.8
        self.J_spx_gold = 0.7

        # SPX bullish targeting bias
        self.target_bias = 0.4

        # Base local fields
        self.h = np.array([-0.5, -0.8, 0.2, 0.3, 0.9, 0.4])


    # ==================================================
    # HAMILTONIAN (Diagonal + Transverse Instability)
    # ==================================================
    def build_hamiltonian(self,
                          oil_shift=0,
                          gold_shift=0,
                          nasdaq_shift=0,
                          vix_shift=0):

        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=np.complex128)

        # Vol-dependent instability strength
        gamma_base = 0.4
        gamma = gamma_base + 0.5 * abs(vix_shift)

        for i in range(dim):

            bits = format(i, f"0{self.n_qubits}b")
            z = np.array([1 if b == "0" else -1 for b in bits])

            # ----- Diagonal terms -----
            local = np.sum(self.h * z)
            local += self.target_bias * z[0]

            local += oil_shift * z[3]
            local += gold_shift * z[4]
            local += nasdaq_shift * z[1]
            local += vix_shift * z[5]

            interaction = (
                self.J_spx_nasdaq * z[0] * z[1] +
                self.J_spx_vix * z[0] * z[5] +
                self.J_spx_oil * z[0] * z[3] +
                self.J_spx_gold * z[0] * z[4]
            )

            H[i, i] = local + interaction

        # ----- Transverse X-flip terms -----
        for i in range(dim):
            bits = format(i, f"0{self.n_qubits}b")

            for q in range(self.n_qubits):
                flipped = list(bits)
                flipped[q] = "1" if bits[q] == "0" else "0"
                j = int("".join(flipped), 2)

                H[i, j] += gamma

        # Ensure strict Hermitian symmetry
        H = (H + H.conj().T) / 2

        return H


    # ==================================================
    # SPECTRUM
    # ==================================================
    def compute_spectrum(self, **kwargs):

        H = self.build_hamiltonian(**kwargs)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        gap = eigenvalues[1] - eigenvalues[0]

        return eigenvalues, eigenvectors, float(gap)


    # ==================================================
    # TIME EVOLUTION
    # ==================================================
    def forward_evolution(self, steps=40, dt=0.07, **kwargs):

        H = self.build_hamiltonian(**kwargs)
        dim = H.shape[0]

        psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        U = expm(-1j * H * dt)

        trajectory = []

        for _ in range(steps):

            psi = U @ psi

            # Normalize for safety
            psi = psi / np.linalg.norm(psi)

            spx_expectation = 0

            for i in range(dim):
                bits = format(i, f"0{self.n_qubits}b")
                z_spx = 1 if bits[0] == "0" else -1
                spx_expectation += z_spx * np.abs(psi[i]) ** 2

            trajectory.append(float(spx_expectation.real))

        return np.array(trajectory)


    # ==================================================
    # SPECTRAL GAP VS VOL
    # ==================================================
    def spectral_gap_curve(self, nasdaq_shift=-0.6):

        vix_range = np.linspace(0, 1.5, 40)
        gaps = []

        for v in vix_range:
            _, _, gap = self.compute_spectrum(
                nasdaq_shift=nasdaq_shift,
                vix_shift=v
            )
            gaps.append(gap)

        return vix_range, np.array(gaps)


    # ==================================================
    # CRISIS DETECTION
    # ==================================================
    def detect_crisis(self, nasdaq_shift=-0.6):

        vix_vals, gaps = self.spectral_gap_curve(nasdaq_shift)

        threshold = 0.05
        critical = vix_vals[gaps < threshold]

        return float(critical[0]) if len(critical) > 0 else None


    # ==================================================
    # REGIME ROTATION SPEED
    # ==================================================
    def regime_rotation_speed(self):

        _, vec1, _ = self.compute_spectrum(nasdaq_shift=-0.2)
        _, vec2, _ = self.compute_spectrum(nasdaq_shift=-0.8)

        overlap = np.abs(np.dot(vec1[:, 0].conj(), vec2[:, 0]))

        return float(1 - overlap)


    # ==================================================
    # MONTE CARLO SHOCK ENGINE
    # ==================================================
    def monte_carlo(self, runs=300):

        results = []

        for _ in range(runs):

            nasdaq = np.random.normal(-0.4, 0.3)
            vix = np.random.normal(0.6, 0.25)

            traj = self.forward_evolution(
                nasdaq_shift=nasdaq,
                vix_shift=vix
            )

            results.append(traj[-1])

        results = np.array(results)

        return {
            "MeanFinalBias": float(results.mean()),
            "StdFinalBias": float(results.std()),
            "BullishProbability": float(np.mean(results > 0))
        }


    # ==================================================
    # DASHBOARD OUTPUT
    # ==================================================
    def dashboard(self):

        eigenvalues, _, gap = self.compute_spectrum(
            nasdaq_shift=-0.6,
            vix_shift=0.6
        )

        trajectory = self.forward_evolution(
            nasdaq_shift=-0.6,
            vix_shift=0.6
        )

        crisis = self.detect_crisis()
        rotation = self.regime_rotation_speed()
        mc = self.monte_carlo(400)

        report = {
            "SpectralGap": gap,
            "CrisisThresholdVol": crisis,
            "RegimeRotationSpeed": rotation,
            "FinalSPXBias": float(trajectory[-1]),
            "MonteCarlo": mc,
            "TargetBias": self.target_bias
        }

        return json.dumps(report, indent=2)


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":

    engine = BloombergMarketsQuantumSimulator()

    print("\n=== LIVE DASHBOARD OUTPUT ===\n")
    print(engine.dashboard())
