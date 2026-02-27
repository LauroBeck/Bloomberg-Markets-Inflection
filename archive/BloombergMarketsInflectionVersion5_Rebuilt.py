import numpy as np
from scipy.linalg import expm
import json


class BloombergMarketsQuantumSimulator:

    def __init__(self):

        # Q0=SPX, Q1=NASDAQ, Q2=Nikkei, Q3=Oil, Q4=Gold, Q5=VIX
        self.n_qubits = 6

        # Couplings
        self.J_spx_nasdaq = 1.2
        self.J_spx_vix = 1.3
        self.J_spx_oil = 0.8
        self.J_spx_gold = 0.7

        # Target SPX positive bias
        self.target_bias = 0.4

        # Local fields
        self.h = np.array([-0.5, -0.8, 0.2, 0.3, 0.9, 0.4])


    # --------------------------------------------------
    # Hamiltonian
    # --------------------------------------------------
    def build_hamiltonian(self,
                          oil_shift=0,
                          gold_shift=0,
                          nasdaq_shift=0,
                          vix_shift=0):

        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim))

        for i in range(dim):
            bits = format(i, f"0{self.n_qubits}b")
            z = np.array([1 if b == "0" else -1 for b in bits])

            # Local field + bullish SPX targeting
            local = np.sum(self.h * z)
            local += self.target_bias * z[0]

            # External macro shifts
            local += oil_shift * z[3]
            local += gold_shift * z[4]
            local += nasdaq_shift * z[1]
            local += vix_shift * z[5]

            # Entanglements
            interaction = (
                self.J_spx_nasdaq * z[0] * z[1] +
                self.J_spx_vix * z[0] * z[5] +
                self.J_spx_oil * z[0] * z[3] +
                self.J_spx_gold * z[0] * z[4]
            )

            H[i, i] = local + interaction

        return H


    # --------------------------------------------------
    # Spectrum + Gap
    # --------------------------------------------------
    def compute_spectrum(self, **kwargs):

        H = self.build_hamiltonian(**kwargs)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        gap = eigenvalues[1] - eigenvalues[0]

        return eigenvalues, eigenvectors, gap


    # --------------------------------------------------
    # Forward Time Evolution
    # --------------------------------------------------
    def forward_evolution(self, steps=30, dt=0.08, **kwargs):

        H = self.build_hamiltonian(**kwargs)

        dim = H.shape[0]
        psi = np.ones(dim) / np.sqrt(dim)

        U = expm(-1j * H * dt)

        trajectory = []

        for _ in range(steps):
            psi = U @ psi

            spx_exp = 0
            for i in range(dim):
                bits = format(i, f"0{self.n_qubits}b")
                z_spx = 1 if bits[0] == "0" else -1
                spx_exp += z_spx * np.abs(psi[i])**2

            trajectory.append(spx_exp.real)

        return np.array(trajectory)


    # --------------------------------------------------
    # Spectral Gap vs Volatility
    # --------------------------------------------------
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


    # --------------------------------------------------
    # Crisis Detection
    # --------------------------------------------------
    def detect_crisis(self, nasdaq_shift=-0.6):

        vix_vals, gaps = self.spectral_gap_curve(nasdaq_shift)

        threshold = 0.03
        crisis_points = vix_vals[gaps < threshold]

        if len(crisis_points) > 0:
            return float(crisis_points[0])
        else:
            return None


    # --------------------------------------------------
    # Regime Rotation Speed
    # --------------------------------------------------
    def regime_rotation_speed(self):

        _, vec1, _ = self.compute_spectrum(nasdaq_shift=-0.2)
        _, vec2, _ = self.compute_spectrum(nasdaq_shift=-0.8)

        overlap = np.abs(np.dot(vec1[:, 0].conj(), vec2[:, 0]))

        return 1 - overlap


    # --------------------------------------------------
    # Monte Carlo Shock Engine
    # --------------------------------------------------
    def monte_carlo(self, runs=200):

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


    # --------------------------------------------------
    # Dashboard JSON Output
    # --------------------------------------------------
    def dashboard(self):

        eigenvalues, _, gap = self.compute_spectrum(
            nasdaq_shift=-0.6,
            vix_shift=0.6
        )

        trajectory = self.forward_evolution(
            nasdaq_shift=-0.6,
            vix_shift=0.6
        )

        crisis_vol = self.detect_crisis()
        rotation = self.regime_rotation_speed()
        mc = self.monte_carlo(300)

        report = {
            "SpectralGap": float(gap),
            "CrisisThresholdVol": crisis_vol,
            "RegimeRotationSpeed": float(rotation),
            "ForwardSPXTrajectory": trajectory.tolist(),
            "MonteCarlo": mc,
            "TargetBias": self.target_bias
        }

        return json.dumps(report, indent=2)


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":

    engine = BloombergMarketsQuantumSimulator()

    print("\n=== LIVE DASHBOARD OUTPUT ===\n")
    print(engine.dashboard())
