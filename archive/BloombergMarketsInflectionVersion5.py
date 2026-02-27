import numpy as np
from scipy.linalg import expm
import yfinance as yf
import json


class BloombergMarketsQuantumSimulatorV5:

    def __init__(self):

        # Q0=SPX, Q1=NASDAQ, Q2=Nikkei, Q3=Oil, Q4=Gold, Q5=VIX
        self.n_qubits = 6

        # Couplings
        self.J_spx_nasdaq = 1.2
        self.J_spx_vix = 1.5
        self.J_spx_oil = 0.8
        self.J_spx_gold = 0.7

        # SPX positive targeting bias
        self.target_bias = 0.4

        self.h = np.array([-0.58, -0.87, 0.16, 0.30, 0.94, 0.5]) * np.pi / 8


    # --------------------------------------------------
    # Hamiltonian
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

            # Positive SPX bias targeting
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

        return H


    # --------------------------------------------------
    # Time Evolution
    # --------------------------------------------------
    def forward_evolution(self, steps=20, dt=0.1, **kwargs):

        H = self.build_hamiltonian(**kwargs)

        dim = H.shape[0]
        psi = np.ones(dim) / np.sqrt(dim)  # initial superposition

        U = expm(-1j * H * dt)

        trajectory = []

        for _ in range(steps):
            psi = U @ psi

            # Compute SPX expectation
            expectation = 0
            for i in range(dim):
                bitstring = format(i, f"0{self.n_qubits}b")
                z_spx = 1 if bitstring[0] == "0" else -1
                expectation += z_spx * np.abs(psi[i])**2

            trajectory.append(expectation.real)

        return np.array(trajectory)


    # --------------------------------------------------
    # Calibration to Real Data
    # --------------------------------------------------
    def calibrate_to_market(self):

        spx = yf.download("^GSPC", period="6mo")["Close"]
        vix = yf.download("^VIX", period="6mo")["Close"]

        spx_returns = spx.pct_change().dropna()
        vix_returns = vix.pct_change().dropna()

        corr = np.corrcoef(spx_returns, vix_returns)[0, 1]

        # Calibrate SPX-VIX coupling
        self.J_spx_vix = abs(corr) * 2

        return {
            "SPX-VIX Correlation": float(corr),
            "Adjusted J_spx_vix": float(self.J_spx_vix)
        }


    # --------------------------------------------------
    # Monte Carlo Shock Engine
    # --------------------------------------------------
    def monte_carlo_simulation(self, simulations=100):

        final_bias = []

        for _ in range(simulations):

            nasdaq_shock = np.random.normal(-0.5, 0.3)
            vix_shock = np.random.normal(0.5, 0.2)

            trajectory = self.forward_evolution(
                nasdaq_shift=nasdaq_shock,
                vix_shift=vix_shock
            )

            final_bias.append(trajectory[-1])

        return {
            "Mean Final SPX Bias": float(np.mean(final_bias)),
            "Std Final SPX Bias": float(np.std(final_bias)),
            "Bullish Probability": float(np.mean(np.array(final_bias) > 0))
        }


    # --------------------------------------------------
    # Dashboard Output
    # --------------------------------------------------
    def build_dashboard_output(self):

        calibration = self.calibrate_to_market()
        mc = self.monte_carlo_simulation(200)

        trajectory = self.forward_evolution(
            nasdaq_shift=-0.5,
            vix_shift=0.5
        )

        output = {
            "Calibration": calibration,
            "MonteCarlo": mc,
            "ForwardTrajectory": trajectory.tolist(),
            "TargetBias": self.target_bias
        }

        return json.dumps(output, indent=2)


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":

    engine = BloombergMarketsQuantumSimulatorV5()

    print("\n=== Dashboard Output ===")
    print(engine.build_dashboard_output())
