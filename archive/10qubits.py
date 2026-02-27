import cirq
import numpy as np

def encode_bloomberg_data(text_data):
    # Initialize 10 qubits
    qubits = cirq.LineQubit.range(10)
    circuit = cirq.Circuit()

    # 1. Russell 2000 Index Value (Qubit 0)
    # Data: 2,677.88 -> Normalize to [0, pi] for Ry gate
    russell_index_value = 2677.88
    # A simple normalization: assume a max value for the index, e.g., 5000
    normalized_russell_value = russell_index_value / 5000 * np.pi
    circuit.append(cirq.ry(normalized_russell_value).on(qubits[0]))

    # 2. Russell 2000 Percentage Change (Qubit 1)
    # Data: 0.55% -> Normalize to [0, pi]. Max change e.g. +/- 5%
    russell_percent_change = 0.55 / 100  # Convert to decimal
    normalized_russell_percent_change = (russell_percent_change + 0.05) / 0.10 * np.pi # Normalize from [-0.05, 0.05] to [0,1]
    circuit.append(cirq.ry(normalized_russell_percent_change).on(qubits[1]))

    # 3. S&P 500 Global Market Status (Qubit 2)
    # Assuming an example numerical value or change for S&P 500, e.g., 4500
    sp500_value = 4500 # Placeholder value
    normalized_sp500 = sp500_value / 5000 * np.pi
    circuit.append(cirq.ry(normalized_sp500).on(qubits[2]))

    # 4. Nasdaq Global Market Status (Qubit 3)
    # Assuming an example numerical value or change for Nasdaq, e.g., 14000
    nasdaq_value = 14000 # Placeholder value
    normalized_nasdaq = nasdaq_value / 20000 * np.pi # Assuming max Nasdaq 20000
    circuit.append(cirq.ry(normalized_nasdaq).on(qubits[3]))

    # 5. Euro Stoxx 50 Global Market Status (Qubit 4)
    # Assuming an example numerical value or change for Euro Stoxx 50, e.g., 4000
    euro_stoxx_value = 4000 # Placeholder value
    normalized_euro_stoxx = euro_stoxx_value / 5000 * np.pi
    circuit.append(cirq.ry(normalized_euro_stoxx).on(qubits[4]))

    # 6. Natural Gas Futures Trend (Qubit 5)
    # Binary flag: 1 for 'up' or 'positive', 0 for 'stable/down' based on 'visible' interpretation.
    # For this example, let's assume it's 'up' if visible implies positive movement.
    natural_gas_trend = 1 # Example: assume positive trend
    circuit.append(cirq.ry(natural_gas_trend * np.pi).on(qubits[5])) # 0 or pi

    # 7. Heating Oil Futures Trend (Qubit 6)
    # Binary flag: 1 for 'up' or 'positive', 0 for 'stable/down'.
    heating_oil_trend = 0 # Example: assume stable/down trend
    circuit.append(cirq.ry(heating_oil_trend * np.pi).on(qubits[6])) # 0 or pi

    # 8. News - Fraud Related (Qubit 7)
    # Binary flag: 1 if fraud-related news is present, 0 otherwise.
    fraud_news_present = 1 # 'guilty plea related to fraud' is present
    circuit.append(cirq.ry(fraud_news_present * np.pi).on(qubits[7])) # 0 or pi

    # 9. News - Economic Concern (Qubit 8)
    # Binary flag: 1 if news indicating economic concern is present, 0 otherwise.
    economic_concern_news = 1 # 'shrinking smartphone market due to a memory crisis' is present
    circuit.append(cirq.ry(economic_concern_news * np.pi).on(qubits[8])) # 0 or pi

    # 10. Live Coverage - High Profile Guest (Qubit 9)
    # Binary flag: 1 if a high-profile guest is featured, 0 otherwise.
    high_profile_guest = 1 # 'Wendy Sherman, former U.S. Deputy Secretary of State' is present
    circuit.append(cirq.ry(high_profile_guest * np.pi).on(qubits[9])) # 0 or pi

    # Add measurements to all qubits
    circuit.append(cirq.measure(*qubits, key='m'))

    return circuit, qubits

# Example usage with the provided text
bloomberg_text_data = """
Russell 2000 Index: Currently at 2,677.88, showing a gain of 14.55 points or 0.55%.
Top Stories: Headlines include a guilty plea related to fraud by a First Brands executive and reports on a shrinking smartphone market due to a memory crisis.
Global Markets: Real-time data for the Euro Stoxx 50, S&P 500, Nasdaq, and futures for natural gas and heating oil are visible.
Live Coverage: The broadcast is titled "Balance of Power" and features Wendy Sherman, former U.S. Deputy Secretary of State.
"""

quantum_circuit, encoded_qubits = encode_bloomberg_data(bloomberg_text_data)
print("Encoded 10-qubit circuit for Bloomberg data:")
print(quantum_circuit)

# Verify the state of one qubit (e.g., Qubit 0 for Russell Index Value)
simulator = cirq.Simulator()

# Simulate with repetitions to get measurement outcomes
print("\nSimulating the circuit with 1000 repetitions...")
results = simulator.run(quantum_circuit, repetitions=1000)

# Get measurement results
measurements = results.measurements['m']

print("\nFirst 10 measurement outcomes (binary strings):")
for i in range(min(10, len(measurements))):
    print(f"  {measurements[i].tolist()}")

# Analyze the histogram of outcomes
print("\nMeasurement Outcome Histograms (proportion of '1' for each qubit):")
for i in range(len(encoded_qubits)):
    # Count how many times qubit i was measured as 1
    count_ones = np.sum(measurements[:, i])
    proportion_ones = count_ones / 1000
    print(f"  Qubit {i}: Proportion of '1' = {proportion_ones:.4f}")


# Let's inspect the rotation angles for better verification (only for Ry gates prior to measurement)
print("\nRotation angles applied to each qubit (in radians) before measurement:")
# Recreate circuit without measurements to inspect angles easily
circuit_no_measure = cirq.Circuit()
for i, op in enumerate(quantum_circuit.all_operations()):
    # Corrected: Check against the gate type cirq.Ry, not the function cirq.ry
    if isinstance(op.gate, cirq.Ry):
        circuit_no_measure.append(op)
        print(f"Qubit {op.qubits[0].x}: {op.gate.exponent * np.pi:.4f} radians")

# Verify sin^2(theta/2) for Qubit 0

# Get the rotation angle for Qubit 0
q0_angle_rads = None
for op in quantum_circuit.all_operations():
    if isinstance(op.gate, cirq.Ry) and op.qubits[0].x == 0:
        q0_angle_rads = op.gate.exponent * np.pi
        break

if q0_angle_rads is not None:
    # Calculate the theoretical probability of measuring |1>
    # P(|1>) = sin^2(theta/2)
    theoretical_prob_q0 = np.sin(q0_angle_rads / 2)**2

    print(f"Qubit 0: Rotation Angle (radians) = {q0_angle_rads:.4f}")
    print(f"Qubit 0: Theoretical Probability of measuring '1' = {theoretical_prob_q0:.4f}")

    # Retrieve the observed proportion from the previous simulation
    # (assuming 'measurements' and 'encoded_qubits' are still in scope)
    # If not, you'd need to re-run the simulation part.
    observed_proportion_q0 = np.sum(measurements[:, 0]) / len(measurements)
    print(f"Qubit 0: Observed Proportion of measuring '1' = {observed_proportion_q0:.4f}")

    if np.isclose(theoretical_prob_q0, observed_proportion_q0, atol=0.05):
        print("The theoretical probability is consistent with the observed proportion (within a small tolerance).")
    else:
        print("There is a significant difference between theoretical and observed probabilities.")
else:
    print("Could not find rotation angle for Qubit 0.")

print("Quantum Circuit Diagram:")
print(quantum_circuit)

!pip install cirq

from PIL import Image

image_files = [
    '/content/Screenshot From 2026-02-26 14-44-18.png',
    '/content/Screenshot From 2026-02-26 14-43-14.png',
    '/content/Screenshot From 2026-02-26 14-44-09.png',
    '/content/Screenshot From 2026-02-26 14-42-02.png'
]

for img_file in image_files:
    print(f"Displaying: {img_file}")
    img = Image.open(img_file)
    display(img)

