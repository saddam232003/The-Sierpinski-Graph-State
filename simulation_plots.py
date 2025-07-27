
"""
Simulation: Entanglement Entropy and VQE Convergence Visualization
Author: [Muhammad Saddam Khokhar]

Dependencies: numpy, matplotlib

This script generates two figures:
1. Entanglement entropy scaling with system size.
2. VQE convergence for two ansatz states.
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Part 1: Entanglement Entropy Scaling
# -------------------------------
n_values = np.array([3, 6, 9, 12, 15, 18, 21])
linear_entropy = np.full_like(n_values, 1)
sierpinski_entropy = np.log(n_values)

plt.figure(figsize=(8, 5))
plt.plot(n_values, linear_entropy, label='Linear Entangled State', linestyle='--', marker='o', color='blue')
plt.plot(n_values, sierpinski_entropy, label='Sierpinski Graph State', linestyle='-', marker='s', color='green')
plt.xlabel('System Size (n)', fontsize=12)
plt.ylabel('Entanglement Entropy S(n)', fontsize=12)
plt.title('Entanglement Entropy vs System Size', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('entanglement_scaling.png', dpi=300)
plt.close()

# -------------------------------
# Part 2: VQE Convergence Simulation
# -------------------------------
iterations = np.arange(0, 50)
energy_sierpinski = -4.5 + 0.5 * np.exp(-0.1 * iterations)  # Simulated fast convergence
energy_linear = -4.5 + 1.2 * np.exp(-0.05 * iterations)     # Simulated slow convergence

plt.figure(figsize=(8, 5))
plt.plot(iterations, energy_sierpinski, label='Sierpinski Graph State', color='green', marker='o', markersize=4)
plt.plot(iterations, energy_linear, label='Linear Entangled State', color='blue', linestyle='--', marker='x', markersize=4)
plt.xlabel('VQE Iteration', fontsize=12)
plt.ylabel('Energy Expectation $\langle \psi(\theta) | H | \psi(\theta) \rangle$', fontsize=12)
plt.title('VQE Convergence: Energy vs Iterations', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('vqe_convergence.png', dpi=300)
plt.close()

print("✅ Figures generated: 'entanglement_scaling.png' and 'vqe_convergence.png'")