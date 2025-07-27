
# Quantum Simulation Visualizations

This Python script generates two figures that visualize the behavior of quantum states used in variational quantum eigensolvers (VQE) and their entanglement properties.

## Files Generated

- `entanglement_scaling.png`: Shows how entanglement entropy scales with system size for a linear entangled state vs. a Sierpinski graph state.
- `vqe_convergence.png`: Simulated VQE energy convergence for two ansatz states, highlighting the faster convergence of the Sierpinski graph state.

## Requirements

Install the required libraries before running:

```bash
pip install numpy matplotlib
```

## How to Run

```bash
python simulation_plots.py
```

## Notes

These figures are simulations based on theoretical models. No quantum hardware was used. The entropy scaling is derived from known scaling laws; the VQE convergence is synthetically modeled using exponential decay to illustrate performance difference.

## Author: Muhammad Saddam Khokhar.
