# Colab-ready  script: copy & paste into one cell and run Author by Muhammad Saddam Khokhar
# Requires: numpy, scipy, matplotlib (all preinstalled in Colab)
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from itertools import combinations
from math import log2
from tqdm.auto import tqdm
import time

# --------------------------
# Utilities: bit ops & indexing
# --------------------------
def int_bit(idx, pos):
    return (idx >> pos) & 1

def flip_bit_index(idx, pos):
    return idx ^ (1 << pos)

# --------------------------
# Build Sierpinski graph
# --------------------------
def build_sierpinski_graph(k):
    """
    Returns (n, edges) for Sierpinski graph of iteration k.
    We label vertices 0..n-1.
    Construction: recursive copy approach.
    """
    if k == 0:
        # base triangle
        n = 3
        edges = [(0,1),(1,2),(2,0)]
        return n, edges
    # build previous
    n_prev, edges_prev = build_sierpinski_graph(k-1)
    # three copies, with offsets
    offsets = [0, n_prev, 2*n_prev]
    # copy edges
    edges = []
    for off in offsets:
        edges += [(u+off, v+off) for (u,v) in edges_prev]
    # corner vertices in each copy: assume corners are 0,1,2 of prev copy
    # map corners to global indices
    corner_global = [0, n_prev, 2*n_prev]  # corners: copy0:0, copy1:n_prev, copy2:2*n_prev
    # connect corner vertices pairwise (the outer triangle edges)
    edges += [(corner_global[0], corner_global[1]),
              (corner_global[1], corner_global[2]),
              (corner_global[2], corner_global[0])]
    n = 3 * n_prev
    return n, edges

# --------------------------
# Circuit primitives acting on statevector
# --------------------------
def apply_ry(state, qubit, theta, n):
    """
    Apply RY on `qubit` to statevector `state` of length 2^n.
    """
    c = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    new = np.copy(state)
    N = state.shape[0]
    for i in range(N):
        if int_bit(i, qubit) == 0:
            j = i | (1 << qubit)
            a = state[i]
            b = state[j]
            new[i] = c*a - s*b
            new[j] = s*a + c*b
    return new

def apply_cz(state, q1, q2, n):
    """
    Apply CZ gate between q1 and q2 on statevector `state`.
    CZ flips phase of basis states where both bits are 1.
    """
    new = np.copy(state)
    N = state.shape[0]
    mask = (1 << q1) | (1 << q2)
    for i in range(N):
        if (i & mask) == mask:
            new[i] = -new[i]
    return new

# --------------------------
# Build ansatz statevector (Sierpinski or chain)
# --------------------------
def ansatz_state(params, n, edges, L=1):
    """
    params: shape (L, n) flattened or (L*n,)
    Apply: for r in [0..L-1]: apply RY on all qubits with params[r,q], then CZ on all edges
    Returns statevector (size 2^n)
    """
    state = np.zeros(1<<n, dtype=complex)
    state[0] = 1.0  # start from |0...0>
    # apply RY(pi/2) on each qubit to prepare |+>
    for q in range(n):
        state = apply_ry(state, q, np.pi/2.0, n)
    # params reshape
    params = np.array(params).reshape((L, n))
    for r in range(L):
        for q in range(n):
            state = apply_ry(state, q, params[r, q], n)
        # apply CZ on all edges
        for (i,j) in edges:
            state = apply_cz(state, i, j, n)
    return state

# --------------------------
# Apply ansatz to a basis state |s>
# --------------------------
def apply_ansatz_to_basisstate(params, n, edges, L, basis_index):
    """
    Return statevector U |basis_index>.
    """
    N = 1 << n
    state = np.zeros(N, dtype=complex)
    state[basis_index] = 1.0
    # apply RY(pi/2) on each qubit to prepare |+> from |s>
    for q in range(n):
        state = apply_ry(state, q, np.pi/2.0, n)
    params = np.array(params).reshape((L, n))
    for r in range(L):
        for q in range(n):
            state = apply_ry(state, q, params[r, q], n)
        for (i,j) in edges:
            state = apply_cz(state, i, j, n)
    return state

def estimate_trace_between_param_sets(params_i, params_j, n, edges, L, sample_basis, use_full=False):
    """
    Estimate Tr(U_i^dagger U_j). If use_full and 2^n is small, compute exact by summing over all basis states.
    Otherwise average over sample_basis (list of basis indices).
    """
    if use_full:
        basis_list = list(range(1<<n))
    else:
        basis_list = sample_basis
    accum = 0.0 + 0.0j
    for s in basis_list:
        psi_i = apply_ansatz_to_basisstate(params_i, n, edges, L, s)
        psi_j = apply_ansatz_to_basisstate(params_j, n, edges, L, s)
        accum += np.vdot(psi_i, psi_j)  # <psi_i|psi_j>
    return ( (1 << n) / len(basis_list) ) * accum  # scale to approximate full trace

# --------------------------
# Frame potential estimator
# --------------------------
def compute_frame_potential(param_list, n, edges, L, basis_sample_size=200):
    """
    param_list: list of param vectors (length M), each vector length L*n
    For each pair (i,j) estimate trace via sampling and compute F2.
    """
    M = len(param_list)
    use_full = ( (1<<n) <= 1024 )  # use exact trace if dimension <= 1024 (~2^10)
    if use_full:
        sample_basis = None
    else:
        rng = np.random.default_rng(12345)
        sample_basis = rng.integers(low=0, high=(1<<n), size=min(basis_sample_size, 1<<n)).tolist()
    F2 = 0.0
    for i in range(M):
        for j in range(M):
            if use_full:
                tr = estimate_trace_between_param_sets(param_list[i], param_list[j], n, edges, L, None, use_full=True)
            else:
                tr = estimate_trace_between_param_sets(param_list[i], param_list[j], n, edges, L, sample_basis, use_full=False)
            F2 += np.abs(tr)**4
    F2 /= (M*M)
    return F2

# --------------------------
# Entanglement entropy (von Neumann) for arbitrary subset
# --------------------------
def partial_trace_entropy(state, n, subsystem_qubits):
    """
    Compute von Neumann entropy S(rho_A) where A = subsystem_qubits (list).
    """
    k = len(subsystem_qubits)
    all_qubits = list(range(n))
    perm = subsystem_qubits + [q for q in all_qubits if q not in subsystem_qubits]
    N = 1 << n
    new_state = np.zeros(N, dtype=complex)
    for idx in range(N):
        bits = [(idx >> q) & 1 for q in range(n)]
        new_idx = 0
        for pos, q in enumerate(perm):
            if bits[q]:
                new_idx |= (1 << pos)
        new_state[new_idx] = state[idx]
    dimA = 1 << k
    dimB = 1 << (n - k)
    mat = new_state.reshape((dimA, dimB), order='C')
    rhoA = mat @ mat.conj().T
    vals = np.linalg.eigvalsh(rhoA)
    vals = np.real(vals)
    vals[vals < 1e-12] = 0.0
    nonzero = vals[vals>0]
    S = -np.sum(nonzero * np.log2(nonzero))
    return S

# --------------------------
# Energy expectation for transverse-field Ising-like Hamiltonian
# --------------------------
def energy_expectation_from_state(state, n, edges, J=1.0, h=1.0):
    N = 1 << n
    probs = np.abs(state)**2
    zz_exp = 0.0
    for (i,j) in edges:
        signs = np.ones(N)
        for idx in range(N):
            bi = 1 - 2*int_bit(idx, i)
            bj = 1 - 2*int_bit(idx, j)
            signs[idx] = bi * bj
        zz_exp += np.sum(probs * signs)
    zz_exp *= J
    x_exp = 0.0
    for i in range(n):
        acc = 0.0 + 0.0j
        for idx in range(N):
            jdx = flip_bit_index(idx, i)
            acc += np.conj(state[idx]) * state[jdx]
        x_exp += np.real(acc)
    x_exp *= h
    return zz_exp + x_exp

# --------------------------
# Simple gradient descent VQE (numerical gradient)
# --------------------------
def numeric_gradient(func, params, eps=1e-6):
    params = np.array(params)
    grad = np.zeros_like(params, dtype=float)
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + eps
        f_plus = func(params)
        params[i] = orig - eps
        f_minus = func(params)
        params[i] = orig
        grad[i] = (f_plus - f_minus) / (2*eps)
    return grad

def vqe_gradient_descent(n, edges, L, init_params, max_iter=60, lr=0.2, J=1.0, h=1.0):
    params = np.array(init_params).astype(float)
    energy_hist = []
    gradnorm_hist = []
    def loss_flat(flat_params):
        st = ansatz_state(flat_params, n, edges, L)
        return energy_expectation_from_state(st, n, edges, J=J, h=h)
    for it in range(max_iter):
        E = loss_flat(params)
        energy_hist.append(E)
        grad = numeric_gradient(loss_flat, params, eps=1e-5)
        gradnorm_hist.append(np.linalg.norm(grad))
        params = params - lr * grad
    return np.array(energy_hist), np.array(gradnorm_hist), params

# --------------------------
# Main experiment function
# --------------------------
def run_experiments(k_list=[0,1,2], L=1, M=20, R=5, basis_sample_size=200):
    results = {}
    for k in k_list:
        print(f"\n=== Running experiments for k={k} ===")
        n, edges = build_sierpinski_graph(k)
        print(f"n={n} qubits, edges={len(edges)}")
        if n > 20:
            print("WARNING: n too large for dense simulation. Skipping k =", k)
            continue
        param_len = L * n
        rng = np.random.default_rng(1234 + k)
        param_list = [rng.uniform(low=-np.pi, high=np.pi, size=(param_len,)) for _ in range(M)]
        t0 = time.time()
        F2 = compute_frame_potential(param_list, n, edges, L, basis_sample_size=basis_sample_size)
        t1 = time.time()
        print(f"Estimated F2 (M={M}): {F2:.4e}  (time {t1-t0:.1f}s)")
        energy_runs = []
        gradnorm_runs = []
        final_params = []
        for r in range(R):
            init = rng.uniform(low=-0.1, high=0.1, size=(param_len,))
            E_hist, g_hist, final_p = vqe_gradient_descent(n, edges, L, init, max_iter=50, lr=0.25)
            energy_runs.append(E_hist)
            gradnorm_runs.append(g_hist)
            final_params.append(final_p)
            print(f" seed {r}: init E {E_hist[0]:.4f} final E {E_hist[-1]:.4f} gradnorm_init {g_hist[0]:.4e}")
        energy_runs = np.array(energy_runs)
        gradnorm_runs = np.array(gradnorm_runs)
        ent_subsys_sizes = list(range(1, min(6, n//2+1)))
        ent_sier = []
        ent_chain = []
        rand_params = rng.uniform(low=-np.pi, high=np.pi, size=(param_len,))
        chain_edges = [(i,i+1) for i in range(n-1)]
        st_sier = ansatz_state(rand_params, n, edges, L)
        st_chain = ansatz_state(rand_params, n, chain_edges, L)
        for s in ent_subsys_sizes:
            subs = list(range(s))
            ent_sier.append(partial_trace_entropy(st_sier, n, subs))
            ent_chain.append(partial_trace_entropy(st_chain, n, subs))
        results[k] = {
            'n': n,
            'edges': edges,
            'param_len': param_len,
            'F2': F2,
            'time_F2': t1-t0,
            'energy_runs': energy_runs,
            'gradnorm_runs': gradnorm_runs,
            'ent_subsys_sizes': ent_subsys_sizes,
            'ent_sier': ent_sier,
            'ent_chain': ent_chain
        }
    return results

# --------------------------
# Run experiments (adjust M, R if you want faster/slower runs)
# --------------------------
if __name__ == "__main__":
    # Parameters you can tune:
    K_LIST = [0,1,2]     # levels to run (k=0,1,2 are safe)
    L = 1                # layers per experiment (keeps parameter count small)
    M = 20               # samples for frame potential
    R = 5                # VQE seeds
    basis_sample_size = 200
    results = run_experiments(k_list=K_LIST, L=L, M=M, R=R, basis_sample_size=basis_sample_size)

    # --------------------------
    # Plot Frame potential vs parameter count
    # --------------------------
    ks = []
    param_counts = []
    F2_vals = []
    for k, v in results.items():
        ks.append(k)
        param_counts.append(v['param_len'])
        F2_vals.append(v['F2'])
    plt.figure(figsize=(6,4))
    plt.errorbar(param_counts, F2_vals, yerr=np.zeros_like(F2_vals), fmt='o-', capsize=4)
    plt.xlabel('Parameter count (L * n)')
    plt.ylabel(r'Estimated $\mathcal{F}_2$')
    plt.title('Frame potential vs parameter count')
    plt.grid(True)
    plt.show()

    # --------------------------
    # Plot mean gradient norm vs iteration (mean ± std across seeds)
    # --------------------------
    plt.figure(figsize=(8,5))
    for k, v in results.items():
        g = v['gradnorm_runs']  # shape (R, iters)
        mean_g = np.mean(g, axis=0)
        std_g = np.std(g, axis=0)
        iters = np.arange(len(mean_g))
        plt.plot(iters, mean_g, label=f'k={k} (n={v["n"]})')
        plt.fill_between(iters, mean_g-std_g, mean_g+std_g, alpha=0.2)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient norm (log scale)')
    plt.title('Mean gradient norm vs iteration')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --------------------------
    # Plot entanglement entropy vs subsystem size (Sierpinski vs chain)
    # --------------------------
    for k, v in results.items():
        sizes = v['ent_subsys_sizes']
        s_sier = v['ent_sier']
        s_chain = v['ent_chain']
        plt.figure(figsize=(6,4))
        plt.plot(sizes, s_sier, 'o-', label='Sierpinski (random params)')
        plt.plot(sizes, s_chain, 's--', label='Chain (same params)')
        plt.xlabel('Subsystem size (qubits)')
        plt.ylabel('Von Neumann entropy (bits)')
        plt.title(f'Entanglement entropy vs subsystem size (k={k}, n={v["n"]})')
        plt.legend()
        plt.grid(True)
        plt.show()

    # --------------------------
    # Basic statistical comparison (example: final energy means Sierpinski seeds)
    # --------------------------
    print("\nStatistical summary (final energies across seeds):")
    for k, v in results.items():
        final_energies = v['energy_runs'][:, -1]
        meanE = np.mean(final_energies)
        stdE = np.std(final_energies)
        print(f" k={k}, n={v['n']}: final energy mean = {meanE:.4f} ± {stdE:.4f} (R={final_energies.size})")

    # Example paired test: compare energies between k=1 and k=2 (if both exist)
    if 1 in results and 2 in results:
        paired1 = results[1]['energy_runs'][:, -1]
        paired2 = results[2]['energy_runs'][:, -1]
        m = min(len(paired1), len(paired2))
        stat, pval = ttest_rel(paired1[:m], paired2[:m])
        print(f"\nPaired t-test k=1 vs k=2 final energies: t={stat:.4f}, p={pval:.4f}")

    print("\nDone. Adjust parameters M (frame potential samples), R (VQE seeds), L (layers) as needed.")
