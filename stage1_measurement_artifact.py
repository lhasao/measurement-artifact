"""
Stage 1: Synthetic measurement-artifact simulation.

Hypothesis under test
----------------------
Sigmoid-shaped "scaling curves" and apparent performance ceilings can arise
purely from how *measurement* works -- the reward function applied to an
outcome, and the spread of difficulty across evaluation items -- and need
NOT reflect any underlying change in how model capability scales.

We model a single evaluation item as a binary outcome: the item is solved
if competence (theta) exceeds the item's difficulty (d). The margin is

    m = theta - d

Reward functions map a margin to a value in [0, 1]:
    - binary:      R = 1 if m >= 0 else 0                (hard step)
    - soft:        R = sigmoid(m / alpha)                 (soft step)
    - continuous:  R = clip((m - m_min) / (m_max - m_min), 0, 1)  (linear ramp)

Three experiments isolate two distinct sources of curve smoothness:
    A. Reward smoothing alone   (sigma = 0, vary the reward function)
    B. Difficulty spread alone  (binary reward, vary sigma of d)
    C. Decomposition            (binary reward, moderate sigma, bin by difficulty)
    D. RLHF Preference Aggregation (simulate binary human choices vs a baseline)

Run with: python stage1_measurement_artifact.py
Finishes in well under 30 seconds on a CPU laptop.
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG -- all tunable knobs live here, nowhere else.
# ============================================================================
SEED = 0                                # fixed RNG seed for reproducibility

THETA_MIN, THETA_MAX, THETA_N = -10.0, 10.0, 401   # competence grid (x-axis)

N_ITEMS = 5000                          # evaluation items sampled per curve
MU = 0.0                                # mean item difficulty

ALPHAS = [0.001, 0.3, 1.0, 3.0]         # soft-reward smoothness sweep (Experiment A)
CONTINUOUS_RANGE = (-2.0, 2.0)          # (m_min, m_max) for the continuous-reward ramp

SIGMAS = [0.05, 0.3, 1.0, 2.0]          # difficulty-spread sweep (Experiment B)

SIGMA_C = 1.0                           # difficulty spread used in Experiment C
N_BINS_C = 5                            # number of difficulty-quantile bins

MC_TOLERANCE = 0.08                     # acceptance tolerance: |simulated - analytic CDF|

FIGURES_DIR = "figures"


# ============================================================================
# Reward functions: map a margin m = theta - d to a value in [0, 1].
# ============================================================================
def reward_binary(m):
    """R = 1 if the margin is non-negative, else 0 -- a hard step at m = 0."""
    return (m >= 0).astype(float)


def reward_soft(m, alpha):
    """
    R = sigmoid(m / alpha).

    d/dm sigmoid(m/alpha) at m=0 equals 1/(4*alpha): alpha sets the
    steepness of the transition without moving its location. As
    alpha -> 0, sigmoid(m/alpha) -> the binary step.

    Computed branch-wise on the sign of the argument to avoid overflow
    in exp() for small alpha / large |m| (the result is mathematically
    identical to 1 / (1 + exp(-m/alpha))).
    """
    z = m / alpha
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def reward_continuous(m, m_min, m_max):
    """R = clip((m - m_min) / (m_max - m_min), 0, 1) -- a linear ramp."""
    return np.clip((m - m_min) / (m_max - m_min), 0.0, 1.0)


# ============================================================================
# Analytic reference: Normal CDF (no scipy dependency -- use math.erf).
# ============================================================================
_erf_vec = np.vectorize(math.erf)


def normal_cdf(x, mu, sigma):
    """
    Phi((x - mu) / sigma), via Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).

    If difficulty d ~ Normal(mu, sigma), an item is solved (binary reward)
    iff theta >= d, so P(solved | theta) = P(d <= theta) = CDF_d(theta).
    This is the analytic prediction that Experiment B's simulation should
    reproduce.
    """
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + _erf_vec(z))


# ============================================================================
# Diagnostics
# ============================================================================
def inflection_point(theta, curve):
    """
    Inflection point := theta of maximum slope dR/dtheta, and that slope.
    For a sigmoid this is its steepest point; for a hard step it is the
    single grid cell containing the jump.
    """
    slope = np.gradient(curve, theta)
    idx = np.argmax(slope)
    return theta[idx], slope[idx]


# ============================================================================
# Experiment A: reward smoothing alone (sigma = 0)
# ============================================================================
def experiment_a(theta):
    """
    Every item has difficulty MU (sigma = 0), so the margin m = theta - MU
    is the same for the whole population. Sweep the reward function's own
    smoothness and compare binary, soft (various alpha), and continuous.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: reward smoothing alone (sigma = 0, difficulty = MU)")
    print("=" * 70)

    m = theta - MU

    curves = {"binary": reward_binary(m)}
    for alpha in ALPHAS:
        curves[f"soft alpha={alpha}"] = reward_soft(m, alpha)
    curves["continuous"] = reward_continuous(m, *CONTINUOUS_RANGE)

    plt.figure(figsize=(7, 5))
    for label, curve in curves.items():
        plt.plot(theta, curve, label=label)
    plt.axvline(MU, color="gray", linestyle=":", linewidth=1, label="item difficulty (mu)")
    plt.xlabel("theta (competence)")
    plt.ylabel("mean reward")
    plt.title("Experiment A: reward smoothing alone (sigma = 0)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "experiment_a_reward_smoothing.png"), dpi=150)
    plt.close()

    diagnostics = {}
    for label, curve in curves.items():
        theta_infl, slope = inflection_point(theta, curve)
        diagnostics[label] = (theta_infl, slope)
        print(f"  {label:18s}: inflection theta = {theta_infl:7.3f}, max slope = {slope:9.3f}")

    # The continuous reward is a *linear* ramp: its slope is constant across
    # the whole ramp, so "the" inflection point is not a single meaningful
    # location -- argmax just picks one point on a flat plateau. Report the
    # ramp's span instead.
    print(f"  {'continuous':18s}: note -- slope is constant (={diagnostics['continuous'][1]:.3f}) "
          f"across the whole ramp m in [{CONTINUOUS_RANGE[0]}, {CONTINUOUS_RANGE[1]}]; "
          f"the inflection-point notion does not apply to a linear ramp.")

    return curves, diagnostics


# ============================================================================
# Experiment B: difficulty spread alone (binary reward)
# ============================================================================
def experiment_b(theta, rng):
    """
    Binary reward only. Draw N_ITEMS difficulties d ~ Normal(MU, sigma) for
    each sigma in SIGMAS and compute the simulated fraction solved:

        frac_solved(theta) = mean_i 1{theta - d_i >= 0} = mean_i 1{d_i <= theta}
                            = empirical CDF of d, evaluated at theta

    Compared against the analytic Normal CDF Phi((theta - MU) / sigma).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: difficulty spread alone (binary reward)")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    sim_curves, analytic_curves, max_errors, diagnostics = {}, {}, {}, {}

    for ax, sigma in zip(axes.flat, SIGMAS):
        d = rng.normal(MU, sigma, size=N_ITEMS)

        # solved[i, j] = True if item j is solved when competence = theta[i]
        solved = (theta[:, None] - d[None, :]) >= 0.0
        frac_solved = solved.mean(axis=1)

        analytic = normal_cdf(theta, MU, sigma)
        err = float(np.max(np.abs(frac_solved - analytic)))

        sim_curves[sigma] = frac_solved
        analytic_curves[sigma] = analytic
        max_errors[sigma] = err

        theta_infl, slope = inflection_point(theta, frac_solved)
        diagnostics[sigma] = (theta_infl, slope)

        ax.plot(theta, frac_solved, label="simulated")
        ax.plot(theta, analytic, "--", label="analytic Normal CDF")
        ax.set_title(f"sigma = {sigma}")
        ax.set_xlabel("theta")
        ax.set_ylabel("fraction solved")
        ax.legend(fontsize=8)

        print(f"  sigma={sigma:5.2f}: inflection theta = {theta_infl:7.3f}, "
              f"max slope = {slope:7.3f}, max |sim - analytic CDF| = {err:.4f}")

    fig.suptitle("Experiment B: difficulty spread alone (binary reward)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "experiment_b_difficulty_spread.png"), dpi=150)
    plt.close(fig)

    return sim_curves, analytic_curves, max_errors, diagnostics


# ============================================================================
# Experiment C: decomposition into difficulty-quantile bins
# ============================================================================
def experiment_c(theta, rng):
    """
    Binary reward, sigma = SIGMA_C. Split the items into N_BINS_C
    difficulty-quantile bins. Each bin spans a narrow difficulty range, so
    its accuracy curve is close to a step at that bin's own difficulty.
    The aggregate over all items is the (weight-averaged) mixture of bin
    curves, which should equal the full-population CDF.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT C: decomposition into {N_BINS_C} difficulty bins (sigma = {SIGMA_C})")
    print("=" * 70)

    d = rng.normal(MU, SIGMA_C, size=N_ITEMS)

    # Sort by difficulty, then split into N_BINS_C contiguous quantile groups.
    order = np.argsort(d)
    bins = np.array_split(order, N_BINS_C)

    plt.figure(figsize=(8, 6))

    bin_curves, weights = [], []
    for i, idx in enumerate(bins):
        d_bin = d[idx]
        solved = (theta[:, None] - d_bin[None, :]) >= 0.0
        frac_solved = solved.mean(axis=1)
        bin_curves.append(frac_solved)
        weights.append(len(idx) / N_ITEMS)

        theta_infl, slope = inflection_point(theta, frac_solved)
        print(f"  bin {i + 1} (d in [{d_bin.min():5.2f}, {d_bin.max():5.2f}]): "
              f"inflection theta = {theta_infl:7.3f}, max slope = {slope:7.3f}")

        plt.plot(theta, frac_solved, "--", alpha=0.6,
                 label=f"bin {i + 1} (d in [{d_bin.min():.2f}, {d_bin.max():.2f}])")

    aggregate = (theta[:, None] - d[None, :] >= 0.0).mean(axis=1)
    # Mixture identity: aggregate accuracy = weighted average of per-bin accuracy
    weighted_avg = sum(w * c for w, c in zip(weights, bin_curves))
    analytic_full = normal_cdf(theta, MU, SIGMA_C)

    agg_vs_cdf = float(np.max(np.abs(aggregate - analytic_full)))
    agg_vs_weighted = float(np.max(np.abs(aggregate - weighted_avg)))

    theta_infl, slope = inflection_point(theta, aggregate)
    print(f"  {'aggregate':28s}: inflection theta = {theta_infl:7.3f}, max slope = {slope:7.3f}")
    print(f"  max |aggregate - weighted bin average| = {agg_vs_weighted:.6f}")
    print(f"  max |aggregate - analytic full CDF|    = {agg_vs_cdf:.4f}")

    plt.plot(theta, aggregate, "k-", linewidth=2.5, label="aggregate (all items)")
    plt.plot(theta, analytic_full, "r:", linewidth=2, label="analytic full CDF")
    plt.xlabel("theta (competence)")
    plt.ylabel("fraction solved")
    plt.title(f"Experiment C: decomposition into {N_BINS_C} difficulty bins (sigma = {SIGMA_C})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "experiment_c_decomposition.png"), dpi=150)
    plt.close()

    return aggregate, analytic_full, agg_vs_cdf, agg_vs_weighted


# ============================================================================
# Experiment D: RLHF Preference Aggregation (Bradley-Terry simulation)
# ============================================================================
def experiment_d(theta, rng):
    """
    Simulates the Bradley-Terry / RLHF evaluation setup.
    A scaling model (theta) is evaluated against a fixed baseline model (theta_base = MU).
    Thousands of human evaluators make a *binary choice* based on a noisy
    perception of capability.
    The aggregate win-rate acts like a softmax/sigmoid mapping latent capability.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT D: RLHF Preference Aggregation (Bradley-Terry effect)")
    print("=" * 70)

    theta_base = MU
    human_noise_sigma = 1.0  # The "human learning variable" / variance

    # Simulate N_ITEMS humans evaluating each theta against theta_base
    # Human perceives A: theta + noise_A, B: theta_base + noise_B
    # Prefers A if perceived_A > perceived_B
    diff_noise = rng.normal(0, math.sqrt(2) * human_noise_sigma, size=N_ITEMS)
    
    # margin = true difference + human perception noise
    margin = (theta[:, None] - theta_base) + diff_noise[None, :]
    win_rate = (margin > 0).mean(axis=1)

    # Analytic expectation is the Normal CDF of the difference
    analytic_win_rate = normal_cdf(theta, theta_base, math.sqrt(2) * human_noise_sigma)

    plt.figure(figsize=(7, 5))
    plt.plot(theta, win_rate, label="simulated human win rate (binary aggregation)")
    plt.plot(theta, analytic_win_rate, "--", label="analytic expectation (sigmoid/CDF)")
    plt.axvline(theta_base, color="gray", linestyle=":", label="baseline model capability")
    plt.xlabel("theta (latent model capability scaling)")
    plt.ylabel("win rate vs baseline")
    plt.title("Experiment D: RL Scaling S-Curve via Binary Preference")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "experiment_d_rlhf_preferences.png"), dpi=150)
    plt.close()


# ============================================================================
# Main: run experiments, then check acceptance criteria.
# ============================================================================
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    theta = np.linspace(THETA_MIN, THETA_MAX, THETA_N)

    curves_a, diag_a = experiment_a(theta)
    sim_b, analytic_b, err_b, diag_b = experiment_b(theta, rng)
    agg_c, analytic_c, err_agg_cdf, err_agg_weighted = experiment_c(theta, rng)
    experiment_d(theta, rng)

    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)

    # 1. sigma=0, binary reward -> hard step (no intermediate reward values).
    binary_curve = curves_a["binary"]
    n_intermediate = int(np.sum((binary_curve > 0) & (binary_curve < 1)))
    crit1 = n_intermediate == 0
    print(f"[{'PASS' if crit1 else 'FAIL'}] 1. sigma=0 + binary reward is a hard step, "
          f"not a smooth sigmoid ({n_intermediate} intermediate-valued grid points)")

    # 2. Experiment B: simulated binary curve matches the analytic Normal CDF.
    worst_b = max(err_b.values())
    crit2 = worst_b < MC_TOLERANCE
    print(f"[{'PASS' if crit2 else 'FAIL'}] 2. Experiment B simulated curves match the analytic "
          f"Normal CDF within {MC_TOLERANCE} (worst max-error = {worst_b:.4f})")

    # 3. Experiment C: aggregate of binned curves equals the full CDF.
    crit3 = err_agg_cdf < MC_TOLERANCE
    print(f"[{'PASS' if crit3 else 'FAIL'}] 3. Experiment C aggregate equals the full CDF within "
          f"{MC_TOLERANCE} (max-error = {err_agg_cdf:.4f})")

    # 4. Both alpha and sigma independently smooth the curve: max slope must
    #    decrease monotonically as alpha (Experiment A) or sigma (Experiment B)
    #    increases -- i.e. either knob alone is sufficient to produce a sigmoid.
    slopes_alpha = [diag_a[f"soft alpha={a}"][1] for a in ALPHAS]
    crit4a = all(slopes_alpha[i] > slopes_alpha[i + 1] for i in range(len(slopes_alpha) - 1))

    slopes_sigma = [diag_b[s][1] for s in SIGMAS]
    crit4b = all(slopes_sigma[i] > slopes_sigma[i + 1] for i in range(len(slopes_sigma) - 1))

    crit4 = crit4a and crit4b
    print(f"[{'PASS' if crit4 else 'FAIL'}] 4. alpha and sigma each independently smooth the "
          f"curve (monotonically decreasing max slope): "
          f"alpha-sweep {'OK' if crit4a else 'FAIL'} {slopes_alpha}, "
          f"sigma-sweep {'OK' if crit4b else 'FAIL'} {slopes_sigma}")

    # Cross-check: inflection location tracks MU (set by difficulty), not the
    # theta grid range, for every curve in Experiments A and B.
    infl_locations = [diag_a[k][0] for k in diag_a if k != "continuous"]
    infl_locations += [diag_b[s][0] for s in SIGMAS]
    max_infl_offset = max(abs(t - MU) for t in infl_locations)
    print(f"\n  Note: inflection theta for all step/sigmoid/CDF curves stayed within "
          f"{max_infl_offset:.3f} of MU={MU} (theta grid spans "
          f"[{THETA_MIN}, {THETA_MAX}]) -- inflection tracks difficulty, not grid range.")

    overall = crit1 and crit2 and crit3 and crit4
    print(f"\n  OVERALL: {'ALL ACCEPTANCE CRITERIA PASSED' if overall else 'SOME CRITERIA FAILED -- see above'}")
    print(f"  Figures saved to: {os.path.abspath(FIGURES_DIR)}")


if __name__ == "__main__":
    main()
