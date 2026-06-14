"""
Stage 2: minimal training for saturate-then-resume (gradient starvation).

Hypothesis under test
----------------------
A binary reward gives zero local gradient once an action is "good enough"
(inside the tolerance band) -- so a policy can stall on a held-out, continuous
oracle metric well before it reaches its actual capability limit. Switching to
a continuous (shaped) reward, which has gradient everywhere, should let the
oracle resume improving.

Task
----
1D regression-as-control: inputs x ~ Uniform(X_LOW, X_HIGH), target t(x) =
sin(x). A small MLP f_w(x) parameterizes the mean of a fixed-std Gaussian
policy a ~ Normal(f_w(x), POLICY_STD). REINFORCE with a batch-mean baseline
is used for the policy gradient.

Training rewards:
    binary : r = 1 if |a - t| < TOL else 0
    shaped : r = exp(-(a - t)^2 / BETA)        (continuous, gradient everywhere)

Held-out oracle (measurement only, NEVER a training reward):
    MAE = mean |f_w(x) - t(x)| over a fixed evaluation grid, using the policy
    MEAN (no sampling noise).

Protocol (per seed)
--------------------
1. Train under binary reward from scratch until the held-out oracle's slope
   over the last PLATEAU_WINDOW eval points drops below
   PLATEAU_SLOPE_THRESH (quantitative plateau detection) -- this is the
   "switch step".
2. From that checkpoint, with a freshly-initialized optimizer ("optimizer
   restart"), continue training under the SHAPED reward (the "resume"
   condition). Measured only on the oracle.
3. Control -- continue-binary: from the same checkpoint, same optimizer
   restart, but keep the BINARY reward. If this also improves the oracle,
   the gain in (2) is from the restart, not the reward change.
4. Reference -- shaped-from-scratch: a model with the SAME initial weights
   as the binary run, trained under the SHAPED reward for the whole budget
   (switch_step + POST_SWITCH_STEPS). This bounds how much of the binary
   "ceiling" was recoverable.

Everything is averaged over SEED_LIST seeds (mean +/- spread band) -- a
single seed cannot separate this effect from REINFORCE's high variance.

Run with: python stage2_saturate_resume.py
CPU only. A few minutes on a laptop.
"""

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG -- all tunable knobs live here, nowhere else.
# ============================================================================
SEED_LIST = [0, 1, 2, 3, 4]        # main protocol: averaged, mean +/- spread band
CALIBRATION_SEED = 12345           # separate seed, used only to set TOL / BETA

# --- task ---
X_LOW, X_HIGH = -3.0, 3.0


def target_fn(x):
    return torch.sin(x)


# --- policy network ---
HIDDEN_WIDTH = 32
N_HIDDEN_LAYERS = 2
POLICY_STD = 0.3                   # FIXED throughout this run. Per protocol,
                                    # std annealing is only introduced once the
                                    # fixed-std version is shown to be stable;
                                    # it is not implemented in this script.

# --- optimization ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-2

# --- reward calibration ---
CALIBRATION_BATCH = 5000
TOL_PERCENTILE = 0.40               # TOL set so initial binary reward rate ~= this
SHAPED_BETA_SCALE = 1.0             # SHAPED_BETA = SHAPED_BETA_SCALE * TOL^2
MODERATE_REWARD_RANGE = (0.10, 0.90)  # acceptance band for "moderate" initial reward

# --- quantitative plateau detection ---
EVAL_INTERVAL = 20                  # steps between held-out oracle evaluations
N_EVAL_POINTS = 200                 # size of the fixed held-out evaluation grid
PLATEAU_WINDOW = 10                 # # of recent eval points used for the slope fit
PLATEAU_SLOPE_THRESH = 2e-4         # |d(oracle MAE)/d(step)| below this => plateaued
MIN_BINARY_STEPS = 300              # plateau check only active after this many steps
MAX_BINARY_STEPS = 2000             # hard cap / forced switch if no plateau found

POST_SWITCH_STEPS = 1200            # steps for resume / continue-binary / scratch

# --- width-scaling addendum (run if time allows) ---
RUN_WIDTH_SCALING = True
WIDTH_SCALING_WIDTHS = [4, 8, 16, 32, 64]
WIDTH_SCALING_SEEDS = [0, 1, 2]

FIGURES_DIR = "figures"


# ============================================================================
# Model, rewards, sampling
# ============================================================================
class PolicyMLP(nn.Module):
    """f_w(x): small MLP giving the mean of a fixed-std Gaussian policy."""

    def __init__(self, hidden_width=HIDDEN_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_width), nn.Tanh()]
            in_dim = hidden_width
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def reward_binary_fn(a, t, tol):
    """r = 1 if |a - t| < tol else 0 -- flat (zero gradient) inside the band."""
    return (torch.abs(a - t) < tol).float()


def reward_shaped_fn(a, t, beta):
    """r = exp(-(a - t)^2 / beta) -- continuous, gradient everywhere."""
    return torch.exp(-(a - t) ** 2 / beta)


def sample_x(batch_size, generator):
    return torch.empty(batch_size, 1).uniform_(X_LOW, X_HIGH, generator=generator)


def oracle_mae(model, x_eval, t_eval):
    """Held-out MAE using the policy MEAN -- measurement only, never a reward."""
    with torch.no_grad():
        mu = model(x_eval)
    return (mu - t_eval).abs().mean().item()


# ============================================================================
# REINFORCE step with a batch-mean baseline
# ============================================================================
def reinforce_step(model, optimizer, std, reward_fn, generator):
    """
    One REINFORCE update:
        advantage = r - mean(r)             <- baseline subtraction
        loss      = -mean(advantage * log pi(a | x))
    The baseline removes the (large) common-mode reward level so the
    gradient estimate isn't swamped by its own variance.
    Returns (mean batch reward, total gradient L2 norm).
    """
    x = sample_x(BATCH_SIZE, generator)
    t = target_fn(x).squeeze(-1)
    mu = model(x)
    dist = torch.distributions.Normal(mu, std)
    a = dist.sample()
    r = reward_fn(a, t)

    baseline = r.mean()
    advantage = (r - baseline).detach()
    log_prob = dist.log_prob(a)
    loss = -(advantage * log_prob).mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.cat([p.grad.detach().flatten() for p in model.parameters()]).norm().item()
    optimizer.step()

    return r.mean().item(), grad_norm


# ============================================================================
# Reward calibration: moderate initial binary reward rate
# ============================================================================
def calibrate_reward_params():
    """
    Choose TOL so that, at initialization (CALIBRATION_SEED, before any
    training), P(|a - t| < TOL) ~= TOL_PERCENTILE -- moderate, not all-zero
    or all-one. SHAPED_BETA is then tied to TOL so the shaped reward operates
    on a comparable error scale (beta = TOL^2 => reward = exp(-1) at |a-t|=TOL).
    """
    torch.manual_seed(CALIBRATION_SEED)
    model = PolicyMLP()
    gen = torch.Generator().manual_seed(CALIBRATION_SEED)
    x = sample_x(CALIBRATION_BATCH, gen)
    t = target_fn(x).squeeze(-1)
    with torch.no_grad():
        mu = model(x)
    dist = torch.distributions.Normal(mu, POLICY_STD)
    a = dist.sample()
    err = (a - t).abs()

    tol = torch.quantile(err, TOL_PERCENTILE).item()
    beta = SHAPED_BETA_SCALE * tol ** 2
    initial_reward_rate = (err < tol).float().mean().item()
    return tol, beta, initial_reward_rate


# ============================================================================
# Training-phase runner with quantitative plateau detection
# ============================================================================
def run_phase(model, optimizer, reward_fn, generator, x_eval, t_eval,
               max_steps, step_offset=0, detect_plateau=False):
    """
    Train for up to max_steps steps (global step = step_offset + local_step).

    Records:
      mae_history    -- (step, oracle MAE) every EVAL_INTERVAL steps
      grad_history   -- (step, grad L2 norm) every step
      reward_history -- (step, mean batch reward) every step

    If detect_plateau, stop early once a least-squares fit of oracle MAE vs.
    step over the last PLATEAU_WINDOW eval points has |slope| <
    PLATEAU_SLOPE_THRESH (and step >= MIN_BINARY_STEPS). Returns the step at
    which this happened as switch_step. If no plateau is found, switch_step
    is the final recorded step (forced switch).
    """
    mae_history, grad_history, reward_history = [], [], []

    for local_step in range(max_steps):
        batch_reward, grad_norm = reinforce_step(model, optimizer, POLICY_STD, reward_fn, generator)
        step = step_offset + local_step
        grad_history.append((step, grad_norm))
        reward_history.append((step, batch_reward))

        if local_step % EVAL_INTERVAL == 0 or local_step == max_steps - 1:
            mae = oracle_mae(model, x_eval, t_eval)
            mae_history.append((step, mae))

            if detect_plateau and len(mae_history) >= PLATEAU_WINDOW and step >= MIN_BINARY_STEPS:
                recent = mae_history[-PLATEAU_WINDOW:]
                steps_arr = np.array([s for s, _ in recent], dtype=float)
                maes_arr = np.array([m for _, m in recent], dtype=float)
                slope = np.polyfit(steps_arr, maes_arr, 1)[0]
                if abs(slope) < PLATEAU_SLOPE_THRESH:
                    return mae_history, grad_history, reward_history, step

    return mae_history, grad_history, reward_history, mae_history[-1][0]


# ============================================================================
# Per-seed protocol: binary -> {resume (shaped), continue-binary, shaped-from-scratch}
# ============================================================================
def run_binary_phase(seed, tol, x_eval, t_eval, hidden_width=HIDDEN_WIDTH):
    """Phase 1, shared by the full protocol and the width-scaling addendum:
    train from scratch under the binary reward until the quantitative
    plateau trigger fires. Returns the trained model, its pre-training
    state dict, the post-plateau checkpoint, the histories, and switch_step."""

    def reward_fn_binary(a, t):
        return reward_binary_fn(a, t, tol)

    torch.manual_seed(seed * 1000)
    model = PolicyMLP(hidden_width=hidden_width)
    init_state = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed * 1000 + 1)

    mae_b, grad_b, rew_b, switch_step = run_phase(
        model, opt, reward_fn_binary, gen, x_eval, t_eval,
        max_steps=MAX_BINARY_STEPS, step_offset=0, detect_plateau=True)
    plateaued = switch_step < (MAX_BINARY_STEPS - 1)
    checkpoint_state = copy.deepcopy(model.state_dict())

    return {
        "init_state": init_state,
        "checkpoint_state": checkpoint_state,
        "switch_step": switch_step,
        "plateaued": plateaued,
        "mae": mae_b, "grad": grad_b, "reward": rew_b,
    }


def run_resume_phase(seed, checkpoint_state, switch_step, beta, x_eval, t_eval,
                      hidden_width=HIDDEN_WIDTH, seed_offset=2):
    """Phase 2a: from checkpoint, fresh optimizer ('restart'), SHAPED reward."""

    def reward_fn_shaped(a, t):
        return reward_shaped_fn(a, t, beta)

    model = PolicyMLP(hidden_width=hidden_width)
    model.load_state_dict(checkpoint_state)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed * 1000 + seed_offset)
    mae, grad, rew, _ = run_phase(
        model, opt, reward_fn_shaped, gen, x_eval, t_eval,
        max_steps=POST_SWITCH_STEPS, step_offset=switch_step, detect_plateau=False)
    return {"mae": mae, "grad": grad, "reward": rew}


def run_seed(seed, tol, beta, x_eval, t_eval, hidden_width=HIDDEN_WIDTH):
    """Full protocol: binary -> {resume (shaped), continue-binary, shaped-from-scratch}."""

    def reward_fn_binary(a, t):
        return reward_binary_fn(a, t, tol)

    def reward_fn_shaped(a, t):
        return reward_shaped_fn(a, t, beta)

    phase1 = run_binary_phase(seed, tol, x_eval, t_eval, hidden_width=hidden_width)
    switch_step = phase1["switch_step"]
    checkpoint_state = phase1["checkpoint_state"]
    init_state = phase1["init_state"]

    # --- Phase 2a: resume from checkpoint under SHAPED reward (optimizer restart) ---
    resume = run_resume_phase(seed, checkpoint_state, switch_step, beta, x_eval, t_eval,
                               hidden_width=hidden_width, seed_offset=2)

    # --- Phase 2b: continue-binary control (same optimizer restart, same reward) ---
    model_cb = PolicyMLP(hidden_width=hidden_width)
    model_cb.load_state_dict(checkpoint_state)
    opt_cb = torch.optim.Adam(model_cb.parameters(), lr=LEARNING_RATE)
    gen_cb = torch.Generator().manual_seed(seed * 1000 + 3)
    mae_cb, grad_cb, rew_cb, _ = run_phase(
        model_cb, opt_cb, reward_fn_binary, gen_cb, x_eval, t_eval,
        max_steps=POST_SWITCH_STEPS, step_offset=switch_step, detect_plateau=False)

    # --- Phase 3: shaped-from-scratch, same initial weights as Phase 1 ---
    model_scratch = PolicyMLP(hidden_width=hidden_width)
    model_scratch.load_state_dict(init_state)
    opt_scratch = torch.optim.Adam(model_scratch.parameters(), lr=LEARNING_RATE)
    gen_scratch = torch.Generator().manual_seed(seed * 1000 + 4)
    mae_s, grad_s, rew_s, _ = run_phase(
        model_scratch, opt_scratch, reward_fn_shaped, gen_scratch, x_eval, t_eval,
        max_steps=switch_step + POST_SWITCH_STEPS, step_offset=0, detect_plateau=False)

    return {
        "switch_step": switch_step,
        "plateaued": phase1["plateaued"],
        "binary": {"mae": phase1["mae"], "grad": phase1["grad"], "reward": phase1["reward"]},
        "resume": resume,
        "continue_binary": {"mae": mae_cb, "grad": grad_cb, "reward": rew_cb},
        "shaped_scratch": {"mae": mae_s, "grad": grad_s, "reward": rew_s},
    }


# ============================================================================
# Aggregation helpers
# ============================================================================
def interp_series(history, query_steps):
    """Linearly interpolate a (step, value) history onto query_steps, with
    flat extrapolation outside the recorded range (e.g. post-plateau)."""
    steps = np.array([s for s, _ in history], dtype=float)
    values = np.array([v for _, v in history], dtype=float)
    return np.interp(query_steps, steps, values)


def block_average(history, block_size, n_blocks):
    """Average a per-step (step, value) history into n_blocks consecutive
    chunks of block_size steps each, starting from the first entry."""
    values = np.array([v for _, v in history], dtype=float)
    values = values[: block_size * n_blocks]
    return values.reshape(n_blocks, block_size).mean(axis=1)


def mean_band(arr_2d):
    """arr_2d: (n_seeds, n_points). Returns (mean, lower, upper) with the
    band = mean +/- 1 std across seeds, clipped at 0 (MAE/grad-norm >= 0)."""
    mean = arr_2d.mean(axis=0)
    std = arr_2d.std(axis=0)
    return mean, np.clip(mean - std, 0, None), mean + std


# ============================================================================
# Main protocol
# ============================================================================
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    x_eval = torch.linspace(X_LOW, X_HIGH, N_EVAL_POINTS).reshape(-1, 1)
    t_eval = target_fn(x_eval).squeeze(-1)

    # ---- Calibration ----
    tol, beta, init_rate = calibrate_reward_params()
    print("=" * 70)
    print("CALIBRATION")
    print("=" * 70)
    print(f"  TOL (binary tolerance)      = {tol:.4f}")
    print(f"  BETA (shaped reward scale)  = {beta:.4f}")
    print(f"  initial binary reward rate  = {init_rate:.3f} "
          f"(target band {MODERATE_REWARD_RANGE})")

    # ---- Main protocol across seeds ----
    print("\n" + "=" * 70)
    print("MAIN PROTOCOL (per seed)")
    print("=" * 70)
    results = []
    for seed in SEED_LIST:
        res = run_seed(seed, tol, beta, x_eval, t_eval)
        results.append(res)
        final_b = res["binary"]["mae"][-1][1]
        final_r = res["resume"]["mae"][-1][1]
        final_cb = res["continue_binary"]["mae"][-1][1]
        final_s = res["shaped_scratch"]["mae"][-1][1]
        print(f"  seed={seed}: switch_step={res['switch_step']:5d} "
              f"plateaued={res['plateaued']!s:5s} | "
              f"MAE binary(end)={final_b:.4f}  resume(end)={final_r:.4f}  "
              f"continue_binary(end)={final_cb:.4f}  shaped_scratch(end)={final_s:.4f}")

    # ============================================================
    # Figure 1: oracle MAE -- binary plateau (full curves, all seeds)
    # ============================================================
    switch_steps = np.array([r["switch_step"] for r in results])
    common_steps = np.arange(0, int(switch_steps.max()) + 1, EVAL_INTERVAL)
    binary_curves = np.array([interp_series(r["binary"]["mae"], common_steps) for r in results])
    mean_bin, lo_bin, hi_bin = mean_band(binary_curves)

    plt.figure(figsize=(7, 5))
    for r in results:
        steps = np.array([s for s, _ in r["binary"]["mae"]])
        maes = np.array([m for _, m in r["binary"]["mae"]])
        plt.plot(steps, maes, color="gray", alpha=0.35, linewidth=1)
        plt.scatter([r["switch_step"]], [r["binary"]["mae"][-1][1]], color="black", s=15, zorder=5)
    plt.plot(common_steps, mean_bin, color="C0", linewidth=2, label="mean over seeds")
    plt.fill_between(common_steps, lo_bin, hi_bin, color="C0", alpha=0.25, label="+/- 1 std")
    plt.xlabel("training step")
    plt.ylabel("held-out oracle MAE")
    plt.title("Stage 2, Fig 1: binary-reward training plateaus on the held-out oracle\n"
              "(gray = per-seed curves, dots = detected switch points)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stage2_fig1_binary_plateau.png"), dpi=150)
    plt.close()

    # ============================================================
    # Figure 2: oracle MAE -- resume vs. controls, aligned at the switch
    # ============================================================
    rel_post = np.arange(0, POST_SWITCH_STEPS, EVAL_INTERVAL)
    W = PLATEAU_WINDOW * EVAL_INTERVAL
    rel_pre = np.arange(-W, 1, EVAL_INTERVAL)

    pre_curves, resume_curves, cb_curves, scratch_curves = [], [], [], []
    for r in results:
        switch = r["switch_step"]
        pre_curves.append(interp_series(r["binary"]["mae"], switch + rel_pre))
        resume_curves.append(interp_series(r["resume"]["mae"], switch + rel_post))
        cb_curves.append(interp_series(r["continue_binary"]["mae"], switch + rel_post))
        scratch_curves.append(interp_series(r["shaped_scratch"]["mae"], switch + rel_post))

    pre_arr = np.array(pre_curves)
    resume_arr = np.array(resume_curves)
    cb_arr = np.array(cb_curves)
    scratch_arr = np.array(scratch_curves)

    mean_pre, lo_pre, hi_pre = mean_band(pre_arr)
    mean_resume, lo_resume, hi_resume = mean_band(resume_arr)
    mean_cb, lo_cb, hi_cb = mean_band(cb_arr)
    mean_scratch, lo_scratch, hi_scratch = mean_band(scratch_arr)

    plt.figure(figsize=(8, 5.5))
    plt.plot(rel_pre, mean_pre, color="gray", linewidth=2, label="binary (pre-switch, plateaued)")
    plt.fill_between(rel_pre, lo_pre, hi_pre, color="gray", alpha=0.25)

    plt.plot(rel_post, mean_resume, color="C1", linewidth=2, label="binary -> shaped (resume)")
    plt.fill_between(rel_post, lo_resume, hi_resume, color="C1", alpha=0.25)

    plt.plot(rel_post, mean_cb, color="C2", linewidth=2, label="continue-binary control")
    plt.fill_between(rel_post, lo_cb, hi_cb, color="C2", alpha=0.25)

    plt.plot(rel_post, mean_scratch, color="C3", linewidth=2, linestyle="--", label="shaped-from-scratch reference")
    plt.fill_between(rel_post, lo_scratch, hi_scratch, color="C3", alpha=0.15)

    plt.axvline(0, color="black", linestyle=":", linewidth=1.5, label="reward switch / optimizer restart")
    plt.xlabel("training step relative to switch")
    plt.ylabel("held-out oracle MAE")
    plt.title(f"Stage 2, Fig 2: does the oracle resume after the reward switch?\n"
              f"(mean +/- 1 std over {len(SEED_LIST)} seeds)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stage2_fig2_resume_vs_controls.png"), dpi=150)
    plt.close()

    # ============================================================
    # Figure 3: gradient norms across the switch
    # ============================================================
    grad_pre_curves, grad_resume_curves, grad_cb_curves = [], [], []
    n_blocks_post = POST_SWITCH_STEPS // EVAL_INTERVAL
    for r in results:
        # last PLATEAU_WINDOW blocks of the binary phase, in reverse-aligned order
        grad_b = np.array([g for _, g in r["binary"]["grad"]])
        n_pre = PLATEAU_WINDOW
        tail = grad_b[-(EVAL_INTERVAL * n_pre):]
        grad_pre_curves.append(tail.reshape(n_pre, EVAL_INTERVAL).mean(axis=1))

        grad_resume_curves.append(block_average(r["resume"]["grad"], EVAL_INTERVAL, n_blocks_post))
        grad_cb_curves.append(block_average(r["continue_binary"]["grad"], EVAL_INTERVAL, n_blocks_post))

    grad_pre_arr = np.array(grad_pre_curves)
    grad_resume_arr = np.array(grad_resume_curves)
    grad_cb_arr = np.array(grad_cb_curves)

    rel_pre_blocks = np.arange(-PLATEAU_WINDOW, 0) * EVAL_INTERVAL
    rel_post_blocks = np.arange(0, n_blocks_post) * EVAL_INTERVAL

    mean_gpre, lo_gpre, hi_gpre = mean_band(grad_pre_arr)
    mean_gres, lo_gres, hi_gres = mean_band(grad_resume_arr)
    mean_gcb, lo_gcb, hi_gcb = mean_band(grad_cb_arr)

    plt.figure(figsize=(8, 5.5))
    plt.plot(rel_pre_blocks, mean_gpre, color="gray", linewidth=2, label="binary (pre-switch)")
    plt.fill_between(rel_pre_blocks, lo_gpre, hi_gpre, color="gray", alpha=0.25)
    plt.plot(rel_post_blocks, mean_gres, color="C1", linewidth=2, label="binary -> shaped (resume)")
    plt.fill_between(rel_post_blocks, lo_gres, hi_gres, color="C1", alpha=0.25)
    plt.plot(rel_post_blocks, mean_gcb, color="C2", linewidth=2, label="continue-binary control")
    plt.fill_between(rel_post_blocks, lo_gcb, hi_gcb, color="C2", alpha=0.25)
    plt.axvline(0, color="black", linestyle=":", linewidth=1.5, label="reward switch / optimizer restart")
    plt.xlabel("training step relative to switch")
    plt.ylabel("gradient L2 norm (block-averaged)")
    plt.title(f"Stage 2, Fig 3: gradient norm across the switch\n"
              f"(mean +/- 1 std over {len(SEED_LIST)} seeds)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stage2_fig3_gradient_norms.png"), dpi=150)
    plt.close()

    # ============================================================
    # Width-scaling addendum
    # ============================================================
    width_results = None
    if RUN_WIDTH_SCALING:
        print("\n" + "=" * 70)
        print("WIDTH-SCALING ADDENDUM")
        print("=" * 70)
        width_results = {"width": [], "binary_final": [], "resume_final": []}
        for width in WIDTH_SCALING_WIDTHS:
            binary_finals, resume_finals = [], []
            for seed in WIDTH_SCALING_SEEDS:
                phase1 = run_binary_phase(seed, tol, x_eval, t_eval, hidden_width=width)
                resume = run_resume_phase(seed, phase1["checkpoint_state"], phase1["switch_step"],
                                           beta, x_eval, t_eval, hidden_width=width, seed_offset=2)
                binary_finals.append(phase1["mae"][-1][1])
                resume_finals.append(resume["mae"][-1][1])
            width_results["width"].append(width)
            width_results["binary_final"].append(binary_finals)
            width_results["resume_final"].append(resume_finals)
            print(f"  width={width:3d}: binary(end) mean={np.mean(binary_finals):.4f}  "
                  f"binary->shaped(end) mean={np.mean(resume_finals):.4f}")

        widths = np.array(width_results["width"])
        binary_arr = np.array(width_results["binary_final"])   # (n_widths, n_seeds)
        resume_arr_w = np.array(width_results["resume_final"])
        mean_bf = binary_arr.mean(axis=1)
        std_bf = binary_arr.std(axis=1)
        mean_rf = resume_arr_w.mean(axis=1)
        std_rf = resume_arr_w.std(axis=1)

        plt.figure(figsize=(7, 5))
        plt.errorbar(widths, mean_bf, yerr=std_bf, marker="o", capsize=3, label="binary reward")
        plt.errorbar(widths, mean_rf, yerr=std_rf, marker="s", capsize=3, label="binary -> shaped (resume)")
        plt.xscale("log", base=2)
        plt.xlabel("MLP hidden width (log scale)")
        plt.ylabel("held-out oracle MAE at end of run")
        plt.title(f"Stage 2, Fig 4: does the binary ceiling flatten across width?\n"
                  f"(mean +/- 1 std over {len(WIDTH_SCALING_SEEDS)} seeds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "stage2_fig4_width_scaling.png"), dpi=150)
        plt.close()

    # ============================================================
    # Console summary + acceptance criteria
    # ============================================================
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)

    # 1. Calibration produced a moderate initial binary reward rate.
    crit1 = MODERATE_REWARD_RANGE[0] <= init_rate <= MODERATE_REWARD_RANGE[1]
    print(f"[{'PASS' if crit1 else 'FAIL'}] 1. initial binary reward rate {init_rate:.3f} is "
          f"within the moderate band {MODERATE_REWARD_RANGE}")

    # 2. The binary phase plateaued (slope-based trigger fired) for every seed.
    n_plateaued = sum(r["plateaued"] for r in results)
    crit2 = n_plateaued == len(results)
    print(f"[{'PASS' if crit2 else 'FAIL'}] 2. quantitative plateau detected for "
          f"{n_plateaued}/{len(results)} seeds (vs. a forced cutoff at MAX_BINARY_STEPS)")

    # 3. Resume (binary -> shaped) beats the continue-binary control at the
    #    end of the post-switch window, on the held-out oracle (lower MAE is better).
    resume_end = resume_arr[:, -1]
    cb_end = cb_arr[:, -1]
    scratch_end = scratch_arr[:, -1]
    n_better = int(np.sum(resume_end < cb_end))
    mean_improve = float(np.mean(cb_end - resume_end))
    crit3 = (resume_end.mean() < cb_end.mean()) and (n_better >= math.ceil(len(results) / 2) + 1)
    print(f"[{'PASS' if crit3 else 'FAIL'}] 3. resume beats continue-binary control: "
          f"resume wins in {n_better}/{len(results)} seeds, "
          f"mean MAE(continue_binary) - MAE(resume) = {mean_improve:+.4f} "
          f"(positive = resume is better)")
    print(f"     mean final MAE: binary(pre-switch)={mean_pre[-1]:.4f}  "
          f"resume={resume_end.mean():.4f}  continue_binary={cb_end.mean():.4f}  "
          f"shaped_scratch={scratch_end.mean():.4f}")

    # 4. Ceiling-gap report (informational): how much of the binary ceiling
    #    was recovered vs. training shaped from scratch with the same budget.
    ceiling_locked_in = float(resume_end.mean() - scratch_end.mean())
    print(f"     ceiling gap (resume - shaped_scratch, final MAE) = {ceiling_locked_in:+.4f} "
          f"(near 0 => fully recovered; positive => some of the binary ceiling was locked in)")

    # 5. Gradient-norm check: is the resume gradient larger only as a brief
    #    transient, or sustained? Compare the post-switch mean grad norm of
    #    resume vs. continue-binary.
    grad_ratio = float(mean_gres.mean() / max(mean_gcb.mean(), 1e-12))
    print(f"     mean post-switch grad-norm ratio (resume / continue_binary) = {grad_ratio:.3f} "
          f"(report only -- used to check the gain isn't just a louder gradient)")

    overall = crit1 and crit2
    verdict = "POSITIVE (oracle resumed and beat the control)" if crit3 else \
              "NEGATIVE (oracle did not resume beyond the control -- still informative: bounds the recoverable plateau)"
    print(f"\n  Setup checks (1,2): {'ALL PASSED' if overall else 'SOME FAILED -- see above'}")
    print(f"  Resume-effect result: {verdict}")
    print(f"  Figures saved to: {os.path.abspath(FIGURES_DIR)}")


if __name__ == "__main__":
    main()
