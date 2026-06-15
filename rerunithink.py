"""
Stage 3: RLHF Preference Simulator and Training Pipeline

Hypothesis under test
----------------------
Does binary human-preference aggregation (a Bradley-Terry setup) induce a
gradient-starvation capability ceiling similar to the absolute binary reward in Stage 2?
Switching from a binary pairwise preference reward to the continuous Bradley-Terry
probability (shaped) should allow the model to resume improving if it plateaued.

Task
----
Same 1D regression-as-control as Stage 2.
Instead of an absolute tolerance, the agent generates two actions a1, a2 for each input.
Reward is based on the difference in true utility (negative error): u = -|a - t|.

Training rewards:
    binary : r1 = 1 if u1 + noise1 > u2 + noise2 else 0
    shaped : r1 = sigmoid((u1 - u2) / temperature)

Held-out oracle: MAE of the policy MEAN.
"""

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
SEED_LIST = [0, 1, 2, 3, 4]
X_LOW, X_HIGH = -3.0, 3.0

def target_fn(x):
    return torch.sin(x)

# --- policy network ---
HIDDEN_WIDTH = 32
N_HIDDEN_LAYERS = 2
POLICY_STD = 0.3

# --- optimization ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-2

# --- reward parameters ---
# Instead of absolute TOL, we have human perception noise and temperature for Bradley-Terry
HUMAN_NOISE_SIGMA = 0.5
TEMPERATURE = math.sqrt(2) * HUMAN_NOISE_SIGMA

# --- quantitative plateau detection ---
EVAL_INTERVAL = 20
N_EVAL_POINTS = 200
PLATEAU_WINDOW = 10
PLATEAU_SLOPE_THRESH = 2e-4
MIN_BINARY_STEPS = 300
MAX_BINARY_STEPS = 2000
POST_SWITCH_STEPS = 1200

RUN_WIDTH_SCALING = False

FIGURES_DIR = "figures"

# ============================================================================
# Model, rewards, sampling
# ============================================================================
class PolicyMLP(nn.Module):
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

def sample_x(batch_size, generator):
    return torch.empty(batch_size, 1).uniform_(X_LOW, X_HIGH, generator=generator)

def oracle_mae(model, x_eval, t_eval):
    with torch.no_grad():
        mu = model(x_eval)
    return (mu - t_eval).abs().mean().item()

# ============================================================================
# Paired REINFORCE step
# ============================================================================
def paired_reinforce_step(model, optimizer, std, reward_type, generator):
    """
    reward_type: 'binary' or 'shaped'
    """
    x = sample_x(BATCH_SIZE, generator)
    t = target_fn(x).squeeze(-1)
    mu = model(x)
    dist = torch.distributions.Normal(mu, std)
    
    a1 = dist.sample()
    a2 = dist.sample()
    
    u1 = -torch.abs(a1 - t)
    u2 = -torch.abs(a2 - t)
    
    if reward_type == 'binary':
        noise1 = torch.empty_like(u1).normal_(0, HUMAN_NOISE_SIGMA, generator=generator)
        noise2 = torch.empty_like(u2).normal_(0, HUMAN_NOISE_SIGMA, generator=generator)
        r1 = (u1 + noise1 > u2 + noise2).float()
    else:
        r1 = torch.sigmoid((u1 - u2) / TEMPERATURE)
        
    r2 = 1.0 - r1
    
    # Baseline for paired comparisons is exactly 0.5 per action
    adv1 = (r1 - 0.5).detach()
    adv2 = (r2 - 0.5).detach()
    
    log_prob1 = dist.log_prob(a1)
    log_prob2 = dist.log_prob(a2)
    
    loss = -(adv1 * log_prob1 + adv2 * log_prob2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.cat([p.grad.detach().flatten() for p in model.parameters()]).norm().item()
    optimizer.step()

    return r1.mean().item(), grad_norm

# ============================================================================
# Training-phase runner
# ============================================================================
def run_phase(model, optimizer, reward_type, generator, x_eval, t_eval,
               max_steps, step_offset=0, detect_plateau=False):
    mae_history, grad_history, reward_history = [], [], []

    for local_step in range(max_steps):
        batch_reward, grad_norm = paired_reinforce_step(model, optimizer, POLICY_STD, reward_type, generator)
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
# Per-seed protocol
# ============================================================================
def run_binary_phase(seed, x_eval, t_eval, hidden_width=HIDDEN_WIDTH):
    torch.manual_seed(seed * 1000)
    model = PolicyMLP(hidden_width=hidden_width)
    init_state = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed * 1000 + 1)

    mae_b, grad_b, rew_b, switch_step = run_phase(
        model, opt, 'binary', gen, x_eval, t_eval,
        max_steps=MAX_BINARY_STEPS, step_offset=0, detect_plateau=True)
    plateaued = switch_step < (MAX_BINARY_STEPS - 1)
    checkpoint_state = copy.deepcopy(model.state_dict())

    return {
        "init_state": init_state, "checkpoint_state": checkpoint_state,
        "switch_step": switch_step, "plateaued": plateaued,
        "mae": mae_b, "grad": grad_b, "reward": rew_b,
    }

def run_resume_phase(seed, checkpoint_state, switch_step, x_eval, t_eval,
                      hidden_width=HIDDEN_WIDTH, seed_offset=2):
    model = PolicyMLP(hidden_width=hidden_width)
    model.load_state_dict(checkpoint_state)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed * 1000 + seed_offset)
    mae, grad, rew, _ = run_phase(
        model, opt, 'shaped', gen, x_eval, t_eval,
        max_steps=POST_SWITCH_STEPS, step_offset=switch_step, detect_plateau=False)
    return {"mae": mae, "grad": grad, "reward": rew}

def run_seed(seed, x_eval, t_eval, hidden_width=HIDDEN_WIDTH):
    phase1 = run_binary_phase(seed, x_eval, t_eval, hidden_width=hidden_width)
    switch_step = phase1["switch_step"]
    checkpoint_state = phase1["checkpoint_state"]
    init_state = phase1["init_state"]

    resume = run_resume_phase(seed, checkpoint_state, switch_step, x_eval, t_eval,
                               hidden_width=hidden_width, seed_offset=2)

    model_cb = PolicyMLP(hidden_width=hidden_width)
    model_cb.load_state_dict(checkpoint_state)
    opt_cb = torch.optim.Adam(model_cb.parameters(), lr=LEARNING_RATE)
    gen_cb = torch.Generator().manual_seed(seed * 1000 + 3)
    mae_cb, grad_cb, rew_cb, _ = run_phase(
        model_cb, opt_cb, 'binary', gen_cb, x_eval, t_eval,
        max_steps=POST_SWITCH_STEPS, step_offset=switch_step, detect_plateau=False)

    model_scratch = PolicyMLP(hidden_width=hidden_width)
    model_scratch.load_state_dict(init_state)
    opt_scratch = torch.optim.Adam(model_scratch.parameters(), lr=LEARNING_RATE)
    gen_scratch = torch.Generator().manual_seed(seed * 1000 + 4)
    mae_s, grad_s, rew_s, _ = run_phase(
        model_scratch, opt_scratch, 'shaped', gen_scratch, x_eval, t_eval,
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
    steps = np.array([s for s, _ in history], dtype=float)
    values = np.array([v for _, v in history], dtype=float)
    return np.interp(query_steps, steps, values)

def block_average(history, block_size, n_blocks):
    values = np.array([v for _, v in history], dtype=float)
    values = values[: block_size * n_blocks]
    return values.reshape(n_blocks, block_size).mean(axis=1)

def mean_band(arr_2d):
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

    print("=" * 70)
    print(f"HUMAN_NOISE_SIGMA = {HUMAN_NOISE_SIGMA:.4f}")
    print(f"TEMPERATURE       = {TEMPERATURE:.4f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("MAIN PROTOCOL (per seed)")
    print("=" * 70)
    results = []
    for seed in SEED_LIST:
        res = run_seed(seed, x_eval, t_eval)
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
    # Figures
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
    plt.title("Stage 3, Fig 1: Binary RLHF training plateaus on the held-out oracle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stage3_fig1_binary_plateau.png"), dpi=150)
    plt.close()

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

    mean_pre, lo_pre, hi_pre = mean_band(np.array(pre_curves))
    mean_resume, lo_resume, hi_resume = mean_band(np.array(resume_curves))
    mean_cb, lo_cb, hi_cb = mean_band(np.array(cb_curves))
    mean_scratch, lo_scratch, hi_scratch = mean_band(np.array(scratch_curves))

    plt.figure(figsize=(8, 5.5))
    plt.plot(rel_pre, mean_pre, color="gray", linewidth=2, label="binary (pre-switch, plateaued)")
    plt.fill_between(rel_pre, lo_pre, hi_pre, color="gray", alpha=0.25)
    plt.plot(rel_post, mean_resume, color="C1", linewidth=2, label="binary -> shaped (resume)")
    plt.fill_between(rel_post, lo_resume, hi_resume, color="C1", alpha=0.25)
    plt.plot(rel_post, mean_cb, color="C2", linewidth=2, label="continue-binary control")
    plt.fill_between(rel_post, lo_cb, hi_cb, color="C2", alpha=0.25)
    plt.plot(rel_post, mean_scratch, color="C3", linewidth=2, linestyle="--", label="shaped-from-scratch reference")
    plt.fill_between(rel_post, lo_scratch, hi_scratch, color="C3", alpha=0.15)
    plt.axvline(0, color="black", linestyle=":", linewidth=1.5, label="reward switch")
    plt.xlabel("training step relative to switch")
    plt.ylabel("held-out oracle MAE")
    plt.title(f"Stage 3, Fig 2: Resume vs Controls\n(mean +/- 1 std over {len(SEED_LIST)} seeds)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stage3_fig2_resume_vs_controls.png"), dpi=150)
    plt.close()

    # ============================================================
    # Console summary + acceptance criteria
    # ============================================================
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA")
    print("=" * 70)

    n_plateaued = sum(r["plateaued"] for r in results)
    crit2 = n_plateaued == len(results)
    print(f"[{'PASS' if crit2 else 'FAIL'}] 1. quantitative plateau detected for "
          f"{n_plateaued}/{len(results)} seeds")

    resume_end = np.array([r["resume"]["mae"][-1][1] for r in results])
    cb_end = np.array([r["continue_binary"]["mae"][-1][1] for r in results])
    n_better = int(np.sum(resume_end < cb_end))
    mean_improve = float(np.mean(cb_end - resume_end))
    crit3 = (resume_end.mean() < cb_end.mean()) and (n_better >= math.ceil(len(results) / 2) + 1)
    
    print(f"[{'PASS' if crit3 else 'FAIL'}] 2. resume beats continue-binary control: "
          f"resume wins in {n_better}/{len(results)} seeds, "
          f"mean improvement = {mean_improve:+.4f}")

    verdict = "POSITIVE (oracle resumed and beat the control)" if crit3 else \
              "NEGATIVE (oracle did not resume beyond the control)"
    print(f"\n  Resume-effect result: {verdict}")
    print(f"  Figures saved to: {os.path.abspath(FIGURES_DIR)}")

if __name__ == "__main__":
    main()
