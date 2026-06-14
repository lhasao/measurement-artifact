"""
Stage 2 follow-up analyses.

This script does NOT change the Stage 2 protocol or controls. It reuses the
exact training functions from stage2_saturate_resume.py (PolicyMLP,
reinforce_step with batch-mean baseline subtraction, run_phase with its
slope-based plateau detector, run_binary_phase, run_resume_phase, and the
continue-binary control as defined in run_seed's Phase 2b). Only the *inputs*
to that protocol change per check: seed count, hidden width + target function
(Check 2 only), and POLICY_STD (Check 3 only).

Check 2 -- gap (continue-binary - resume) vs width.
    Priority check. First confirms width is a real scaling axis (does
    shaped-from-scratch MAE fall as width grows from 4 to 64?). If not with
    the original target sin(x), switches to a harder target and recalibrates
    TOL/BETA, then re-checks. Only once that passes does it compute the gap
    and plot it.

Check 1 -- paired significance of the resume effect.
    Per-seed paired difference (resume final MAE) - (continue-binary final
    MAE), for 5 seeds and then 15 seeds. Reports mean and std plainly; if std
    is comparable to the mean, says so.

Check 3 -- exploration-grain control.
    Re-runs the headline resume-vs-control comparison with POLICY_STD = 0.1
    (everything else, including TOL/BETA, unchanged) and reports whether the
    resume effect survives.

Run order: Check 2, then Check 1, then Check 3 (as specified).

Run with: python stage2_followup_analyses.py
"""

import contextlib
import math
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

import stage2_saturate_resume as s2

# ============================================================================
# CONFIG
# ============================================================================
FIGURES_DIR = s2.FIGURES_DIR

# --- Check 2: gap vs width ---
WIDTH_TARGET_FREQ = 4.0          # candidate harder target: sin(WIDTH_TARGET_FREQ * x)
CONFIRM_STEPS = 1000              # shaped-from-scratch training budget for the width-confirmation step
CONFIRM_SEEDS = [0, 1]
SCALING_RATIO_THRESHOLD = 1.5     # "decreases visibly" => MAE(width=4) / MAE(width=64) > this
GAP_WIDTHS = s2.WIDTH_SCALING_WIDTHS      # [4, 8, 16, 32, 64]
GAP_SEEDS = s2.WIDTH_SCALING_SEEDS        # [0, 1, 2]

# --- Check 1: paired significance ---
SEEDS_5 = list(range(5))
SEEDS_15 = list(range(15))

# --- Check 3: exploration-grain control ---
POLICY_STD_SMALL = 0.1


def target_fn_width(x):
    return torch.sin(WIDTH_TARGET_FREQ * x)


# ============================================================================
# Helpers: monkeypatch the two module globals (target_fn, POLICY_STD) that
# stage2_saturate_resume's reinforce_step() reads at call time. The training
# algorithm itself (run_phase / reinforce_step / baseline subtraction /
# plateau detection / optimizer restart) is untouched.
# ============================================================================
@contextlib.contextmanager
def patched_target(target_fn):
    old = s2.target_fn
    s2.target_fn = target_fn
    try:
        yield
    finally:
        s2.target_fn = old


@contextlib.contextmanager
def patched_policy_std(std):
    old = s2.POLICY_STD
    s2.POLICY_STD = std
    try:
        yield
    finally:
        s2.POLICY_STD = old


def shaped_from_scratch_mae(seed, width, beta, target_fn, x_eval, t_eval, n_steps):
    """Train a fresh model under the SHAPED reward for n_steps; return final oracle MAE."""
    def reward_fn_shaped(a, t):
        return s2.reward_shaped_fn(a, t, beta)

    torch.manual_seed(seed * 1000)
    model = s2.PolicyMLP(hidden_width=width)
    opt = torch.optim.Adam(model.parameters(), lr=s2.LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed * 1000 + 4)
    with patched_target(target_fn):
        mae_hist, _, _, _ = s2.run_phase(model, opt, reward_fn_shaped, gen, x_eval, t_eval,
                                          max_steps=n_steps, step_offset=0, detect_plateau=False)
    return mae_hist[-1][1]


def run_continue_binary_phase(seed, checkpoint_state, switch_step, tol, x_eval, t_eval,
                               hidden_width=s2.HIDDEN_WIDTH, seed_offset=3):
    """Same as run_seed's Phase 2b control: from the binary-plateau checkpoint, fresh
    optimizer ('restart'), but keep the BINARY reward."""
    def reward_fn_binary(a, t):
        return s2.reward_binary_fn(a, t, tol)

    model = s2.PolicyMLP(hidden_width=hidden_width)
    model.load_state_dict(checkpoint_state)
    opt = torch.optim.Adam(model.parameters(), lr=s2.LEARNING_RATE)
    gen = torch.Generator().manual_seed(seed * 1000 + seed_offset)
    mae, grad, rew, _ = s2.run_phase(model, opt, reward_fn_binary, gen, x_eval, t_eval,
                                      max_steps=s2.POST_SWITCH_STEPS, step_offset=switch_step,
                                      detect_plateau=False)
    return {"mae": mae, "grad": grad, "reward": rew}


def reward_rate_at_std(tol, std, target_fn):
    """P(|a - t| < tol) at initialization (CALIBRATION_SEED), for a given std,
    WITHOUT changing tol -- used to report what a fixed tolerance implies under a
    different exploration scale."""
    torch.manual_seed(s2.CALIBRATION_SEED)
    model = s2.PolicyMLP()
    gen = torch.Generator().manual_seed(s2.CALIBRATION_SEED)
    x = s2.sample_x(s2.CALIBRATION_BATCH, gen)
    t = target_fn(x).squeeze(-1)
    with torch.no_grad():
        mu = model(x)
    dist = torch.distributions.Normal(mu, std)
    a = dist.sample()
    err = (a - t).abs()
    return (err < tol).float().mean().item()


# ============================================================================
# Check 2: gap (continue-binary - resume) vs width
# ============================================================================
def check2_gap_vs_width():
    print("=" * 70)
    print("CHECK 2: gap (continue-binary - resume) vs width")
    print("=" * 70)

    x_eval = torch.linspace(s2.X_LOW, s2.X_HIGH, s2.N_EVAL_POINTS).reshape(-1, 1)

    def confirm(name, target_fn):
        t_eval = target_fn(x_eval).squeeze(-1)
        with patched_target(target_fn):
            tol, beta, rate = s2.calibrate_reward_params()
        means = []
        for width in GAP_WIDTHS:
            vals = [shaped_from_scratch_mae(seed, width, beta, target_fn, x_eval, t_eval, CONFIRM_STEPS)
                    for seed in CONFIRM_SEEDS]
            means.append(float(np.mean(vals)))
        ratio = means[0] / means[-1] if means[-1] > 0 else float("inf")
        passed = ratio > SCALING_RATIO_THRESHOLD
        print(f"\n  {name}: TOL={tol:.4f} BETA={beta:.4f} init_rate={rate:.3f}")
        print(f"    shaped-from-scratch MAE by width ({CONFIRM_STEPS} steps, seeds={CONFIRM_SEEDS}):")
        print("      " + "  ".join(f"w={w}:{m:.4f}" for w, m in zip(GAP_WIDTHS, means)))
        print(f"    width=4 -> width=64 ratio = {ratio:.2f}x  "
              f"=> [{'PASS' if passed else 'FAIL'}] {'decreases visibly' if passed else 'does NOT decrease visibly'} "
              f"(threshold {SCALING_RATIO_THRESHOLD}x)")
        return tol, beta, means, passed

    print("\nStep A: confirm width is a real scaling axis (shaped-from-scratch MAE vs width)")
    tol_a, beta_a, means_a, passed_a = confirm("(a) original target sin(x)", s2.target_fn)

    if passed_a:
        target_fn_gap, tol_w, beta_w, gap_target_name = s2.target_fn, tol_a, beta_a, "sin(x)"
    else:
        print("\n  sin(x) is essentially saturated at width=4 (the task is too easy to use as a "
              "width-scaling axis). Switching to a harder target and recalibrating.")
        tol_b, beta_b, means_b, passed_b = confirm(f"(b) harder target sin({WIDTH_TARGET_FREQ:.0f}x)", target_fn_width)
        if not passed_b:
            print("\n  Neither target shows a clear width-scaling trend. Proceeding with the harder "
                  "target anyway, but the gap-vs-width plot below should be read with that caveat.")
        target_fn_gap, tol_w, beta_w = target_fn_width, tol_b, beta_b
        gap_target_name = f"sin({WIDTH_TARGET_FREQ:.0f}x)"

    print(f"\nStep B: gap (continue-binary - resume) vs width, target = {gap_target_name}, "
          f"TOL={tol_w:.4f}, BETA={beta_w:.4f}, seeds={GAP_SEEDS}")

    t_eval_gap = target_fn_gap(x_eval).squeeze(-1)
    gap_means, gap_stds = [], []
    with patched_target(target_fn_gap):
        for width in GAP_WIDTHS:
            gaps, plateaued_flags, switch_steps = [], [], []
            for seed in GAP_SEEDS:
                phase1 = s2.run_binary_phase(seed, tol_w, x_eval, t_eval_gap, hidden_width=width)
                resume = s2.run_resume_phase(seed, phase1["checkpoint_state"], phase1["switch_step"],
                                              beta_w, x_eval, t_eval_gap, hidden_width=width, seed_offset=2)
                cb = run_continue_binary_phase(seed, phase1["checkpoint_state"], phase1["switch_step"],
                                                tol_w, x_eval, t_eval_gap, hidden_width=width, seed_offset=3)
                gaps.append(cb["mae"][-1][1] - resume["mae"][-1][1])
                plateaued_flags.append(phase1["plateaued"])
                switch_steps.append(phase1["switch_step"])
            gap_means.append(float(np.mean(gaps)))
            gap_stds.append(float(np.std(gaps)))
            print(f"  width={width:3d}: gap mean={gap_means[-1]:+.4f} std={gap_stds[-1]:.4f}  "
                  f"per-seed={[round(g, 4) for g in gaps]}  "
                  f"switch_steps={switch_steps}  plateaued={plateaued_flags}")

    widths_arr = np.array(GAP_WIDTHS, dtype=float)
    means_arr = np.array(gap_means)
    stds_arr = np.array(gap_stds)

    plt.figure(figsize=(7, 5))
    plt.errorbar(widths_arr, means_arr, yerr=stds_arr, marker="o", capsize=3, color="C0")
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.xscale("log", base=2)
    plt.xlabel("MLP hidden width (log scale)")
    plt.ylabel("gap = MAE(continue-binary) - MAE(resume)")
    plt.title(f"Stage 2 follow-up, Fig 5: resume advantage vs width (target = {gap_target_name})\n"
              f"(mean +/- 1 std over {len(GAP_SEEDS)} seeds; positive = resume better)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stage2_fig5_gap_vs_width.png"), dpi=150)
    plt.close()

    end_diff = float(means_arr[-1] - means_arr[0])
    pooled_std = float(np.mean(stds_arr))
    if abs(end_diff) < pooled_std:
        trend = (f"roughly FLAT (width=64 minus width=4 gap = {end_diff:+.4f}, smaller than "
                 f"the typical per-width std {pooled_std:.4f})")
    elif end_diff > 0:
        trend = (f"GROWS with width (width=64 minus width=4 gap = {end_diff:+.4f}, "
                 f"larger than the typical per-width std {pooled_std:.4f})")
    else:
        trend = (f"SHRINKS with width (width=64 minus width=4 gap = {end_diff:+.4f}, "
                 f"larger than the typical per-width std {pooled_std:.4f})")
    print(f"\n  => the resume-vs-control gap {trend}.")
    print(f"  Figure saved: figures/stage2_fig5_gap_vs_width.png")

    return {
        "gap_target_name": gap_target_name,
        "passed_a": passed_a,
        "means_a": means_a,
        "gap_widths": GAP_WIDTHS,
        "gap_means": gap_means,
        "gap_stds": gap_stds,
        "trend": trend,
    }


# ============================================================================
# Check 1: paired significance of the resume effect
# ============================================================================
def check1_paired_significance():
    print("\n" + "=" * 70)
    print("CHECK 1: paired significance of the resume effect")
    print("=" * 70)

    x_eval = torch.linspace(s2.X_LOW, s2.X_HIGH, s2.N_EVAL_POINTS).reshape(-1, 1)
    t_eval = s2.target_fn(x_eval).squeeze(-1)
    tol, beta, _ = s2.calibrate_reward_params()

    def paired_diffs(seed_list):
        diffs = []
        for seed in seed_list:
            phase1 = s2.run_binary_phase(seed, tol, x_eval, t_eval)
            resume = s2.run_resume_phase(seed, phase1["checkpoint_state"], phase1["switch_step"],
                                          beta, x_eval, t_eval, seed_offset=2)
            cb = run_continue_binary_phase(seed, phase1["checkpoint_state"], phase1["switch_step"],
                                            tol, x_eval, t_eval, hidden_width=s2.HIDDEN_WIDTH, seed_offset=3)
            diffs.append(resume["mae"][-1][1] - cb["mae"][-1][1])
        return np.array(diffs)

    def report(seed_list, diffs):
        print(f"\n  {len(seed_list)} seeds: per-seed (resume - continue_binary) final MAE "
              f"[negative = resume better]:")
        print("    " + "  ".join(f"seed{s}:{d:+.4f}" for s, d in zip(seed_list, diffs)))
        mean, std = float(diffs.mean()), float(diffs.std())
        print(f"    mean = {mean:+.4f}   std = {std:.4f}")
        if std >= abs(mean):
            print(f"    => std ({std:.4f}) >= |mean| ({abs(mean):.4f}): the effect is within "
                  f"noise at {len(seed_list)} seeds, not yet supported by this test alone.")
        else:
            print(f"    => |mean| ({abs(mean):.4f}) > std ({std:.4f}): the effect is larger than "
                  f"seed-to-seed spread at {len(seed_list)} seeds.")
        return mean, std

    diffs_5 = paired_diffs(SEEDS_5)
    mean_5, std_5 = report(SEEDS_5, diffs_5)

    diffs_15 = paired_diffs(SEEDS_15)
    mean_15, std_15 = report(SEEDS_15, diffs_15)

    return {"diffs_5": diffs_5, "mean_5": mean_5, "std_5": std_5,
            "diffs_15": diffs_15, "mean_15": mean_15, "std_15": std_15}


# ============================================================================
# Check 3: exploration-grain control (smaller fixed POLICY_STD)
# ============================================================================
def check3_exploration_grain():
    print("\n" + "=" * 70)
    print(f"CHECK 3: exploration-grain control (POLICY_STD = {POLICY_STD_SMALL}, "
          f"vs headline POLICY_STD = {s2.POLICY_STD})")
    print("=" * 70)

    x_eval = torch.linspace(s2.X_LOW, s2.X_HIGH, s2.N_EVAL_POINTS).reshape(-1, 1)
    t_eval = s2.target_fn(x_eval).squeeze(-1)

    # TOL/BETA are NOT recalibrated -- "everything else fixed" means reusing the
    # same calibration as the headline (POLICY_STD=0.3) run.
    tol, beta, rate_headline = s2.calibrate_reward_params()
    rate_small = reward_rate_at_std(tol, POLICY_STD_SMALL, s2.target_fn)
    print(f"\n  TOL={tol:.4f}, BETA={beta:.4f} (unchanged, calibrated at POLICY_STD={s2.POLICY_STD})")
    print(f"  initial binary reward rate: {rate_headline:.3f} at POLICY_STD={s2.POLICY_STD}  "
          f"vs {rate_small:.3f} at POLICY_STD={POLICY_STD_SMALL}  "
          f"(moderate band {s2.MODERATE_REWARD_RANGE})")
    if not (s2.MODERATE_REWARD_RANGE[0] <= rate_small <= s2.MODERATE_REWARD_RANGE[1]):
        print(f"  NOTE: at POLICY_STD={POLICY_STD_SMALL} the same TOL gives an initial reward rate "
              f"OUTSIDE the moderate band -- the calibration done for POLICY_STD=0.3 does not "
              f"transfer to this std.")

    switch_steps, plateaued_flags, resume_finals, cb_finals = [], [], [], []
    with patched_policy_std(POLICY_STD_SMALL):
        for seed in SEEDS_5:
            phase1 = s2.run_binary_phase(seed, tol, x_eval, t_eval)
            resume = s2.run_resume_phase(seed, phase1["checkpoint_state"], phase1["switch_step"],
                                          beta, x_eval, t_eval, seed_offset=2)
            cb = run_continue_binary_phase(seed, phase1["checkpoint_state"], phase1["switch_step"],
                                            tol, x_eval, t_eval, hidden_width=s2.HIDDEN_WIDTH, seed_offset=3)
            switch_steps.append(phase1["switch_step"])
            plateaued_flags.append(phase1["plateaued"])
            resume_finals.append(resume["mae"][-1][1])
            cb_finals.append(cb["mae"][-1][1])

    resume_finals = np.array(resume_finals)
    cb_finals = np.array(cb_finals)
    n_better = int(np.sum(resume_finals < cb_finals))
    mean_improve = float(np.mean(cb_finals - resume_finals))

    print(f"\n  per-seed switch_step={switch_steps}")
    print(f"  per-seed plateaued={plateaued_flags}")
    print(f"  mean final MAE: resume={resume_finals.mean():.4f}  continue_binary={cb_finals.mean():.4f}")
    print(f"  resume wins in {n_better}/{len(SEEDS_5)} seeds, "
          f"mean improvement (MAE(continue_binary) - MAE(resume)) = {mean_improve:+.4f} "
          f"(positive = resume better)")

    survives = (resume_finals.mean() < cb_finals.mean()) and (n_better >= math.ceil(len(SEEDS_5) / 2) + 1)
    if survives:
        print(f"  => the resume effect SURVIVES at POLICY_STD={POLICY_STD_SMALL} by the same "
              f"win-count/mean-improvement criterion used for the headline result.")
    else:
        print(f"  => the resume effect DOES NOT survive at POLICY_STD={POLICY_STD_SMALL} by that "
              f"criterion -- consistent with the original effect being (at least partly) about "
              f"exploration grain rather than reward shape alone.")

    return {"rate_headline": rate_headline, "rate_small": rate_small,
            "switch_steps": switch_steps, "plateaued": plateaued_flags,
            "resume_finals": resume_finals, "cb_finals": cb_finals,
            "n_better": n_better, "mean_improve": mean_improve, "survives": survives}


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    check2_gap_vs_width()
    check1_paired_significance()
    check3_exploration_grain()
