"""
Entry point. Validates scoring math, runs the full alpha sweep, prints summary, saves plots.

Usage:
  python main.py                    # full run (M=500)
  python main.py --quick            # M=50 for testing
  python main.py --N 100 --q 0.8
  python main.py --output results/
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import SimConfig, compute_consensus_prob
from scoring import (bayesian_update, predict_reference_report,
                     expected_payment_honest, expected_payment_consensus,
                     compute_equilibrium_gap)
from experiments import run_all_experiments, print_summary_table, find_alpha_star
from plots import generate_all_plots


def validate_scoring(cfg: SimConfig) -> bool:
    """Sanity-check scoring.py before the full run."""
    print("Running scoring sanity checks...")

    for s in [0, 1]:
        post = bayesian_update(s, cfg)
        assert abs(post.sum() - 1.0) < 1e-10
        assert post[s] > post[1 - s], f"Posterior doesn't favor theta={s} given s={s}"
    print("  [OK] Bayesian posteriors correct")

    for s in [0, 1]:
        g = predict_reference_report(s, cfg)
        assert abs(g.sum() - 1.0) < 1e-10
    print("  [OK] Prediction distributions correct")

    consensus_prob = compute_consensus_prob(cfg)
    for s in [0, 1]:
        h_pay = expected_payment_honest(s, cfg)
        c_pay = expected_payment_consensus(s, consensus_prob, cfg)
        assert h_pay > c_pay, (
            f"Honest payment ({h_pay:.4f}) <= consensus ({c_pay:.4f}) at s={s}, alpha=0"
        )
    print("  [OK] Honest payment dominates consensus at alpha=0")

    h, c = compute_equilibrium_gap(0.0, cfg)
    assert h > c
    print(f"  [OK] Equilibrium gap at alpha=0: E[honest]={h:.4f}, E[LLM]={c:.4f}, gap={h-c:.4f}")

    print("All checks passed.\n")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Peer prediction simulation: honest vs LLM-assisted workers'
    )
    parser.add_argument('--quick', action='store_true', help='M=50 rounds (for testing)')
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--q', type=float, default=0.7)
    parser.add_argument('--M', type=int, default=None)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--save-csv', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = SimConfig(N=args.N, q=args.q)
    num_rounds = args.M if args.M is not None else (50 if args.quick else 500)
    num_alpha = len(np.arange(0, 1.05, 0.05))

    print("=" * 60)
    print("PEER PREDICTION SIMULATION")
    print("=" * 60)
    print(f"Config: N={cfg.N}, q={cfg.q}, prior={cfg.prior}, M={num_rounds}")
    print(f"alpha sweep: 0.00 -> 1.00 in steps of 0.05 ({num_alpha} values)")
    print(f"Output: {args.output}\n")

    validate_scoring(cfg)

    print(f"Starting simulation ({num_rounds} x {num_alpha} x 5 = "
          f"{num_rounds * num_alpha * 5:,} total rounds)...")

    df = run_all_experiments(cfg=cfg, M=num_rounds, verbose=True)
    print_summary_table(df)

    alpha_star = find_alpha_star(df, 'baseline')
    if alpha_star is not None:
        print(f"RESULT: Truthful equilibrium collapses at alpha* ~= {alpha_star:.2f}")
        print(f"  -> When more than {alpha_star * 100:.0f}% of workers use LLMs,")
        print("     honest reporting is no longer individually rational.")
    else:
        print("RESULT: Equilibrium holds for all tested alpha values")

    print("\nTheoretical equilibrium gap by alpha:")
    for a in np.arange(0, 1.05, 0.05):
        cfg_a = SimConfig(N=cfg.N, q=cfg.q, alpha=a)
        h, c = compute_equilibrium_gap(a, cfg_a)
        if h <= c:
            print(f"  Theoretical alpha* ~= {a:.2f} (E[honest]={h:.4f} <= E[LLM]={c:.4f})")
            break
    else:
        print("  Mechanism remains theoretically robust across full alpha range")

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nResults saved to: {args.save_csv}")

    if not args.no_plots:
        print("\nGenerating plots...")
        out_dir = args.output
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
        generate_all_plots(df, output_dir=out_dir)

    print("\nSimulation complete.")
    return df


if __name__ == '__main__':
    main()
