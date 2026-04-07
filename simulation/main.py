"""
main.py -- Entry point for the peer prediction simulation.

Runs all experiments in sequence:
  1. Validates scoring.py math (quick sanity check)
  2. Runs full alpha sweep with M=500 rounds per configuration
  3. Prints summary table with alpha* highlighted
  4. Generates and saves all 4 plots

Usage:
  python main.py                    # Full run (default settings)
  python main.py --quick            # Fast run: M=50 rounds (for testing)
  python main.py --N 100 --q 0.8   # Custom worker count and signal accuracy
  python main.py --output results/  # Custom output directory for plots

Expected runtime: ~2-5 minutes for M=500, N=200 (full run).
"""

import argparse
import sys
import os
import numpy as np

# Ensure simulation/ is in path when run from parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import SimConfig, compute_consensus_prob
from scoring import (bayesian_update, predict_reference_report,
                     expected_payment_honest, expected_payment_consensus,
                     compute_equilibrium_gap)
from experiments import run_all_experiments, print_summary_table, find_alpha_star
from plots import generate_all_plots


def validate_scoring(cfg: SimConfig) -> bool:
    """
    Sanity-check the scoring module before running the full simulation.

    Tests that:
    1. Bayesian posteriors sum to 1 and favor the matching state
    2. g(sj | si) prediction distributions sum to 1
    3. E[honest payment] > E[consensus payment] at alpha=0 (equilibrium holds)
    4. Equilibrium gap is positive at alpha=0

    Returns True if all checks pass, raises AssertionError otherwise.
    """
    print("Running scoring sanity checks...")

    # Check 1: posteriors sum to 1 and favor the matching state
    for s in [0, 1]:
        post = bayesian_update(s, cfg)
        assert abs(post.sum() - 1.0) < 1e-10, f"Posterior doesn't sum to 1 for s={s}"
        assert post[s] > post[1 - s], f"Posterior doesn't favor theta={s} given s={s}"
    print("  [OK] Bayesian posteriors correct")

    # Check 2: g(sj | si) sums to 1
    for s in [0, 1]:
        g = predict_reference_report(s, cfg)
        assert abs(g.sum() - 1.0) < 1e-10, f"g doesn't sum to 1 for s={s}"
    print("  [OK] Prediction distributions correct")

    # Check 3: honest payment dominates at alpha=0
    consensus_prob = compute_consensus_prob(cfg)

    for s in [0, 1]:
        h_pay = expected_payment_honest(s, cfg)
        c_pay = expected_payment_consensus(s, consensus_prob, cfg)
        assert h_pay > c_pay, (
            f"Honest payment ({h_pay:.4f}) not > consensus payment ({c_pay:.4f}) "
            f"for s={s}. Equilibrium broken at alpha=0 -- check scoring logic."
        )
    print("  [OK] Honest payment dominates consensus at alpha=0 (equilibrium holds)")

    # Check 4: equilibrium gap is positive
    h, c = compute_equilibrium_gap(0.0, cfg)
    assert h > c, f"Equilibrium gap negative at alpha=0: h={h:.4f}, c={c:.4f}"
    print(f"  [OK] Equilibrium gap at alpha=0: E[honest]={h:.4f}, E[consensus]={c:.4f}, "
          f"gap={h - c:.4f}")

    print("All scoring checks passed.\n")
    return True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Peer prediction simulation: honest vs LLM-assisted workers'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: M=50 rounds (for testing)')
    parser.add_argument('--N', type=int, default=200,
                        help='Number of workers (default: 200)')
    parser.add_argument('--q', type=float, default=0.7,
                        help='Signal accuracy (default: 0.7)')
    parser.add_argument('--M', type=int, default=None,
                        help='Simulation rounds per configuration (default: 500)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for plots (default: results/)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--save-csv', type=str, default=None,
                        help='Save results to CSV file at this path')
    return parser.parse_args()


def main():
    """Run the full peer prediction simulation suite."""
    args = parse_args()

    # Configuration
    cfg = SimConfig(N=args.N, q=args.q)
    num_rounds = args.M if args.M is not None else (50 if args.quick else 500)
    num_alpha = len(np.arange(0, 1.05, 0.05))

    print("=" * 60)
    print("PEER PREDICTION SIMULATION")
    print("=" * 60)
    print(f"Config: N={cfg.N} workers, q={cfg.q} signal accuracy, "
          f"prior={cfg.prior}, M={num_rounds} rounds per alpha")
    print(f"alpha sweep: 0.00 -> 1.00 in steps of 0.05 ({num_alpha} values)")
    print("Methods: baseline + Fix A (k=3,5) + Fix B + Fix C")
    print(f"Output directory: {args.output}\n")

    # Step 1: Validate scoring math
    validate_scoring(cfg)

    # Step 2: Run full experiment sweep
    total_rounds = num_rounds * num_alpha * 5
    print(f"Starting full simulation ({num_rounds} rounds x {num_alpha} alpha values "
          f"x 5 methods = {total_rounds:,} total rounds)...")

    df = run_all_experiments(cfg=cfg, M=num_rounds, verbose=True)

    # Step 3: Print summary table
    print_summary_table(df)

    # Step 4: Find and report alpha*
    alpha_star = find_alpha_star(df, 'baseline')
    if alpha_star is not None:
        print(f"RESULT: Truthful equilibrium collapses at alpha* ~= {alpha_star:.2f}")
        print(f"  -> When more than {alpha_star * 100:.0f}% of workers use LLMs,")
        print("     honest reporting is no longer individually rational.")
    else:
        print("RESULT: Equilibrium holds for all tested alpha values (no alpha* found)")

    # Also report theoretical alpha* from scoring.py
    print("\nTheoretical analysis (from scoring.py):")
    for a in np.arange(0, 1.05, 0.05):
        cfg_a = SimConfig(N=cfg.N, q=cfg.q, alpha=a)
        h, c = compute_equilibrium_gap(a, cfg_a)
        if h <= c:
            print(f"  Theoretical alpha* ~= {a:.2f} (E[honest]={h:.4f} <= E[LLM]={c:.4f})")
            break
    else:
        print("  Mechanism remains robust theoretically across full alpha range")

    # Step 5: Save CSV if requested
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nResults saved to: {args.save_csv}")

    # Step 6: Generate plots
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
