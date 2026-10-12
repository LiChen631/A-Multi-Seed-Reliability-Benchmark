# Result-file dictionary

Each file in `results/` is the single source of truth for the numbers reported in the paper.
All runs use the 3-UAV testbed described in the README.

| File | Top-level keys | Contents | Paper location |
|---|---|---|---|
| `qbias_fixed_results.json` | `meta`, `maddpg`, `matd3`, `muav_mtd3`, `muav_std3` | Per-algorithm signed TD-target discrepancy (mean/SD per seed) **and** absolute discrepancy statistics (mean, p50, p90) over sampled state-action pairs | Table 3, Figure 2, Section 5.5 |
| `robustness_multi_seed.json` | `meta`, `maddpg`, `matd3`, `muav_mtd3`, `muav_std3` | Mean reward under observation perturbation: additive observation noise ("delay" conditions), dropout and additive Gaussian noise, aggregated over seeds | Table 6, Figure 5, Section 5.8 |
| `robustness_multi_seed.before_rerun.json` | same | Earlier version of the same experiment, retained for provenance (superseded) | — |
| `metrics_split.json` | `meta`, `<algo>_seed<k>` | Separated task metrics per run: individual arrival, team arrival, true collision (d < 25 m), safety violation (25 m <= d < 45 m), penalty trigger (d < 20 m) | Table 9, Section 6.7 |
| `interagent_diag.json` | `meta`, `<algo>_seed<k>` | Inter-agent coupling frequency (fraction of steps with d_ij < 20 m and < 25 m), minimum inter-agent distance, formation error, formation-centroid distance | Table 10, Section 6.7 |
| `云服务器实测数据_20261001.json` | `_meta`, `degeneracy_5seed`, `metrics_split_seeds42_123`, `timing_rollout_only`, `timing_clean_20261001`, `qbias_seeds42_123_20261001` | Consolidated measurements from the evaluation server: per-seed degeneracy/saturation, split metrics, timing of gradient updates and rollout steps, discrepancy on seeds 42/123 | Tables 3, 5, 6, 8; Section 4.4 |
| `log_robustness_5seed.txt`, `log_A_diag.txt`, `diag_5seed.txt`, `time_clean.txt`, `time_training.txt`, `_数据核对报告.txt` | — | Raw logs and cross-check notes kept as an audit trail | — |

## Conventions

* Seeds: **42, 123, 456, 789, 1024** (primary evaluation); the discrepancy and robustness experiments use 42, 123, 456.
* Seeds 456, 789 and 1024 contain degenerate (fully saturated) controllers; check `degeneracy_5seed` before interpreting any aggregate.
* All distances are in metres and expressed as **centre-to-centre** distances; an obstacle has radius r = 25 m, so a centre distance of 45 m corresponds to a 20 m clearance from the obstacle boundary.
* "Collision" is a centre-distance criterion, **not** a physical contact model; the paper does not claim verified collision avoidance.
