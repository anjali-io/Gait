# Results

## Best Result
- Best Rank-1 accuracy: 25.13% at iteration 60,000.
- Best Rank-5 accuracy: 39.57% at iteration 60,000.

## Training Metrics

*Note: In Metric Learning for gait recognition, accuracy is not typically a training metric. We report the losses instead.*

| Iteration | Train Loss (Triplet) | Train Loss (CE) | Total Loss |
|--------:|-------------------:|--------------:|---------:|
| 10,000 | [Fill in] | [Fill in] | [Fill in] |
| 20,000 | [Fill in] | [Fill in] | [Fill in] |
| 30,000 | [Fill in] | [Fill in] | [Fill in] |
| 40,000 | [Fill in] | [Fill in] | [Fill in] |
| 50,000 | [Fill in] | [Fill in] | [Fill in] |
| 60,000 | [Fill in] | [Fill in] | [Fill in] |

## Testing Metrics (GREW Distractor Benchmark)

| Iteration | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
|--------:|-------:|-------:|--------:|--------:|
| 60,000 | 25.13% | 39.57% | 45.47% | 51.20% |

## Notes
- Loss used: Triplet Loss + Cross-Entropy Loss (with Reliability Gating).
- Training Dataset: GREW subset (approx 500 subjects).
- Testing Dataset: GREW Distractor Benchmark (6,000 probes vs 239,857 gallery).
- Training stopped at iteration 60,000.
