# tiny-lineup-round0: 20 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.725 [0.369, 1.145] | 0.438 [0.263, 0.600] | 1.000 [1.000, 1.000] | 0.897 [0.880, 0.912] | 1.875 [1.513, 2.288] | 702 |
| 1 | 1.025 [0.568, 1.566] | 0.525 [0.352, 0.700] | 0.537 [0.352, 0.737] | 0.892 [0.867, 0.911] | 1.788 [1.239, 2.395] | 695 |
| 4 | 0.475 [0.188, 0.816] | 0.275 [0.119, 0.452] | 0.287 [0.138, 0.474] | 0.898 [0.884, 0.912] | 0.812 [0.434, 1.250] | 697 |
| 16 | 0.338 [0.091, 0.655] | 0.212 [0.075, 0.382] | 0.025 [0.000, 0.079] | 0.891 [0.875, 0.907] | 0.450 [0.138, 0.829] | 702 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.375 [0.138, 0.631] | 0.300 [0.125, 0.475] | 1.000 [1.000, 1.000] |
| 1 | 0.575 [0.263, 0.975] | 0.388 [0.217, 0.566] | 0.537 [0.352, 0.737] |
| 4 | 0.237 [0.071, 0.440] | 0.212 [0.071, 0.395] | 0.225 [0.071, 0.421] |
| 16 | 0.100 [0.000, 0.237] | 0.100 [0.000, 0.237] | 0.025 [0.000, 0.079] |

Seconds per stage: {'generate': 32.6, 'reconstruct': 23.1, 'score_names': 13.0}. Intervals: bootstrap over entities (10k resamples).