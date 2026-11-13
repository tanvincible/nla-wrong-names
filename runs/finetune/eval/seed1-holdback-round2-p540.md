# seed1-holdback-round2: 540 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.660 [0.574, 0.751] | 0.428 [0.383, 0.472] | 1.000 [1.000, 1.000] | 0.900 [0.896, 0.904] | 1.820 [1.708, 1.938] | 705 |
| 1 | 1.037 [0.901, 1.180] | 0.523 [0.476, 0.570] | 0.534 [0.477, 0.591] | 0.891 [0.888, 0.895] | 1.746 [1.573, 1.924] | 704 |
| 4 | 0.512 [0.427, 0.601] | 0.324 [0.281, 0.368] | 0.216 [0.183, 0.249] | 0.891 [0.888, 0.895] | 0.853 [0.741, 0.971] | 697 |
| 16 | 0.321 [0.261, 0.388] | 0.222 [0.187, 0.258] | 0.074 [0.054, 0.096] | 0.893 [0.889, 0.896] | 0.473 [0.400, 0.550] | 698 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.376 [0.313, 0.441] | 0.290 [0.247, 0.334] | 1.000 [1.000, 1.000] |
| 1 | 0.556 [0.468, 0.650] | 0.368 [0.321, 0.415] | 0.514 [0.455, 0.572] |
| 4 | 0.226 [0.175, 0.281] | 0.169 [0.135, 0.205] | 0.186 [0.156, 0.217] |
| 16 | 0.121 [0.096, 0.148] | 0.100 [0.080, 0.122] | 0.060 [0.040, 0.082] |

Seconds per stage: {'generate': 718.7, 'reconstruct': 185.4, 'score_names': 112.5}. Intervals: bootstrap over entities (10k resamples).