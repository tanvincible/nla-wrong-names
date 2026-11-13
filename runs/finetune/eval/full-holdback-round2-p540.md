# full-holdback-round2: 540 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.667 [0.584, 0.755] | 0.437 [0.395, 0.478] | 1.000 [1.000, 1.000] | 0.900 [0.896, 0.904] | 1.837 [1.725, 1.957] | 709 |
| 1 | 1.006 [0.876, 1.143] | 0.515 [0.467, 0.564] | 0.528 [0.474, 0.582] | 0.891 [0.888, 0.895] | 1.708 [1.548, 1.876] | 708 |
| 4 | 0.462 [0.382, 0.547] | 0.302 [0.261, 0.345] | 0.212 [0.178, 0.245] | 0.891 [0.888, 0.895] | 0.799 [0.693, 0.909] | 698 |
| 16 | 0.309 [0.250, 0.373] | 0.221 [0.187, 0.256] | 0.081 [0.056, 0.108] | 0.893 [0.890, 0.897] | 0.464 [0.391, 0.544] | 700 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.360 [0.304, 0.416] | 0.284 [0.244, 0.325] | 1.000 [1.000, 1.000] |
| 1 | 0.554 [0.463, 0.650] | 0.361 [0.314, 0.408] | 0.502 [0.445, 0.559] |
| 4 | 0.210 [0.163, 0.263] | 0.160 [0.128, 0.194] | 0.183 [0.151, 0.216] |
| 16 | 0.119 [0.088, 0.150] | 0.095 [0.073, 0.117] | 0.064 [0.041, 0.090] |

Seconds per stage: {'generate': 717.4, 'reconstruct': 183.4, 'score_names': 111.7}. Intervals: bootstrap over entities (10k resamples).