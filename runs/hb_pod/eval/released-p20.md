# released: 20 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.700 [0.395, 1.053] | 0.450 [0.289, 0.613] | 1.000 [1.000, 1.000] | 0.896 [0.880, 0.912] | 1.825 [1.488, 2.197] | 700 |
| 1 | 0.912 [0.536, 1.355] | 0.512 [0.345, 0.684] | 0.500 [0.303, 0.713] | 0.890 [0.865, 0.910] | 1.650 [1.145, 2.211] | 700 |
| 4 | 0.463 [0.188, 0.788] | 0.325 [0.150, 0.525] | 0.300 [0.118, 0.513] | 0.898 [0.883, 0.913] | 0.850 [0.440, 1.316] | 702 |
| 16 | 0.312 [0.060, 0.650] | 0.188 [0.053, 0.350] | 0.062 [0.000, 0.175] | 0.891 [0.875, 0.907] | 0.463 [0.118, 0.900] | 706 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.312 [0.118, 0.560] | 0.250 [0.105, 0.412] | 1.000 [1.000, 1.000] |
| 1 | 0.412 [0.175, 0.711] | 0.312 [0.143, 0.512] | 0.475 [0.263, 0.711] |
| 4 | 0.275 [0.095, 0.500] | 0.250 [0.091, 0.447] | 0.287 [0.100, 0.500] |
| 16 | 0.037 [0.000, 0.118] | 0.037 [0.000, 0.118] | 0.050 [0.000, 0.158] |

Seconds per stage: {'reconstruct': 23.1, 'score_names': 11.8}. Intervals: bootstrap over entities (10k resamples).