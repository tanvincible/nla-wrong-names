# tiny-holdback-round0: 20 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.787 [0.474, 1.163] | 0.475 [0.329, 0.614] | 1.000 [1.000, 1.000] | 0.897 [0.880, 0.912] | 1.962 [1.618, 2.355] | 701 |
| 1 | 1.025 [0.583, 1.539] | 0.550 [0.359, 0.737] | 0.575 [0.381, 0.776] | 0.892 [0.867, 0.912] | 1.812 [1.250, 2.408] | 694 |
| 4 | 0.450 [0.184, 0.763] | 0.287 [0.132, 0.461] | 0.263 [0.102, 0.452] | 0.898 [0.883, 0.912] | 0.775 [0.400, 1.211] | 703 |
| 16 | 0.400 [0.145, 0.737] | 0.263 [0.119, 0.429] | 0.025 [0.000, 0.079] | 0.892 [0.876, 0.908] | 0.512 [0.211, 0.868] | 700 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.287 [0.092, 0.523] | 0.237 [0.079, 0.412] | 1.000 [1.000, 1.000] |
| 1 | 0.400 [0.179, 0.684] | 0.325 [0.163, 0.500] | 0.550 [0.338, 0.775] |
| 4 | 0.212 [0.075, 0.382] | 0.212 [0.075, 0.382] | 0.250 [0.087, 0.447] |
| 16 | 0.150 [0.000, 0.350] | 0.125 [0.000, 0.289] | 0.025 [0.000, 0.079] |

Seconds per stage: {'generate': 32.6, 'reconstruct': 23.5, 'score_names': 11.8}. Intervals: bootstrap over entities (10k resamples).