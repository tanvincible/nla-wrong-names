# tiny-recon-round0: 20 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.775 [0.417, 1.211] | 0.463 [0.300, 0.625] | 1.000 [1.000, 1.000] | 0.896 [0.880, 0.912] | 1.950 [1.566, 2.395] | 698 |
| 1 | 0.950 [0.500, 1.476] | 0.463 [0.286, 0.650] | 0.525 [0.325, 0.737] | 0.891 [0.867, 0.911] | 1.725 [1.163, 2.329] | 698 |
| 4 | 0.487 [0.167, 0.895] | 0.275 [0.113, 0.461] | 0.287 [0.120, 0.487] | 0.897 [0.882, 0.912] | 0.850 [0.424, 1.342] | 701 |
| 16 | 0.388 [0.150, 0.688] | 0.250 [0.132, 0.382] | 0.025 [0.000, 0.079] | 0.891 [0.876, 0.907] | 0.487 [0.224, 0.787] | 698 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.400 [0.150, 0.711] | 0.287 [0.132, 0.461] | 1.000 [1.000, 1.000] |
| 1 | 0.525 [0.214, 0.917] | 0.312 [0.155, 0.487] | 0.525 [0.325, 0.737] |
| 4 | 0.275 [0.053, 0.579] | 0.200 [0.050, 0.375] | 0.263 [0.095, 0.463] |
| 16 | 0.138 [0.000, 0.381] | 0.087 [0.000, 0.225] | 0.025 [0.000, 0.079] |

Seconds per stage: {'generate': 32.5, 'reconstruct': 23.4, 'score_names': 12.1}. Intervals: bootstrap over entities (10k resamples).