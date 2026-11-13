# released: 540 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.719 [0.627, 0.813] | 0.453 [0.407, 0.499] | 1.000 [0.999, 1.000] | 0.900 [0.896, 0.904] | 1.876 [1.764, 1.992] | 704 |
| 1 | 1.097 [0.954, 1.249] | 0.543 [0.495, 0.592] | 0.537 [0.483, 0.592] | 0.892 [0.888, 0.895] | 1.815 [1.636, 1.997] | 705 |
| 4 | 0.517 [0.432, 0.605] | 0.326 [0.281, 0.373] | 0.217 [0.182, 0.252] | 0.891 [0.888, 0.895] | 0.868 [0.757, 0.982] | 695 |
| 16 | 0.334 [0.268, 0.409] | 0.225 [0.190, 0.263] | 0.075 [0.054, 0.098] | 0.893 [0.890, 0.897] | 0.485 [0.405, 0.573] | 696 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.387 [0.321, 0.454] | 0.289 [0.245, 0.334] | 1.000 [0.999, 1.000] |
| 1 | 0.609 [0.508, 0.715] | 0.382 [0.334, 0.431] | 0.511 [0.454, 0.567] |
| 4 | 0.232 [0.182, 0.283] | 0.177 [0.142, 0.212] | 0.193 [0.159, 0.227] |
| 16 | 0.126 [0.099, 0.156] | 0.102 [0.081, 0.123] | 0.061 [0.041, 0.083] |

Seconds per stage: {'reconstruct': 165.8, 'score_names': 122.3}. Intervals: bootstrap over entities (10k resamples).