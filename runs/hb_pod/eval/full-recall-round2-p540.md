# full-recall-round2: 540 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.670 [0.581, 0.761] | 0.429 [0.382, 0.475] | 1.000 [1.000, 1.000] | 0.900 [0.896, 0.904] | 1.836 [1.725, 1.951] | 700 |
| 1 | 1.079 [0.941, 1.222] | 0.534 [0.487, 0.581] | 0.545 [0.492, 0.599] | 0.892 [0.888, 0.895] | 1.806 [1.629, 1.990] | 702 |
| 4 | 0.514 [0.430, 0.602] | 0.324 [0.284, 0.366] | 0.218 [0.182, 0.254] | 0.892 [0.888, 0.895] | 0.859 [0.749, 0.975] | 692 |
| 16 | 0.326 [0.257, 0.402] | 0.219 [0.181, 0.261] | 0.075 [0.054, 0.098] | 0.893 [0.890, 0.897] | 0.481 [0.398, 0.575] | 693 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.376 [0.311, 0.444] | 0.283 [0.238, 0.330] | 1.000 [1.000, 1.000] |
| 1 | 0.582 [0.487, 0.681] | 0.374 [0.325, 0.422] | 0.520 [0.465, 0.575] |
| 4 | 0.251 [0.197, 0.308] | 0.188 [0.153, 0.225] | 0.194 [0.161, 0.230] |
| 16 | 0.118 [0.090, 0.146] | 0.096 [0.075, 0.117] | 0.058 [0.037, 0.081] |

Seconds per stage: {'generate': 724.5, 'reconstruct': 182.9, 'score_names': 113.2}. Intervals: bootstrap over entities (10k resamples).