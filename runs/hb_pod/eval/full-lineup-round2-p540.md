# full-lineup-round2: 540 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.718 [0.633, 0.805] | 0.452 [0.406, 0.498] | 1.000 [0.999, 1.000] | 0.899 [0.895, 0.903] | 1.883 [1.774, 1.997] | 703 |
| 1 | 1.082 [0.944, 1.224] | 0.539 [0.491, 0.588] | 0.528 [0.473, 0.582] | 0.891 [0.887, 0.894] | 1.782 [1.611, 1.959] | 703 |
| 4 | 0.523 [0.434, 0.617] | 0.324 [0.279, 0.372] | 0.209 [0.172, 0.245] | 0.891 [0.887, 0.894] | 0.864 [0.745, 0.988] | 695 |
| 16 | 0.330 [0.262, 0.406] | 0.226 [0.190, 0.265] | 0.080 [0.056, 0.105] | 0.892 [0.889, 0.896] | 0.483 [0.401, 0.573] | 695 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.393 [0.325, 0.462] | 0.293 [0.247, 0.340] | 1.000 [0.999, 1.000] |
| 1 | 0.585 [0.494, 0.679] | 0.381 [0.334, 0.429] | 0.502 [0.446, 0.558] |
| 4 | 0.241 [0.190, 0.297] | 0.175 [0.143, 0.209] | 0.184 [0.148, 0.220] |
| 16 | 0.121 [0.091, 0.153] | 0.101 [0.078, 0.125] | 0.064 [0.042, 0.090] |

Seconds per stage: {'generate': 711.3, 'reconstruct': 183.7, 'score_names': 113.4}. Intervals: bootstrap over entities (10k resamples).