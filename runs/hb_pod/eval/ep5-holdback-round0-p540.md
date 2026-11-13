# ep5-holdback-round0: 540 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.665 [0.585, 0.748] | 0.427 [0.385, 0.470] | 1.000 [1.000, 1.000] | 0.900 [0.896, 0.904] | 1.826 [1.725, 1.929] | 704 |
| 1 | 1.007 [0.877, 1.140] | 0.518 [0.470, 0.566] | 0.526 [0.469, 0.582] | 0.892 [0.889, 0.895] | 1.702 [1.535, 1.871] | 705 |
| 4 | 0.488 [0.405, 0.575] | 0.309 [0.267, 0.351] | 0.205 [0.169, 0.241] | 0.892 [0.888, 0.895] | 0.815 [0.712, 0.924] | 694 |
| 16 | 0.332 [0.260, 0.410] | 0.223 [0.183, 0.263] | 0.075 [0.053, 0.099] | 0.894 [0.890, 0.897] | 0.485 [0.401, 0.580] | 696 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.396 [0.334, 0.460] | 0.287 [0.247, 0.327] | 1.000 [1.000, 1.000] |
| 1 | 0.574 [0.480, 0.672] | 0.369 [0.320, 0.419] | 0.501 [0.442, 0.560] |
| 4 | 0.242 [0.188, 0.300] | 0.186 [0.149, 0.223] | 0.184 [0.150, 0.219] |
| 16 | 0.138 [0.105, 0.173] | 0.106 [0.083, 0.130] | 0.061 [0.040, 0.083] |

Seconds per stage: {'generate': 725.4, 'reconstruct': 185.4, 'score_names': 114.9}. Intervals: bootstrap over entities (10k resamples).