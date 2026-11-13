# full-recon-round2: 540 held-out test passages x 4 distances x 4 samples (natural text)

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | any same-type names per explanation | length (chars) |
|---|---|---|---|---|---|---|
| 0 | 0.697 [0.605, 0.794] | 0.438 [0.393, 0.483] | 1.000 [1.000, 1.000] | 0.900 [0.896, 0.904] | 1.864 [1.748, 1.987] | 704 |
| 1 | 1.052 [0.915, 1.194] | 0.528 [0.483, 0.574] | 0.537 [0.478, 0.595] | 0.892 [0.888, 0.895] | 1.761 [1.584, 1.941] | 705 |
| 4 | 0.541 [0.452, 0.636] | 0.333 [0.292, 0.375] | 0.216 [0.182, 0.250] | 0.892 [0.888, 0.895] | 0.886 [0.769, 1.008] | 695 |
| 16 | 0.342 [0.274, 0.414] | 0.227 [0.190, 0.265] | 0.078 [0.056, 0.101] | 0.893 [0.890, 0.897] | 0.500 [0.416, 0.589] | 696 |

With the filter "keep a name only if at least 2 of the 4 samples give it" applied to the same samples:

| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |
|---|---|---|---|
| 0 | 0.383 [0.316, 0.453] | 0.288 [0.242, 0.334] | 1.000 [1.000, 1.000] |
| 1 | 0.558 [0.460, 0.662] | 0.351 [0.304, 0.400] | 0.511 [0.450, 0.572] |
| 4 | 0.254 [0.200, 0.312] | 0.187 [0.154, 0.222] | 0.189 [0.157, 0.222] |
| 16 | 0.120 [0.094, 0.147] | 0.099 [0.079, 0.120] | 0.060 [0.039, 0.082] |

Seconds per stage: {'generate': 620.6, 'reconstruct': 168.0, 'score_names': 130.0}. Intervals: bootstrap over entities (10k resamples).