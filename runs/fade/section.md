Passages: 1800 (1260 train, 540 test), 60 entities. All natural text.

### Steps 2 and 3: top-1 accuracy among the 10 same-type entities (natural, test passages, chance 0.10)

| tokens after entity | readout A (concept vectors) | trained probe |
|---|---|---|
| 0 | 0.487 [0.417, 0.559] | 0.993 [0.985, 0.998] |
| 1 | 0.317 [0.256, 0.381] | 0.428 [0.372, 0.485] |
| 2 | 0.294 [0.241, 0.348] | 0.357 [0.306, 0.413] |
| 4 | 0.228 [0.174, 0.285] | 0.298 [0.250, 0.348] |
| 8 | 0.226 [0.180, 0.274] | 0.243 [0.196, 0.293] |
| 16 | 0.207 [0.170, 0.246] | 0.233 [0.193, 0.278] |

![fade plot](runs/fade/fade.png)
