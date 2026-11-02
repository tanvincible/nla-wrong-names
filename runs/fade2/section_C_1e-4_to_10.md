Passages: 6000 (3360 fit, 840 validation, 1800 test), 60 entities, 100 per entity (every entity capped at the same count). All natural text.

### Steps 2 and 3: top-1 accuracy among the 10 same-type entities (natural, test passages, chance 0.10)

Probe: logistic regression, C chosen on validation from [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] per distance and type, refit on fit+validation.

| tokens after entity | readout A (concept vectors) | trained probe (swept) | C chosen per type |
|---|---|---|---|
| 0 | 0.491 [0.421, 0.562] | 0.999 [0.998, 1.000] | 0.0001, 0.001, 0.01, 0.001, 0.001, 0.001 |
| 1 | 0.302 [0.251, 0.355] | 0.592 [0.553, 0.630] | 10, 0.1, 10, 10, 0.1, 10 |
| 2 | 0.272 [0.227, 0.318] | 0.461 [0.417, 0.506] | 10, 0.1, 0.01, 0.1, 0.0001, 0.001 |
| 4 | 0.258 [0.213, 0.306] | 0.396 [0.354, 0.438] | 0.001, 0.1, 0.001, 0.001, 0.0001, 1 |
| 8 | 0.228 [0.185, 0.274] | 0.370 [0.331, 0.412] | 0.01, 0.001, 0.001, 0.001, 0.0001, 0.001 |
| 16 | 0.224 [0.187, 0.265] | 0.317 [0.275, 0.363] | 0.001, 0.01, 0.0001, 0.0001, 0.0001, 0.001 |

**Prediction, stated before the step 4 results.** If the verbalizer names the true entity at distances 4 and 16 much more often than the probe's top-1 accuracy there, the fade is a limitation of the probe, not a property of the activation. If it names the true entity no more often than the probe reads it, and its same-type confabulations rise as the probe falls, the fade is real and the verbalizer is guessing.

![fade plot](runs/fade2/fade.png)
