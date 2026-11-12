All prompts are natural (nothing planted). Intervals: bootstrap over entities, 10k resamples.

| | group a: entity is the next token | group b: entity comes 6 tokens later | couplets: entity ends the next line |
|---|---|---|---|
| prompts (valid / total) | 110 / 120 | 118 / 120 | 4 / 12 |
| logit lens at layer 20: entity is top-1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| logit lens at layer 20: entity in top-5 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| final layer: entity is the top next token | 0.573 [0.445, 0.694] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| verbalizer names the entity (per sample) | 0.691 [0.607, 0.770] | 0.375 [0.284, 0.471] | 0.000 [0.000, 0.000] |
| verbalizer names the entity (any of 4) | 0.882 [0.811, 0.945] | 0.517 [0.414, 0.619] | 0.000 [0.000, 0.000] |
| verbalizer names a different same-type entity | 0.514 [0.416, 0.611] | 0.542 [0.452, 0.633] | 0.125 [0.000, 0.250] |
| verbalizer names it, when logit lens top-5 has it | n/a | n/a | n/a |
| verbalizer names it, when logit lens top-5 lacks it | 0.691 [0.607, 0.770] | 0.375 [0.284, 0.471] | 0.000 [0.000, 0.000] |
| B/F: explanations tested | 178 | 109 | 0 |
| B: true explanation beats rival-swapped, by cosine | 0.882 [0.834, 0.926] | 0.810 [0.737, 0.876] | n/a |
| B: true beats rival-swapped, by KL over next 12 tokens | 0.581 [0.514, 0.645] | 0.361 [0.287, 0.442] | n/a |
| B: cosine margin (true minus rival) | 0.005 [0.004, 0.006] | 0.002 [0.001, 0.003] | n/a |
| B: KL penalty for the rival (nats) | 0.003 [0.001, 0.005] | -0.002 [-0.004, -0.000] | n/a |
| B: KL of the true rebuild itself (nats) | 0.027 [0.013, 0.046] | 0.032 [0.013, 0.057] | n/a |
| F, no steering: model names the rival | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a |
| F, no steering: model still names the original | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a |
| F, steering x1: model names the rival | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a |
| F, steering x1: model still names the original | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a |
| F, steering x4: model names the rival | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a |
| F, steering x4: model still names the original | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a |
| F, steering x8: model names the rival | 0.034 [0.006, 0.066] | 0.000 [0.000, 0.000] | n/a |
| F, steering x8: model still names the original | 0.888 [0.830, 0.938] | 1.000 [1.000, 1.000] | n/a |