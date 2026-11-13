Natural text. Each unit is one name in one explanation (its first mention). 540 passages, 8,640 explanations.

### AUC for telling true names from wrong ones

| tokens after entity | names (true / wrong) | probability of the name, one explanation (equal-length names compared) | first token only, one explanation | agreement count over 4 explanations |
|---|---|---|---|---|
| 0 | 2159 / 1539 | 0.917 [0.901, 0.931] | 0.915 [0.901, 0.929] | 0.906 [0.888, 0.924] |
| 1 | 1159 / 2343 | 0.755 [0.722, 0.787] | 0.774 [0.742, 0.804] | 0.828 [0.799, 0.855] |
| 4 | 468 / 1109 | 0.729 [0.685, 0.773] | 0.756 [0.713, 0.798] | 0.798 [0.766, 0.830] |
| 16 | 160 / 720 | 0.703 [0.640, 0.761] | 0.726 [0.665, 0.780] | 0.760 [0.695, 0.814] |
| 1, 4 and 16 pooled | 1787 / 4172 | 0.746 [0.716, 0.774] | 0.769 [0.739, 0.797] | 0.817 [0.790, 0.841] |

### One explanation against four: wrong names removed at the same share of true names kept

The confidence threshold (on the raw joint probability, which also benefits from true names being shorter) is set so that it keeps the same share of true names as the agreement filter does.

| tokens after entity | filter | true names kept | wrong names removed, agreement (4 explanations) | wrong names removed, confidence (1 explanation) |
|---|---|---|---|---|
| 1 | at least 2 of 4 | 95.0% | 0.446 [0.402, 0.494] | 0.364 [0.318, 0.414] |
| 1 | at least 3 of 4 | 89.0% | 0.624 [0.580, 0.669] | 0.481 [0.437, 0.530] |
| 1 | at least 4 of 4 | 79.4% | 0.804 [0.768, 0.838] | 0.633 [0.596, 0.673] |
| 4 | at least 2 of 4 | 88.9% | 0.553 [0.494, 0.614] | 0.489 [0.435, 0.544] |
| 4 | at least 3 of 4 | 78.6% | 0.719 [0.669, 0.771] | 0.624 [0.574, 0.673] |
| 4 | at least 4 of 4 | 58.1% | 0.859 [0.820, 0.898] | 0.800 [0.763, 0.835] |
| 16 | at least 2 of 4 | 81.2% | 0.621 [0.550, 0.688] | 0.554 [0.471, 0.631] |
| 16 | at least 3 of 4 | 62.5% | 0.807 [0.739, 0.869] | 0.732 [0.669, 0.789] |
| 16 | at least 4 of 4 | 32.5% | 0.911 [0.869, 0.949] | 0.928 [0.895, 0.954] |
| 1, 4 and 16 pooled | at least 2 of 4 | 92.2% | 0.505 [0.457, 0.555] | 0.423 [0.373, 0.476] |
| 1, 4 and 16 pooled | at least 3 of 4 | 83.9% | 0.681 [0.639, 0.724] | 0.566 [0.522, 0.613] |
| 1, 4 and 16 pooled | at least 4 of 4 | 69.6% | 0.837 [0.809, 0.865] | 0.726 [0.693, 0.760] |