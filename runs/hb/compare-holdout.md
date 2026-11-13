Natural text. Each checkpoint minus the released verbalizer on the same 540 held-out passages, 4 samples each. Intervals: bootstrap over entities (10k). An interval that excludes 0 is a real difference.

| checkpoint | tokens after entity | wrong names per explanation, released | difference | relative | true-name rate, released | difference |
|---|---|---|---|---|---|---|
| full-recall-round2 | 0 | 0.719 | -0.048 [-0.081, -0.015] | -6.7% | 100.0% | +0.0% [+0.0%, +0.1%] |
| full-recall-round2 | 1 | 1.097 | -0.018 [-0.065, +0.029] | -1.6% | 53.8% | +0.8% [-0.7%, +2.3%] |
| full-recall-round2 | 4 | 0.517 | -0.003 [-0.042, +0.034] | -0.5% | 21.7% | +0.0% [-1.5%, +1.7%] |
| full-recall-round2 | 16 | 0.334 | -0.008 [-0.035, +0.020] | -2.4% | 7.5% | +0.0% [-1.0%, +1.1%] |
| full-recall-round2 | all four | 0.667 | -0.019 [-0.039, +0.001] | -2.9% | 45.7% | +0.2% [-0.4%, +0.8%] |
| full-recall-round2 (entities trained on) | 0 | 0.699 | -0.041 [-0.078, -0.006] | -5.9% | 99.9% | +0.1% [+0.0%, +0.2%] |
| full-recall-round2 (entities trained on) | 1 | 1.092 | -0.016 [-0.068, +0.036] | -1.5% | 52.3% | +1.2% [-0.4%, +2.9%] |
| full-recall-round2 (entities trained on) | 4 | 0.515 | -0.013 [-0.058, +0.030] | -2.5% | 21.3% | +0.2% [-1.6%, +2.2%] |
| full-recall-round2 (entities trained on) | 16 | 0.318 | +0.002 [-0.028, +0.035] | +0.7% | 7.4% | +0.3% [-0.8%, +1.5%] |
| full-recall-round2 (entities trained on) | all four | 0.656 | -0.017 [-0.039, +0.006] | -2.6% | 45.2% | +0.5% [-0.2%, +1.1%] |
| full-recall-round2 (entities held out of training) | 0 | 0.796 | -0.076 [-0.155, +0.005] | -9.6% | 100.0% | +0.0% [+0.0%, +0.0%] |
| full-recall-round2 (entities held out of training) | 1 | 1.118 | -0.025 [-0.137, +0.081] | -2.3% | 59.5% | -0.9% [-4.9%, +3.0%] |
| full-recall-round2 (entities held out of training) | 4 | 0.523 | +0.037 [-0.025, +0.106] | +7.1% | 23.4% | -0.7% [-3.2%, +2.1%] |
| full-recall-round2 (entities held out of training) | 16 | 0.400 | -0.049 [-0.104, +0.002] | -12.1% | 7.6% | -1.2% [-3.2%, +0.7%] |
| full-recall-round2 (entities held out of training) | all four | 0.709 | -0.028 [-0.072, +0.014] | -4.0% | 47.6% | -0.7% [-2.2%, +0.9%] |