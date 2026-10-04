# Model tendencies

> Research/education only. Descriptive analytics, **not** betting advice, and nothing here changes the model or the staking rules automatically.

*Generated 2026-10-04 18:42 UTC.* For each segment we test whether home teams win more or less often than the probabilities say (`bias` = actual minus predicted home-win frequency, in percentage points). With many segments, some look significant by luck, so every p-value is corrected across all tests (Benjamini-Hochberg false-discovery rate) and a segment is only a **tendency** if the same-direction bias also shows up in both halves of the data (first/second half by date).

## Model vs outcomes (walk-forward, out of sample)

**2,269 games, 94 segments tested.** Overall bias -0.63 pts. At a 10% false-discovery rate about 1 of the flagged segments could still be flukes.

**No replicated tendency.** Every apparent pattern is either consistent with chance after correcting for the number of tests, or did not hold in both halves. That is the normal, expected result and is good news for the model's calibration.

### Strongest signals (for context; most are noise)

| Segment | Games | Bias | p | q | Verdict |
|---|---|---|---|---|---|
| team at home: VAN | 71 | -18.5 pts | 0.001 | 0.135 | no reliable tendency |
| team at home: NYR | 72 | -14.5 pts | 0.013 | 0.590 | no reliable tendency |
| team away: MTL | 73 | -12.7 pts | 0.025 | 0.775 | no reliable tendency |
| home win probability: 0.00 to 0.45 | 347 | +4.6 pts | 0.076 | 0.812 | no reliable tendency |
| team at home: BOS | 70 | +10.0 pts | 0.088 | 0.812 | no reliable tendency |
| team away: BOS | 71 | +9.6 pts | 0.098 | 0.812 | no reliable tendency |
| home win probability: 0.60 to 1.00 | 710 | -2.9 pts | 0.105 | 0.812 | no reliable tendency |
| favourite: home team favoured | 1587 | -2.0 pts | 0.105 | 0.812 | no reliable tendency |

*Market comparison starts at 300 settled games with odds; so far 29.*
