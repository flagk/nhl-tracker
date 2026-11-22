# Model tendencies

> Research/education only. Descriptive analytics, **not** betting advice, and nothing here changes the model or the staking rules automatically.

*Generated 2026-11-22 21:08 UTC.* For each segment we test whether home teams win more or less often than the probabilities say (`bias` = actual minus predicted home-win frequency, in percentage points). With many segments, some look significant by luck, so every p-value is corrected across all tests (Benjamini-Hochberg false-discovery rate) and a segment is only a **tendency** if the same-direction bias also shows up in both halves of the data (first/second half by date).

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

## Betting market vs outcomes (live odds)

**367 games, 18 segments tested.** Overall bias -0.33 pts. At a 10% false-discovery rate about 1 of the flagged segments could still be flukes.

**No replicated tendency.** Every apparent pattern is either consistent with chance after correcting for the number of tests, or did not hold in both halves. That is the normal, expected result and is good news for the model's calibration.

### Strongest signals (for context; most are noise)

| Segment | Games | Bias | p | q | Verdict |
|---|---|---|---|---|---|
| season stage: games 15 to 60 | 121 | -9.1 pts | 0.040 | 0.490 | no reliable tendency |
| month: November | 146 | -7.1 pts | 0.077 | 0.490 | no reliable tendency |
| home win probability: 0.50 to 0.55 | 68 | -9.8 pts | 0.107 | 0.490 | no reliable tendency |
| home win probability: 0.55 to 0.60 | 68 | +8.8 pts | 0.142 | 0.490 | no reliable tendency |
| home win probability: 0.45 to 0.50 | 68 | -8.0 pts | 0.184 | 0.490 | no reliable tendency |
| season stage: first 15 games (ratings lean on last year) | 246 | +4.0 pts | 0.201 | 0.490 | no reliable tendency |
| month: October | 218 | +4.1 pts | 0.214 | 0.490 | no reliable tendency |
| weekday: Tue | 76 | -7.0 pts | 0.218 | 0.490 | no reliable tendency |
