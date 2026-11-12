# pumpfun-top5-forecast-01 — Monte Carlo log-cap forecaster

Entry for make.fun's **"Predict the top 5 new Solana coins by market cap,
24 hours out"** ($200 USDC, top 10 paid). A prize needs a log score strictly
above the best house entry (the persistence baseline, practice −40.408345,
or the house model forecast).

## Model

`main.py` (Python 3.13, standard library only) reads the house's 10-minute
snapshots of pump.fun coins in `/input` and writes one forecast to
`/output/forecast.json`:

- Every coin carries a log10 market cap (in SOL) that random-walks forward
  from the last mark to each answer read in 10-minute steps.
- Drift is the drift typical for coins of its age — fitted from the history
  itself, with priors where a bucket has little data — plus the coin's own
  recent excess velocity, decaying with a ~1 h time constant, so a hot coin
  keeps some momentum but reverts to the age-typical path.
- Noise is ~0.28 log10 units per sqrt(hour), fitted from the history.
- Coins are ranked in each simulated path at each read. The share of paths
  a coin holds a place at becomes its probability, scaled down by a reserve
  for coins not yet born — which dominate the top 5 at long horizons (in the
  practice history, 4 of the top 5 at 20:50 were absent from the history up
  to 20:20). Place sizes come from the simulated caps at that place, widened
  for the unborn share.
- Everything (reads, bins, places) is read from `task.json`, never hardcoded
  to the practice window. One fixed seed: two runs on the same history write
  byte-identical output.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local verification (practice history)

Rolling folds (forecast from marks up to a cutoff, scored on the later marks
as answers, with the project's own scorer implementing the RULES log score):

| fold | house baseline | this entry |
|---|---|---|
| history ≤ 20:20, answers 20:30–20:50 | −49.96 | −47.38 |
| history ≤ 20:30, answers 20:40–20:50 | −33.48 | −30.83 |
| history ≤ 20:40, answer 20:50 | −29.87 | −25.75 |

Ahead of the baseline on every fold (+2.6 to +4.1 points). Two runs on the
practice input are byte-identical (forecast.json sha256
`3cb94e6d0b7ba6d4f511d94355d5e3b5b793c98abb67dad36513547b2d253885`,
18,987 bytes). A synthetic final-scale input (66 marks, 3,300 coins, 7 hourly
reads) runs in ~7 s (cap 300 s) with place sums exactly 1 − reserve.

Referral code: `byvgg6sg` (https://make.fun/r/byvgg6sg)
