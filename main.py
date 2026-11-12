#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: Monte Carlo log-cap forecaster (Python 3.13, stdlib only).

Model, in one paragraph: every coin carries a log10 market cap (in SOL) that
random-walks forward from the last mark to each answer read in 10-minute steps.
Its drift is the drift typical for coins of its age (fitted from the history
it is given, with priors when a bucket has little data) plus its own recent
excess velocity, which decays with a ~1 h time constant, so a hot coin keeps
some momentum but reverts to the age-typical path. Noise is ~0.28 log10 units
per sqrt(hour). Coins are ranked in each simulated path at each read; the
share of paths a coin holds a place at becomes its probability, scaled down
by a reserve for coins not yet born (which dominate the top 5 at long
horizons: in the practice history, 4 of the top 5 at 20:50 were absent from
the history up to 20:20). Place sizes come from the simulated caps at that
place, widened for the unborn share. Everything is read from task.json
(reads, bins, places), never hardcoded to the practice window.

Deterministic: one fixed seed; two runs on the same history write byte-identical
forecast.json.
"""
# A fixed share of probability is reserved for coins not yet launched, since newcomers frequently outcompete all currently-visible coins at horizons beyond ~30 minutes.
import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
PATHS = 300
SIGMA_H = 0.28          # log10 units per sqrt(hour)
VEL_TAU_H = 1.0         # decay of a coin's excess velocity
RESERVE_R0 = 0.30       # reserve at zero horizon
RESERVE_RINF = 0.85     # reserve at long horizon
RESERVE_TAU_H = 1.0     # horizon scale of the reserve
SIZE_EXTRA_SD = 0.60    # sd of the wide size prior for the unborn share
MIN_P = 0.006           # rows below this join the reserve

# Prior drift (log10 cap per hour) by coin age in hours; overridden by the
# history's own buckets when a bucket holds enough observations.
AGE_EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
AGE_PRIOR = [0.25, 0.15, 0.05, -0.02, -0.05, -0.05, -0.03]


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def bucket_idx(age_h):
    for i in range(len(AGE_EDGES) - 1):
        if AGE_EDGES[i] <= age_h < AGE_EDGES[i + 1]:
            return i
    return len(AGE_EDGES) - 2


def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json")))
    edges = task["size"]["edges"]
    places = task["places"]
    reads = [parse_ts(r) for r in task["reads"]]
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")
    marks.sort(key=lambda m: m["at"])
    t_last = parse_ts(marks[-1]["at"])

    # --- per-coin series: mint -> [(t, logcap, rank, venue, created_at)] ---
    series = {}
    for m in marks:
        t = parse_ts(m["at"])
        for r in m["rows"]:
            try:
                x = math.log10(int(r["market_cap_lamports"]) / 1e9)
            except (KeyError, ValueError):
                continue
            series.setdefault(r["mint"], []).append(
                (t, x, r["rank"], r.get("venue", "curve"), r.get("created_at"))
            )
    for v in series.values():
        v.sort()

    # --- fit age-bucketed drift from consecutive-mark pairs ---
    # mean dlog10 per hour, per age bucket (age at the earlier mark).
    sums = [0.0] * (len(AGE_EDGES) - 1)
    ns = [0] * (len(AGE_EDGES) - 1)
    all_d = []
    for mint, pts in series.items():
        created = None
        for p in pts:
            if p[4]:
                created = parse_ts(p[4])
                break
        for a, b in zip(pts, pts[1:]):
            dt_h = (b[0] - a[0]) / 3600.0
            if dt_h <= 0:
                continue
            d = (b[1] - a[1]) / dt_h
            all_d.append(d)
            age = ((a[0] - created) / 3600.0) if created else 1.0
            bi = bucket_idx(max(age, 0.0))
            sums[bi] += d
            ns[bi] += 1
    if all_d:
        all_d.sort()
        global_drift = all_d[len(all_d) // 2]  # median, robust to outliers
    else:
        global_drift = 0.0
    drift_rate = []
    for i in range(len(AGE_EDGES) - 1):
        if ns[i] >= 30:
            # shrink the bucket mean toward the global median
            drift_rate.append((sums[i] + 20 * global_drift) / (ns[i] + 20))
        else:
            drift_rate.append(AGE_PRIOR[i])
    # volatility: sd of per-hour changes, floored
    if len(all_d) >= 20:
        mean_d = sum(all_d) / len(all_d)
        var = sum((d - mean_d) ** 2 for d in all_d) / len(all_d)
        sigma = max(0.15, min(0.45, math.sqrt(var)))
    else:
        sigma = SIGMA_H

    # --- candidate universe: last mark's rows + recent top-30 coins ---
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    universe = {}
    for r in last_rows:
        universe[r["mint"]] = True
    for m in marks[-6:]:
        for r in sorted(m["rows"], key=lambda rr: rr["rank"])[:30]:
            universe[r["mint"]] = True
    coins = []
    for mint in universe:
        pts = series[mint]
        t0, x0 = pts[-1][0], pts[-1][1]
        created = None
        for p in pts:
            if p[4]:
                created = parse_ts(p[4])
                break
        if created is None:
            created = t0
        # recent velocity: recency-weighted slope over the last <=3 intervals
        vel = None
        if len(pts) >= 2:
            num = 0.0
            den = 0.0
            w = 1.0
            for a, b in list(zip(pts, pts[1:]))[-3:][::-1]:
                dt_h = (b[0] - a[0]) / 3600.0
                if dt_h > 0:
                    num += w * (b[1] - a[1]) / dt_h
                    den += w
                w *= 0.6
            if den:
                vel = num / den
        age_last = max((t_last - created) / 3600.0, 0.0)
        base = drift_rate[bucket_idx(age_last)]
        if vel is None:
            v_excess = 0.0
        else:
            # shrink the raw velocity toward the bucket rate by observation count
            k = min(len(pts) - 1, 3)
            vel_s = (k * vel + 2 * base) / (k + 2)
            v_excess = vel_s - base
        coins.append({
            "mint": mint, "x0": x0, "created": created,
            "age_last": age_last, "v_excess": v_excess,
        })
    # trim universe by plausibility (current cap + momentum) to bound runtime
    coins.sort(key=lambda c: c["x0"] + max(c["v_excess"], 0.0), reverse=True)
    coins = coins[:600]

    # --- simulate in 10-minute steps (the marks' cadence) ---
    rng = random.Random(SEED)
    n_reads = len(reads)
    step_min = 10.0
    horizon_h = (reads[-1] - t_last) / 3600.0
    n_steps = max(1, int(math.ceil(horizon_h * 60.0 / step_min)))
    read_step = []
    for rt in reads:
        read_step.append(min(n_steps, max(1, int(round((rt - t_last) / (step_min * 60.0))))))
    # per-coin, per-step drift (deterministic given the history)
    steps_per_hour = 60.0 / step_min
    for c in coins:
        drifts = []
        for s in range(1, n_steps + 1):
            t_mid = t_last + (s - 0.5) * step_min * 60.0
            age = max((t_mid - c["created"]) / 3600.0, 0.0)
            base = drift_rate[bucket_idx(age)]
            excess = c["v_excess"] * math.exp(-(s - 0.5) / (VEL_TAU_H * steps_per_hour))
            drifts.append((base + excess) / steps_per_hour)
        c["drifts"] = drifts
    sig_step = sigma / math.sqrt(steps_per_hour)
    occ = [[dict() for _ in range(places)] for _ in range(n_reads)]
    size_samples = [[[] for _ in range(places)] for _ in range(n_reads)]
    gauss = rng.gauss
    n_coins = len(coins)
    drift_lists = [c["drifts"] for c in coins]
    mints = [c["mint"] for c in coins]
    # adaptive path count: keep the Monte Carlo work bounded for the final's
    # 66-mark history (78 steps x up to 600 coins) while staying deterministic
    n_paths = max(120, min(PATHS, 6_000_000 // max(1, n_steps * n_coins)))
    for _ in range(n_paths):
        xs = [c["x0"] for c in coins]
        step = 0
        for ri in range(n_reads):
            target = read_step[ri]
            while step < target:
                for i in range(n_coins):
                    xs[i] += drift_lists[i][step] + sig_step * gauss(0.0, 1.0)
                step += 1
            order = sorted(range(n_coins), key=lambda i: xs[i], reverse=True)
            for pl in range(places):
                idx = order[pl]
                m = mints[idx]
                d = occ[ri][pl]
                d[m] = d.get(m, 0) + 1
                size_samples[ri][pl].append(xs[idx])
    # --- reserve by mean horizon ---
    h_mean = sum((rt - t_last) / 3600.0 for rt in reads) / max(n_reads, 1)
    reserve = RESERVE_RINF - (RESERVE_RINF - RESERVE_R0) * math.exp(-h_mean / RESERVE_TAU_H)
    reserve = max(0.05, min(0.90, reserve))

    probabilities = []
    sizes = []
    top_scores = {}
    # place probabilities: average occupancy over reads, scaled by (1 - reserve)
    for pl in range(places):
        agg = {}
        for ri in range(n_reads):
            for m, cnt in occ[ri][pl].items():
                agg[m] = agg.get(m, 0) + cnt
        total = n_paths * n_reads
        rows = []
        for m, cnt in agg.items():
            p = (cnt / total) * (1.0 - reserve)
            if p >= MIN_P:
                rows.append((m, p))
            top_scores[m] = top_scores.get(m, 0.0) + cnt / total
        s = sum(p for _, p in rows)
        # the named mass must sum to exactly (1 - reserve): rows dropped
        # below MIN_P rejoin the named mass proportionally, never vanish
        if s > 0:
            rows = [(m, p * (1.0 - reserve) / s) for m, p in rows]
        rows.sort(key=lambda r: -r[1])
        for m, p in rows[:100]:
            probabilities.append({"mint": m, "place": pl + 1, "p": round(p, 6)})
    # top5 by total occupancy across places: the coins most likely to be
    # anywhere in the top 5 (the tie-break rewards being in the top 5 first,
    # place accuracy second; an expected-place ordering scored worse on the
    # practice folds, so it was reverted)
    top5 = [m for m, _ in sorted(top_scores.items(), key=lambda kv: -kv[1])[:5]]
    # sizes: simulated caps per place, mixed with a wide prior for the unborn share
    for pl in range(places):
        samples = []
        for ri in range(n_reads):
            samples.extend(size_samples[ri][pl])
        samples.sort()
        med = samples[len(samples) // 2] if samples else 3.0
        # histogram over the 82 bins: simulated mass (1-reserve) + wide normal (reserve)
        bin_p = [0.0] * 82
        bw = (edges[1] - edges[0]) if len(edges) > 1 else 0.05
        KW = 2.5  # kernel half-width, in bins
        for x in samples:
            b = 0
            for i, e in enumerate(edges):
                if x >= e:
                    b = i + 1
                else:
                    break
            if 1 <= b <= 80:
                u = b + (x - edges[b - 1]) / bw
            elif b == 0:
                u = max(0.01, 1.0 + (x - edges[0]) / bw)
            else:
                u = min(80.99, 81.0 + (x - edges[80]) / bw)
            ws = []
            wsum = 0.0
            for j in range(82):
                w = 1.0 - abs(u - (j + 0.5)) / KW
                if w > 0:
                    ws.append((j, w)); wsum += w
            for j, w in ws:
                bin_p[j] += (1.0 - reserve) / len(samples) * (w / wsum)
        sd_wide = SIZE_EXTRA_SD
        for b in range(82):
            lo = -float("inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == 81 else edges[b]
            pb = norm_cdf(hi, med, sd_wide) - norm_cdf(lo, med, sd_wide)
            bin_p[b] += reserve * pb
        tot = sum(bin_p)
        for b in range(82):
            p = bin_p[b] / tot if tot else 1.0 / 82
            if p >= 0.0005:
                sizes.append({"place": pl + 1, "bin": b, "p": round(p, 6)})
    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    blob = json.dumps(forecast, sort_keys=True).encode()
    if len(blob) > 1_048_576:
        raise SystemExit("forecast exceeds 1MB")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(blob.decode() + "\n")


def norm_cdf(x, mean, sd):
    if x == float("inf"):
        return 1.0
    if x == -float("inf"):
        return 0.0
    return 0.5 * math.erfc((mean - x) / (sd * math.sqrt(2.0)))


# Each coin's excess velocity is projected forward with exponential decay so that a recent surge influences early steps strongly but its effect fades to zero over roughly one hour.
if __name__ == "__main__":
    main()
