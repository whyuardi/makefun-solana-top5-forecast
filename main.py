#!/usr/bin/env python3
"""pumpfun-top5-forecast-01: forecast the 5 largest pump.fun coins by market
cap at each of seven answer reads, with honest probabilities.

Model: per-coin log10 market-cap (in SOL) follows a random walk with
momentum that decays over a few hours toward a mildly negative long-run
drift (pump.fun coins fade within hours; pool/graduated coins fade slower
than curve coins). A deterministic Monte Carlo (fixed seed) simulates each
answer read independently, ranks the coins, and turns place frequencies and
place-cap distributions into probabilities. A reserve of each place is left
unnamed for coins born after the history cut, which no forecast can name.

Reads /input/task.json + /input/history.ndjson (or $MAKEFUN_INPUT_DIR),
writes /output/forecast.json (or $MAKEFUN_OUTPUT_DIR).
Standard library only. Deterministic: same input -> same output.
"""
import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SEED = 20261112
N_SIMS = 4000

# --- tunables (fixed in advance; see notes.md for the backtest) ---
MOMENTUM_HALFLIFE_H = 3.0      # recent slope decays with this half-life
LONG_DRIFT_POOL = -0.03         # log10 SOL per hour, long-run drift (pool)
LONG_DRIFT_CURVE = -0.08        # log10 SOL per hour, long-run drift (curve)
VOL_FLOOR = 0.10                # log10 per sqrt(hour)
VOL_CAP = 0.55
YOUNG_VOL_MULT = 1.4            # coins younger than 1h are noisier
YOUNG_AGE_H = 1.0
SLOPE_WINDOW = 6                # marks used for the recent slope
SLOPE_PRIOR_W = 3.0             # shrinkage weight toward the group slope
MAX_P = 0.40                    # cap on any single place probability
MIN_P = 0.002                   # drop place rows below this after scaling
SIZE_LEAVE = 0.10               # unnamed mass left on each place's sizes
SMOOTH_ALPHA = 5.0              # additive smoothing counts (places)
SIZE_SMOOTH = 2.0               # additive smoothing counts (size bins)
RESERVE_FLOOR = 0.50            # unnamed mass per place (newcomers after cut)
RESERVE_LONG = 0.10             # extra reserve at long horizons
DRIFT_CLIP = 0.4                # clip per-coin drift (log10/hr)
MOM_GAIN_CAP = 0.6              # cap on integrated momentum gain (log10)
MOM_LOSS_CAP = -1.0             # cap on integrated momentum loss (log10)
HIGH_CAP_TAPER = 3.5            # soften expected caps above this log10 SOL


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap_sol(lamports):
    return math.log10(int(lamports)) - 9.0


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    reads = [parse_time(r) for r in task["reads"]]
    out_name = task.get("output", "forecast.json")

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark")
    mark_times = [parse_time(m["at"]) for m in marks]
    last_t = mark_times[-1]

    # --- per-coin series -------------------------------------------------
    coins = {}  # mint -> dict
    for i, m in enumerate(marks):
        for r in m["rows"]:
            c = coins.setdefault(r["mint"], {"caps": {}, "venue": {}, "created": r["created_at"]})
            c["caps"][i] = logcap_sol(r["market_cap_lamports"])
            c["venue"][i] = r.get("venue", "curve")
    last_rows = marks[-1]["rows"]
    last_rank = {r["mint"]: r["rank"] for r in last_rows}

    # --- group drift / vol priors ---------------------------------------
    def ols_slope(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 0:
            return 0.0, 0.0
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
        sd = (sum(v * v for v in resid) / max(1, n - 2)) ** 0.5 if n > 2 else 0.0
        return b, sd

    group = {"pool": {"slopes": [], "vols": []}, "curve": {"slopes": [], "vols": []}}
    for mint, c in coins.items():
        idx = sorted(c["caps"])
        if len(idx) < 2:
            continue
        xs = [(mark_times[i] - mark_times[0]).total_seconds() / 3600.0 for i in idx]
        ys = [c["caps"][i] for i in idx]
        b, sd = ols_slope(xs, ys)
        g = "pool" if c["venue"][idx[-1]] == "pool" else "curve"
        group[g]["slopes"].append(b)
        group[g]["vols"].append(sd)

    def median(v):
        if not v:
            return 0.0
        v = sorted(v)
        return v[len(v) // 2]

    group_prior = {}
    for g in ("pool", "curve"):
        # group slope prior, clipped: the raw median is survivor-biased
        # (pool median +0.39/hr in practice is a pump-phase artefact)
        sl = max(-0.05, min(0.15, median(group[g]["slopes"])))
        group_prior[g] = {
            "slope": sl,
            "vol": max(VOL_FLOOR, median(group[g]["vols"]) * 2.0),
        }

    # --- candidate coins: present in the last mark -----------------------
    cands = []
    for r in last_rows:
        mint = r["mint"]
        c = coins[mint]
        if r.get("mayhem") and c["venue"][max(c["caps"])] == "curve":
            continue  # mayhem on curve is never ranked until its pool prices it
        idx = sorted(c["caps"])
        use = idx[-SLOPE_WINDOW:]
        xs = [(mark_times[i] - mark_times[0]).total_seconds() / 3600.0 for i in use]
        ys = [c["caps"][i] for i in use]
        slope, sd = ols_slope(xs, ys) if len(use) >= 2 else (0.0, 0.0)
        venue = "pool" if c["venue"][idx[-1]] == "pool" else "curve"
        prior = group_prior[venue]
        drift0 = (len(use) * slope + SLOPE_PRIOR_W * prior["slope"]) / (len(use) + SLOPE_PRIOR_W)
        drift0 = max(-DRIFT_CLIP, min(DRIFT_CLIP, drift0))
        vol = sd * 2.0 if sd > 0 else prior["vol"]
        vol = max(VOL_FLOOR, min(VOL_CAP, vol))
        age_h = (last_t - parse_time(c["created"])).total_seconds() / 3600.0
        if age_h < YOUNG_AGE_H:
            vol = min(VOL_CAP, vol * YOUNG_VOL_MULT)
        cands.append({
            "mint": mint,
            "last": c["caps"][idx[-1]],
            "drift0": drift0,
            "vol": vol,
            "venue": venue,
            "rank": last_rank[mint],
        })
    if len(cands) < places:
        raise SystemExit("fewer candidate coins than places")

    # --- Monte Carlo over answer reads (independent per read) ------------
    rng = random.Random(SEED)
    gauss = rng.gauss
    dts = [(rt - last_t).total_seconds() / 3600.0 for rt in reads]
    dt_mean = sum(dts) / len(dts)
    reserve = RESERVE_FLOOR + RESERVE_LONG * (1.0 - math.exp(-dt_mean / 6.0))
    hl = MOMENTUM_HALFLIFE_H / math.log(2)  # exponential decay constant
    n_c = len(cands)

    place_counts = [[0] * n_c for _ in range(places)]
    size_counts = [[0] * (len(edges) + 1) for _ in range(places)]
    top5_score = [0.0] * n_c  # expected-rank weight

    for rt_i, dt in enumerate(dts):
        # expected logcap and sd per coin at this read
        mus = []
        sds = []
        mom = hl * (1.0 - math.exp(-dt / hl))  # integrated momentum
        for c in cands:
            g = LONG_DRIFT_POOL if c["venue"] == "pool" else LONG_DRIFT_CURVE
            mgain = c["drift0"] * mom
            mgain = max(MOM_LOSS_CAP, min(MOM_GAIN_CAP, mgain))
            mu = c["last"] + mgain + g * dt
            if mu > HIGH_CAP_TAPER:
                mu = HIGH_CAP_TAPER + (mu - HIGH_CAP_TAPER) * 0.5
            mus.append(mu)
            sds.append(c["vol"] * math.sqrt(min(dt, 8.0)))
        for _ in range(N_SIMS):
            vals = [mus[i] + sds[i] * gauss(0.0, 1.0) for i in range(n_c)]
            order = sorted(range(n_c), key=lambda i: vals[i], reverse=True)
            for p in range(places):
                ci = order[p]
                place_counts[p][ci] += 1
                top5_score[ci] += (places - p)
            for p in range(places):
                v = vals[order[p]]
                b = 0
                for e in edges:
                    if v >= e:
                        b += 1
                    else:
                        break
                size_counts[p][b] += 1

    # --- convert counts to probabilities ---------------------------------
    # Name a coin at a place only while its modelled probability beats the
    # unnamed per-coin rate (reserve shared by the chart coins not named
    # there); naming more coins concentrates the unnamed mass on the rest.
    def floor6(x):
        return math.floor(x * 1e6) / 1e6

    probabilities = []
    for p in range(places):
        counts = place_counts[p]
        named = [i for i in range(n_c) if counts[i] > 0]
        n_named = len(named)
        raw = {}
        for i in named:
            raw[i] = (counts[i] + SMOOTH_ALPHA) / (N_SIMS + SMOOTH_ALPHA * n_named)
        tot = sum(raw.values())
        if tot <= 0:
            continue
        shrunk = {}
        for i, v in raw.items():
            shrunk[i] = min(0.9 * (v / tot) + 0.1 * (1.0 / n_named), MAX_P)
        order = sorted(shrunk, key=lambda i: -shrunk[i])
        keep = []
        for i in order:
            k = len(keep) + 1
            u = reserve / max(1, 100 - k + 1)
            if shrunk[i] >= u or k <= 5:
                keep.append(i)
            else:
                break
        s = sum(shrunk[i] for i in keep)
        if s <= 0:
            continue
        target = 1.0 - reserve
        final = {}
        for i in keep:
            pv = floor6(shrunk[i] / s * target)
            if pv >= MIN_P:
                final[i] = pv
        while sum(final.values()) > target and final:
            i = max(final, key=lambda k: final[k])
            final[i] = floor6(final[i] - 1e-6)
            if final[i] <= 0:
                del final[i]
        for i, pv in final.items():
            if pv > 0:
                probabilities.append({"mint": cands[i]["mint"], "place": p + 1, "p": pv})

    # sizes: lognormal fit of the simulated place caps, sd floored and grown
    # with horizon (caps move a lot over 7-13h, little over minutes)
    sizes = []
    for p in range(places):
        counts = size_counts[p]
        tot = sum(counts)
        nb = len(edges) + 1
        mean = 0.0
        for b, c in enumerate(counts):
            mid = 1.975 if b == 0 else (6.025 if b == nb - 1 else (edges[b - 1] + edges[b]) / 2.0)
            mean += c * mid
        mean /= tot
        var = 0.0
        for b, c in enumerate(counts):
            mid = 1.975 if b == 0 else (6.025 if b == nb - 1 else (edges[b - 1] + edges[b]) / 2.0)
            var += c * (mid - mean) ** 2
        sd = math.sqrt(var / tot)
        sd = max(sd, 0.30 + 0.06 * math.sqrt(dt_mean))
        sd = min(sd, 0.9)
        target = 1.0 - SIZE_LEAVE
        final = {}
        for b in range(nb):
            lo = -1e9 if b == 0 else edges[b - 1]
            hi = 1e9 if b == nb - 1 else edges[b]
            pr = 0.5 * (math.erf((hi - mean) / (sd * math.sqrt(2.0))) - math.erf((lo - mean) / (sd * math.sqrt(2.0))))
            pv = floor6(pr * target)
            if pv >= MIN_P:
                final[b] = pv
        while sum(final.values()) > target and final:
            b = max(final, key=lambda k: final[k])
            final[b] = floor6(final[b] - 1e-6)
            if final[b] <= 0:
                del final[b]
        for b, pv in sorted(final.items()):
            if pv > 0:
                sizes.append({"place": p + 1, "bin": b, "p": pv})

    # top5 by expected rank: higher top5_score first, tie-break by last rank
    ranked = sorted(range(n_c), key=lambda i: (-top5_score[i], cands[i]["rank"]))
    top5 = [cands[i]["mint"] for i in ranked[:places]]

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
