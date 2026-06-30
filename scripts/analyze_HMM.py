import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR

# ── 1. Load and construct emission matrix ──────────────────────────────────────
df = pd.read_parquet(OUTPUT_DIR / "momentum_table.parquet").sort_values("date").dropna(
    subset=["gex_total", "morning_ret", "afternoon_ret", "n_contracts"]
)

# Interaction term: captures joint signal strength
df["gex_x_morning"] = df["gex_total"] * df["morning_ret"]

# Normalize GEX with a 63-day rolling z-score to remove the structural upward
# trend in GEX over 2022-2026 (~4x growth). Without this, the HMM separates
# time periods (early vs late sample) rather than market regimes.
df["gex_zscore"] = (
    df["gex_total"]
    .rolling(63, min_periods=20)
    .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-12))
)
df = df.dropna(subset=["gex_zscore"])

features = ["morning_ret", "afternoon_ret", "gex_zscore", "n_contracts"]
X_raw = df[features].values

# Standardize — Gaussian HMM fits means/covariances in emission space;
# unscaled gex_total (~1e9) will dwarf morning_ret (~1e-3) and dominate
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── 2. Fit HMM for K = 2, 3, 4 and select via BIC ────────────────────────────
def fit_hmm(X, n_states, n_iter=200, n_init=20):
    """
    Multiple random restarts (n_init) are critical — EM is not convex.
    Return best model by log-likelihood across restarts.
    """
    best_model, best_ll = None, -np.inf
    for _ in range(n_init):
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",   # NOT diagonal — we need cross-correlations
            n_iter=n_iter,
            tol=1e-5,
            random_state=np.random.randint(0, 10000),
        )
        try:
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll, best_model = ll, model
        except Exception:
            continue
    return best_model, best_ll

def bic(model, X):
    n, d = X.shape
    k = model.n_components
    # params: transition (k²-k) + means (k*d) + covariance full (k*d*(d+1)/2)
    n_params = (k**2 - k) + k * d + k * d * (d + 1) // 2
    return -2 * model.score(X) + n_params * np.log(n)

results = {}
for k in [2, 3, 4]:
    model, ll = fit_hmm(X, k)
    results[k] = {"model": model, "ll": ll, "bic": bic(model, X)}
    print(f"K={k}  LL={ll:.2f}  BIC={results[k]['bic']:.2f}")

best_k = min(results, key=lambda k: results[k]["bic"])
best_model = results[best_k]["model"]
print(f"\nSelected K={best_k} by BIC")

# ── 3. Viterbi decoding — most likely state sequence ──────────────────────────
# predict() runs the Viterbi algorithm: finds the globally optimal state path,
# not greedy argmax at each step. This matters for temporal consistency.
state_seq = best_model.predict(X)
df["hmm_state"] = state_seq

# ── 4. Label states by economic content ───────────────────────────────────────
# Inspect each state's mean morning→afternoon correlation and GEX level
for s in range(best_k):
    mask = df["hmm_state"] == s
    sub = df[mask]
    corr = sub[["morning_ret", "afternoon_ret"]].corr().iloc[0, 1]
    mean_gex = sub["gex_total"].mean() / 1e9   # in $B
    n = mask.sum()
    print(f"State {s}: n={n}, morning/afternoon corr={corr:.3f}, "
          f"mean GEX=${mean_gex:.2f}B")

# Manually assign labels after inspecting output — example:
# State 0: corr ≈ +0.4, mean GEX = -$1.2B  → AMPLIFYING
# State 1: corr ≈ -0.3, mean GEX = +$0.8B  → DAMPENING
# State 2: corr ≈ +0.0, mean GEX = -$0.1B  → NOISE
STATE_LABELS = {0: "amplifying", 1: "dampening", 2: "noise"}  # adjust after inspection
df["regime"] = df["hmm_state"].map(STATE_LABELS)