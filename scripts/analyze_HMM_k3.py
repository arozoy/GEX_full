import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_parquet(OUTPUT_DIR / "momentum_table.parquet").sort_values("date").dropna(
    subset=["gex_total", "morning_ret", "afternoon_ret", "n_contracts"]
)

df["gex_x_morning"] = df["gex_total"] * df["morning_ret"]

# 63-day rolling z-score removes the structural 4x GEX growth over 2022-2026
# so the HMM learns regime structure, not time-period separation
df["gex_zscore"] = (
    df["gex_total"]
    .rolling(63, min_periods=20)
    .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-12))
)
df = df.dropna(subset=["gex_zscore"])

features = ["morning_ret", "afternoon_ret", "gex_zscore", "n_contracts"]
X_raw = df[features].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── 2. Fit HMM with K=3 ───────────────────────────────────────────────────────
N_STATES  = 3
N_INIT    = 20   # multiple restarts — EM is not convex
N_ITER    = 200

best_model, best_ll = None, -np.inf
for _ in range(N_INIT):
    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
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

print(f"K=3  LL={best_ll:.2f}")

# ── 3. Viterbi decoding ───────────────────────────────────────────────────────
state_seq = best_model.predict(X)
df["hmm_state"] = state_seq

# ── 4. State economic content ─────────────────────────────────────────────────
print("\n--- State profiles ---")
for s in range(N_STATES):
    mask = df["hmm_state"] == s
    sub  = df[mask]
    corr = sub[["morning_ret", "afternoon_ret"]].corr().iloc[0, 1]
    mean_gex   = sub["gex_total"].mean() / 1e9
    mean_zscore = sub["gex_zscore"].mean()
    n = mask.sum()
    print(
        f"State {s}: n={n:>3d} ({n/len(df)*100:.0f}%)  "
        f"corr={corr:+.3f}  mean GEX=${mean_gex:.1f}B  "
        f"mean GEX-zscore={mean_zscore:+.2f}"
    )

# ── 5. Transition matrix ──────────────────────────────────────────────────────
# Diagonal values > ~0.85 = states are persistent (genuine regimes).
# Values near 1/K = no temporal persistence = just clustering, not sequencing.
print("\n--- Transition matrix ---")
print("     ", "  ".join(f"->S{j}" for j in range(N_STATES)))
for i, row in enumerate(best_model.transmat_):
    print(f"S{i}:  " + "  ".join(f"{v:.3f}" for v in row))

# ── 6. Temporal distribution by year ─────────────────────────────────────────
# If one state dominates a single year, the HMM is still separating time
# periods rather than market regimes despite normalization.
df["year"] = pd.to_datetime(df["date"]).dt.year
print("\n--- State counts by year ---")
print(df.groupby(["year", "hmm_state"]).size().unstack(fill_value=0).to_string())

# ── 7. State labels ───────────────────────────────────────────────────────────
# Update these after inspecting the profiles above
STATE_LABELS = {0: "amplifying", 1: "dampening", 2: "noise"}
df["regime"] = df["hmm_state"].map(STATE_LABELS)
