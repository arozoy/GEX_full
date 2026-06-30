"""
iv_gamma.py
-----------
Single-contract implied volatility (Jäckel / py_lets_be_rational) + BSM gamma,
with the numerical guards from `Jackel_BSM_IV_Guide.pdf` and
`Gamma_Computation_0DTE.pdf`.

Public API:
    compute_iv_gamma(option_bid, option_ask, S, K, T, r, q, flag) -> ContractResult

CLI:
    python iv_gamma.py --bid 1.20 --ask 1.25 --S 580 --K 580 \
                       --T 0.000913 --r 0.0525 --q 0.013 --flag C
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Optional

import py_lets_be_rational as lbr
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants (per the IV + Gamma guides)
# ---------------------------------------------------------------------------
SECONDS_PER_YEAR = 365.25 * 24 * 3600
T_FLOOR          = 60.0 / SECONDS_PER_YEAR     # 1 minute in years
SIGMA_MIN        = 0.01
SIGMA_MAX        = 5.00
D1_OVERFLOW      = 38.0
CONTRACT_MULT    = 100
SQRT_2PI         = math.sqrt(2.0 * math.pi)

N  = norm.cdf
Np = norm.pdf

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ContractResult:
    sigma:        Optional[float]
    gamma:        Optional[float]
    dollar_gamma: Optional[float]
    weight:       float
    status:       str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# BSM helpers (used for round-trip validation)
# ---------------------------------------------------------------------------
def bsm_price(S, K, T, r, q, sigma, flag):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if flag == 'C' else max(K - S, 0.0)
    F      = S * math.exp((r - q) * T)
    vsqrtT = sigma * math.sqrt(T)
    d1     = (math.log(F / K) + 0.5 * vsqrtT * vsqrtT) / vsqrtT
    d2     = d1 - vsqrtT
    disc   = math.exp(-r * T)
    if flag == 'C':
        return disc * (F * N(d1) - K * N(d2))
    return disc * (K * N(-d2) - F * N(-d1))


# ---------------------------------------------------------------------------
# Guarded gamma — six error checks from `Gamma_Computation_0DTE.pdf`
# ---------------------------------------------------------------------------
def _stable_log_ratio(S: float, K: float) -> float:
    ratio = S / K
    if abs(ratio - 1.0) < 1e-3:
        x = ratio - 1.0
        return x - 0.5 * x * x + (x ** 3) / 3.0
    return math.log(ratio)


def _gamma_cap(S: float) -> float:
    return 1.0 / (S * SIGMA_MIN * math.sqrt(T_FLOOR))


def compute_gamma(S, K, T, r, q, sigma) -> float:
    if sigma is None or not math.isfinite(sigma):
        return float('nan')
    if sigma < SIGMA_MIN or sigma > SIGMA_MAX:
        return float('nan')
    if S <= 0 or K <= 0:
        return float('nan')
    if T <= 0:
        return float('nan')
    if T < T_FLOOR:
        return 0.0

    sqrtT = math.sqrt(T)
    logSK = _stable_log_ratio(S, K)
    if abs(logSK) > 3.0 * sigma * sqrtT:
        return 0.0

    d1 = (logSK + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    if abs(d1) > D1_OVERFLOW:
        return 0.0

    Nd1 = math.exp(-0.5 * d1 * d1) / SQRT_2PI
    if Nd1 < 1e-300:
        return 0.0

    gamma = math.exp(-q * T) * Nd1 / (S * sigma * sqrtT)
    if gamma > _gamma_cap(S):
        return float('nan')
    return gamma


def dollar_gamma(gamma: float, S: float) -> float:
    if gamma is None or not math.isfinite(gamma):
        return float('nan')
    return gamma * S * S * 0.01 * CONTRACT_MULT


def quote_quality_weight(bid: float, ask: float) -> float:
    mid = 0.5 * (bid + ask)
    if mid <= 0:
        return 0.0
    spread_pct = max(ask - bid, 0.0) / mid
    return 1.0 / (1.0 + 10.0 * spread_pct)


# ---------------------------------------------------------------------------
# Quote-validity filter (time-of-day spread thresholds from the IV guide)
# ---------------------------------------------------------------------------
def _spread_threshold(T: float) -> float:
    if T > 0.0013:  return 0.50
    if T > 0.0006:  return 0.40
    if T > 0.0002:  return 0.30
    if T > 0.0001:  return 0.20
    return 0.10


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_iv_gamma(option_bid: float, option_ask: float,
                     S: float, K: float, T: float,
                     r: float, q: float, flag: str) -> ContractResult:
    """Solve Jäckel IV and BSM gamma for one option contract.

    Parameters
    ----------
    option_bid, option_ask : NBBO bid / ask of the option (mid is what we invert).
    S        : underlying mid price.
    K        : strike.
    T        : time to expiry in years (use exact wall-clock seconds for 0DTE).
    r, q     : continuously compounded risk-free rate and dividend yield.
    flag     : 'C' or 'P'.

    Returns
    -------
    ContractResult(sigma, gamma, dollar_gamma, weight, status).
    """
    flag = (flag or '').upper()[:1]
    if flag not in ('C', 'P'):
        return ContractResult(None, None, None, 0.0, "bad_flag")

    if option_bid <= 0 or option_ask <= option_bid:
        return ContractResult(None, None, None, 0.0, "bad_quote")
    if T <= 0 or S <= 0:
        return ContractResult(None, None, None, 0.0, "bad_T_or_S")

    mid = 0.5 * (option_bid + option_ask)
    if mid < 0.05:
        return ContractResult(None, None, None, 0.0, "mid_too_low")
    if (option_ask - option_bid) / mid > _spread_threshold(T):
        return ContractResult(None, None, None, 0.0, "spread_too_wide")

    F        = S * math.exp((r - q) * T)
    disc     = math.exp(-r * T)
    intrinsic = disc * (max(F - K, 0.0) if flag == 'C' else max(K - F, 0.0))
    if mid <= intrinsic + 1e-8:
        return ContractResult(None, None, None, 0.0, "below_intrinsic")

    # --- Jäckel via py_lets_be_rational ---
    # The library's `price` argument is the *undiscounted* (Black-76 forward)
    # price: it checks `price >= F-K` for calls and divides by sqrt(F*K) with
    # no exp(-rT) factor. Convert market mid -> undiscounted by * exp(r*T).
    #   BSM_mid = exp(-rT) * (F·N(d1) - K·N(d2))
    #   => BSM_mid * exp(r*T) = F·N(d1) - K·N(d2)  (= Black-76 forward price)
    undisc_price = mid * math.exp(r * T)
    call_put     = 1.0 if flag == 'C' else -1.0
    try:
        sigma = lbr.implied_volatility_from_a_transformed_rational_guess(
            undisc_price, F, K, T, call_put
        )
    except Exception as e:
        return ContractResult(None, None, None, 0.0, f"solver_error:{e}")

    if sigma is None or not math.isfinite(sigma) or not (SIGMA_MIN < sigma < SIGMA_MAX):
        return ContractResult(None, None, None, 0.0, "sigma_out_of_bounds")

    g  = compute_gamma(S, K, T, r, q, sigma)
    dg = dollar_gamma(g, S)
    w  = quote_quality_weight(option_bid, option_ask)

    # Round-trip sanity
    bsm_chk = bsm_price(S, K, T, r, q, sigma, flag)
    status  = "ok" if abs(bsm_chk - mid) / mid < 1e-3 else "round_trip_suspicious"
    return ContractResult(sigma, g, dg, w, status)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    p = argparse.ArgumentParser(description="Single-contract IV + gamma")
    p.add_argument("--bid",  type=float, required=True)
    p.add_argument("--ask",  type=float, required=True)
    p.add_argument("--S",    type=float, required=True, help="Underlying mid")
    p.add_argument("--K",    type=float, required=True, help="Strike")
    p.add_argument("--T",    type=float, required=True, help="Years to expiry")
    p.add_argument("--r",    type=float, default=0.0525)
    p.add_argument("--q",    type=float, default=0.013)
    p.add_argument("--flag", type=str,   required=True, choices=["C", "P", "c", "p"])
    args = p.parse_args()

    res = compute_iv_gamma(args.bid, args.ask, args.S, args.K, args.T,
                            args.r, args.q, args.flag)
    print(f"sigma         : {res.sigma}")
    print(f"gamma         : {res.gamma}")
    print(f"dollar_gamma  : {res.dollar_gamma}")
    print(f"weight        : {res.weight}")
    print(f"status        : {res.status}")


if __name__ == "__main__":
    _cli()
