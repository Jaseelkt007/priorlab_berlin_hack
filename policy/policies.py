"""Speed policies for the delta pick line, and their offline replay on a counterfactual pick bank.

The decision (docs/handoff/session-2.md §2): after the 35 ms vacuum dwell, before the loaded move, choose the speed
setting alpha (ABB AccSet: 20–100 %) for this pick. Every policy sees only what a real line knows at that moment (the
logged features, including the vacuum sensor's seal_kpa and evac_ms) and, afterwards, the outcome at the alpha it chose.

Replay: the bank holds every pick instance simulated at all alphas (exact counterfactuals, sim.picklog.simulate_instance),
so each policy is evaluated on the same items, line states and vision errors, with honest bandit feedback: only the
chosen alpha's outcome is appended to its log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sim.picklog import ALPHAS, FEATURES

X_COLS = FEATURES + ["alpha"]


class Policy:
    name = "policy"

    def start(self, log: pd.DataFrame) -> None:
        """The line's log before the stream (all alphas from commissioning speed tests)."""

    def choose(self, cand: pd.DataFrame) -> tuple[float, dict]:
        """cand: one row per alpha (same pick, pre-move features only). Returns (alpha, diagnostics)."""
        raise NotImplementedError

    def observe(self, row: pd.Series) -> None:
        """The outcome of the executed pick (its features, chosen alpha, dropped, place_offset_mm)."""


def value_rate(p_drop, cycle_s, cost: float):
    """Expected value per second of a pick: a held pick counts 1, a drop costs `cost` picks' worth (scrap, clean-up,
    possible stop), divided by the cycle time."""
    return (1.0 - (1.0 + cost) * np.asarray(p_drop)) / np.asarray(cycle_s)


class FixedRecipe(Policy):
    """Today's practice: one speed per product, set from the commissioning speed tests.
    target rule: the fastest alpha whose observed drop rate, and that of every slower setting, is at most `target`;
    `margin` steps slower for safety. cost rule: the alpha with the best average value per second on the log
    (value_rate with that product's drop rate and cycle time at each alpha). Unknown products run at the slowest alpha."""

    def __init__(self, target: float | None = None, margin: int = 0, cost: float | None = None):
        self.target, self.margin, self.cost = target, margin, cost
        rule = f"cost={cost:g}" if cost is not None else f"target={target:g}"
        self.name = f"fixed({rule}{', margin ' + str(margin) if margin else ''})"
        self.recipe: dict[str, float] = {}

    def start(self, log):
        for prod, g in log.groupby("product"):
            if self.cost is not None:
                st = g.groupby("alpha").agg(d=("dropped", "mean"), c=("cycle_s", "mean")).reindex(ALPHAS)
                v = value_rate(st["d"].fillna(1.0), st["c"].fillna(np.inf), self.cost)
                k = int(np.nanargmax(v))
            else:
                rate = g.groupby("alpha")["dropped"].mean().reindex(ALPHAS)
                ok = [a for i, a in enumerate(ALPHAS) if (rate.iloc[: i + 1].fillna(1.0) <= self.target).all()]
                k = list(ALPHAS).index(ok[-1]) if ok else 0
            self.recipe[prod] = float(ALPHAS[max(0, k - self.margin)])

    def choose(self, cand):
        prod = cand["product"].iloc[0]
        return self.recipe.get(prod, float(ALPHAS[0])), {}


class Oracle(Policy):
    """Hindsight upper bound: the fastest alpha that does not drop this very item (slowest if all drop)."""
    name = "oracle"

    def choose(self, cand):
        ok = cand.loc[~cand["dropped"].astype(bool), "alpha"]
        return (float(ok.max()) if len(ok) else float(cand["alpha"].min())), {}


class RiskPolicy(Policy):
    """Learned risk model in the loop: P(drop | pre-move features, alpha) for all alphas of this pick, then either the
    fastest alpha with P <= target (the slowest alpha if none qualifies, as an operator would), or, with `cost`, the
    alpha with the best expected value per second (value_rate). The model is refit on the growing log every
    `refit_every` picks (a live line would refresh its speed map about once a second, i.e. every 2–3 picks).
    Exploration (optional): with probability eps, try one setting above the choice when its predicted risk is below
    explore_max (the line can learn that a setting is safe only by using it).
    frozen: never add the stream's picks (the commissioning log only), to isolate in-context adaptation.
    safe_step: safe-set expansion (as in SafeOpt, and as engineers ramp up a new product): never more than one setting
    above the fastest setting this product has already been seen to hold in the log (the slowest setting if none).
    Probabilities far outside the logged speeds are extrapolations; this bounds what one wrong extrapolation costs.
    budget: maximum log rows; beyond it, evict_log() keeps every drop, the most recent picks and an even thinning of
    older held picks per product × alpha."""

    def __init__(self, target: float | None = None, refit_every: int = 5, eps: float = 0.0,
                 explore_max: float | None = None, seed: int = 0, frozen: bool = False, budget: int | None = None,
                 cost: float | None = None, safe_step: bool = False):
        self.target, self.cost, self.refit_every, self.eps = target, cost, refit_every, eps
        self.safe_step = safe_step
        self.explore_max = explore_max if explore_max is not None else (2 * target if target is not None else 0.05)
        self.rng = np.random.default_rng(seed)
        self.frozen, self.budget = frozen, budget
        self.n_since = 0

    def rule(self) -> str:
        r = f"cost={self.cost:g}" if self.cost is not None else f"target={self.target:g}"
        return r + (", safe-step" if self.safe_step else "")

    # model hooks ---------------------------------------------------------------
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def _proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # policy ----------------------------------------------------------------------
    def start(self, log):
        self.log = log[X_COLS + ["product", "dropped"]].copy()
        self._refit()

    def _refit(self):
        if self.budget is not None and len(self.log) > self.budget:
            self.log = evict_log(self.log, self.budget)
        y = self.log["dropped"].astype(int).to_numpy()
        self.const = None if 0 < y.sum() < len(y) else float(y.mean())   # one class only: no model to fit
        if self.const is None:
            self._fit(self.log[X_COLS].to_numpy(float), y)
        self.n_since = 0

    def choose(self, cand):
        X = cand[X_COLS].to_numpy(float)
        p = np.full(len(X), self.const) if self.const is not None else self._proba(X)
        al = cand["alpha"].to_numpy(float)
        allowed = np.ones(len(al), bool)
        if self.safe_step:
            prod = cand["product"].iloc[0]
            held = self.log.loc[(self.log["product"] == prod) & ~self.log["dropped"].astype(bool), "alpha"]
            top = float(held.max()) if len(held) else float(al.min()) - 0.1
            allowed = al <= top + 0.1 + 1e-9
        if self.cost is not None:      # best expected value per second; the cycle time is planned before the move
            v = value_rate(p, cand["cycle_s"].to_numpy(float), self.cost)
            i = int(np.argmax(np.where(allowed, v, -np.inf)))
        else:
            ok = np.flatnonzero((p <= self.target) & allowed)
            i = ok[np.argmax(al[ok])] if len(ok) else int(np.argmin(al))  # nothing qualifies: the slowest setting
        explored = False
        if self.eps > 0 and self.rng.uniform() < self.eps:
            up = np.flatnonzero(al > al[i])
            if len(up):
                j = up[np.argmin(al[up])]
                if p[j] <= self.explore_max:
                    i, explored = j, True
        return float(al[i]), dict(p_drop=float(p[i]), explored=explored)

    def observe(self, row):
        if self.frozen:
            return
        self.log = pd.concat([self.log, row[X_COLS + ["product", "dropped"]].to_frame().T], ignore_index=True)
        self.n_since += 1
        if self.n_since >= self.refit_every:
            self._refit()


_TABPFN: dict = {}     # one loaded classifier per (version, device, n_estimators), shared by sequential policies


class TabPFNPolicy(RiskPolicy):
    """TabPFN-3.5 classifier with the line's log as in-context data: no training, the log is the model."""

    def __init__(self, target: float | None = None, version: str = "3.5", device: str = "cuda", n_estimators: int = 1,
                 **kw):
        super().__init__(target, **kw)
        key = (version, device, n_estimators)
        if key not in _TABPFN:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            ver = {"3.5": ModelVersion.V3_5, "3.5-fast": ModelVersion.V3_5_FAST}[version]
            _TABPFN[key] = TabPFNClassifier.create_default_for_version(ver, n_estimators=n_estimators, device=device,
                                                                       fit_mode="fit_with_cache", random_state=0)
        self.clf = _TABPFN[key]
        tags = [self.rule()] + (["frozen"] if self.frozen else []) + ([f"eps={self.eps:g}"] if self.eps else [])
        self.name = f"tabpfn-{version}({', '.join(tags)})"

    def _fit(self, X, y):
        self.clf.fit(X, y)

    def _proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


class GBDTPolicy(RiskPolicy):
    """Gradient-boosted trees (scikit-learn HistGradientBoosting) retrained on the same log: the conventional
    trained-model baseline."""

    def __init__(self, target: float | None = None, **kw):
        super().__init__(target, **kw)
        self.name = f"gbdt({self.rule()})"

    def _fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(random_state=0).fit(X, y)

    def _proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


class LogisticPolicy(RiskPolicy):
    """Logistic regression on standardized features (median-imputed, with a missing indicator for evac_ms):
    the simplest trained baseline, monotone in alpha by construction."""

    def __init__(self, target: float | None = None, **kw):
        super().__init__(target, **kw)
        self.name = f"logistic({self.rule()})"

    def _fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.clf = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                 LogisticRegression(max_iter=2000)).fit(X, y)

    def _proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


def evict_log(log: pd.DataFrame, budget: int, recent: float = 0.5) -> pd.DataFrame:
    """Keep the log within `budget` rows (row order = time): every drop (rare and informative), the most recent
    `recent`·budget rows, and the rest of the budget as an even thinning of the older held picks within each product ×
    alpha cell (so no speed setting disappears from the context)."""
    n_recent = int(recent * budget)
    old, new = log.iloc[:-n_recent], log.iloc[-n_recent:]
    keep_fail = old[old["dropped"].astype(bool)]
    held = old[~old["dropped"].astype(bool)]
    room = max(budget - len(new) - len(keep_fail), 0)
    if len(held) > room:
        frac = room / len(held)
        held = held.groupby(["product", "alpha"], group_keys=False).apply(
            lambda g: g.iloc[np.unique(np.linspace(0, len(g) - 1, max(1, int(round(frac * len(g))))).round().astype(int))])
    return pd.concat([keep_fail, held]).sort_index().pipe(lambda d: pd.concat([d, new]))


def replay(bank: pd.DataFrame, policy: Policy, log: pd.DataFrame) -> pd.DataFrame:
    """Run a policy over one stream of the bank (rows: every alpha of each pick, ordered by `idx`). Returns the executed
    picks with the policy's diagnostics."""
    policy.start(log)
    out = []
    pre = [c for c in bank.columns if not c.startswith("_") and c not in ("dropped", "place_offset_mm", "slip_mm",
                                                                           "vac_min_kpa", "move_ms", "a_peak")]
    for idx, cand in bank.groupby("idx", sort=True):
        cand = cand.sort_values("alpha")
        visible = cand[pre] if not isinstance(policy, Oracle) else cand
        alpha, diag = policy.choose(visible)
        row = cand.loc[np.isclose(cand["alpha"], alpha)].iloc[0]
        policy.observe(row)
        out.append(dict(row) | diag | dict(policy=policy.name))
    return pd.DataFrame(out)


def summarize(runs: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Good picks per minute (60 · held picks / Σ cycle time) and drop rate per policy, with bootstrap 95 % intervals
    over picks."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in runs.groupby("policy", sort=False):
        held = (~g["dropped"].astype(bool)).to_numpy(float)
        cyc = g["cycle_s"].to_numpy(float)
        gpm = lambda ix: 60 * held[ix].sum() / cyc[ix].sum()
        idx = rng.integers(0, len(g), (n_boot, len(g)))
        b_gpm = np.array([gpm(i) for i in idx])
        b_dr = 1 - held[idx].mean(axis=1)
        rows.append(dict(policy=name, picks=len(g), good_per_min=gpm(np.arange(len(g))),
                         gpm_lo=np.percentile(b_gpm, 2.5), gpm_hi=np.percentile(b_gpm, 97.5),
                         drop_rate=1 - held.mean(), dr_lo=np.percentile(b_dr, 2.5), dr_hi=np.percentile(b_dr, 97.5),
                         mean_alpha=g["alpha"].mean()))
    return pd.DataFrame(rows)
