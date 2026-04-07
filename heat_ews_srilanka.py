"""
Heat Impact-Based Early Warning System — Sri Lanka
===================================================
Adaptation of Economou et al. (2016) Bayesian decision framework
for temperature / heat index thresholds in Colombo District.

Original framework: Economou T, Stephenson DB, Rougier JC, Neal RA,
Mylne KR (2016) Proc. R. Soc. A 472: 20160295

Sectors addressed (Rogers 2026 prescriptive models):
  - Human Health (heat-health action plans, hospital surge)
  - Agriculture (heat stress, irrigation, crop calendars)
  - Outdoor Labour (WBGT / work-rest cycles)

Usage (demo):
    python heat_ews_srilanka.py

Usage (with your own data):
    from heat_ews_srilanka import HeatWarningSystem, HeatData
    data = HeatData(observed_categories=..., forecast_probs=...)
    sys  = HeatWarningSystem.for_sector('health')
    sys.fit(data)
    result = sys.issue_warning([0.10, 0.35, 0.40, 0.15])
    print(result)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import warnings

# ─────────────────────────────────────────────────────────────
# 1. HEAT SEVERITY CATEGORIES
#    Based on Sri Lanka DoM Heat Index thresholds (2018)
#    and UTCI thermal stress classes
# ─────────────────────────────────────────────────────────────

# Heat Index boundaries (°C) for categories 1→2→3→4
HI_THRESHOLDS  = [39, 46, 52]   # Normal | Caution | Extreme | Danger

# UTCI boundaries (°C) for categories 1→2→3→4
UTCI_THRESHOLDS = [26, 32, 38]  # No/Mod stress | Strong | Very Strong | Extreme+

# Tmax boundaries (°C)
TMAX_THRESHOLDS = [33, 36, 39]

STATE_LABELS = {
    1: "Normal      (HI < 39°C)",
    2: "Caution     (HI 39–45°C)",
    3: "Extreme     (HI 46–51°C)",
    4: "Danger      (HI ≥ 52°C)",
}

WARNING_LABELS = {
    1: "GREEN  — No Warning",
    2: "YELLOW — Heat Watch",
    3: "AMBER  — Heat Alert",
    4: "RED    — Heat Warning",
}


def categorise_hi(hi_value: float) -> int:
    """Convert a Heat Index value (°C) to a severity category 1–4."""
    if hi_value < HI_THRESHOLDS[0]:
        return 1
    elif hi_value < HI_THRESHOLDS[1]:
        return 2
    elif hi_value < HI_THRESHOLDS[2]:
        return 3
    else:
        return 4


def categorise_utci(utci_value: float) -> int:
    """Convert a UTCI value (°C) to a severity category 1–4."""
    if utci_value < UTCI_THRESHOLDS[0]:
        return 1
    elif utci_value < UTCI_THRESHOLDS[1]:
        return 2
    elif utci_value < UTCI_THRESHOLDS[2]:
        return 3
    else:
        return 4


# ─────────────────────────────────────────────────────────────
# 2. DATA STRUCTURE
# ─────────────────────────────────────────────────────────────

@dataclass
class HeatData:
    """Paired observed/forecast data for heat categories.

    Attributes
    ----------
    observed_categories : array of int, shape (n,)
        Observed heat severity category (1–4) for each day.
    forecast_probs : array of float, shape (n, 4)
        Forecast probability vector [p1, p2, p3, p4] for each day.
        Each row should sum to 1.
    dates : list of str, optional
        ISO date strings for each observation (for reporting).
    """
    observed_categories: np.ndarray
    forecast_probs: np.ndarray       # shape (n, 4)
    dates: Optional[List[str]] = None

    def __post_init__(self):
        self.observed_categories = np.asarray(self.observed_categories, dtype=int)
        self.forecast_probs = np.asarray(self.forecast_probs, dtype=float)
        assert self.observed_categories.ndim == 1
        assert self.forecast_probs.shape == (len(self.observed_categories), 4), \
            "forecast_probs must be shape (n, 4)"
        # Normalise rows
        row_sums = self.forecast_probs.sum(axis=1, keepdims=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.forecast_probs = np.where(row_sums > 0,
                                           self.forecast_probs / row_sums,
                                           0.25)

    @property
    def n(self):
        return len(self.observed_categories)

    @classmethod
    def from_hi_series(cls, observed_hi: list, forecast_hi_scenarios: list,
                       dates: Optional[List[str]] = None):
        """Convenience constructor from raw Heat Index values.

        Parameters
        ----------
        observed_hi : list of float
            Observed daily maximum Heat Index (°C).
        forecast_hi_scenarios : list of lists
            For each day, a list of ensemble member HI values.
        """
        obs_cats = np.array([categorise_hi(v) for v in observed_hi])
        n = len(observed_hi)
        probs = np.zeros((n, 4))
        for t, members in enumerate(forecast_hi_scenarios):
            for m in members:
                cat = categorise_hi(m) - 1
                probs[t, cat] += 1
            if probs[t].sum() > 0:
                probs[t] /= probs[t].sum()
        return cls(obs_cats, probs, dates)


# ─────────────────────────────────────────────────────────────
# 3. LOSS FUNCTIONS  (Economou et al. Eq. 4.6–4.7)
#    Adapted for three sector profiles
# ─────────────────────────────────────────────────────────────

SECTOR_LOSS_PARAMS = {
    # (c, l, gamma_c, gamma_l, gamma_d)
    # c  = max protection cost (mobilising early response)
    # l  = max potential damage if no action taken
    # Calibrated from Rogers (2026) sectoral value estimates
    "health":    (20,  150, 1.40, 0.80, 0.32),   # high damage asymmetry; lives at stake
    "agri":      (30,   90, 1.90, 0.50, 0.32),   # high cost of mobilising irrigation/advisory
    "labour":    (22,  120, 1.60, 0.70, 0.32),   # WBGT-based work-rest cycles
    "balanced":  (25,  100, 1.74, 0.60, 0.32),   # Economou et al. baseline
}


def build_loss_matrix(c=25, l=100, gc=1.74, gl=0.60, gd=0.32, I=4, J=4) -> np.ndarray:
    """Construct the I×J loss matrix L(a_i, x_j).

    Returns
    -------
    L : np.ndarray, shape (I, J)
        L[i,j] = loss when warning level i+1 issued, state j+1 occurs.
    """
    L = np.zeros((I, J))
    for i in range(I):
        a = i / (I - 1) if I > 1 else 0.0
        for j in range(J):
            x = j / (J - 1) if J > 1 else 0.0
            Ca  = c * (a ** gc) if a > 0 else 0.0
            LRa = l * (1 - (a ** gl)) if a > 0 else l
            Dx  = x ** gd if x > 0 else 0.0
            L[i, j] = Ca + LRa * Dx
    return np.round(L).astype(int)


# ─────────────────────────────────────────────────────────────
# 4. PROBABILITY MODELS
# ─────────────────────────────────────────────────────────────

class ClimatologicalModel:
    """Baseline: p(x|forecast) = p(x) — ignores forecast signal."""

    def __init__(self):
        self.p_x = np.ones(4) / 4

    def fit(self, data: HeatData):
        counts = np.array([(data.observed_categories == j).sum() for j in range(1, 5)],
                          dtype=float)
        self.p_x = counts / counts.sum()
        return self

    def predict(self, forecast_p: np.ndarray = None) -> np.ndarray:
        return self.p_x.copy()


class CalibratedModel:
    """Bayesian calibration: p(x|f) ∝ p(f|x) × p(x).

    Uses historical forecast–observation contingency to calibrate
    raw ensemble probabilities against actual heat outcomes.
    """

    def __init__(self):
        self.contingency = None   # 4×4 table: [forecast modal cat, obs cat]
        self.p_x = None

    def _modal(self, prob_vec: np.ndarray) -> int:
        """Return modal category (1-indexed) with add-one smoothing."""
        return int(np.argmax(prob_vec + 1e-6)) + 1

    def fit(self, data: HeatData):
        self.contingency = np.zeros((4, 4))
        modals = np.array([self._modal(data.forecast_probs[t]) for t in range(data.n)])
        for k in range(4):
            for j in range(4):
                self.contingency[k, j] = np.sum((modals == k+1) & (data.observed_categories == j+1))
        counts = np.array([(data.observed_categories == j).sum() for j in range(1,5)], dtype=float)
        self.p_x = counts / counts.sum()
        return self

    def predict(self, forecast_p: np.ndarray) -> np.ndarray:
        """Compute p(x|forecast) using Bayes' theorem with add-one smoothing."""
        k = self._modal(forecast_p) - 1
        p_f_given_x = np.array([
            (self.contingency[k, j] + 1) / (self.contingency[:, j].sum() + 4)
            for j in range(4)
        ])
        numerator = p_f_given_x * self.p_x
        denom = numerator.sum()
        return numerator / denom if denom > 0 else np.ones(4) / 4


class EnsembleModel:
    """Direct ensemble: uses forecast probabilities as-is."""

    def predict(self, forecast_p: np.ndarray) -> np.ndarray:
        p = np.asarray(forecast_p, dtype=float)
        return p / p.sum() if p.sum() > 0 else np.ones(4) / 4


# ─────────────────────────────────────────────────────────────
# 5. BAYES DECISION RULE  (Economou et al. Eq. 3.1)
# ─────────────────────────────────────────────────────────────

def bayes_warning(p_x: np.ndarray, loss_matrix: np.ndarray) -> int:
    """Return the warning level that minimises expected loss.

    a*(y) = argmin_a  Σ_x  L(a, x) · p(x|y)
    """
    expected = loss_matrix @ p_x
    return int(np.argmin(expected)) + 1


def expected_losses(p_x: np.ndarray, loss_matrix: np.ndarray) -> np.ndarray:
    return loss_matrix @ p_x


# ─────────────────────────────────────────────────────────────
# 6. VERIFICATION  (Brier score)
# ─────────────────────────────────────────────────────────────

def brier_scores(data: HeatData, prob_model) -> np.ndarray:
    """Compute Brier score B_j for each severity category."""
    p_all = np.vstack([prob_model.predict(data.forecast_probs[t]) for t in range(data.n)])
    scores = np.zeros(4)
    for j in range(4):
        indicator = (data.observed_categories == j + 1).astype(float)
        scores[j] = np.mean((p_all[:, j] - indicator) ** 2)
    return scores


def hit_rates(data: HeatData, loss_matrix: np.ndarray, prob_model) -> dict:
    """Compute warning contingency statistics."""
    issued   = np.array([bayes_warning(prob_model.predict(data.forecast_probs[t]), loss_matrix)
                         for t in range(data.n)])
    observed = data.observed_categories

    # High impact events = categories 3 or 4
    high_obs  = (observed >= 3)
    high_warn = (issued  >= 3)

    hits   = int(np.sum(high_obs  & high_warn))
    misses = int(np.sum(high_obs  & ~high_warn))
    fa     = int(np.sum(~high_obs & high_warn))
    cn     = int(np.sum(~high_obs & ~high_warn))

    pod  = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far  = fa   / (fa   + hits)   if (fa   + hits)   > 0 else 0.0
    csi  = hits / (hits + misses + fa) if (hits + misses + fa) > 0 else 0.0

    return {'hits': hits, 'misses': misses, 'false_alarms': fa, 'correct_negatives': cn,
            'POD': pod, 'FAR': far, 'CSI': csi}


# ─────────────────────────────────────────────────────────────
# 7. MAIN CLASS — FULL PIPELINE
# ─────────────────────────────────────────────────────────────

class HeatWarningSystem:
    """End-to-end Bayesian Heat Early Warning System for Sri Lanka.

    Parameters
    ----------
    sector : str
        One of 'health', 'agri', 'labour', 'balanced'.
    model_type : str
        'calibrated' (recommended), 'ensemble', or 'climatological'.
    """

    def __init__(self, sector: str = 'balanced', model_type: str = 'calibrated'):
        params = SECTOR_LOSS_PARAMS[sector]
        self.loss_matrix = build_loss_matrix(*params)
        self.model_type  = model_type
        self.sector      = sector
        if model_type == 'calibrated':
            self.model = CalibratedModel()
        elif model_type == 'ensemble':
            self.model = EnsembleModel()
        else:
            self.model = ClimatologicalModel()
        self._fitted = (model_type == 'ensemble')

    @classmethod
    def for_sector(cls, sector: str, model_type: str = 'calibrated') -> 'HeatWarningSystem':
        return cls(sector=sector, model_type=model_type)

    def fit(self, data: HeatData) -> 'HeatWarningSystem':
        """Train the probability model on historical data."""
        self.model.fit(data)
        self._fitted = True
        return self

    def issue_warning(self, forecast_p: list) -> dict:
        """Issue an optimal warning for a single forecast.

        Parameters
        ----------
        forecast_p : list of 4 floats
            Probability vector [p(cat1), p(cat2), p(cat3), p(cat4)].

        Returns
        -------
        dict with warning level, label, probabilities and expected losses.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit(data) before issuing warnings.")
        p = self.model.predict(np.asarray(forecast_p, dtype=float))
        el = expected_losses(p, self.loss_matrix)
        w  = int(np.argmin(el)) + 1
        return {
            'warning':          w,
            'warning_label':    WARNING_LABELS[w],
            'p_by_state':       dict(zip(STATE_LABELS.values(), p.tolist())),
            'expected_losses':  dict(zip(WARNING_LABELS.values(), el.tolist())),
            'min_expected_loss': float(el[w-1]),
        }

    def evaluate(self, data: HeatData) -> dict:
        """Evaluate the system over an evaluation dataset."""
        if not self._fitted:
            raise RuntimeError("Call .fit(data) before evaluating.")
        issued = np.array([bayes_warning(self.model.predict(data.forecast_probs[t]),
                                         self.loss_matrix) for t in range(data.n)])
        actual_losses = np.array([self.loss_matrix[issued[t]-1, data.observed_categories[t]-1]
                                   for t in range(data.n)])
        bs = brier_scores(data, self.model)
        hr = hit_rates(data, self.loss_matrix, self.model)
        return {
            'n_days':            data.n,
            'warning_issued':    issued.tolist(),
            'actual_losses':     actual_losses.tolist(),
            'total_loss':        float(actual_losses.sum()),
            'mean_daily_loss':   float(actual_losses.mean()),
            'brier_scores':      bs.tolist(),
            'mean_brier':        float(bs.mean()),
            **hr,
        }

    def print_loss_matrix(self):
        """Pretty-print the sector loss matrix."""
        print(f"\nLoss matrix — {self.sector} sector")
        print("-" * 60)
        header = f"{'':>12}" + "".join(f"{'Cat '+str(j+1):>14}" for j in range(4))
        print(header)
        for i in range(4):
            row = f"{WARNING_LABELS[i+1]:>12}" + "".join(f"{self.loss_matrix[i,j]:>14}" for j in range(4))
            print(row)
        print()


# ─────────────────────────────────────────────────────────────
# 8. SECTOR ACTION LOOKUP  (Rogers 2026)
# ─────────────────────────────────────────────────────────────

SECTOR_ACTIONS = {
    1: {  # Green
        'Health':    ['Routine surveillance. Ensure ORS stocks at district facilities.'],
        'Agri':      ['Normal field operations. Monitor soil moisture.'],
        'Labour':    ['Normal schedules. Reinforce hydration messaging.'],
        'Education': ['Normal activities. Ensure water access at schools.'],
        'DMC':       ['Routine monitoring. Heat desk on standby.'],
    },
    2: {  # Yellow — Watch
        'Health':    ['Activate heat-health surveillance.', 'Pre-position ORS and electrolyte supplies.',
                      'Brief PHIs in at-risk divisions.', 'Alert elderly care facilities.'],
        'Agri':      ['Advise fieldwork before 10:00 and after 16:00.',
                      'Check irrigation systems.', 'Increase water for livestock.'],
        'Labour':    ['Issue workplace heat advisory.', '10-min rest break per hour outdoors.',
                      'Ensure shaded rest areas and water at worksites.'],
        'Education': ['Limit outdoor sports to early morning.', 'Distribute hydration guidance.'],
        'DMC':       ['Activate district heat response plan.', 'Test SMS alert system.',
                      'Brief GN officers.'],
    },
    3: {  # Amber — Alert
        'Health':    ['Open emergency heat treatment pathways at Teaching Hospital.',
                      'Deploy mobile health teams to high-risk divisions.',
                      'Alert A&E departments for heat-stroke cases.',
                      'Broadcast public guidance (symptoms and first aid).'],
        'Agri':      ['Suspend non-essential fieldwork 10:00–16:00.',
                      'Issue crop stress advisory (paddy, vegetables).',
                      'Activate emergency irrigation if drought co-occurs.'],
        'Labour':    ['Mandate rest-in-shade 11:00–15:00.',
                      'Provide 1 L water per worker per 2 hrs.',
                      'Report heat illness cases to MOH office.'],
        'Education': ['Cancel afternoon outdoor activities.', 'Consider early school closure.',
                      'Increase water rations: 500 ml/child/hr.'],
        'DMC':       ['Convene inter-agency heat task force.',
                      'Activate EOC if >10 districts affected.',
                      'Coordinate media messaging (Sinhala, Tamil, English).'],
    },
    4: {  # Red — Warning
        'Health':    ['Declare public health emergency if fatalities confirmed.',
                      'Maximise ICU/A&E capacity at Teaching Hospitals.',
                      'Activate Ministry of Health Heat Emergency Protocol.'],
        'Agri':      ['Suspend all outdoor fieldwork until further notice.',
                      'Emergency water trucking for livestock.',
                      'Activate crop insurance notifications.'],
        'Labour':    ['ALL outdoor work SUSPENDED (except essential services).',
                      'LECO: monitor grid for cooling-load surge.',
                      'Enforce emergency heat regulations via Labour Ministry.'],
        'Education': ['Close schools in affected districts.',
                      'Arrange safe transport for boarding students.'],
        'DMC':       ['Activate National Emergency Operations Centre.',
                      'Brief Presidential Secretariat.',
                      'Notify international partners (IFRC/WHO) if mass casualties.'],
    },
}


def print_sector_actions(warning_level: int):
    """Print recommended actions for all sectors at a given warning level."""
    print(f"\n{'='*60}")
    print(f"  SECTOR ACTIONS — {WARNING_LABELS[warning_level]}")
    print(f"{'='*60}")
    for sector, actions in SECTOR_ACTIONS[warning_level].items():
        print(f"\n  [{sector}]")
        for a in actions:
            print(f"    • {a}")
    print()


# ─────────────────────────────────────────────────────────────
# 9. SYNTHETIC DEMO DATA  (Colombo climatological profile)
# ─────────────────────────────────────────────────────────────

def generate_demo_data(n_train: int = 180, n_test: int = 90,
                       seed: int = 42) -> tuple:
    """Generate synthetic heat data mimicking Colombo urban area.

    Climatological distribution skewed toward Cat 1–2 in cool season,
    Cat 3–4 in Mar–May inter-monsoonal peak.

    Returns
    -------
    train_data, test_data : HeatData
    """
    rng = np.random.default_rng(seed)

    def _gen(n, hot_season_frac=0.35):
        # Climatological mix: mostly Cat 1–2, with elevated Cat 3–4
        p_base = np.array([0.40, 0.35, 0.15, 0.10])
        obs = rng.choice([1, 2, 3, 4], size=n, p=p_base)

        probs = np.zeros((n, 4))
        for t in range(n):
            x = obs[t]
            # Forecast quality: moderate skill (ensemble)
            base = np.zeros(4)
            base[x-1] = 0.50              # correct category weight
            base += 0.15                  # spread across others
            noise = rng.dirichlet(base * 5 + 1)
            probs[t] = noise
        return obs, probs

    tr_obs, tr_probs = _gen(n_train)
    te_obs, te_probs = _gen(n_test, hot_season_frac=0.55)  # slightly hotter test period

    return (HeatData(tr_obs, tr_probs),
            HeatData(te_obs, te_probs))


# ─────────────────────────────────────────────────────────────
# 10. DEMO RUNNER
# ─────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("  HEAT IBF — BAYESIAN WARNING SYSTEM DEMO")
    print("  Colombo District, Sri Lanka")
    print("=" * 60)

    train_data, test_data = generate_demo_data()
    print(f"\nTraining period: {train_data.n} days")
    print(f"Evaluation period: {test_data.n} days")

    # ── Compare three model types ──────────────────────────
    print("\n── Model Comparison (Balanced sector loss) ──")
    for model_type in ['climatological', 'ensemble', 'calibrated']:
        sys = HeatWarningSystem(sector='balanced', model_type=model_type)
        if model_type != 'ensemble':
            sys.fit(train_data)
        results = sys.evaluate(test_data)
        print(f"\n  [{model_type.capitalize()}]")
        print(f"    Total loss:      {results['total_loss']:.0f}")
        print(f"    Mean daily loss: {results['mean_daily_loss']:.2f}")
        print(f"    Mean Brier score:{results['mean_brier']:.4f}")
        print(f"    POD (high heat): {results['POD']:.2f} | "
              f"FAR: {results['FAR']:.2f} | CSI: {results['CSI']:.2f}")

    # ── Sector-specific loss matrices ──────────────────────
    print("\n── Loss Matrices by Sector ──")
    for sector in ['health', 'agri', 'labour']:
        sys = HeatWarningSystem(sector=sector, model_type='calibrated')
        sys.fit(train_data)
        sys.print_loss_matrix()

    # ── Single forecast example ────────────────────────────
    print("\n── Single Forecast Decision ──")
    print("  Forecast probabilities: [Cat1=10%, Cat2=30%, Cat3=40%, Cat4=20%]")
    forecast = [0.10, 0.30, 0.40, 0.20]

    for sector in ['health', 'agri', 'labour', 'balanced']:
        sys = HeatWarningSystem(sector=sector, model_type='calibrated')
        sys.fit(train_data)
        result = sys.issue_warning(forecast)
        print(f"\n  Sector [{sector:10s}]: {result['warning_label']}")
        print(f"    Min expected loss: {result['min_expected_loss']:.1f}")

    # ── Show actions for the balanced-sector warning ───────
    sys = HeatWarningSystem(sector='balanced', model_type='calibrated')
    sys.fit(train_data)
    result = sys.issue_warning(forecast)
    print_sector_actions(result['warning'])

    # ── Threshold display ──────────────────────────────────
    print("\n── Current Heat Index Thresholds (Colombo Pilot) ──")
    print(f"  Heat Watch  (Yellow): Heat Index ≥ {HI_THRESHOLDS[0]}°C")
    print(f"  Heat Alert  (Amber):  Heat Index ≥ {HI_THRESHOLDS[1]}°C")
    print(f"  Heat Warning (Red):   Heat Index ≥ {HI_THRESHOLDS[2]}°C")
    print("\n  To refine thresholds for your data:")
    print("  1. Replace generate_demo_data() with real Colombo observations")
    print("  2. Call sys.fit(your_train_data)")
    print("  3. Evaluate with sys.evaluate(your_test_data)")
    print("  4. Adjust HI_THRESHOLDS until POD > 0.85 and FAR < 0.30")
    print()


if __name__ == "__main__":
    run_demo()
